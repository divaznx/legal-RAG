"""Retrieval orchestration: multi-query fusion, comparison routing,
version awareness, and the bounded agentic retry ladder.

Sits between the engine and vector_store.hybrid_search. The single-query
path is unchanged from before (one dense + one BM25 leg, RRF-fused); this
module adds, all config-gated:

- decomposition sub-queries and rewrite variants retrieved in parallel and
  fused with RRF across query lists (Features: Query Rewriting/Decomposition)
- per-document retrieval for comparison queries (Feature: Query Classification)
- latest-version preference metadata (Feature: Version-aware Retrieval)
- an agentic retry ladder that rewrites/broadens before giving up
  (Feature: Agentic Retrieval) — bounded by settings.agentic_max_retries

Latency: every extra query costs one embedding (~10ms warm) plus two
sub-millisecond Qdrant legs; the ladder only runs when the initial gate
fails, so the happy path pays nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import embeddings, query as query_mod, vector_store
from .config import settings
from .query import QueryClass, RetrievalStrategy
from .vector_store import RetrievedChunk

_RRF_K = 60


@dataclass
class RetrievalDebug:
    """Internal diagnostics (Feature: Retrieval Explainability). Exposed only
    via debug mode / developer logs — never in normal user responses."""
    query_class: str = QueryClass.GENERAL.value
    sub_questions: list[str] = field(default_factory=list)
    variant_queries: list[str] = field(default_factory=list)
    per_document: bool = False
    agentic_attempts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)  # per-chunk diagnostics
    overview_target: str | None = None       # resolved document for overview queries
    overview_alternatives: list[str] = field(default_factory=list)  # ambiguous targets


def _chunk_key(c: RetrievedChunk) -> tuple:
    return (c.document, c.page, c.index, c.text[:80])


def _fuse_query_lists(result_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """RRF across per-query result lists. Keeps each chunk's best per-leg
    scores/ranks so diagnostics and the dense refusal gate stay meaningful."""
    fused: dict[tuple, RetrievedChunk] = {}
    scores: dict[tuple, float] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            kept = fused.get(key)
            if kept is None:
                fused[key] = chunk
            else:
                kept.dense_score = max(kept.dense_score, chunk.dense_score)
                kept.bm25_score = max(kept.bm25_score, chunk.bm25_score)
                for leg in chunk.matched_on:
                    if leg not in kept.matched_on:
                        kept.matched_on.append(leg)
    ranked = sorted(fused.items(), key=lambda kv: scores[kv[0]], reverse=True)
    out = []
    for key, chunk in ranked:
        chunk.score = round(scores[key], 5)
        out.append(chunk)
    return out


def _search_queries(
    queries: list[str],
    strategy: RetrievalStrategy,
    document: str | None = None,
    precomputed: dict[str, tuple[list[float], object]] | None = None,
) -> list[list[RetrievedChunk]]:
    precomputed = precomputed or {}
    result_lists = []
    for q in queries:
        if q in precomputed:
            dense, sparse = precomputed[q]
        else:
            dense = embeddings.embed_query(q)
            sparse = embeddings.embed_query_sparse(q)
        result_lists.append(
            vector_store.hybrid_search(
                dense, sparse,
                limit=strategy.resolved_candidate_k() if document is None
                else settings.comparison_per_doc_k,
                dense_weight=strategy.dense_weight,
                bm25_weight=strategy.bm25_weight,
                document=document,
            )
        )
    return result_lists


# --- Version awareness (Feature: Version-aware Retrieval) -------------------

_VERSION_TOKEN_RE = re.compile(r"[_\-]v\d+(?:\.\d+)*", re.IGNORECASE)


def base_document(document: str) -> str:
    """Filename with the version token stripped: MSA_Acme_v2.1.txt -> MSA_Acme."""
    return _VERSION_TOKEN_RE.sub("", Path(document).stem).strip("_- ")


def _version_tuple(version: str) -> tuple:
    try:
        return tuple(int(p) for p in version.split("."))
    except ValueError:
        return (0,)


def latest_version_documents() -> set[str]:
    """The newest document of each version family, per the ingest manifest."""
    from .ingest import load_manifest

    best: dict[str, tuple[tuple, str]] = {}
    for name, info in load_manifest().items():
        base = base_document(name)
        vt = _version_tuple(str(info.get("version", "0")))
        if base not in best or vt > best[base][0]:
            best[base] = (vt, name)
    return {name for _, name in best.values()}


def apply_version_preference(chunks: list[RetrievedChunk], qclass: QueryClass) -> None:
    """Bump latest-version chunks after reranking. Comparison queries are
    exempt — they need every version represented, not the newest preferred."""
    if not settings.prefer_latest_version or qclass == QueryClass.COMPARISON:
        return
    families = {base_document(c.document) for c in chunks}
    multi_version = {
        base for base in families
        if len({c.document for c in chunks if base_document(c.document) == base}) > 1
    }
    if not multi_version:
        return
    latest = latest_version_documents()
    for c in chunks:
        if c.document in latest and base_document(c.document) in multi_version:
            c.rerank_score = (c.rerank_score or 0.0) + settings.version_boost
            if "latest-version" not in c.matched_on:
                c.matched_on.append("latest-version")


def select_final(
    chunks: list[RetrievedChunk], strategy: RetrievalStrategy, per_document: bool
) -> list[RetrievedChunk]:
    """Final-k cut. For comparison queries, round-robin across documents so
    every document keeps representation instead of one dominating."""
    k = strategy.resolved_final_k()
    if not per_document:
        return chunks[:k]
    by_doc: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:  # chunks arrive rank-ordered
        by_doc.setdefault(c.document, []).append(c)
    out: list[RetrievedChunk] = []
    round_idx = 0
    while len(out) < max(k, len(by_doc)) and any(by_doc.values()):
        for doc in list(by_doc):
            if by_doc[doc]:
                out.append(by_doc[doc].pop(0))
        round_idx += 1
        if round_idx > k:
            break
    return out[: max(k, len(by_doc))]


# --- Document overview retrieval (Feature: Document Overview) ---------------

_STOP_TARGET_WORDS = frozenset(
    "the this that give show tell explain describe summarize summary about "
    "overview agreement contract document deal what walk through me and of".split()
)

# Section buckets an overview must cover, in presentation order. A chunk
# lands in a bucket when its section/heading/leading text matches.
_OVERVIEW_BUCKETS: list[tuple[str, re.Pattern]] = [
    ("parties", re.compile(r"(?i)\b(parties|by and between|recitals|whereas|preamble)\b")),
    ("definitions", re.compile(r'(?i)\bdefinitions?\b|"\s*[A-Z][^"]{2,40}"\s+means')),
    ("purpose", re.compile(r"(?i)\b(purpose|scope|services|the merger\b|transaction)\b")),
    ("financial", re.compile(
        r"(?i)\b(consideration|payment|fees?|price|retainer|per share|purchase price|compensation)\b")),
    ("obligations", re.compile(r"(?i)\b(covenants?|obligations?|shall deliver|responsibilit)\b")),
    ("termination", re.compile(r"(?i)\b(terminat\w+|survival)\b")),
    ("closing", re.compile(r"(?i)\b(closing|conditions precedent|effective time)\b")),
    ("confidentiality", re.compile(r"(?i)\b(confidential|non-?disclosure)\b")),
]


def target_documents(question: str) -> list[str]:
    """Documents the question names, matched on filename tokens ("twitter",
    "merger", "msa acme"). Empty when nothing matches."""
    from .ingest import load_manifest

    words = {
        w for w in re.findall(r"[a-z0-9]{3,}", question.lower())
        if w not in _STOP_TARGET_WORDS
    }
    # Score by how many query words a filename matches: "MSA Acme" must
    # resolve to MSA_Acme_* (2 words), not also to Form8K_Acme_* (1 word).
    scores: dict[str, int] = {}
    for document in load_manifest():
        condensed = re.sub(r"[^a-z0-9]", "", document.lower())
        hits = sum(1 for w in words if w in condensed)
        if hits:
            scores[document] = hits
    if not scores:
        return []
    best = max(scores.values())
    return sorted(d for d, hits in scores.items() if hits == best)


def resolve_overview_target(question: str) -> tuple[str | None, list[str]]:
    """(target document, ambiguous alternatives). Prefers the latest version
    within a named family; a bare "summarize this contract" resolves only
    when the corpus holds a single document family."""
    from .ingest import load_manifest

    named = target_documents(question)
    requested = query_mod.requested_version_in(question)
    if named:
        if requested:
            pinned = [d for d in named if _version_matches(d, requested)]
            named = pinned or named
        families = {base_document(d) for d in named}
        if len(families) > 1:
            return None, named
        latest = latest_version_documents()
        preferred = [d for d in named if d in latest] if not requested else named
        return (preferred or named)[-1], []
    manifest = sorted(load_manifest())
    families = {base_document(d) for d in manifest}
    if len(families) == 1 and manifest:
        latest = latest_version_documents()
        preferred = [d for d in manifest if d in latest]
        return (preferred or manifest)[-1], []
    return None, manifest


def _version_matches(document: str, requested: str) -> bool:
    from .parsing import detect_version

    v = detect_version(document)
    return v == requested or v.startswith(requested + ".") or requested.startswith(v + ".")


def overview_chunks(document: str) -> list[RetrievedChunk]:
    """Stratified evidence across `document`: one best chunk per overview
    bucket (parties, definitions, financial, termination, ...) plus the
    opening chunk, capped at settings.overview_final_k. Coverage over
    similarity — a summary answered from the single top-ranked chunk misses
    the document."""
    all_chunks = vector_store.all_document_chunks(document)
    if not all_chunks:
        return []
    selected: list[RetrievedChunk] = []
    seen: set[tuple] = set()

    def take(chunk: RetrievedChunk, bucket: str) -> None:
        key = _chunk_key(chunk)
        if key in seen or len(selected) >= settings.overview_final_k:
            return
        chunk.matched_on = [f"overview:{bucket}"]
        seen.add(key)
        selected.append(chunk)

    take(all_chunks[0], "opening")  # title / preamble / parties block
    for bucket, pattern in _OVERVIEW_BUCKETS:
        for chunk in all_chunks:
            probe = f"{chunk.section or ''}\n{chunk.heading or ''}\n{chunk.text[:300]}"
            if pattern.search(probe):
                take(chunk, bucket)
                break
    # Under-filled (sparse headings): spread evenly across the document span.
    if len(selected) < settings.overview_final_k and len(all_chunks) > len(selected):
        step = max(1, len(all_chunks) // (settings.overview_final_k - len(selected) + 1))
        for chunk in all_chunks[::step]:
            take(chunk, "spread")
    return selected


# --- Agentic retry ladder (Feature: Agentic Retrieval) ----------------------

_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or the to "
    "what when where which who why with".split()
)


def _keyword_broadening(question: str) -> str:
    """Strip stopwords down to content keywords — a BM25-shaped last resort."""
    words = [w for w in re.findall(r"[A-Za-z0-9.'-]+", question) if w.lower() not in _STOPWORDS]
    return " ".join(words)


def _agentic_attempts(question: str) -> list[tuple[str, list[str]]]:
    """(label, queries) attempts tried in order once the gate fails."""
    attempts: list[tuple[str, list[str]]] = []
    terms = query_mod.defined_terms_in(question)
    if terms:
        attempts.append(
            ("definitions-search", [f'"{t}" means' for t in terms[:3]])
        )
    broadened = _keyword_broadening(question)
    if broadened and broadened.lower() != question.lower():
        attempts.append(("synonym-broadening", [broadened]))
    return attempts


def _passes_gate(chunks: list[RetrievedChunk]) -> bool:
    return any(c.dense_score >= settings.min_dense_score for c in chunks)


# --- Main entry -------------------------------------------------------------

def retrieve(
    question: str,
    query_dense: list[float],
    query_sparse,
) -> tuple[list[RetrievedChunk], RetrievalStrategy, RetrievalDebug]:
    """Full candidate retrieval for a question. Returns (candidates,
    strategy, debug); candidates are RRF-ranked, gate-checked by the caller."""
    qclass = query_mod.classify(question)
    strategy = query_mod.strategy_for(qclass)
    debug = RetrievalDebug(query_class=qclass.value)

    # Document overview: stratified coverage of the target document, with the
    # semantic candidates fused in behind it (they carry the dense scores the
    # refusal gate and confidence read).
    if strategy.overview:
        target, alternatives = resolve_overview_target(question)
        debug.overview_target, debug.overview_alternatives = target, alternatives
        if target is not None:
            stratified = overview_chunks(target)
            debug.notes.append(
                f"overview: {len(stratified)} stratified chunk(s) from {target} "
                f"({', '.join(c.matched_on[0].split(':', 1)[1] for c in stratified)})"
            )
            semantic = _fuse_query_lists(
                _search_queries([question], strategy, None,
                                {question: (query_dense, query_sparse)})
            )
            present = {_chunk_key(c) for c in stratified}
            extras = [c for c in semantic if _chunk_key(c) not in present]
            return stratified + extras[: settings.final_k], strategy, debug

    sub_questions = query_mod.decompose(question)
    if len(sub_questions) > 1:
        debug.sub_questions = sub_questions

    variants = query_mod.rewrite(question)
    if strategy.definitions_first and not any('"' in v for v in variants):
        # Definition lookups probe the Definitions text shape first.
        variants = [f'"{t}" means' for t in query_mod.defined_terms_in(question)[:2]] + variants
    if qclass == QueryClass.CONSEQUENCE:
        # Breach questions retrieve the consequence provisions in parallel
        # with the user's question; RRF fusion merges the evidence. Generic
        # rewrites are REPLACED — entity/definition variants ("client name",
        # '"Agreement" means') pull preambles up and drown the provisions.
        variants = list(query_mod.CONSEQUENCE_PROBES)
        debug.notes.append("consequence-probes: retrieving termination/remedies/liability provisions")
    elif settings.concept_expansion:
        # Legal concept graph: related provisions (confidentiality->survival,
        # payment->late interest, ...) retrieved alongside the question.
        probes = query_mod.concept_probes(question)
        if probes:
            variants = variants + [p for p in probes if p not in variants]
            debug.notes.append(f"concept-expansion: {probes}")
    debug.variant_queries = variants

    queries = list(dict.fromkeys(sub_questions + variants))
    if question not in queries and not debug.sub_questions:
        queries.insert(0, question)
    precomputed = {question: (query_dense, query_sparse)}

    if strategy.per_document:
        from .ingest import load_manifest

        debug.per_document = True
        result_lists: list[list[RetrievedChunk]] = []
        for document in sorted(load_manifest()):
            result_lists.extend(_search_queries(queries, strategy, document, precomputed))
        candidates = _fuse_query_lists(result_lists)
    else:
        candidates = _fuse_query_lists(_search_queries(queries, strategy, None, precomputed))

    # Clause injection (Feature: Clause-aware Retrieval): when the user names
    # a clause, guarantee the actual clause chunk of every document is in the
    # candidate set — hybrid search can miss short clauses on paraphrase.
    clause_pairs = query_mod.clause_reference_pairs(question)
    if clause_pairs and settings.clause_injection:
        from .ingest import load_manifest

        candidates = candidates[: strategy.resolved_candidate_k()]
        present = {_chunk_key(c) for c in candidates}
        injected = 0
        # A question that names a document only gets that document's clause
        # injected — never a same-numbered clause from an unrelated agreement.
        named = target_documents(question)
        for document in named or sorted(load_manifest()):
            for kw, ref in clause_pairs[:3]:
                fetched = vector_store.fetch_clause_by_number(document, ref, kw=kw)
                if fetched is not None and _chunk_key(fetched) not in present:
                    fetched.matched_on = ["clause-injection"]
                    candidates.append(fetched)
                    present.add(_chunk_key(fetched))
                    injected += 1
        if injected:
            labels = [f"{(kw or 'clause').capitalize()} {num}" for kw, num in clause_pairs]
            debug.notes.append(f"clause-injection added {injected} chunk(s) for {labels}")

    # Consequence injection (Feature: Consequence Analysis): when a breach
    # question names a document, its consequence provisions (termination,
    # liability, remedies, ...) are fetched deterministically — hybrid fusion
    # over a mixed corpus can crowd them out of the candidate set entirely.
    if qclass == QueryClass.CONSEQUENCE:
        named = target_documents(question)
        if named:
            candidates = candidates[: strategy.resolved_candidate_k()]
            present = {_chunk_key(c) for c in candidates}
            added = 0
            for document in named:
                per_doc = 0
                for chunk in vector_store.all_document_chunks(document):
                    probe = f"{chunk.section or ''} {chunk.heading or ''} {chunk.text[:120]}"
                    if (
                        query_mod.CONSEQUENCE_SECTION_RE.search(probe)
                        and _chunk_key(chunk) not in present
                    ):
                        chunk.matched_on = ["consequence-injection"]
                        candidates.append(chunk)
                        present.add(_chunk_key(chunk))
                        added += 1
                        per_doc += 1
                        if per_doc >= 4:
                            break
            if added:
                debug.notes.append(
                    f"consequence-injection added {added} provision chunk(s) from {named}"
                )

    # Obligation injection (Feature: Legal Intelligence Layer): "the Client's
    # obligations" fetches sentences where the Client (or both parties)
    # shall/must from the named documents — obligations are defined by duty
    # language, and fusion over a mixed corpus can crowd those chunks out.
    if qclass == QueryClass.OBLIGATION:
        named = target_documents(question)
        subject = query_mod.obligation_subject(question)
        if named and subject:
            duty = query_mod.obligation_pattern(subject)
            candidates = candidates[: strategy.resolved_candidate_k()]
            present = {_chunk_key(c) for c in candidates}
            added = 0
            for document in named:
                per_doc = 0
                for chunk in vector_store.all_document_chunks(document):
                    if duty.search(chunk.text) and _chunk_key(chunk) not in present:
                        chunk.matched_on = ["obligation-injection"]
                        candidates.append(chunk)
                        present.add(_chunk_key(chunk))
                        added += 1
                        per_doc += 1
                        if per_doc >= 4:
                            break
            if added:
                debug.notes.append(
                    f"obligation-injection added {added} duty chunk(s) for '{subject}' from {named}"
                )

    # Agentic ladder: rewrite/broaden before refusing, bounded, only on miss.
    if not _passes_gate(candidates) and settings.agentic_retrieval:
        for label, retry_queries in _agentic_attempts(question)[: settings.agentic_max_retries]:
            debug.agentic_attempts.append(label)
            retry_lists = _search_queries(retry_queries, strategy, None, precomputed)
            candidates = _fuse_query_lists([candidates] + retry_lists)
            if _passes_gate(candidates):
                debug.notes.append(f"gate passed after {label}")
                break

    # Keep injected clause chunks even when the RRF-ranked head is full —
    # injection exists precisely for chunks hybrid search under-ranked.
    limit = strategy.resolved_candidate_k()
    head = candidates[:limit]
    tail = [
        c for c in candidates[limit:]
        if any(m.endswith("-injection") for m in c.matched_on)
    ]
    return head + tail, strategy, debug
