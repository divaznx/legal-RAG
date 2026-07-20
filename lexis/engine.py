"""The answering engine: cache -> hybrid retrieve -> rerank -> stream -> verify.

Latency design (the production legal-RAG playbook):
- semantic answer cache short-circuits repeated questions in milliseconds
- hybrid (BM25 + dense) retrieval fans out to candidate_k, fused with RRF
- a cross-encoder reranks candidates down to final_k — fewer chunks means a
  smaller prompt and faster LLM prefill
- the answer streams token-by-token; citation verification runs on the
  completed text
- every stage is timed; timings ride along on the result

Confidence is computed here, mechanically, from reranker scores, OCR
quality of the evidence, and the citation-verification verdict — never from
the model's self-assessment.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Iterator

from . import cache, embeddings, llm, packing, prompts, query as query_mod, rerank, retrieval, vector_store
from .config import settings
from .llm import CitationReport
from .prompts import REFUSAL
from .query import QueryClass
from .vector_store import RetrievedChunk


@dataclass
class AskResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: CitationReport = field(default_factory=CitationReport)
    confidence: str = "Low"
    limitations: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    query_class: str = ""
    regenerated: bool = False          # answer was regenerated after a failed verification
    debug: dict | None = None          # retrieval diagnostics (debug mode only)

    @property
    def refused(self) -> bool:
        return REFUSAL in self.answer or self.citations.refusal


class _Timer:
    def __init__(self) -> None:
        self._start = self._last = perf_counter()
        self.stages: dict[str, float] = {}

    def mark(self, stage: str) -> None:
        now = perf_counter()
        self.stages[stage] = round((now - self._last) * 1000, 1)
        self._last = now

    def done(self) -> dict[str, float]:
        self.stages["total"] = round((perf_counter() - self._start) * 1000, 1)
        return self.stages


def _boost_fact_chunks(question: str, chunks: list[RetrievedChunk]) -> None:
    """Deterministic metadata boost: a fact-block chunk whose key (e.g.
    "client", "contact email") appears in the question outranks generic
    prose.

    Needed because redaction replaces fact *values* with placeholders, and
    cross-encoders score placeholder-only key-value blocks near zero against
    fluent prose (measured: 0.0035 vs 0.98) — no amount of text massaging at
    rerank time fixes that, so the ranking signal must come from indexed
    structure instead.
    """
    q = question.lower()
    for chunk in chunks:
        for key in chunk.fact_keys:
            if re.search(rf"\b{re.escape(key)}\b", q):
                chunk.rerank_score = max(chunk.rerank_score or 0.0, 0.99)
                chunk.boosted = True
                if "fact-key" not in chunk.matched_on:
                    chunk.matched_on.append("fact-key")
                break


def _boost_definition_chunks(question: str, chunks: list[RetrievedChunk]) -> None:
    """Definition-aware boost (Feature: Definition-aware Retrieval): a chunk
    that *defines* a term used in the question outranks chunks that merely
    use the term — explicitly defined meanings take precedence over ordinary
    language. Deterministic, from indexed `defined_terms` metadata."""
    if not settings.definition_boost:
        return
    terms = [t.lower() for t in query_mod.defined_terms_in(question)]
    if not terms:
        return
    for chunk in chunks:
        defined = {t.lower() for t in chunk.defined_terms}
        in_definitions_section = "definition" in (chunk.section or "").lower()
        if any(t in defined for t in terms) or (
            in_definitions_section and any(t in chunk.text.lower() for t in terms)
        ):
            chunk.rerank_score = max(chunk.rerank_score or 0.0, 0.98)
            chunk.boosted = True
            if "definition-boost" not in chunk.matched_on:
                chunk.matched_on.append("definition-boost")


def _boost_clause_chunks(question: str, chunks: list[RetrievedChunk]) -> None:
    """Clause-aware boost (Feature: Clause-aware Retrieval): when the user
    names "Clause 7.2", the chunk whose section numbering IS 7.2 gets the top
    deterministic tier — a chunk that merely *references* the clause never
    outranks the actual clause (it gets no boost at all).

    An explicit version in the question ("v1.0") restricts the boost to
    matching documents, so asking about the old version cannot be hijacked
    by the latest-version preference."""
    if not settings.clause_boost:
        return
    pairs = query_mod.clause_reference_pairs(question)
    if not pairs:
        return
    requested_version = query_mod.requested_version_in(question)
    named = retrieval.target_documents(question)
    for chunk in chunks:
        if named and chunk.document not in named:
            continue  # user named a document — same-numbered clauses elsewhere get nothing
        num = vector_store.section_number(chunk.section)
        sec_kw = vector_store.section_keyword(chunk.section)
        if not any(
            vector_store.number_matches(num, ref_num)
            and vector_store.keyword_compatible(ref_kw, sec_kw)
            for ref_kw, ref_num in pairs
        ):
            continue
        if requested_version and not vector_store.number_matches(chunk.version, requested_version):
            continue
        chunk.rerank_score = max(chunk.rerank_score or 0.0, 0.995)
        chunk.boosted = True
        if "clause-exact" not in chunk.matched_on:
            chunk.matched_on.append("clause-exact")


def _boost_jurisdiction_chunks(question: str, chunks: list[RetrievedChunk]) -> None:
    """Jurisdiction preference (Feature: Legal Intelligence Layer): a question
    naming a jurisdiction ("under Delaware law") bumps chunks whose
    jurisdiction metadata matches, so legal systems are not silently mixed.
    No-op when the question names none."""
    if not settings.jurisdiction_preference:
        return
    q = question.lower()
    for chunk in chunks:
        jur = (chunk.jurisdiction or "").lower()
        if not jur:
            continue
        tokens = [t for t in re.findall(r"[a-z]{4,}", jur) if t not in ("state", "laws")]
        if tokens and any(t in q for t in tokens):
            chunk.rerank_score = (chunk.rerank_score or 0.0) + settings.jurisdiction_boost
            chunk.boosted = True
            if "jurisdiction-match" not in chunk.matched_on:
                chunk.matched_on.append("jurisdiction-match")


def _boost_consequence_chunks(question: str, chunks: list[RetrievedChunk]) -> None:
    """Pin provisions that define consequences above semantic noise for
    breach questions. Floor (0.9) sits below clause-exact (0.995) so an
    explicitly named clause still ranks first.

    Scoped to named documents: a 95-page merger agreement is saturated with
    termination/liability language and would flood the floor tier when the
    user asked about a different, named agreement."""
    named = retrieval.target_documents(question)
    for chunk in chunks:
        if named and chunk.document not in named:
            continue
        probe = f"{chunk.section or ''} {chunk.heading or ''} {chunk.text[:120]}"
        if query_mod.CONSEQUENCE_SECTION_RE.search(probe):
            chunk.rerank_score = max(chunk.rerank_score or 0.0, 0.9)
            chunk.boosted = True
            if "consequence-boost" not in chunk.matched_on:
                chunk.matched_on.append("consequence-boost")


def _rank_and_boost(
    question: str, qclass: QueryClass, candidates: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Cross-encoder rerank + deterministic boosts, with before/after ranks
    recorded for explainability. Boost order is deliberate: clause-exact
    (0.995) over fact-key (0.99) over definitions (0.98) over the
    latest-version bump (+0.05)."""
    for i, c in enumerate(candidates, start=1):
        c.rank_before_rerank = i
    reranked = rerank.rerank_chunks(question, candidates)
    # Stratified overview chunks are pinned above semantic noise: they were
    # selected for document *coverage*, which a cross-encoder scoring a broad
    # "explain this agreement" question cannot judge.
    for c in reranked:
        if any(m.startswith("overview:") or m == "obligation-injection" for m in c.matched_on):
            c.rerank_score = max(c.rerank_score or 0.0, 0.9)
    _boost_clause_chunks(question, reranked)
    if qclass == QueryClass.CONSEQUENCE:
        # Breach questions need provisions, not parties/definitions: the
        # fact-key boost ("Client:") and definition boost ("Agreement")
        # would drown Termination/Liability under preamble blocks.
        _boost_consequence_chunks(question, reranked)
    elif qclass == QueryClass.OBLIGATION:
        # Same reasoning: "the Client's obligations" asks for duty language,
        # not the preamble block naming the Client.
        pass
    else:
        _boost_fact_chunks(question, reranked)
        _boost_definition_chunks(question, reranked)
    _boost_jurisdiction_chunks(question, reranked)
    # Named-document preference: when the question names an agreement, its
    # own provisions outrank other documents that merely describe it (e.g.
    # an 8-K summarizing the MSA must not beat the MSA itself).
    named_docs = retrieval.target_documents(question)
    if named_docs:
        for c in reranked:
            if c.document in named_docs:
                c.rerank_score = (c.rerank_score or 0.0) + 0.05
                if "named-document" not in c.matched_on:
                    c.matched_on.append("named-document")
    # An explicit version request suspends the latest-version preference —
    # the user's choice of version always wins.
    if not query_mod.requested_version_in(question):
        retrieval.apply_version_preference(reranked, qclass)
    reranked = sorted(
        reranked,
        key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score),
        reverse=True,
    )
    for i, c in enumerate(reranked, start=1):
        c.rank_after_rerank = i
    return reranked


# Retrieval-internal labels must never surface in answers: the compact
# (Source: ...) citation is the only sanctioned way to reference evidence.
_CHUNK_BRACKET_RE = re.compile(r"\[(?:Parent context for )?Chunk\s+\d+[^\]]*\]\s*")
_CHUNK_BARE_RE = re.compile(r"\bChunks?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\b")


def _scrub_internal_labels(answer: str) -> str:
    """Strip "[Chunk N]" artifacts a small model may echo despite the prompt.

    The bracketed form is deleted outright; a bare "Chunk 2" mid-sentence is
    replaced with a neutral phrase so grammar survives. Runs before citation
    verification (which never relies on chunk labels)."""
    scrubbed = _CHUNK_BRACKET_RE.sub("", answer)
    scrubbed = _CHUNK_BARE_RE.sub("the retrieved evidence", scrubbed)
    return re.sub(r"[ \t]{2,}", " ", scrubbed)


def _mentions_any_document(question: str, families: set[str]) -> bool:
    condensed = re.sub(r"[^a-z0-9]", "", question.lower())
    return any(
        token and token in condensed
        for token in (re.sub(r"[^a-z0-9]", "", f.lower()) for f in families)
    )


def _clause_ambiguity(
    question: str, reranked: list[RetrievedChunk], selected: list[RetrievedChunk]
) -> tuple[list[str] | None, list[RetrievedChunk], list[str]]:
    """Ambiguity detection (Feature: Ambiguity Detection).

    Returns (clarification_documents, extra_chunks, notes):
    - clarification_documents: the requested clause exists in multiple
      UNRELATED documents and the question names none of them — ask the user
      to specify instead of guessing.
    - extra_chunks: the clause exists in multiple VERSIONS of the same
      agreement — ensure every version is represented so the answer compares
      them instead of silently selecting one.
    """
    if not settings.ambiguity_detection or not query_mod.clause_references_in(question):
        return None, [], []
    exact = [c for c in reranked if "clause-exact" in c.matched_on]
    if not exact:
        return None, [], []

    families: dict[str, list[RetrievedChunk]] = {}
    for c in exact:
        families.setdefault(retrieval.base_document(c.document), []).append(c)

    if len(families) > 1 and not _mentions_any_document(question, set(families)):
        # One label per agreement family, versions collapsed — two versions
        # of the same agreement are not two unrelated documents.
        labels = []
        for base, fam in sorted(families.items()):
            docs = sorted({c.document for c in fam})
            if len(docs) == 1:
                labels.append(docs[0])
            else:
                versions = ", ".join("v" + v for v in sorted({c.version for c in fam}))
                labels.append(f"{base} ({versions})")
        return labels, [], []

    extra: list[RetrievedChunk] = []
    notes: list[str] = []
    if query_mod.requested_version_in(question):
        return None, [], []  # the user pinned a version — nothing ambiguous
    for base, fam_chunks in families.items():
        versions = sorted({c.version for c in fam_chunks})
        if len(versions) < 2:
            continue
        selected_versions = {
            c.version for c in selected if retrieval.base_document(c.document) == base
        }
        for version in versions:
            if version not in selected_versions:
                top = next(c for c in fam_chunks if c.version == version)
                if top not in selected:
                    extra.append(top)
        notes.append(
            f"The requested clause exists in multiple versions of {base} "
            f"({', '.join('v' + v for v in versions)}); evidence from each version "
            "is included so differences are presented rather than one being chosen silently."
        )
    return None, extra, notes


def _system_limitations(chunks: list[RetrievedChunk], citations: CitationReport) -> list[str]:
    limitations: list[str] = []
    low_ocr = sorted({(c.document, c.page) for c in chunks if c.ocr_confidence < 0.5})
    if low_ocr:
        pages = ", ".join(f"{d} p.{p}" for d, p in low_ocr)
        limitations.append(f"Low OCR confidence (possible scans): {pages}.")
    if citations.fabricated:
        limitations.append(
            f"{len(citations.fabricated)} citation(s) did not match any retrieved chunk "
            f"and should not be trusted: {'; '.join(citations.fabricated)}"
        )
    if citations.uncited_answer and not citations.refusal:
        limitations.append("The answer contains no verifiable citations.")
    return limitations


def _confidence(chunks: list[RetrievedChunk], citations: CitationReport) -> str:
    """Computed from the calibrated dense signal + evidence quality —
    reranker scores are ordering-only and deliberately excluded."""
    if not chunks or not citations.passed or citations.refusal:
        return "Low"
    top_dense = max(c.dense_score for c in chunks)
    min_ocr = min(c.ocr_confidence for c in chunks)
    if top_dense >= 0.6 and min_ocr >= 0.5 and citations.verified == citations.total:
        return "High"
    if top_dense >= settings.min_dense_score:
        return "Medium"
    return "Low"


def _cache_payload(result: AskResult) -> dict:
    return {
        "answer": result.answer,
        "chunks": [asdict(c) for c in result.chunks],
        "citations": asdict(result.citations),
        "confidence": result.confidence,
        "limitations": result.limitations,
        "query_class": result.query_class,
    }


def _from_cache_entry(question: str, entry: dict) -> AskResult:
    return AskResult(
        question=question,
        answer=entry["answer"],
        chunks=[RetrievedChunk(**c) for c in entry.get("chunks", [])],
        citations=CitationReport(**entry.get("citations", {})),
        confidence=entry.get("confidence", "Low"),
        limitations=entry.get("limitations", []),
        query_class=entry.get("query_class", ""),
        cached=True,
    )


def retrieve_only(question: str) -> tuple[list[RetrievedChunk], "retrieval.RetrievalDebug", dict[str, float]]:
    """Retrieval + rerank without generation — powers the evaluation harness
    and debug tooling. Returns (ranked candidates, debug, timings)."""
    timer = _Timer()
    query_dense = embeddings.embed_query(question)
    query_sparse = embeddings.embed_query_sparse(question)
    timer.mark("embed_query")

    candidates, strategy, rdebug = retrieval.retrieve(question, query_dense, query_sparse)
    timer.mark("hybrid_retrieve")

    qclass = QueryClass(rdebug.query_class)
    reranked = _rank_and_boost(question, qclass, candidates)
    # Mirror the answer path's final selection (comparison round-robin etc.)
    # at the head of the list so evaluation measures what the LLM would see;
    # the remaining candidates follow for deeper-K metrics.
    final = retrieval.select_final(reranked, strategy, rdebug.per_document)
    if settings.clause_exact_only and query_mod.clause_references_in(question):
        exact = [c for c in reranked if "clause-exact" in c.matched_on]
        if exact:
            final = exact[: strategy.resolved_final_k()]
    rest = [c for c in reranked if c not in final]
    timer.mark("rerank")
    return final + rest, rdebug, timer.done()


def ask_stream(question: str, debug: bool = False) -> Iterator[tuple[str, object]]:
    """Yield ("delta", text) fragments as they stream, then ("result", AskResult)."""
    timer = _Timer()

    query_dense = embeddings.embed_query(question)
    timer.mark("embed_query")

    hit = cache.lookup(question, query_dense)
    if hit is not None:
        result = _from_cache_entry(question, hit)
        result.timings_ms = timer.done()
        yield ("delta", result.answer)
        yield ("result", result)
        return
    timer.mark("cache_lookup")

    query_sparse = embeddings.embed_query_sparse(question)
    # Orchestrated retrieval: classification-driven strategy, decomposition
    # sub-queries + rewrite variants fused with RRF, per-document routing for
    # comparisons, and the bounded agentic ladder on a gate miss.
    candidates, strategy, rdebug = retrieval.retrieve(question, query_dense, query_sparse)
    qclass = QueryClass(rdebug.query_class)
    timer.mark("hybrid_retrieve")

    # Refusal gate: the calibrated dense signal decides whether the corpus
    # is even in-domain for this question. The reranker below only orders.
    # An overview with an explicitly resolved target document is in-domain
    # by construction — its stratified chunks carry no dense scores.
    in_domain = (
        any(c.dense_score >= settings.min_dense_score for c in candidates)
        or rdebug.overview_target is not None
    )

    # Ambiguous overview target: ask which document, don't guess (pre-LLM).
    if (
        qclass == QueryClass.OVERVIEW
        and rdebug.overview_target is None
        and rdebug.overview_alternatives
    ):
        alternatives = rdebug.overview_alternatives
        result = AskResult(
            question=question,
            answer=(
                "The corpus contains multiple documents: "
                f"{'; '.join(alternatives)}. Please specify which one to "
                'summarize — for example: "Give me an overview of '
                f'{alternatives[0]}".'
            ),
            citations=CitationReport(refusal=True),
            confidence="Low",
            limitations=["Ambiguous overview target — asked for clarification instead of guessing."],
            timings_ms=timer.done(),
            query_class=qclass.value,
            debug=_debug_payload(rdebug, []) if (debug or settings.debug_retrieval) else None,
        )
        yield ("delta", result.answer)
        yield ("result", result)
        return

    reranked = _rank_and_boost(question, qclass, candidates)
    chunks = retrieval.select_final(reranked, strategy, rdebug.per_document)

    # Clause lookups with exact matches: only the actual clause (plus its
    # attached parent context) reaches the LLM — unrelated clauses, however
    # semantically similar, are excluded from packing. Consequence questions
    # are exempt: "what if I break Clause 3" needs the termination/remedies/
    # liability provisions alongside the clause, not the clause alone.
    if (
        settings.clause_exact_only
        and qclass != QueryClass.CONSEQUENCE
        and query_mod.clause_references_in(question)
    ):
        exact = [c for c in reranked if "clause-exact" in c.matched_on]
        if exact:
            chunks = exact[: strategy.resolved_final_k()]
            rdebug.notes.append(f"clause-exact-only: context narrowed to {len(chunks)} chunk(s)")
    timer.mark("rerank")

    if not chunks or not in_domain:
        limitations = ["No retrieved chunk was semantically relevant to the question "
                       f"(best dense score below {settings.min_dense_score})."]
        if rdebug.agentic_attempts:
            limitations.append(
                "Agentic retrieval retried with: "
                + ", ".join(rdebug.agentic_attempts) + " — still no relevant evidence."
            )
        result = AskResult(
            question=question,
            answer=REFUSAL,
            citations=CitationReport(refusal=True),
            confidence="Low",
            limitations=limitations,
            timings_ms=timer.done(),
            query_class=qclass.value,
            debug=_debug_payload(rdebug, chunks) if (debug or settings.debug_retrieval) else None,
        )
        yield ("delta", result.answer)
        yield ("result", result)
        return

    # Ambiguity detection: clarify across unrelated documents; represent
    # every version of the same agreement instead of silently picking one.
    clarification_docs, extra_chunks, ambiguity_notes = _clause_ambiguity(
        question, reranked, chunks
    )
    if clarification_docs:
        pairs = query_mod.clause_reference_pairs(question)
        label = " / ".join(f"{(kw or 'clause').capitalize()} {num}" for kw, num in pairs)
        example = clarification_docs[0].split(" (")[0]
        answer = (
            f"The requested {label} appears in multiple agreements: "
            f"{'; '.join(clarification_docs)}. Please specify which document "
            f'you mean — for example: "{label.split(" / ")[0].strip()} of '
            f'{example}" (and, where several versions exist, which version).'
        )
        rdebug.notes.append(f"ambiguity: {label} matched {len(clarification_docs)} agreement families")
        result = AskResult(
            question=question,
            answer=answer,
            citations=CitationReport(refusal=True),
            confidence="Low",
            limitations=[
                f"Ambiguous clause reference — {label} matched "
                f"{len(clarification_docs)} unrelated agreements; asked for "
                "clarification instead of guessing."
            ],
            timings_ms=timer.done(),
            query_class=qclass.value,
            debug=_debug_payload(rdebug, chunks) if (debug or settings.debug_retrieval) else None,
        )
        yield ("delta", result.answer)
        yield ("result", result)
        return
    if extra_chunks:
        chunks = chunks + extra_chunks
        rdebug.notes.append(f"ambiguity: added {len(extra_chunks)} version-representation chunk(s)")
    if rdebug.overview_target:
        ambiguity_notes.append(f"Overview generated from {rdebug.overview_target}.")

    # Parent-child attachment, cross-reference resolution, then context
    # packing: dedupe, merge overlaps, strip repeated headings, doc order.
    packing.attach_parents(chunks)
    chunks = packing.expand_references(chunks)
    resolved = [
        c.matched_on[0] for c in chunks
        if c.matched_on and c.matched_on[0].startswith("graph-reference:")
    ]
    if resolved:
        rdebug.notes.append(
            "cross-references resolved: "
            + ", ".join(r.removeprefix("graph-reference:") for r in resolved)
        )
    packed = packing.pack(chunks)
    timer.mark("pack")

    format_hint = None
    gen_max_tokens = None
    if qclass == QueryClass.OVERVIEW:
        format_hint = prompts.OVERVIEW_FORMAT
        gen_max_tokens = settings.overview_max_tokens
    elif qclass == QueryClass.CONSEQUENCE:
        format_hint = prompts.CONSEQUENCE_FORMAT

    parts: list[str] = []
    for delta in llm.stream_answer(
        question, packed, format_hint=format_hint, max_tokens=gen_max_tokens
    ):
        parts.append(delta)
        yield ("delta", delta)
    answer = _scrub_internal_labels("".join(parts).strip())
    timer.mark("generate")

    citations = llm.verify_citations(answer, chunks)
    regenerated = False
    if (
        not citations.passed
        and not citations.refusal
        and settings.regenerate_on_citation_failure
    ):
        # One-shot regeneration constrained to the verified citation set
        # (Feature: Answer Verification). Streamed so UIs stay live.
        regenerated = True
        reasons = citations.failure_reasons()
        rdebug.notes.append("citation verification failed: " + "; ".join(reasons))
        yield ("delta", "\n\n[Citation verification failed — regenerating from verified evidence]\n\n")
        parts = []
        note = llm.corrective_note_for(packed, reasons)
        for delta in llm.stream_answer(
            question, packed, corrective_note=note,
            format_hint=format_hint, max_tokens=gen_max_tokens,
        ):
            parts.append(delta)
            yield ("delta", delta)
        answer = _scrub_internal_labels("".join(parts).strip())
        citations = llm.verify_citations(answer, chunks)
        if not citations.passed and settings.strict_verification:
            answer = REFUSAL
            citations = CitationReport(refusal=True)
            yield ("delta", f"\n\n{REFUSAL}")
        timer.mark("regenerate")

    result = AskResult(
        question=question,
        answer=answer,
        chunks=chunks,
        citations=citations,
        confidence=_confidence(chunks, citations),
        limitations=_system_limitations(chunks, citations) + ambiguity_notes,
        timings_ms=timer.done(),
        query_class=qclass.value,
        regenerated=regenerated,
        debug=_debug_payload(rdebug, chunks) if (debug or settings.debug_retrieval) else None,
    )
    if citations.passed and not result.refused:
        cache.store(question, query_dense, _cache_payload(result))
    yield ("result", result)


def _debug_payload(rdebug: "retrieval.RetrievalDebug", chunks: list[RetrievedChunk]) -> dict:
    """Retrieval diagnostics for debug mode / developer logs — never part of
    the user-facing answer (Feature: Retrieval Explainability)."""
    return {
        "query_class": rdebug.query_class,
        "sub_questions": rdebug.sub_questions,
        "variant_queries": rdebug.variant_queries,
        "per_document": rdebug.per_document,
        "agentic_attempts": rdebug.agentic_attempts,
        "notes": rdebug.notes,
        "chunks": [c.diagnostics() for c in chunks],
    }


def ask(question: str, debug: bool = False) -> AskResult:
    result: AskResult | None = None
    for kind, payload in ask_stream(question, debug=debug):
        if kind == "result":
            result = payload  # type: ignore[assignment]
    assert result is not None
    return result


def warmup(include_llm: bool = True) -> None:
    """Load every model + the vector store off the hot path (call at startup)."""
    embeddings.warm()
    rerank.warm()
    vector_store.client()
    if include_llm:
        llm.warm()
