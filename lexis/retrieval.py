"""Legal retrieval: executes a QueryPlan and assembles the evidence set.

    document-filtered hybrid search (one probe per legal concept)
      + exact clause lookup      (legal address, not similarity)
      + definition-first seeding (defined terms are variables, not English)
      + concept-filtered pull    (deterministic, from the indexed ontology)
      + cross-reference expansion("subject to Clause 6" -> fetch Clause 6)
      + parent/sibling context
      -> rerank -> budgeted assembly

Two properties are non-negotiable here, because both are the difference
between a useful answer and a dangerous one:

1. Every chunk carries `retrieval_reason`, so the lawyer sees WHY a clause is
   in the answer's evidence — "exact clause reference", "referenced by
   Clause 1", "legally related concept".
2. The refusal gate reads only the PRIMARY query's dense score. Concept
   probes are deliberately broad and will always match something in a
   contract corpus; letting them vote on relevance would dissolve the
   system's ability to say "not in these documents".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import embeddings, rerank, vector_store
from .config import settings
from .legal import xref as xref_mod
from .legal.entities import clause_sort_key
from .legal.planner import QueryPlan
from .vector_store import RetrievedChunk

_RRF_K = 60

# Assembly order = the order a lawyer reads evidence in: what the terms mean,
# then the clause asked about, then the clauses that qualify it, then the
# operative clauses, then surrounding context.
#
# This is also the specificity ranking used to assign a chunk to a budget
# bucket, and STRUCTURAL roles must outrank semantic ones. A clause found both
# by embedding similarity and by "it expressly disapplies the clause you are
# reading" is in the evidence set for the second reason; filing it under
# "semantic match" costs it its reserved slot and it drops out entirely.
_ROLE_ORDER = {
    "definition": 0,
    "exact_clause": 1,
    "sibling": 2,
    "xref": 3,
    "concept": 4,
    "primary": 5,
    "context": 6,
}

# Structural roles get reserved slots; the rest compete on rerank score.
# `concept` is structural: a clause is in that bucket because it covers a
# named link of the legal chain (cure -> termination -> remedies -> damages ->
# liability cap), which is a stronger claim on a slot than "the cross-encoder
# liked it". Left to compete on rerank, the chain loses every time — the
# clauses that answer "what happens if" barely mention the triggering event.
_STRUCTURAL_ROLES = ("exact_clause", "definition", "sibling", "xref", "concept")

# Language by which one clause switches off another. A clause containing it,
# and pointing at the clause being read, is the single most important thing to
# put in front of a lawyer — it is the difference between "liability is capped
# at 150% of fees" and "...except for the indemnity, which is uncapped".
_DISAPPLIES_RE = re.compile(
    r"(?i)\b(?:does not apply|shall not apply|do not apply|not\s+subject to|"
    r"notwithstanding|except (?:for|as|in)|save (?:for|that)|"
    r"nothing in this (?:agreement|clause|section)|without prejudice to)\b"
)


@dataclass
class Evidence:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    primary_best_dense: float = 0.0
    stats: dict = field(default_factory=dict)


def _weighted_fuse(runs: list[tuple[float, list[RetrievedChunk]]]) -> dict[str, RetrievedChunk]:
    """Weighted reciprocal-rank fusion across the plan's sub-queries."""
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for weight, hits in runs:
        for rank, chunk in enumerate(hits):
            key = chunk.key
            scores[key] = scores.get(key, 0.0) + weight / (_RRF_K + rank + 1)
            existing = fused.get(key)
            if existing is None:
                fused[key] = chunk
            else:
                # keep the best calibrated dense score seen for this chunk
                existing.dense_score = max(existing.dense_score, chunk.dense_score)
    for key, chunk in fused.items():
        chunk.score = round(scores[key], 6)
    return fused


def _merge(pool: dict[str, RetrievedChunk], roles: dict[str, str],
           chunks: list[RetrievedChunk], role: str, reason: str | None = None) -> None:
    """Add chunks to the candidate pool, keeping the strongest role claim.

    A clause that is both the exact clause asked about and a semantic match
    should be presented as the former — the more specific reason is the more
    useful one to show a lawyer. The reason is carried over from the incoming
    chunk (each structured lookup stamps its own) so the promoted role and the
    displayed reason can never disagree.
    """
    for chunk in chunks:
        key = chunk.key
        existing = pool.get(key)
        if reason:
            chunk.retrieval_reason = reason
        if existing is None:
            pool[key] = chunk
            roles[key] = role
        else:
            # A chunk reached by several routes keeps the strongest reference
            # force regardless of which role wins, so budget ordering sees it.
            if chunk.xref_force < existing.xref_force:
                existing.xref_force = chunk.xref_force
                existing.xref_target = chunk.xref_target
            if _ROLE_ORDER[role] < _ROLE_ORDER[roles[key]]:
                roles[key] = role
                existing.retrieval_reason = chunk.retrieval_reason


def retrieve(plan: QueryPlan) -> Evidence:
    documents = plan.resolution.documents or None
    policy = plan.policy
    stats: dict = {}
    notes: list[str] = []

    pool: dict[str, RetrievedChunk] = {}
    roles: dict[str, str] = {}

    # --- 1. multi-probe hybrid search --------------------------------------
    texts = [s.text for s in plan.subqueries]
    dense_vectors = embeddings.embed_queries(texts)
    runs: list[tuple[float, list[RetrievedChunk]]] = []
    primary_best_dense = 0.0

    for i, (sub, dense_vector) in enumerate(zip(plan.subqueries, dense_vectors)):
        # The primary probe searches with the fully expanded term bag on the
        # BM25 leg; concept probes use their own phrasing on both legs.
        sparse_text = plan.sparse_query if sub.purpose == "primary" else sub.text
        sparse_vector = embeddings.embed_query_sparse(sparse_text)
        hits = vector_store.hybrid_search(
            dense_vector, sparse_vector,
            limit=policy.candidate_k,
            documents=documents,
        )
        if sub.purpose == "primary":
            primary_best_dense = max([primary_best_dense, *(h.dense_score for h in hits)])
        runs.append((sub.weight, hits))

    fused = _weighted_fuse(runs)
    stats["candidates"] = len(fused)
    stats["probes"] = len(plan.subqueries)
    _merge(pool, roles, list(fused.values()), "primary")

    # Promote each concept probe's best hits out of the undifferentiated
    # semantic pool. Fusing every probe into one ranked list is what buries
    # the chain: the primary query contributes far more mass than any single
    # concept probe, so the clause that is the ONLY answer for "remedies"
    # ranks below three near-duplicates of the question.
    for sub, (_, hits) in zip(plan.subqueries, runs):
        if not sub.purpose.startswith("concept:"):
            continue
        name = sub.purpose.split(":", 1)[1]
        promoted = []
        for chunk in hits[:2]:
            pooled = pool.get(chunk.key)
            if pooled is None:
                continue
            pooled.retrieval_reason = f"covers '{name.replace('_', ' ')}' in the legal chain"
            promoted.append(pooled)
        _merge(pool, roles, promoted, "concept")

    # --- 2. exact clause lookup --------------------------------------------
    if plan.clause_targets:
        numbers = [ref.number for ref in plan.clause_targets]
        pinned = vector_store.fetch_clauses(numbers, documents)
        _merge(pool, roles, pinned, "exact_clause")
        stats["exact_clauses"] = len(pinned)
        if not pinned and not plan.resolution.missing_clause:
            labels = ", ".join(ref.label for ref in plan.clause_targets)
            notes.append(
                f"{labels} was referenced in the question but no clause with that "
                f"number was found in the selected document(s)."
            )

    # --- 3. deterministic concept pull -------------------------------------
    if policy.related_depth or policy.consequence_depth:
        concept_hits = vector_store.fetch_by_concepts(
            plan.expanded_concepts[:12], documents, limit=policy.candidate_k
        )
        _merge(pool, roles, concept_hits, "concept")
        stats["concept_matches"] = len(concept_hits)

    # --- 4. definition-first ------------------------------------------------
    if policy.definitions_first:
        terms = list(plan.definition_targets)
        if not terms:
            # Nothing was asked to be defined, so define the terms the leading
            # clauses USE. Seeding from the terms those clauses DEFINE instead
            # pulls in the title block on every query, because the preamble
            # parenthetically defines "Supplier" and "Customer" — party labels
            # a lawyer never needs spelled out mid-answer.
            top_text = " ".join(
                c.text.lower()
                for c in sorted(pool.values(), key=lambda c: c.score, reverse=True)[:6]
            )
            terms = [
                term for term in plan.known_terms
                if " " in term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", top_text)
            ]
        definitions = vector_store.fetch_definitions(
            list(dict.fromkeys(terms))[:8], documents
        )
        if not definitions and plan.definition_targets:
            definitions = vector_store.fetch_definition_sections(documents)
        _merge(pool, roles, definitions, "definition")
        stats["definitions"] = len(definitions)

    # --- 5. cross-reference expansion (both directions) --------------------
    absent_targets: list[str] = []
    if policy.follow_xrefs:
        # Only the strongest evidence gets its references followed. Widening
        # this spends the cross-reference budget on pointers out of clauses
        # the answer does not rely on.
        seeds = sorted(pool.values(), key=lambda c: c.score, reverse=True)[:4]

        # Outbound: clauses the evidence points to, strongest force first.
        # "Subject to Clause 3" changes what the host clause means; a passing
        # "as described in Clause 9" does not. When the cross-reference budget
        # is smaller than the number of references, the qualifying ones must
        # win the slots.
        force_by_target: dict[str, int] = {}
        for chunk in seeds:
            for ref in xref_mod.extract(chunk.text):
                if ref.target_kind not in ("clause", "section", "article", "paragraph"):
                    continue
                rank = xref_mod.force_rank(ref.force)
                if rank < force_by_target.get(ref.target, 99):
                    force_by_target[ref.target] = rank

        wanted = sorted(force_by_target, key=lambda t: force_by_target[t])
        if wanted:
            linked = vector_store.fetch_clauses(wanted, documents, limit=40)
            for chunk in linked:
                chunk.retrieval_reason = "referenced by a retrieved clause"
                # sub-clauses inherit their parent reference's force
                number = chunk.clause_number or ""
                matches = [(r, t) for t, r in force_by_target.items()
                           if number == t or number.startswith(f"{t}.")]
                if matches:
                    chunk.xref_force, chunk.xref_target = min(matches)
            _merge(pool, roles, linked, "xref")
            stats["xref_clauses"] = len(linked)
            # A target with no chunk anywhere in the document is genuinely
            # absent, as opposed to merely not selected — only the former is
            # worth warning a lawyer about. A reference to "Clause 3" is
            # satisfied by 3.1-3.3: the parent number itself is never indexed
            # when the clause is fully subdivided.
            found = {c.clause_number for c in linked if c.clause_number}
            absent_targets = [
                t for t in wanted
                if t not in found and not any(f.startswith(f"{t}.") for f in found)
            ]

        # Inbound: clauses that qualify or disapply the evidence.
        seed_keys = list(dict.fromkeys(c.clause_key for c in seeds if c.clause_key))
        if seed_keys:
            referring = vector_store.fetch_referring_clauses(seed_keys, documents)
            for chunk in referring:
                # An inbound reference is retrieved precisely because it acts
                # on the clause being read, so it ranks with conditions — and
                # ahead of everything when it switches that clause off.
                chunk.xref_force = 0 if _DISAPPLIES_RE.search(chunk.text) else 1
            _merge(pool, roles, referring, "xref")
            stats["referring_clauses"] = len(referring)

    # --- 6. parent / sibling context ---------------------------------------
    if policy.parent_context:
        # Complete the ONE provision the answer is most about, rather than
        # sampling sub-clauses from several. A partial provision is the
        # dangerous outcome: returning 11.2, 11.3 and 11.4 while dropping 11.1
        # answers "how much notice?" with every termination right except the
        # notice period itself.
        top = sorted(pool.values(), key=lambda c: c.score, reverse=True)
        parents = list(dict.fromkeys(c.parent_section for c in top[:4] if c.parent_section))
        if parents:
            siblings = vector_store.fetch_siblings(parents[:1], documents)
            _merge(pool, roles, siblings, "sibling")
            stats["siblings"] = len(siblings)

    # --- 7. rerank + budgeted assembly -------------------------------------
    candidates = list(pool.values())
    reranked = rerank.rerank_chunks(plan.question, candidates)
    order = {c.key: i for i, c in enumerate(reranked)}

    chosen = _assemble(candidates, roles, order, policy, plan.expanded_concepts)
    stats["selected"] = len(chosen)

    # Gaps are reported against the FINAL evidence set, so the answer only
    # warns about a missing exhibit when it actually relied on a clause that
    # incorporates one.
    if policy.follow_xrefs:
        notes.extend(_missing_reference_notes(chosen, absent_targets))
    notes.extend(_qualification_notes(chosen))

    return Evidence(chunks=chosen, notes=notes,
                    primary_best_dense=primary_best_dense, stats=stats)


def _missing_reference_notes(chosen: list[RetrievedChunk],
                             absent_targets: list[str]) -> list[str]:
    """Flag binding attachments and clauses the corpus does not contain.

    A contract that incorporates Exhibit A by reference is legally incomplete
    without it. Answering "the Provider shall deliver the services described
    in Exhibit A" as if it were a full answer is the kind of omission that
    gets relied on in advice.

    Only genuinely absent targets are reported. A cross-referenced clause that
    exists but lost the evidence budget is not a gap in the corpus, and saying
    so would train the reader to ignore these warnings.
    """
    notes: list[str] = []

    incorporated: list[str] = []
    for chunk in chosen:
        incorporated.extend(chunk.incorporates)

    if incorporated:
        unique = list(dict.fromkeys(incorporated))
        singular = len(unique) == 1
        notes.append(
            f"The retrieved clauses incorporate {', '.join(unique)} by reference. "
            f"{'It is' if singular else 'They are'} binding contract text but "
            f"{'was' if singular else 'were'} not found in the ingested corpus, "
            f"so the answer may be incomplete."
        )
    if absent_targets:
        notes.append(
            "Cross-referenced but absent from the selected document(s): "
            + ", ".join(f"Clause {t}" for t in dict.fromkeys(absent_targets)) + "."
        )
    return notes


def _spread_by_target(chunks: list[RetrievedChunk], order: dict[str, int]) -> list[RetrievedChunk]:
    """Round-robin cross-referenced clauses across the clauses they came from.

    "Subject to Clause 3 ... except as permitted in Clause 14" must surface
    both. Ranking the pooled sub-clauses flat lets Clause 3's three
    sub-clauses take every slot and Clause 14 — the one that answers the
    question — never appears.
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.xref_target or chunk.clause_number or "?", []).append(chunk)
    for group in groups.values():
        group.sort(key=lambda c: clause_sort_key(c.clause_number or ""))

    ordered_groups = sorted(
        groups.values(),
        key=lambda g: (min(c.xref_force for c in g), min(order.get(c.key, 10_000) for c in g)),
    )
    spread: list[RetrievedChunk] = []
    for depth in range(max((len(g) for g in ordered_groups), default=0)):
        for group in ordered_groups:
            if depth < len(group):
                spread.append(group[depth])
    return spread


def _spread_by_concept(chunks: list[RetrievedChunk], expanded: list[str],
                       order: dict[str, int]) -> list[RetrievedChunk]:
    """One clause per link of the legal chain before a second from any link.

    `expanded` arrives in chain order (breach, cure, termination, remedies,
    damages, liability cap...), so this also presents the consequence in the
    order it legally unfolds. Without the round-robin, one richly-tagged
    provision supplies every concept slot and the chain has a hole in it.
    """
    position = {name: i for i, name in enumerate(expanded)}

    groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        covered = min((position[c] for c in chunk.concepts if c in position), default=10_000)
        groups.setdefault(expanded[covered] if covered < len(expanded) else "?", []).append(chunk)

    ordered_groups = sorted(groups.items(), key=lambda kv: position.get(kv[0], 10_000))
    for _, group in ordered_groups:
        group.sort(key=lambda c: order.get(c.key, 10_000))

    spread: list[RetrievedChunk] = []
    for depth in range(max((len(g) for _, g in ordered_groups), default=0)):
        for _, group in ordered_groups:
            if depth < len(group):
                spread.append(group[depth])
    return spread


def _qualification_notes(chosen: list[RetrievedChunk]) -> list[str]:
    """Name the clauses in the evidence that switch other clauses off.

    Retrieving the carve-out is necessary but not sufficient: a model asked
    for "the liability cap" will quote the cap and stop, leaving the reader
    with a figure that does not apply to the largest exposure in the contract.
    Detecting the relationship mechanically — clause A contains disapplying
    language AND references clause B, both of which are in the evidence — puts
    it in Limitations whether or not the model noticed.
    """
    present = {c.clause_number: c for c in chosen if c.clause_number}
    notes: list[str] = []
    for chunk in chosen:
        if not chunk.clause_number or not _DISAPPLIES_RE.search(chunk.text):
            continue
        for key in chunk.xrefs:
            kind, _, target = key.partition(":")
            if kind not in ("clause", "section", "article", "paragraph"):
                continue
            if target in present and target != chunk.clause_number:
                notes.append(
                    f"{chunk.section or f'Clause {chunk.clause_number}'} qualifies or "
                    f"disapplies {present[target].section or f'Clause {target}'}: any "
                    f"statement of that clause's effect must be read subject to it."
                )
    return list(dict.fromkeys(notes))


def _assemble(candidates: list[RetrievedChunk], roles: dict[str, str],
              order: dict[str, int], policy,
              expanded_concepts: list[str]) -> list[RetrievedChunk]:
    """Fill the evidence set under per-role budgets.

    Definitions and cross-references get reserved slots rather than competing
    on rerank score, because both lose that competition and both change the
    meaning of the clauses that win it.
    """
    limit = min(policy.final_k, settings.max_evidence_chunks)

    buckets: dict[str, list[RetrievedChunk]] = {role: [] for role in _ROLE_ORDER}
    for chunk in candidates:
        buckets[roles[chunk.key]].append(chunk)
    for role in buckets:
        buckets[role].sort(key=lambda c: order.get(c.key, 10_000))
    # Sub-clauses of one provision read in document order — 11.1 before 11.2.
    # Relevance order would present a provision shuffled, which is how a
    # reader loses the thread of a conditional chain.
    buckets["sibling"].sort(key=lambda c: clause_sort_key(c.clause_number or ""))
    # Cross-references are spent by legal force, not by embedding similarity —
    # a cross-encoder has no way to know that "subject to" matters more than
    # "as described in" — and are spread across distinct targets so that one
    # heavily subdivided reference cannot consume the whole budget.
    buckets["xref"] = _spread_by_target(buckets["xref"], order)
    buckets["concept"] = _spread_by_concept(buckets["concept"], expanded_concepts, order)

    chosen: list[RetrievedChunk] = []
    seen: set[str] = set()

    def take(items: list[RetrievedChunk], count: int, ceiling: int) -> None:
        for chunk in items:
            if count <= 0 or len(chosen) >= ceiling:
                return
            if chunk.key in seen:
                continue
            seen.add(chunk.key)
            chosen.append(chunk)
            count -= 1

    # The reserved roles are filled FIRST. Filling them last — after the
    # semantic bucket, which is always the largest — leaves no slots and makes
    # the budgets silently do nothing. That is the difference between quoting
    # a liability cap and quoting the carve-out that disapplies it.
    #
    # They are also capped, so that a heavily cross-referenced clause cannot
    # crowd out the passages that actually answer the question: at least a
    # third of the evidence always comes from relevance ranking.
    structural_ceiling = max(limit - max(2, limit // 3), 1)

    # Order within the structural roles matters as much as the caps. Siblings
    # go last: they are the widest bucket (a whole provision) and, taken
    # first, they consume the ceiling and starve the cross-references — which
    # are the ones that reach OUTSIDE the provision to the clause that
    # qualifies it.
    concept_budget = (
        settings.concept_budget if policy.consequence_depth
        else max(1, settings.concept_budget // 2)
    )

    take(buckets["exact_clause"], limit, structural_ceiling)
    if policy.definitions_first:
        take(buckets["definition"], settings.definition_budget, structural_ceiling)
    take(buckets["xref"], settings.xref_budget, structural_ceiling)
    take(buckets["concept"], concept_budget, structural_ceiling)
    take(buckets["sibling"], settings.sibling_budget, structural_ceiling)
    take(buckets["primary"], limit, limit)
    # leftover structural evidence, then the best-reranked remainder
    for role in _STRUCTURAL_ROLES:
        take(buckets[role], limit, limit)
    take(buckets["context"], limit, limit)
    take(sorted(candidates, key=lambda c: order.get(c.key, 10_000)), limit, limit)

    # Group by role for presentation, but preserve the order each role was
    # taken in — re-sorting by rerank score here would scramble the sub-clause
    # sequence the sibling stage deliberately put into document order.
    position = {chunk.key: i for i, chunk in enumerate(chosen)}
    chosen.sort(key=lambda c: (_ROLE_ORDER[roles[c.key]], position[c.key]))
    return chosen
