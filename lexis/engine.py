"""The answering engine.

    cache -> plan (resolve document, classify intent, expand concepts)
          -> legal retrieval -> stream -> verify

Pipeline (the legal-RAG playbook):
- semantic answer cache short-circuits repeated questions in milliseconds
- the legal intelligence layer resolves WHICH agreement and WHAT KIND of
  legal answer before any vector search happens
- retrieval fans out across concept probes, exact clause lookups, definitions
  and cross-references, then reranks and assembles under per-role budgets
- the answer streams token-by-token; citation verification runs on the
  completed text
- every stage is timed; timings ride along on the result

Two paths deliberately never reach the LLM: an ambiguous target document, and
a question citing a clause no ingested agreement contains. In both cases
generating anything at all would be worse than answering the question that
was actually asked.

Confidence is computed here, mechanically, from the calibrated dense signal,
OCR quality of the evidence, and the citation-verification verdict — never
from the model's self-assessment.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Iterator

from . import cache, embeddings, ingest, llm, rerank, retrieval, security, vector_store
from .config import settings
from .legal import planner
from .legal.planner import QueryPlan
from .llm import CitationReport
from .prompts import (
    REFUSAL,
    clarification_answer,
    finalize_answer,
    missing_clause_answer,
)
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
    # legal intelligence trace — auditable in the API/UI
    legal: dict = field(default_factory=dict)
    needs_clarification: bool = False

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


def _boost_fact_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
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
                break
    return chunks


def _system_limitations(
    chunks: list[RetrievedChunk],
    citations: CitationReport,
    plan: QueryPlan | None = None,
    notes: list[str] | None = None,
) -> list[str]:
    limitations: list[str] = list(notes or [])
    suspect = sorted({f"{c.document} {c.section or '-'}" for c in chunks if c.is_suspect})
    if suspect:
        limitations.append(
            "Evidence includes text that attempts to instruct the AI system rather "
            f"than state contractual terms ({', '.join(suspect)}). It was treated as "
            "document content only; review that passage manually."
        )
    low_ocr = sorted({(c.document, c.page) for c in chunks if c.ocr_confidence < 0.5})
    if low_ocr:
        pages = ", ".join(f"{d} p.{p}" for d, p in low_ocr)
        limitations.append(f"Low OCR confidence (possible scans): {pages}.")
    if plan is not None and plan.resolution.assumption:
        # Medium-confidence document resolution: the target agreement was
        # assumed, not certain — the user must see the assumption to override it.
        limitations.append(plan.resolution.assumption)
    if plan is not None and plan.resolution.superseded:
        limitations.append(
            "Superseded version(s) excluded from the evidence: "
            + ", ".join(plan.resolution.superseded) + "."
        )
    documents = {c.document for c in chunks}
    if len(documents) > 1:
        limitations.append(
            "Evidence spans multiple documents ("
            + ", ".join(sorted(documents))
            + "); each statement is attributed to its own agreement."
        )
    if citations.fabricated:
        limitations.append(
            f"{len(citations.fabricated)} citation(s) did not match any retrieved chunk "
            f"and should not be trusted: {'; '.join(citations.fabricated)}"
        )
    if citations.uncited_answer and not citations.refusal:
        limitations.append("The answer contains no verifiable citations.")
    return list(dict.fromkeys(limitations))


def _confidence(
    chunks: list[RetrievedChunk],
    citations: CitationReport,
    best_dense: float,
) -> str:
    """Computed from the calibrated dense signal + evidence quality —
    reranker scores are ordering-only and deliberately excluded."""
    if not chunks or not citations.passed or citations.refusal:
        return "Low"
    min_ocr = min(c.ocr_confidence for c in chunks)
    if best_dense >= 0.6 and min_ocr >= 0.5 and citations.verified == citations.total:
        return "High"
    if best_dense >= settings.min_dense_score:
        return "Medium"
    return "Low"


def _cache_payload(result: AskResult) -> dict:
    return {
        "answer": result.answer,
        "chunks": [asdict(c) for c in result.chunks],
        "citations": asdict(result.citations),
        "confidence": result.confidence,
        "limitations": result.limitations,
        "legal": result.legal,
    }


def _from_cache_entry(question: str, entry: dict) -> AskResult:
    return AskResult(
        question=question,
        answer=entry["answer"],
        chunks=[RetrievedChunk(**c) for c in entry.get("chunks", [])],
        citations=CitationReport(**entry.get("citations", {})),
        confidence=entry.get("confidence", "Low"),
        limitations=entry.get("limitations", []),
        legal=entry.get("legal", {}),
        cached=True,
    )


def _plain_retrieve(question: str) -> tuple[list[RetrievedChunk], float]:
    """Legacy single-query path, used when legal_intelligence is disabled."""
    query_dense = embeddings.embed_query(question)
    query_sparse = embeddings.embed_query_sparse(question)
    candidates = vector_store.hybrid_search(query_dense, query_sparse)
    best_dense = max((c.dense_score for c in candidates), default=0.0)
    reranked = _boost_fact_chunks(question, rerank.rerank_chunks(question, candidates))
    reranked.sort(key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score),
                  reverse=True)
    return reranked[: settings.final_k], best_dense


def _short_circuit(question: str, answer: str, plan: QueryPlan, timer: _Timer,
                   limitations: list[str]) -> AskResult:
    return AskResult(
        question=question,
        answer=answer,
        citations=CitationReport(refusal=True),
        confidence="Low" if plan.resolution.needs_clarification else "High",
        limitations=limitations,
        timings_ms=timer.done(),
        legal=plan.explain(),
        needs_clarification=plan.resolution.needs_clarification,
    )


def _validate(question: str) -> str | None:
    """Reject input that cannot produce a meaningful answer.

    Without this an empty question reaches document resolution, matches every
    agreement equally, and comes back as a clarification request listing the
    corpus — so a stray Enter in the client's UI looks like a considered
    response. An unbounded question is also a cost and latency vector: it is
    embedded, expanded into the sparse query, and prepended to every prompt.
    """
    stripped = question.strip()
    if not stripped:
        return "Please enter a question."
    if len(stripped) < settings.min_question_chars:
        return (f"That question is too short to answer reliably "
                f"(minimum {settings.min_question_chars} characters).")
    if len(stripped) > settings.max_question_chars:
        return (f"That question is too long ({len(stripped)} characters; "
                f"maximum {settings.max_question_chars}). Ask it in parts.")
    return None


def _rejected(question: str, message: str, timer: _Timer) -> AskResult:
    return AskResult(
        question=question,
        answer=f"## Answer\n{message}\n",
        citations=CitationReport(refusal=True),
        confidence="Low",
        limitations=[message],
        timings_ms=timer.done(),
    )


def ask_stream(question: str, history: list[str] | None = None) -> Iterator[tuple[str, object]]:
    """Yield ("delta", text) fragments as they stream, then ("result", AskResult).

    `history` is the user's prior questions in this conversation (oldest
    first). It never reaches the LLM — the engine stays single-turn — but the
    document-resolution layer uses it to keep an "active agreement" across
    follow-up questions that don't name one.
    """
    timer = _Timer()

    problem = _validate(question)
    if problem is not None:
        result = _rejected(question, problem, timer)
        yield ("delta", result.answer)
        yield ("result", result)
        return

    query_dense = embeddings.embed_query(question)
    timer.mark("embed_query")

    plan: QueryPlan | None = None
    notes: list[str] = []

    if settings.legal_intelligence:
        plan = planner.plan(
            question,
            ingest.load_profiles(),
            allow_clarification=settings.allow_clarification,
            history=history,
        )
        timer.mark("legal_planning")

    # Cache lookup happens AFTER planning so the key can include the resolved
    # agreement. A question-embedding-only key is document-blind: two
    # near-identical questions that resolve to different contracts would share
    # a cache entry, and a cached answer would be served from the wrong
    # agreement — reintroducing, through the cache, exactly the failure the
    # resolution layer exists to prevent.
    scope = plan.resolution.documents if plan else None
    hit = cache.lookup(question, query_dense, scope)
    if hit is not None:
        result = _from_cache_entry(question, hit)
        result.timings_ms = timer.done()
        yield ("delta", result.answer)
        yield ("result", result)
        return
    timer.mark("cache_lookup")

    if plan is not None:
        # Ambiguous target agreement -> ask, never guess.
        if plan.resolution.needs_clarification:
            result = _short_circuit(
                question, clarification_answer(plan.resolution), plan, timer,
                [f"{plan.resolution.reason} Clarification requested instead of answering."],
            )
            yield ("delta", result.answer)
            yield ("result", result)
            return

        # A cited clause that exists nowhere in the corpus is a factual answer,
        # not a retrieval problem.
        if plan.resolution.missing_clause:
            result = _short_circuit(
                question, missing_clause_answer(plan.resolution, question), plan, timer,
                [plan.resolution.reason],
            )
            yield ("delta", result.answer)
            yield ("result", result)
            return

        evidence = retrieval.retrieve(plan)
        chunks = _boost_fact_chunks(question, evidence.chunks)
        best_dense = evidence.primary_best_dense
        notes = evidence.notes
        timer.mark("legal_retrieve")
    else:
        chunks, best_dense = _plain_retrieve(question)
        timer.mark("retrieve")

    # Refusal gate: the calibrated dense signal of the PRIMARY query decides
    # whether the corpus is in-domain. Concept probes are intentionally broad
    # and never vote here.
    if not chunks or best_dense < settings.min_dense_score:
        result = AskResult(
            question=question,
            answer=REFUSAL,
            citations=CitationReport(refusal=True),
            confidence="Low",
            limitations=["No retrieved chunk was semantically relevant to the question "
                         f"(best dense score below {settings.min_dense_score})."],
            timings_ms=timer.done(),
            legal=plan.explain() if plan else {},
        )
        yield ("delta", result.answer)
        yield ("result", result)
        return

    parts: list[str] = []
    for delta in llm.stream_answer(question, chunks, plan, notes):
        parts.append(delta)
        yield ("delta", delta)
    answer = "".join(parts).strip()
    timer.mark("generate")

    # Remove any attacker-specified string the model was induced to emit, and
    # treat compliance as a compromised answer rather than a cosmetic issue:
    # if an injection reached the output at all, nothing else in it is
    # trustworthy enough to carry a confidence above Low.
    answer, injected = security.strip_injected_output(
        answer, [c.text for c in chunks if c.is_suspect]
    )

    citations = llm.verify_citations(answer, chunks)
    confidence = _confidence(chunks, citations, best_dense)
    limitations = _system_limitations(chunks, citations, plan, notes)
    if injected:
        confidence = "Low"
        limitations.insert(0, (
            "A passage in the source documents instructed the model to emit specific "
            f"text, and it complied. The inserted text ({', '.join(repr(i) for i in injected)}) "
            "was removed automatically. Treat this answer as unverified and review the "
            "source document."
        ))

    result = AskResult(
        question=question,
        # The stored/displayed answer carries the SYSTEM's confidence and
        # limitations, never the model's own account of them.
        answer=finalize_answer(answer, limitations, confidence),
        chunks=chunks,
        citations=citations,
        confidence=confidence,
        limitations=limitations,
        timings_ms=timer.done(),
        legal=plan.explain() if plan else {},
    )
    if citations.passed and not result.refused:
        cache.store(question, query_dense, _cache_payload(result), scope)
    yield ("result", result)


def ask(question: str, history: list[str] | None = None) -> AskResult:
    result: AskResult | None = None
    for kind, payload in ask_stream(question, history):
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
