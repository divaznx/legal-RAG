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

from . import cache, embeddings, llm, rerank, vector_store
from .config import settings
from .llm import CitationReport
from .prompts import REFUSAL
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
    return sorted(chunks, key=lambda c: (c.rerank_score if c.rerank_score is not None else c.score), reverse=True)


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
    }


def _from_cache_entry(question: str, entry: dict) -> AskResult:
    return AskResult(
        question=question,
        answer=entry["answer"],
        chunks=[RetrievedChunk(**c) for c in entry.get("chunks", [])],
        citations=CitationReport(**entry.get("citations", {})),
        confidence=entry.get("confidence", "Low"),
        limitations=entry.get("limitations", []),
        cached=True,
    )


def ask_stream(question: str) -> Iterator[tuple[str, object]]:
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
    candidates = vector_store.hybrid_search(query_dense, query_sparse)
    timer.mark("hybrid_retrieve")

    # Refusal gate: the calibrated dense signal decides whether the corpus
    # is even in-domain for this question. The reranker below only orders.
    in_domain = any(c.dense_score >= settings.min_dense_score for c in candidates)

    reranked = _boost_fact_chunks(question, rerank.rerank_chunks(question, candidates))
    chunks = reranked[: settings.final_k]
    timer.mark("rerank")

    if not chunks or not in_domain:
        result = AskResult(
            question=question,
            answer=REFUSAL,
            citations=CitationReport(refusal=True),
            confidence="Low",
            limitations=["No retrieved chunk was semantically relevant to the question "
                         f"(best dense score below {settings.min_dense_score})."],
            timings_ms=timer.done(),
        )
        yield ("delta", result.answer)
        yield ("result", result)
        return

    parts: list[str] = []
    for delta in llm.stream_answer(question, chunks):
        parts.append(delta)
        yield ("delta", delta)
    answer = "".join(parts).strip()
    timer.mark("generate")

    citations = llm.verify_citations(answer, chunks)
    result = AskResult(
        question=question,
        answer=answer,
        chunks=chunks,
        citations=citations,
        confidence=_confidence(chunks, citations),
        limitations=_system_limitations(chunks, citations),
        timings_ms=timer.done(),
    )
    if citations.passed and not result.refused:
        cache.store(question, query_dense, _cache_payload(result))
    yield ("result", result)


def ask(question: str) -> AskResult:
    result: AskResult | None = None
    for kind, payload in ask_stream(question):
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
