"""Cross-encoder reranking — the production second stage.

Hybrid retrieval casts a wide net (candidate_k); the cross-encoder scores
each (question, chunk) pair jointly and keeps only the best final_k. Fewer,
better chunks reach the LLM, which both improves grounding and shrinks the
prompt — prefill time is the dominant LLM cost, so this is a latency win,
not just a quality one.

Scores are sigmoid-squashed logits (0..1) so thresholds are stable.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .config import settings
from .vector_store import RetrievedChunk


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.reranker_model)


def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks or not settings.use_reranker:
        return chunks
    scores = list(_model().rerank(question, [c.text for c in chunks]))
    for chunk, logit in zip(chunks, scores):
        chunk.rerank_score = 1.0 / (1.0 + math.exp(-float(logit)))
    return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)


def warm() -> None:
    if settings.use_reranker:
        list(_model().rerank("warmup", ["warmup"]))
