"""Embedding wrappers around fastembed (ONNX — no torch required).

Dense: bge uses asymmetric prefixes — `query_embed` for questions,
`passage_embed` for chunks; fastembed applies them automatically.

Sparse: BM25 term vectors for the hybrid (keyword + semantic) retrieval leg;
the collection applies IDF server-side (Modifier.IDF).

Both models are process-lifetime singletons so repeated queries pay zero
model-load cost; call `warm()` at startup to move the load off the first
request.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models as qm

from .config import settings

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


@lru_cache(maxsize=1)
def _dense() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def _sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _dense().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_dense().query_embed(text))).tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Batch form for multi-probe retrieval — the legal planner issues one
    probe per expanded concept, and batching keeps that a single ONNX call."""
    if not texts:
        return []
    return [v.tolist() for v in _dense().query_embed(texts)]


def _to_sparse_vector(embedding) -> qm.SparseVector:
    return qm.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def embed_passages_sparse(texts: list[str]) -> list[qm.SparseVector]:
    return [_to_sparse_vector(e) for e in _sparse().passage_embed(texts)]


def embed_query_sparse(text: str) -> qm.SparseVector:
    return _to_sparse_vector(next(iter(_sparse().query_embed(text))))


def warm() -> None:
    embed_query("warmup")
    embed_query_sparse("warmup")
