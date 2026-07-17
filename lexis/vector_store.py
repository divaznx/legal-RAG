"""Qdrant wrapper — the only module that talks to the vector DB.

v2 schema: named dense vector ("dense", cosine) + BM25 sparse vector
("bm25", IDF modifier). `hybrid_search` runs both legs as server-side
prefetches fused with Reciprocal Rank Fusion — the standard production
hybrid-retrieval pattern.

Runs embedded (on-disk) by default; set QDRANT_URL to use a server.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .chunking import Chunk
from .config import settings
from .embeddings import EMBEDDING_DIM


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    if settings.qdrant_url:
        c = QdrantClient(url=settings.qdrant_url)
    else:
        c = QdrantClient(path=settings.qdrant_path)
    if not c.collection_exists(settings.qdrant_collection):
        c.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={"dense": qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
    # close before interpreter teardown so the embedded store's __del__
    # doesn't fire after sys.meta_path is gone
    atexit.register(c.close)
    return c


@dataclass
class RetrievedChunk:
    score: float                      # RRF-fused retrieval score (rank-based)
    dense_score: float                # cosine similarity — the calibrated relevance signal
    document: str
    page: int
    section: str | None
    version: str
    ocr_confidence: float
    ocr_source: str
    text: str
    rerank_score: float | None = None  # cross-encoder score, 0..1 (set by rerank stage)
    fact_keys: list[str] = field(default_factory=list)  # keys of a key-value fact block
    boosted: bool = False  # fact-key metadata boost applied (see engine)


def _doc_filter(document: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="document", match=qm.MatchValue(value=document))])


def upsert_chunks(
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[qm.SparseVector],
) -> None:
    client().upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qm.PointStruct(
                id=chunk.id,
                vector={"dense": dense, "bm25": sparse},
                payload=chunk.payload(),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ],
    )


def delete_document(document: str) -> None:
    client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(filter=_doc_filter(document)),
    )


_RRF_K = 60  # standard reciprocal-rank-fusion constant


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: qm.SparseVector,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 legs fused client-side with RRF.

    Fusion is done here (not server-side FusionQuery) so every chunk keeps
    its dense cosine score — the only calibrated relevance signal in the
    stack, needed for the refusal gate. Reranker and RRF scores are
    rank-based/uncalibrated and must never gate refusals.
    """
    limit = limit or settings.candidate_k
    common = dict(collection_name=settings.qdrant_collection, limit=limit, with_payload=True)
    dense_hits = client().query_points(query=dense_vector, using="dense", **common).points
    sparse_hits = client().query_points(query=sparse_vector, using="bm25", **common).points

    fused: dict[str, dict] = {}
    for leg_rank, hits in (("dense", dense_hits), ("bm25", sparse_hits)):
        for rank, hit in enumerate(hits):
            entry = fused.setdefault(
                str(hit.id), {"payload": hit.payload or {}, "rrf": 0.0, "dense": 0.0}
            )
            entry["rrf"] += 1.0 / (_RRF_K + rank + 1)
            if leg_rank == "dense":
                entry["dense"] = float(hit.score)

    results = []
    for entry in sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:limit]:
        p = entry["payload"]
        results.append(
            RetrievedChunk(
                score=round(entry["rrf"], 5),
                dense_score=entry["dense"],
                document=p.get("document", "?"),
                page=int(p.get("page", 0)),
                section=p.get("section"),
                version=p.get("version", "?"),
                ocr_confidence=float(p.get("ocr_confidence", 1.0)),
                ocr_source=p.get("ocr_source", "text"),
                text=p.get("text", ""),
                fact_keys=p.get("fact_keys") or [],
            )
        )
    return results
