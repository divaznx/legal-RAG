"""Qdrant wrapper — the only module that talks to the vector DB.

v2 schema: named dense vector ("dense", cosine) + BM25 sparse vector
("bm25", IDF modifier). `hybrid_search` runs both legs as server-side
prefetches fused with Reciprocal Rank Fusion — the standard production
hybrid-retrieval pattern.

Runs embedded (on-disk) by default; set QDRANT_URL to use a server.
"""

from __future__ import annotations

import atexit
import re
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
    # --- Structure / hierarchy (all optional; legacy points default) ---
    index: int = -1                       # chunk position within the document
    parent_section: str | None = None     # one hierarchy level up ("Clause 4" for 4.2)
    heading: str | None = None            # full heading line
    is_table: bool = False
    defined_terms: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    parent_context: str | None = None     # parent-section text attached at retrieval
    # --- Document-level enrichment (Feature: Metadata Enrichment) ---
    document_type: str | None = None
    agreement_type: str | None = None
    jurisdiction: str | None = None
    effective_date: str | None = None
    confidentiality_level: str | None = None
    language: str | None = None
    # --- Retrieval explainability (Feature: Retrieval Explainability) ---
    bm25_score: float = 0.0               # raw BM25 leg score (uncalibrated)
    rank_dense: int | None = None         # 1-based rank in the dense leg
    rank_bm25: int | None = None          # 1-based rank in the BM25 leg
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None
    matched_on: list[str] = field(default_factory=list)  # why this chunk is here

    def diagnostics(self) -> dict:
        """Internal retrieval diagnostics — debug mode / developer logs only."""
        return {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "bm25_score": round(self.bm25_score, 4),
            "dense_score": round(self.dense_score, 4),
            "rrf_score": self.score,
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "rank_dense": self.rank_dense,
            "rank_bm25": self.rank_bm25,
            "rank_before_rerank": self.rank_before_rerank,
            "rank_after_rerank": self.rank_after_rerank,
            "matched_on": self.matched_on,
            "metadata_used": {
                k: v
                for k, v in {
                    "fact_keys": self.fact_keys,
                    "defined_terms": self.defined_terms,
                    "parent_section": self.parent_section,
                    "agreement_type": self.agreement_type,
                    "version": self.version,
                }.items()
                if v
            },
        }


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


def _search_filter(document: str | None) -> qm.Filter | None:
    """Pre-retrieval filtering: document scoping + tenant isolation.

    The tenant condition is only applied when settings.tenant is set —
    single-tenant deployments (tenant=None) apply no filter, so points
    indexed before the tenant field existed keep matching (Feature: Access
    Control Ready)."""
    must: list[qm.Condition] = []
    if document:
        must.append(qm.FieldCondition(key="document", match=qm.MatchValue(value=document)))
    if settings.tenant:
        must.append(qm.FieldCondition(key="tenant", match=qm.MatchValue(value=settings.tenant)))
    return qm.Filter(must=must) if must else None


def _from_payload(p: dict, **scores) -> RetrievedChunk:
    return RetrievedChunk(
        document=p.get("document", "?"),
        page=int(p.get("page", 0)),
        section=p.get("section"),
        version=p.get("version", "?"),
        ocr_confidence=float(p.get("ocr_confidence", 1.0)),
        ocr_source=p.get("ocr_source", "text"),
        text=p.get("text", ""),
        fact_keys=p.get("fact_keys") or [],
        index=int(p.get("index", -1)),
        parent_section=p.get("parent_section"),
        heading=p.get("heading"),
        is_table=bool(p.get("is_table", False)),
        defined_terms=p.get("defined_terms") or [],
        references=p.get("references") or [],
        document_type=p.get("document_type"),
        agreement_type=p.get("agreement_type"),
        jurisdiction=p.get("jurisdiction"),
        effective_date=p.get("effective_date"),
        confidentiality_level=p.get("confidentiality_level"),
        language=p.get("language"),
        **scores,
    )


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: qm.SparseVector,
    limit: int | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    document: str | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 legs fused client-side with RRF.

    Fusion is done here (not server-side FusionQuery) so every chunk keeps
    its dense cosine score — the only calibrated relevance signal in the
    stack, needed for the refusal gate. Reranker and RRF scores are
    rank-based/uncalibrated and must never gate refusals.

    Per-leg weights let query classification tilt the fusion (e.g. entity
    lookups weight BM25 up); both default to 1.0 — identical to unweighted
    RRF. `document` scopes retrieval to one document (comparison queries).
    """
    limit = limit or settings.candidate_k
    common = dict(
        collection_name=settings.qdrant_collection,
        limit=limit,
        with_payload=True,
        query_filter=_search_filter(document),
    )
    dense_hits = client().query_points(query=dense_vector, using="dense", **common).points
    sparse_hits = client().query_points(query=sparse_vector, using="bm25", **common).points

    fused: dict[str, dict] = {}
    for leg, weight, hits in (("dense", dense_weight, dense_hits), ("bm25", bm25_weight, sparse_hits)):
        for rank, hit in enumerate(hits):
            entry = fused.setdefault(
                str(hit.id),
                {"payload": hit.payload or {}, "rrf": 0.0, "dense": 0.0, "bm25": 0.0,
                 "rank_dense": None, "rank_bm25": None, "legs": []},
            )
            entry["rrf"] += weight / (_RRF_K + rank + 1)
            entry["legs"].append(leg)
            if leg == "dense":
                entry["dense"] = float(hit.score)
                entry["rank_dense"] = rank + 1
            else:
                entry["bm25"] = float(hit.score)
                entry["rank_bm25"] = rank + 1

    results = []
    for entry in sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:limit]:
        results.append(
            _from_payload(
                entry["payload"],
                score=round(entry["rrf"], 5),
                dense_score=entry["dense"],
                bm25_score=entry["bm25"],
                rank_dense=entry["rank_dense"],
                rank_bm25=entry["rank_bm25"],
                matched_on=entry["legs"],
            )
        )
    return results


def fetch_section_chunk(document: str, section: str) -> RetrievedChunk | None:
    """Earliest chunk for `section` of `document` (exact or numbering-prefix
    match). Powers parent-context attachment and graph expansion."""
    hits, _ = client().scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_search_filter(document),
        limit=256,
        with_payload=True,
        with_vectors=False,
    )
    target = section.strip().lower()
    best: dict | None = None
    for hit in hits:
        p = hit.payload or {}
        sec = (p.get("section") or "").strip().lower()
        if sec == target or sec.startswith(target + ".") or sec.startswith(target + " "):
            if best is None or int(p.get("index", 1 << 30)) < int(best.get("index", 1 << 30)):
                best = p
    if best is None:
        return None
    return _from_payload(best, score=0.0, dense_score=0.0)


_SECTION_NUM_RE = re.compile(r"(\d+(?:\.\d+)*)")


def section_number(section: str | None) -> str | None:
    """The numbering of a section label: "Clause 7.2 Termination" -> "7.2"."""
    if not section:
        return None
    m = _SECTION_NUM_RE.search(section)
    return m.group(1) if m else None


def number_matches(a: str | None, b: str | None) -> bool:
    """Numbering equality with dot-boundary prefix both ways, so a "Clause 7"
    lookup matches a "Clause 7.2" chunk and vice versa."""
    if not a or not b:
        return False
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


# clause/section/article are interchangeable drafting synonyms; "item"
# (SEC-filing numbering) is a distinct system and never cross-matches them.
_KW_GROUPS = {
    "clause": "csa", "section": "csa", "article": "csa",
    "art": "csa", "sec": "csa", "item": "item",
}


def section_keyword(section: str | None) -> str | None:
    if not section:
        return None
    first = section.strip().split()[0].rstrip(".").lower()
    return first if first in _KW_GROUPS else None


def keyword_compatible(query_kw: str | None, section_kw: str | None) -> bool:
    if not query_kw or not section_kw:
        return True  # unlabeled on either side: fall back to number matching
    return _KW_GROUPS.get(query_kw) == _KW_GROUPS.get(section_kw)


def fetch_clause_by_number(document: str, number: str, kw: str | None = None) -> RetrievedChunk | None:
    """Earliest chunk of `document` whose section numbering matches `number`
    (and whose section keyword is compatible with `kw` when given).

    Powers clause-injection: guarantees the actual clause reaches the
    candidate set when the user names it, even if hybrid search missed it."""
    hits, _ = client().scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_search_filter(document),
        limit=256,
        with_payload=True,
        with_vectors=False,
    )
    best: dict | None = None
    for hit in hits:
        p = hit.payload or {}
        if not keyword_compatible(kw, section_keyword(p.get("section"))):
            continue
        if number_matches(section_number(p.get("section")), number):
            if best is None or int(p.get("index", 1 << 30)) < int(best.get("index", 1 << 30)):
                best = p
    if best is None:
        return None
    return _from_payload(best, score=0.0, dense_score=0.0)


def all_document_chunks(document: str, limit: int = 1000) -> list[RetrievedChunk]:
    """Every chunk of a document in reading order — powers stratified
    document-overview retrieval (no vectors needed, scroll only)."""
    hits, _ = client().scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_search_filter(document),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    chunks = [_from_payload(h.payload or {}, score=0.0, dense_score=0.0) for h in hits]
    chunks.sort(key=lambda c: (c.page, c.index))
    return chunks


def fetch_section(document: str, section: str, max_chars: int | None = None) -> str | None:
    chunk = fetch_section_chunk(document, section)
    if chunk is None or not chunk.text:
        return None
    limit = max_chars or settings.parent_context_max_chars
    return chunk.text[:limit]
