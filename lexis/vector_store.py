"""Qdrant wrapper — the only module that talks to the vector DB.

v3 schema: named dense vector ("dense", cosine) + BM25 sparse vector
("bm25", IDF modifier), plus an indexed legal-structure payload
(clause_number, defined_terms, concepts, is_definition, xrefs).

`hybrid_search` runs both legs as server-side prefetches fused with
Reciprocal Rank Fusion — the standard production hybrid-retrieval pattern.

The structured lookups beside it (`fetch_clauses`, `fetch_definitions`,
`fetch_by_key`) are not similarity search at all: they are exact retrieval by
legal address. "Show me Clause 6.2" is a lookup, and answering it with the
nearest neighbour in embedding space is how a system ends up quoting Clause 5
under a Clause 6 heading.

Every read and write is scoped to the calling tenant's own collection
(`lexis/tenancy.py`), so cross-tenant leakage is not something a query can
get wrong.

Runs embedded (on-disk) by default; set QDRANT_URL to use a server.
"""

from __future__ import annotations

import atexit
import threading
import warnings
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from . import tenancy
from .chunking import Chunk
from .config import settings
from .embeddings import EMBEDDING_DIM

# Payload fields worth an index: every one of them is filtered on in the hot
# path by the legal retrieval stages.
_INDEXED_FIELDS = (
    ("document", qm.PayloadSchemaType.KEYWORD),
    ("clause_number", qm.PayloadSchemaType.KEYWORD),
    ("section", qm.PayloadSchemaType.KEYWORD),
    ("parent_section", qm.PayloadSchemaType.KEYWORD),
    ("version", qm.PayloadSchemaType.KEYWORD),
    ("defined_terms", qm.PayloadSchemaType.KEYWORD),
    ("concepts", qm.PayloadSchemaType.KEYWORD),
    ("xrefs", qm.PayloadSchemaType.KEYWORD),
    ("is_definition", qm.PayloadSchemaType.BOOL),
)


class StoreUnavailable(RuntimeError):
    """Raised with actionable guidance when the vector store cannot be opened."""


_client: QdrantClient | None = None
_client_lock = threading.Lock()


def client() -> QdrantClient:
    """The process-wide store handle, opened once.

    Double-checked locking rather than `lru_cache`, which is not atomic under
    contention: two threads missing the cache together both run the factory,
    and for the embedded store the second one hits the directory's exclusive
    lock and fails. That is exactly the shape of the API's startup — a
    background warmup thread opening the store while the first request opens
    it too — so it showed up as an intermittent 503 on the first call.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = _open_client()
        return _client


def _open_client() -> QdrantClient:
    if settings.qdrant_url:
        c = QdrantClient(url=settings.qdrant_url)
    else:
        try:
            c = QdrantClient(path=settings.qdrant_path)
        except Exception as exc:
            # The embedded store takes an exclusive lock on its directory.
            # Running the API and the Streamlit UI at once is the obvious
            # first thing a new operator does, and the raw error names a
            # storage folder rather than the cause.
            raise StoreUnavailable(
                f"Could not open the embedded vector store at "
                f"{settings.qdrant_path}: {exc}\n"
                "The embedded store allows a single process at a time. If the "
                "API and the UI are both running, stop one, or run a Qdrant "
                "server and set QDRANT_URL to use both together."
            ) from exc
    # close before interpreter teardown so the embedded store's __del__
    # doesn't fire after sys.meta_path is gone
    atexit.register(c.close)
    return c


# Collections already known to exist in this process. The existence check is
# a round trip, and every retrieval stage below needs the collection name, so
# it would otherwise run several times per question.
_ensured: set[str] = set()
_ensure_lock = threading.Lock()


def ensure_collection(name: str) -> str:
    if name in _ensured:
        return name
    # Serialized for the same reason `client()` is: two threads racing the
    # first question of a new tenant would both pass the existence check and
    # both call create_collection.
    with _ensure_lock:
        if name in _ensured:
            return name
        c = client()
        if not c.collection_exists(name):
            c.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE)
                },
                sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            # Indexes are an optimisation only — the embedded store ignores
            # them and filters by scan, which is correct but slower. Never let
            # index creation (or its warning) surface as an ingestion failure.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for field_name, schema in _INDEXED_FIELDS:
                    try:
                        c.create_payload_index(
                            collection_name=name, field_name=field_name, field_schema=schema
                        )
                    except Exception:
                        pass
        _ensured.add(name)
        return name


def _collection() -> str:
    """The current tenant's collection, created on first use.

    Every read and write below goes through this. Tenant isolation is the
    collection boundary itself, so there is no filter to forget.
    """
    return ensure_collection(tenancy.collection())


def drop_tenant(tenant: str) -> bool:
    """Delete a tenant's entire collection (offboarding). Returns False if it
    was never created. Does not touch the tenant's manifest or the audit log —
    the record of what was held has to outlive the data."""
    name = tenancy.collection(tenant)
    _ensured.discard(name)
    if not client().collection_exists(name):
        return False
    client().delete_collection(name)
    return True


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
    # --- legal structure (see chunking.Chunk) ---
    clause_number: str | None = None
    heading: str | None = None
    parent_section: str | None = None
    defined_terms: list[str] = field(default_factory=list)
    defined_term_labels: list[str] = field(default_factory=list)
    xrefs: list[str] = field(default_factory=list)
    xref_labels: list[str] = field(default_factory=list)
    incorporates: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    is_definition: bool = False
    is_suspect: bool = False
    # why this chunk is in the evidence set — shown to the user
    retrieval_reason: str = "semantic match"
    # strength of the cross-reference that pulled this chunk in (lower =
    # stronger; 99 = not reached by cross-reference). See legal.xref.force_rank.
    xref_force: int = 99
    # the referenced clause number that pulled this chunk in ("14" for 14.2)
    xref_target: str = ""

    @property
    def key(self) -> str:
        return f"{self.document}|{self.page}|{self.section}|{self.text[:40]}"

    @property
    def clause_key(self) -> str:
        return f"clause:{self.clause_number}".lower() if self.clause_number else ""


def _doc_filter(document: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="document", match=qm.MatchValue(value=document))])


def _documents_condition(documents: list[str] | None) -> list[qm.FieldCondition]:
    if not documents:
        return []
    return [qm.FieldCondition(key="document", match=qm.MatchAny(any=list(documents)))]


def upsert_chunks(
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[qm.SparseVector],
) -> None:
    client().upsert(
        collection_name=_collection(),
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
        collection_name=_collection(),
        points_selector=qm.FilterSelector(filter=_doc_filter(document)),
    )


def _from_payload(payload: dict, score: float = 0.0, dense: float = 0.0,
                  reason: str = "semantic match") -> RetrievedChunk:
    return RetrievedChunk(
        score=score,
        dense_score=dense,
        document=payload.get("document", "?"),
        page=int(payload.get("page", 0)),
        section=payload.get("section"),
        version=payload.get("version", "?"),
        ocr_confidence=float(payload.get("ocr_confidence", 1.0)),
        ocr_source=payload.get("ocr_source", "text"),
        text=payload.get("text", ""),
        fact_keys=payload.get("fact_keys") or [],
        clause_number=payload.get("clause_number"),
        heading=payload.get("heading"),
        parent_section=payload.get("parent_section"),
        defined_terms=payload.get("defined_terms") or [],
        defined_term_labels=payload.get("defined_term_labels") or [],
        xrefs=payload.get("xrefs") or [],
        xref_labels=payload.get("xref_labels") or [],
        incorporates=payload.get("incorporates") or [],
        concepts=payload.get("concepts") or [],
        is_definition=bool(payload.get("is_definition", False)),
        is_suspect=bool(payload.get("is_suspect", False)),
        retrieval_reason=reason,
    )


_RRF_K = 60  # standard reciprocal-rank-fusion constant


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: qm.SparseVector,
    limit: int | None = None,
    documents: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 legs fused client-side with RRF.

    Fusion is done here (not server-side FusionQuery) so every chunk keeps
    its dense cosine score — the only calibrated relevance signal in the
    stack, needed for the refusal gate. Reranker and RRF scores are
    rank-based/uncalibrated and must never gate refusals.

    `documents` restricts the search to the agreements chosen by the document
    resolution layer, which is what stops evidence being blended across
    unrelated contracts.
    """
    limit = limit or settings.candidate_k
    conditions = _documents_condition(documents)
    query_filter = qm.Filter(must=conditions) if conditions else None
    common = dict(
        collection_name=_collection(),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
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
        results.append(_from_payload(entry["payload"], round(entry["rrf"], 5), entry["dense"]))
    return results


def _scroll(conditions: list, limit: int, reason: str) -> list[RetrievedChunk]:
    if not conditions:
        return []
    points, _ = client().scroll(
        collection_name=_collection(),
        scroll_filter=qm.Filter(must=conditions),
        limit=limit,
        with_payload=True,
    )
    return [_from_payload(p.payload or {}, reason=reason) for p in points]


def expand_clause_numbers(
    clause_numbers: list[str],
    documents: list[str] | None = None,
) -> list[str]:
    """Add every sub-clause of the requested numbers.

    "Subject to Clause 10" refers to the whole of Clause 10, and its
    substance lives in 10.1 and 10.2 — the chunk numbered plain "10" is only
    the heading. Fetching the literal number alone retrieves a title and none
    of the terms it was cited for.
    """
    inventory: set[str] = set()
    for document in documents or []:
        inventory |= clause_inventory(document)
    if not inventory:
        return list(dict.fromkeys(clause_numbers))

    expanded: list[str] = []
    for number in clause_numbers:
        expanded.append(number)
        prefix = f"{number}."
        expanded.extend(sorted(c for c in inventory if c.startswith(prefix)))
    return list(dict.fromkeys(expanded))


def fetch_clauses(
    clause_numbers: list[str],
    documents: list[str] | None = None,
    limit: int = 12,
    expand: bool = True,
) -> list[RetrievedChunk]:
    """Exact clause lookup by legal address — no similarity involved."""
    if not clause_numbers:
        return []
    if expand:
        clause_numbers = expand_clause_numbers(clause_numbers, documents)
    conditions = [
        qm.FieldCondition(key="clause_number", match=qm.MatchAny(any=list(clause_numbers)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="exact clause reference")


def fetch_definitions(
    terms: list[str],
    documents: list[str] | None = None,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Chunks defining any of `terms` (normalized keys)."""
    if not terms:
        return []
    conditions = [
        qm.FieldCondition(key="defined_terms", match=qm.MatchAny(any=list(terms)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="defines a term used in the question")


def fetch_definition_sections(
    documents: list[str] | None = None,
    limit: int = 6,
) -> list[RetrievedChunk]:
    conditions = [
        qm.FieldCondition(key="is_definition", match=qm.MatchValue(value=True))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="definitions section")


def fetch_by_concepts(
    concepts: list[str],
    documents: list[str] | None = None,
    limit: int = 12,
) -> list[RetrievedChunk]:
    """Clauses tagged with any of the given legal concepts.

    The deterministic complement to the concept sub-queries: a clause indexed
    under `limitation_of_liability` is retrieved for a breach question even if
    its wording is a poor embedding match for the question.
    """
    if not concepts:
        return []
    conditions = [
        qm.FieldCondition(key="concepts", match=qm.MatchAny(any=list(concepts)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="legally related concept")


def fetch_referring_clauses(
    clause_keys: list[str],
    documents: list[str] | None = None,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Clauses that point AT the given clauses (inbound cross-references).

    Outbound references alone are not enough. "Clause 12.3 caps liability at
    150% of fees" reads as a complete answer until you find Clause 12.4: "the
    limit in Clause 12.3 does not apply to the Supplier's indemnity". The
    qualifying clause never mentions the subject matter and is a poor
    embedding match — the only reliable way to find it is to ask which clauses
    reference the one being read.
    """
    if not clause_keys:
        return []
    conditions = [
        qm.FieldCondition(key="xrefs", match=qm.MatchAny(any=list(clause_keys)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="qualifies a retrieved clause")


def fetch_siblings(
    parent_sections: list[str],
    documents: list[str] | None = None,
    limit: int = 12,
) -> list[RetrievedChunk]:
    """Every sub-clause sharing a parent with the given evidence.

    Sub-clauses of one clause are a single legal provision split for
    readability, and they routinely qualify each other: 12.3 sets a liability
    cap, 12.4 disapplies it for the indemnity. Retrieving 12.3 alone yields a
    confident, cited, wrong answer. Filtering on `parent_section` (not
    `section`, which holds the chunk's own label) is what makes the whole
    provision come back together.
    """
    if not parent_sections:
        return []
    conditions = [
        qm.FieldCondition(key="parent_section", match=qm.MatchAny(any=list(parent_sections)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="sub-clause of the same provision")


def fetch_sections(
    sections: list[str],
    documents: list[str] | None = None,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Every chunk of the named sections."""
    if not sections:
        return []
    conditions = [
        qm.FieldCondition(key="section", match=qm.MatchAny(any=list(sections)))
    ] + _documents_condition(documents)
    return _scroll(conditions, limit, reason="parent section context")


def clause_inventory(document: str) -> set[str]:
    """Clause numbers actually present in a document."""
    numbers: set[str] = set()
    offset = None
    while True:
        points, offset = client().scroll(
            collection_name=_collection(),
            scroll_filter=_doc_filter(document),
            limit=256,
            with_payload=["clause_number"],
            offset=offset,
        )
        for p in points:
            number = (p.payload or {}).get("clause_number")
            if number:
                numbers.add(str(number))
        if offset is None:
            break
    return numbers
