"""Ingestion pipeline: parse -> redact -> chunk -> embed -> upsert.

A JSON manifest under data/manifest.json records every ingested document
(version, page count, chunk count, redaction summary) and backs the
document-listing endpoints.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import chunking, embeddings, parsing, redaction, vector_store
from .config import settings


@dataclass
class IngestReport:
    document: str
    version: str
    pages: int
    chunks: int
    redactions: dict[str, int]
    low_ocr_pages: list[int]
    ingested_at: str


def _manifest_path() -> Path:
    return settings.data_path / "manifest.json"


def load_manifest() -> dict[str, dict]:
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ingest_file(path: str | Path) -> IngestReport:
    path = Path(path)
    document = path.name
    version = parsing.detect_version(document)

    pages = parsing.parse(path)

    redaction_counts: Counter = Counter()
    for page in pages:
        result = redaction.redact(page.text)
        page.text = result.text
        redaction_counts.update(result.counts)

    chunks = chunking.chunk_pages(pages, document=document, version=version)
    if not chunks:
        raise ValueError(f"No extractable text in {document}")

    texts = [c.text for c in chunks]
    dense_vectors = embeddings.embed_passages(texts)
    sparse_vectors = embeddings.embed_passages_sparse(texts)

    # Re-ingest safely: clear any previous vectors for this document first.
    vector_store.delete_document(document)
    vector_store.upsert_chunks(chunks, dense_vectors, sparse_vectors)

    report = IngestReport(
        document=document,
        version=version,
        pages=len(pages),
        chunks=len(chunks),
        redactions=dict(redaction_counts),
        low_ocr_pages=[p.number for p in pages if p.ocr_confidence < 0.5],
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    manifest = load_manifest()
    manifest[document] = asdict(report)
    _save_manifest(manifest)
    return report


def delete_document(document: str) -> bool:
    manifest = load_manifest()
    if document not in manifest:
        return False
    vector_store.delete_document(document)
    del manifest[document]
    _save_manifest(manifest)
    return True
