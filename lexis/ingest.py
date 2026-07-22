"""Ingestion pipeline: parse -> redact -> chunk -> embed -> upsert.

A JSON manifest under data/manifest.json records every ingested document
(version, page count, chunk count, redaction summary) and backs the
document-listing endpoints.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import chunking, embeddings, parsing, redaction, security, vector_store
from .config import settings
from .extraction import pipeline as extraction_pipeline
from .legal import profile as legal_profile
from .legal.profile import DocumentProfile
from .store import repository as store_repository


@dataclass
class IngestReport:
    document: str
    version: str
    pages: int
    chunks: int
    redactions: dict[str, int]
    low_ocr_pages: list[int]
    ingested_at: str
    # Adversarial-content scan. Legal documents routinely arrive from the
    # other side of a deal, so this is recorded per document and surfaced to
    # whoever uploaded it.
    injection: dict = field(default_factory=dict)
    # Counts from the ingest-time extraction into the structured store.
    extraction: dict = field(default_factory=dict)
    # Document-level legal profile: doc type, parties, jurisdiction, clause
    # inventory, defined terms, version lineage. Read back on every question
    # by the document-resolution layer, so it is computed once here rather
    # than re-derived per query.
    profile: dict = field(default_factory=dict)


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

    # Scan the raw text before redaction rewrites it, so an injection hidden
    # inside a value that redaction would replace is still seen.
    injection = security.scan("\n\n".join(p.text for p in pages))

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
        profile=legal_profile.build(chunks, document=document, version=version).as_dict(),
        injection=injection.as_dict(),
    )

    # Decompose into the structured store. This is what turns the corpus from
    # something you can question into something you can query: obligation
    # registers, renewal calendars, and portfolio risk are reads over these
    # rows, and computing them per question would be both unusably slow and
    # non-reproducible.
    extraction = extraction_pipeline.extract_document(
        filename=document,
        chunks=chunks,
        profile=report.profile,
        page_count=len(pages),
        injection_flagged=injection.flagged,
    )
    report.extraction = extraction.as_dict()

    manifest = load_manifest()
    manifest[document] = asdict(report)
    _save_manifest(manifest)
    return report


def load_profiles() -> list[DocumentProfile]:
    """Legal profiles of every ingested document.

    Documents ingested before the profile layer existed have no stored
    profile; they are skipped rather than half-built, so resolution never
    reasons from an empty clause inventory (which would let it wrongly
    conclude that a referenced clause does not exist).
    """
    profiles: list[DocumentProfile] = []
    for entry in load_manifest().values():
        data = entry.get("profile")
        if data:
            profiles.append(legal_profile.from_dict(data))
    return profiles


def delete_document(document: str) -> bool:
    manifest = load_manifest()
    if document not in manifest:
        return False
    vector_store.delete_document(document)
    store_repository.delete_document(document)   # cascades to all extracted facts
    del manifest[document]
    _save_manifest(manifest)
    return True
