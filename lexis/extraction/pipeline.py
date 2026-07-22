"""Ingest-time extraction: document -> structured legal facts.

This runs ONCE per document, at ingest, not per question. That is the whole
architectural point. Obligation registers, renewal calendars, and portfolio
risk views are reads over pre-extracted rows; computing them per query would
be both impossibly slow and non-reproducible.

It is also what makes answers fast. Generation dominates query latency, and
the way to spend less of it is to have already done the work.

EXTRACTOR_VERSION is the contract with the corpus: bump it whenever any
extractor's output could change, and `repository.stale_documents` will name
exactly which documents need re-running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store import repository
from . import dates, money, obligations

# Bump when any extractor changes. Composed from the parts so a change to one
# extractor cannot be forgotten here.
EXTRACTOR_VERSION = (
    obligations.EXTRACTOR_VERSION * 100
    + dates.EXTRACTOR_VERSION * 10
    + money.EXTRACTOR_VERSION
)


@dataclass
class ExtractionReport:
    document: str
    document_id: str
    clauses: int = 0
    obligations: int = 0
    key_dates: int = 0
    money_terms: int = 0
    extractor_version: int = EXTRACTOR_VERSION
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "document": self.document,
            "clauses": self.clauses,
            "obligations": self.obligations,
            "key_dates": self.key_dates,
            "money_terms": self.money_terms,
            "extractor_version": self.extractor_version,
            "warnings": self.warnings,
        }


def extract_document(
    filename: str,
    chunks,
    profile: dict,
    page_count: int,
    injection_flagged: bool = False,
    tenant_id: str = "default",
    matter_id: str | None = None,
) -> ExtractionReport:
    """Decompose one already-chunked document into the object model."""
    document_id = repository.upsert_document(
        filename=filename, profile=profile, page_count=page_count,
        clause_count=len(chunks), injection_flagged=injection_flagged,
        tenant_id=tenant_id, matter_id=matter_id,
    )
    clause_ids = repository.replace_clauses(document_id, chunks, tenant_id)

    all_obligations, all_dates, all_money = [], [], []
    for chunk in chunks:
        clause_id = clause_ids[chunk.id]
        # A clause flagged as carrying instructions aimed at the AI is not
        # trustworthy source material for a register a lawyer will act on.
        # It stays searchable and citable; it does not silently become a row
        # in the obligation table.
        if chunk.is_suspect:
            continue
        all_obligations.extend(obligations.extract(chunk.text, document_id, clause_id))
        all_dates.extend(dates.extract(chunk.text, document_id, clause_id))
        all_money.extend(money.extract(chunk.text, document_id, clause_id))

    repository.replace_obligations(document_id, all_obligations, EXTRACTOR_VERSION, tenant_id)
    repository.replace_key_dates(document_id, all_dates, EXTRACTOR_VERSION, tenant_id)
    repository.replace_money_terms(document_id, all_money, EXTRACTOR_VERSION, tenant_id)
    repository.mark_extracted(document_id, EXTRACTOR_VERSION)

    report = ExtractionReport(
        document=filename, document_id=document_id, clauses=len(chunks),
        obligations=len(all_obligations), key_dates=len(all_dates),
        money_terms=len(all_money),
    )
    suspect = sum(1 for c in chunks if c.is_suspect)
    if suspect:
        report.warnings.append(
            f"{suspect} clause(s) flagged as containing instructions aimed at the AI "
            f"system were excluded from extraction."
        )
    if not all_obligations:
        report.warnings.append(
            "No obligations extracted. Check that the document has recognisable "
            "clause structure — a scanned page with no text layer will look empty."
        )
    return report


def stale(tenant_id: str = "default") -> list[str]:
    """Documents whose facts were produced by an older extractor."""
    return repository.stale_documents(EXTRACTOR_VERSION, tenant_id)
