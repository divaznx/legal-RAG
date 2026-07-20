"""Document-level metadata enrichment (Feature: Metadata Enrichment).

Heuristic extraction of legal metadata from a parsed document's text —
agreement type, jurisdiction, effective date, confidentiality marking,
language. Runs once per document at ingest; the resulting DocMeta is
stamped onto every chunk payload so it is filterable/citable at retrieval.

All fields are optional and default to None/"unspecified" — chunks indexed
before this module existed simply lack the payload keys, and RetrievedChunk
defaults cover them (backward compatible).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Ordered: first match wins, so more specific phrases go first.
_AGREEMENT_TYPES: list[tuple[str, str]] = [
    (r"amended and restated master services agreement", "MSA (Amended and Restated)"),
    (r"master services agreement", "MSA"),
    (r"non[- ]disclosure agreement|confidentiality agreement", "NDA"),
    (r"statement of work", "SOW"),
    (r"service order", "Service Order"),
    (r"data processing (?:agreement|addendum)", "DPA"),
    (r"employment agreement", "Employment Agreement"),
    (r"license agreement", "License Agreement"),
    (r"amendment", "Amendment"),
]

_JURISDICTION_RE = re.compile(
    r"governed by(?: and construed in accordance with)? the laws of (?:the )?"
    r"(?P<j>[A-Z][A-Za-z .,]{2,60}?)(?:[.,;\n]|$)"
)

_EFFECTIVE_DATE_RES = [
    re.compile(r"Effective Date[:\s]+(?P<d>[A-Za-z0-9, /.\-]{6,30}?)(?:[.;\n]|$)"),
    re.compile(r"effective as of (?P<d>[A-Za-z0-9, /.\-]{6,30}?)(?:[.;\n]|$)"),
    re.compile(r"dated (?:as of )?(?P<d>[A-Za-z]+ \d{1,2},? \d{4})"),
]

_CONFIDENTIAL_RE = re.compile(
    r"^\s*(STRICTLY )?(CONFIDENTIAL|PRIVILEGED( AND CONFIDENTIAL)?)\s*$", re.MULTILINE
)


@dataclass
class DocMeta:
    document_type: str = "unknown"        # file format: pdf | docx | txt | md
    agreement_type: str | None = None     # MSA | NDA | SOW | ...
    jurisdiction: str | None = None       # "the State of Delaware" -> "State of Delaware"
    effective_date: str | None = None     # verbatim as found; never inferred
    confidentiality_level: str = "unspecified"
    language: str = "en"                  # heuristic; corpus is English-first

    # Access-control readiness (Feature: Access Control Ready). Defaults keep
    # single-tenant deployments untouched: no tenant, open permissions.
    tenant: str | None = None
    client: str | None = None
    matter: str | None = None
    permissions: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "document_type": self.document_type,
            "agreement_type": self.agreement_type,
            "jurisdiction": self.jurisdiction,
            "effective_date": self.effective_date,
            "confidentiality_level": self.confidentiality_level,
            "language": self.language,
            "tenant": self.tenant,
            "client": self.client,
            "matter": self.matter,
            "permissions": self.permissions,
        }


def _looks_english(text: str) -> bool:
    common = ("the", "and", "of", "to", "shall", "agreement")
    sample = text[:2000].lower()
    return sum(f" {w} " in sample for w in common) >= 2


def extract_doc_meta(path: Path | str, full_text: str, tenant: str | None = None) -> DocMeta:
    """Extract document-level metadata from the first ~4000 chars (legal
    metadata — titles, effective dates, governing law — overwhelmingly sits
    in the opening recitals or the final boilerplate, so scan head + tail)."""
    head = full_text[:4000]
    tail = full_text[-2000:] if len(full_text) > 4000 else ""
    scan = head + "\n" + tail

    meta = DocMeta(document_type=Path(path).suffix.lstrip(".").lower() or "unknown", tenant=tenant)

    lowered = scan.lower()
    for pattern, label in _AGREEMENT_TYPES:
        if re.search(pattern, lowered):
            meta.agreement_type = label
            break

    m = _JURISDICTION_RE.search(scan)
    if m:
        meta.jurisdiction = m.group("j").strip().rstrip(".,;")

    for pattern in _EFFECTIVE_DATE_RES:
        m = pattern.search(scan)
        if m:
            candidate = m.group("d").strip().rstrip(".,;")
            # Skip pure boilerplate ("the Effective Date and continues...")
            if any(ch.isdigit() for ch in candidate):
                meta.effective_date = candidate
                break

    if _CONFIDENTIAL_RE.search(scan):
        meta.confidentiality_level = "confidential"

    if not _looks_english(full_text):
        meta.language = "und"  # undetermined — flag, never guess wrong

    return meta
