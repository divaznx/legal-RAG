"""Document-level legal profile, built once at ingest.

Retrieval decisions that must happen *before* any vector search — which
agreement does this question target, is this version superseded, does a
"Clause 8" even exist — cannot be answered from chunk embeddings. They need a
document-level index: type, parties, jurisdiction, clause inventory, defined
terms, and version lineage.

That index is this dataclass. It is written into the ingest manifest and read
back by the document-resolution layer on every question.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import entities, ontology, xref

# Longest/most specific first — "master services agreement" must not be
# swallowed by the generic "services agreement".
_DOC_TYPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("MSA", "Master Services Agreement",
     ("master services agreement", "master service agreement", "msa")),
    ("NDA", "Non-Disclosure Agreement",
     ("non-disclosure agreement", "nondisclosure agreement", "non disclosure agreement",
      "confidentiality agreement", "mutual nda", "nda")),
    ("SOW", "Statement of Work",
     ("statement of work", "scope of work", "work order", "sow")),
    ("ServiceOrder", "Service Order",
     ("service order", "order form", "ordering document")),
    ("DPA", "Data Processing Agreement",
     ("data processing agreement", "data protection agreement", "dpa")),
    ("SLA", "Service Level Agreement",
     ("service level agreement", "sla")),
    ("SaaS", "Software as a Service Agreement",
     ("software as a service agreement", "software-as-a-service agreement",
      "saas agreement", "subscription agreement", "cloud services agreement", "saas")),
    ("Reseller", "Reseller Agreement",
     ("reseller agreement", "distribution agreement", "channel partner agreement")),
    ("Supply", "Supply Agreement",
     ("supply agreement", "manufacturing agreement", "vendor agreement")),
    ("Employment", "Employment Agreement",
     ("employment agreement", "offer letter", "contract of employment")),
    ("Lease", "Lease Agreement",
     ("lease agreement", "lease deed", "rental agreement", "tenancy agreement")),
    ("License", "Licence Agreement",
     ("license agreement", "licence agreement", "end user license agreement", "eula")),
    ("Purchase", "Purchase Agreement",
     ("purchase agreement", "purchase order", "sale agreement", "sales agreement")),
    ("Consulting", "Consulting Agreement",
     ("consulting agreement", "consultancy agreement")),
    ("Partnership", "Partnership Agreement",
     ("partnership agreement", "joint venture agreement", "shareholders agreement")),
    ("Settlement", "Settlement Agreement",
     ("settlement agreement", "release agreement")),
    ("Amendment", "Amendment",
     ("amendment agreement", "amendment no", "addendum", "change order")),
    ("LOI", "Letter of Intent",
     ("letter of intent", "memorandum of understanding", "term sheet", "loi", "mou")),
)

# Runs against whitespace-collapsed text. The capture deliberately runs to the
# sentence boundary rather than stopping at "and", so "England and Wales" and
# "New York" both survive intact.
_GOVERNING_LAW_RE = re.compile(
    r"(?i)governed by(?:\s+and construed in accordance with)?\s+the\s+laws?\s+of\s+"
    r"(?:the\s+)?(?:State\s+of\s+|Commonwealth\s+of\s+|Republic\s+of\s+)?"
    r"(?P<place>[A-Z][\w\s]{2,50}?)(?=[.,;)]|$)"
)

_SUPERSEDES_RE = re.compile(
    r"(?i)(?:supersedes?|amends and restates|amended and restated)\b[^.]{0,120}?"
    r"(?:version|v\.?)\s*(?P<version>\d+(?:\.\d+)*)"
)

_AMENDED_RE = re.compile(r"(?i)\bamended and restated\b|\bamendment\b|\brestatement\b")

_VERSION_SUFFIX_RE = re.compile(r"[_\-\s]*v\.?\d+(?:\.\d+)*\s*$", re.IGNORECASE)


@dataclass
class DocumentProfile:
    document: str
    version: str
    family: str                                   # version-lineage key
    doc_type: str = "Unknown"                     # short code, e.g. "MSA"
    doc_type_label: str = "Unknown Document"
    title: str = ""
    parties: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    jurisdictions: list[str] = field(default_factory=list)
    governing_law: str | None = None
    clause_numbers: list[str] = field(default_factory=list)
    clause_index: dict[str, str] = field(default_factory=dict)   # "4" -> "Clause 4. Termination"
    defined_terms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)         # referenced exhibits/schedules
    incorporated: list[str] = field(default_factory=list)        # binding, incorporated by reference
    supersedes_version: str | None = None
    is_amendment: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def family_key(document: str, doc_type: str) -> str:
    """Version-lineage key: MSA_Acme_v1.0.txt and MSA_Acme_v2.1.txt share one.

    Keyed on the version-stripped filename stem so unrelated agreements of the
    same type (two different NDAs) stay in separate families.
    """
    stem = Path(document).stem
    base = _VERSION_SUFFIX_RE.sub("", stem).strip("_- ").lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return f"{doc_type.lower()}:{base}" if base else doc_type.lower()


def detect_doc_type(text: str, document: str) -> tuple[str, str]:
    """(code, label) — matched against the document's opening text, then its
    filename. Opening text wins: filenames lie, title pages rarely do."""
    head = re.sub(r"\s+", " ", text[:1500]).lower()
    name = Path(document).stem.lower().replace("_", " ").replace("-", " ")
    for code, label, forms in _DOC_TYPES:
        for form in forms:
            if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", head):
                return code, label
    for code, label, forms in _DOC_TYPES:
        for form in forms:
            if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", name):
                return code, label
    return "Unknown", "Unknown Document"


def _title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 6 and not line.endswith(("；", ":")):
            return line[:120]
    return ""


def build(chunks, document: str, version: str) -> DocumentProfile:
    """Assemble the profile from already-chunked, already-redacted text."""
    full_text = "\n\n".join(c.text for c in chunks)
    flat_text = entities.collapse_whitespace(full_text)
    doc_type, doc_type_label = detect_doc_type(flat_text, document)

    ents = entities.extract(full_text)

    clause_index: dict[str, str] = {}
    for chunk in chunks:
        if chunk.clause_number and chunk.clause_number not in clause_index:
            clause_index[chunk.clause_number] = chunk.heading or (chunk.section or "")

    # Register parent numbers implied by sub-clauses. A contract with 12.1-12.4
    # and no standalone "Clause 12" body still HAS a Clause 12, and the
    # document-resolution layer must not conclude it is missing when a lawyer
    # asks about it.
    for number in list(clause_index):
        parts = number.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            clause_index.setdefault(parent, f"Clause {parent}")

    defined_terms: list[str] = []
    for chunk in chunks:
        defined_terms.extend(chunk.defined_terms)

    attachments: list[str] = []
    incorporated: list[str] = []
    for chunk in chunks:
        for ref in xref.extract(chunk.text):
            if ref.target_kind not in ("clause", "section", "article", "paragraph"):
                attachments.append(ref.label)
        incorporated.extend(xref.incorporated_attachments(chunk.text))

    law = _GOVERNING_LAW_RE.search(flat_text)
    supersedes = _SUPERSEDES_RE.search(flat_text)

    return DocumentProfile(
        document=document,
        version=version,
        family=family_key(document, doc_type),
        doc_type=doc_type,
        doc_type_label=doc_type_label,
        title=_title(full_text),
        parties=ents.parties,
        organizations=ents.organizations,
        jurisdictions=ents.jurisdictions,
        governing_law=law.group("place").strip() if law else None,
        clause_numbers=sorted(clause_index, key=entities.clause_sort_key),
        clause_index=clause_index,
        defined_terms=sorted(dict.fromkeys(defined_terms)),
        concepts=ontology.detect_concepts(full_text)[:20],
        attachments=sorted(dict.fromkeys(attachments)),
        incorporated=sorted(dict.fromkeys(incorporated)),
        supersedes_version=supersedes.group("version") if supersedes else None,
        is_amendment=bool(_AMENDED_RE.search(flat_text[:2000])),
    )


def version_key(version: str) -> tuple[int, ...]:
    """Sortable numeric version, so v2.1 > v1.0 and v10 > v9."""
    return tuple(int(p) for p in re.findall(r"\d+", str(version))) or (0,)


def from_dict(data: dict) -> DocumentProfile:
    known = {f for f in DocumentProfile.__dataclass_fields__}
    return DocumentProfile(**{k: v for k, v in data.items() if k in known})


__all__ = ["DocumentProfile", "build", "from_dict", "family_key", "version_key",
           "detect_doc_type"]
