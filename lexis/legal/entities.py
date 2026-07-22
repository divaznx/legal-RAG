"""Legal entity recognition — regex/gazetteer based, deliberately not an LLM.

Lawyers need the extraction layer to be deterministic and inspectable: the
same contract must yield the same clause references and the same monetary
values on every run, and a wrong extraction must be traceable to a specific
pattern rather than to a sampling temperature.

Recognised: parties, agreement titles, clause/section/article references,
exhibits and schedules, courts, statutes and regulations, monetary values,
dates, durations, and jurisdictions.

Note on redaction: `lexis.redaction` runs *before* this, so client names
appear as [CLIENT_NAME]. Entity recognition therefore keys off legal ROLES
("Provider", "Receiving Party") and registered-entity suffixes ("LLC",
"Ltd"), which survive redaction and are what clauses actually refer to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def collapse_whitespace(text: str) -> str:
    """All runs of whitespace -> a single space.

    Contract text arrives hard-wrapped, so a phrase like "subject to Clause 6"
    routinely straddles a line break as "subject to\\nClause 6" — or worse,
    "subject\\nto Clause 6". Every multi-word legal pattern in this package
    would silently miss those. Matching runs against the collapsed text
    instead; the stored chunk text is never modified.
    """
    return re.sub(r"\s+", " ", text).strip()


def collapse_lines(text: str) -> str:
    """Join wrapped lines but keep paragraph breaks.

    Used where paragraph structure carries meaning — a defined term must not
    be allowed to run across a blank line into the following paragraph, but it
    must survive an ordinary line wrap.
    """
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]+", " ", joined)


# --- clause / section / article references --------------------------------
# "Clause 8", "Section 4.2", "Article IX", "§ 12.1", "para 3"
_CLAUSE_RE = re.compile(
    r"(?<!\w)(?P<kind>clause|section|article|paragraph|para\.?|art\.?|sec\.?|§)\s*"
    r"(?P<num>\d+(?:\.\d+)*[a-z]?|[IVXLC]+(?:\.\d+)*)(?!\w)",
    re.IGNORECASE,
)

_EXHIBIT_RE = re.compile(
    r"(?<!\w)(?P<kind>exhibit|schedule|annex|annexure|appendix|addendum|attachment)\s*"
    r"(?P<id>[A-Z0-9](?:[A-Z0-9.\-]{0,5}[A-Z0-9])?)(?!\w)",
    re.IGNORECASE,
)

# Party role nouns as they appear in defined-party form: ("Provider")
_PARTY_ROLE_RE = re.compile(
    r"(?<!\w)(?P<role>provider|supplier|vendor|contractor|service provider|client|"
    r"customer|purchaser|buyer|seller|licensor|licensee|disclosing party|"
    r"receiving party|indemnifying party|indemnified party|company|employer|"
    r"employee|landlord|tenant|lessor|lessee|borrower|lender|guarantor)(?!\w)",
    re.IGNORECASE,
)

# Registered legal entities: "Acme Consulting LLC", "Foo Bar Pvt. Ltd."
# The entity SUFFIX is matched case-insensitively while the name itself must
# still be capitalised. Party blocks are conventionally set in full capitals —
# "HELIOS TECHNOLOGIES PRIVATE LIMITED" — and a case-sensitive suffix silently
# drops every party from exactly the part of the document that names them.
_ENTITY_RE = re.compile(
    r"(?<!\w)(?P<name>(?:[A-Z][\w&.\-]*\s+){0,4}[A-Z][\w&.\-]*\s+"
    r"(?i:LLC|L\.L\.C\.|Inc\.?|Incorporated|Corp\.?|Corporation|Ltd\.?|Limited|"
    r"LLP|LP|PLC|GmbH|S\.A\.|B\.V\.|N\.V\.|Pvt\.?\s*Ltd\.?|Private\s+Limited|"
    r"Pte\.?\s*Ltd\.?|S\.p\.A\.|A/S|AB|AG))(?!\w)"
)

_AGREEMENT_RE = re.compile(
    r"(?<!\w)(?P<title>(?:[A-Z][\w\-]*\s+){0,5}"
    r"(?:Agreement|Contract|Deed|Lease|Licence|License|Memorandum|Undertaking|"
    r"Order|Amendment|Addendum|Policy))(?!\w)"
)

_COURT_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:the\s+)?(?:Supreme|High|District|Federal|State|Commercial|Chancery|Circuit|"
    r"Superior|Magistrate|County|Appellate)\s+Court(?:\s+of\s+[A-Z][\w\s]{2,30})?"
    r"|Court\s+of\s+Appeals?(?:\s+for\s+the\s+[\w\s]{2,30})?"
    r"|National\s+Company\s+Law\s+Tribunal|NCLT|Arbitral\s+Tribunal"
    r"|ICC|LCIA|SIAC|JAMS|AAA|UNCITRAL|ICDR"
    r")(?!\w)"
)

_STATUTE_RE = re.compile(
    r"(?<!\w)(?:"
    r"GDPR|CCPA|CPRA|HIPAA|FCPA|UKBA|SOX|DPDP(?:\s+Act)?|PIPEDA|UCC|DMCA"
    r"|Regulation\s*\(EU\)\s*\d{4}/\d+"
    r"|Directive\s*\d{4}/\d+/\w+"
    r"|\d+\s+U\.S\.C\.\s*§*\s*\d+[\w\-]*"
    r"|[A-Z][\w\s]{2,40}\s+Act,?\s+\d{4}"
    r"|Companies\s+Act|Arbitration\s+and\s+Conciliation\s+Act"
    r"|Indian\s+Contract\s+Act|Information\s+Technology\s+Act"
    r")(?!\w)"
)

_MONEY_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:USD|EUR|GBP|INR|AUD|CAD|SGD|JPY|CHF|Rs\.?|₹|\$|€|£)\s*"
    r"\d[\d,.\s]*(?:\s*(?:million|billion|lakh|crore|k|m|bn))?"
    r"|\d[\d,.]*\s*(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)"
    r"|\d+(?:\.\d+)?\s*(?:x|times)\s+the\s+fees"
    r")(?!\w)",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December),?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r")(?!\w)",
    re.IGNORECASE,
)

# "thirty (30) days", "twelve (12) months", "5 years", "45 days"
_DURATION_RE = re.compile(
    r"(?<!\w)(?:[a-z\-]+\s*)?\(?\d{1,4}\)?\s*"
    r"(?:calendar\s+|business\s+|working\s+)?(?:day|days|week|weeks|month|months|"
    r"year|years|hour|hours)(?!\w)",
    re.IGNORECASE,
)

_JURISDICTIONS = (
    "Delaware", "New York", "California", "Texas", "Florida", "Illinois",
    "Massachusetts", "Washington", "Nevada", "New Jersey", "Virginia", "Georgia",
    "England and Wales", "England", "Scotland", "Northern Ireland",
    "Republic of Ireland", "Ireland", "Singapore", "Hong Kong", "India",
    "Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Germany", "France",
    "Netherlands", "Switzerland", "Australia", "New South Wales", "Victoria",
    "Canada", "Ontario", "British Columbia", "United Arab Emirates", "DIFC",
    "Japan", "United States", "United Kingdom", "European Union",
)
_JURISDICTION_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(j) for j in
                             sorted(_JURISDICTIONS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(token: str) -> int | None:
    token = token.lower()
    if not token or any(ch not in _ROMAN for ch in token):
        return None
    total = 0
    previous = 0
    for ch in reversed(token):
        value = _ROMAN[ch]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total


def clause_sort_key(number: str) -> tuple:
    """Document order for clause numbers: 2 < 10, and 11.2 < 11.10."""
    parts = []
    for piece in str(number or "").split("."):
        digits = re.match(r"\d+", piece)
        parts.append((int(digits.group(0)) if digits else 0, piece))
    return tuple(parts)


def normalize_clause_number(raw: str) -> str:
    """"IX" -> "9", "4.2" -> "4.2", "8a" -> "8a". The canonical key used for
    exact clause lookup, so "Article IX" and "Clause 9" resolve alike."""
    raw = raw.strip().rstrip(".")
    if re.fullmatch(r"[IVXLCDMivxlcdm]+", raw):
        value = _roman_to_int(raw)
        if value:
            return str(value)
    return raw.lower()


@dataclass
class ClauseRef:
    kind: str      # "clause" | "section" | "article" | "paragraph"
    number: str    # normalized, e.g. "4.2"
    raw: str

    @property
    def label(self) -> str:
        return f"{self.kind.capitalize()} {self.number}"


@dataclass
class LegalEntities:
    parties: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    agreements: list[str] = field(default_factory=list)
    clause_refs: list[ClauseRef] = field(default_factory=list)
    exhibits: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    monetary_values: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    durations: list[str] = field(default_factory=list)
    jurisdictions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "parties": self.parties,
            "organizations": self.organizations,
            "agreements": self.agreements,
            "clause_refs": [c.label for c in self.clause_refs],
            "exhibits": self.exhibits,
            "courts": self.courts,
            "statutes": self.statutes,
            "monetary_values": self.monetary_values,
            "dates": self.dates,
            "durations": self.durations,
            "jurisdictions": self.jurisdictions,
        }


def _unique(values) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def extract_clause_refs(text: str) -> list[ClauseRef]:
    text = collapse_whitespace(text)
    kinds = {"art": "article", "sec": "section", "para": "paragraph", "§": "clause"}
    refs: list[ClauseRef] = []
    seen: set[tuple[str, str]] = set()
    for m in _CLAUSE_RE.finditer(text):
        kind = m.group("kind").lower().rstrip(".")
        kind = kinds.get(kind, kind)
        number = normalize_clause_number(m.group("num"))
        if (kind, number) in seen:
            continue
        seen.add((kind, number))
        refs.append(ClauseRef(kind=kind, number=number, raw=m.group(0).strip()))
    return refs


def extract(text: str) -> LegalEntities:
    """Full entity sweep over a question or a clause."""
    text = collapse_whitespace(text)
    exhibits = [f"{m.group('kind').capitalize()} {m.group('id').upper()}"
                for m in _EXHIBIT_RE.finditer(text)]
    return LegalEntities(
        parties=_unique(m.group("role").title() for m in _PARTY_ROLE_RE.finditer(text)),
        organizations=_unique(m.group("name") for m in _ENTITY_RE.finditer(text)),
        agreements=_unique(m.group("title") for m in _AGREEMENT_RE.finditer(text)),
        clause_refs=extract_clause_refs(text),
        exhibits=_unique(exhibits),
        courts=_unique(m.group(0) for m in _COURT_RE.finditer(text)),
        statutes=_unique(m.group(0) for m in _STATUTE_RE.finditer(text)),
        monetary_values=_unique(m.group(0) for m in _MONEY_RE.finditer(text)),
        dates=_unique(m.group(0) for m in _DATE_RE.finditer(text)),
        durations=_unique(m.group(0) for m in _DURATION_RE.finditer(text)),
        jurisdictions=_unique(m.group(0) for m in _JURISDICTION_RE.finditer(text)),
    )
