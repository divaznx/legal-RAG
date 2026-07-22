"""Cross-reference detection and resolution.

Contracts are graphs, not documents. Clause 6 caps liability "except as
provided in Clause 8"; Clause 1 delivers services "described in Exhibit A".
Answering from Clause 6 alone is not merely incomplete — it is wrong, and a
lawyer relying on it would advise wrongly.

So every chunk records its outgoing references at ingest time, and the
retrieval stage pulls the referenced clauses in as first-class evidence.

Equally important is the negative case: if Exhibit A is incorporated by
reference but was never ingested, the honest answer says so. `unresolved()`
produces exactly that list, which the engine surfaces under Limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .entities import collapse_whitespace, normalize_clause_number

# Trigger phrases that introduce a cross-reference, grouped by legal force.
# `force` drives retrieval priority: an overriding reference ("notwithstanding")
# changes the meaning of the host clause, so it outranks a merely
# informational one ("see also").
_TRIGGERS: tuple[tuple[str, str, str], ...] = (
    ("notwithstanding", "override", r"notwithstanding"),
    ("subject_to", "condition", r"subject to"),
    ("except_as", "exception", r"except as (?:otherwise )?(?:provided|set forth|permitted|required) (?:in|under|by)"),
    ("except_as", "exception", r"except (?:as|for) (?:provided in|under)"),
    ("pursuant_to", "authority", r"pursuant to"),
    ("in_accordance", "authority", r"in accordance with"),
    ("as_defined", "definition", r"as defined in"),
    ("as_set_forth", "reference", r"as (?:set forth|described|specified|provided|stated) in"),
    ("see", "reference", r"see(?: also)?"),
    ("refer_to", "reference", r"refer(?:ence)?(?:red)? to in"),
    ("under", "reference", r"under"),
    ("in", "reference", r"in"),
)

_CLAUSE_TARGET = (
    r"(?P<kind>clause|section|article|paragraph|para\.?|art\.?|sec\.?|§)\s*"
    r"(?P<num>\d+(?:\.\d+)*[a-z]?|[IVXLC]+(?:\.\d+)*)"
)

_XREF_RES: tuple[tuple[str, str, re.Pattern], ...] = tuple(
    (name, force, re.compile(rf"(?<!\w){trigger}\s+{_CLAUSE_TARGET}(?!\w)", re.IGNORECASE))
    for name, force, trigger in _TRIGGERS
)

# The identifier must not end on punctuation, or "Exhibit A." and "Exhibit A"
# become two distinct attachments.
_ATTACHMENT_RE = re.compile(
    r"(?<!\w)(?P<kind>exhibit|schedule|annex|annexure|appendix|addendum|attachment)\s+"
    r"(?P<id>[A-Z0-9](?:[A-Z0-9.\-]{0,5}[A-Z0-9])?)(?!\w)",
    re.IGNORECASE,
)

_INCORPORATION_RE = re.compile(
    r"(?<!\w)(?:incorporated (?:herein )?by reference|incorporation by reference"
    r"|attached hereto and (?:incorporated|made a part)"
    r"|forms? (?:an integral )?part of this agreement"
    r"|annexed hereto)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XRef:
    target_kind: str   # "clause" | "exhibit" | "schedule" | "annex" | ...
    target: str        # normalized identifier: "4.2", "A", "1"
    trigger: str       # which phrase introduced it
    force: str         # "override" | "condition" | "exception" | "authority" | ...
    raw: str

    @property
    def label(self) -> str:
        return f"{self.target_kind.capitalize()} {self.target.upper() if len(self.target) <= 2 else self.target}"

    def key(self) -> str:
        return f"{self.target_kind}:{self.target}".lower()


_FORCE_RANK = {
    "override": 0, "condition": 1, "exception": 2,
    "definition": 3, "authority": 4, "reference": 5,
}


def force_rank(force: str) -> int:
    """How much a reference changes the meaning of its host clause.

    Lower is stronger. Used to spend a limited cross-reference budget on the
    references that alter the answer ("subject to", "except as provided in")
    rather than on incidental pointers.
    """
    return _FORCE_RANK.get(force, 9)


def extract(text: str) -> list[XRef]:
    """Outgoing cross-references in a clause, strongest legal force first.

    A span already claimed by a stronger trigger is not re-matched, so
    "subject to Clause 6" yields one condition reference rather than also a
    bare "in Clause 6" reference.
    """
    text = collapse_whitespace(text)
    refs: list[XRef] = []
    claimed: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()

    ordered = sorted(_XREF_RES, key=lambda r: _FORCE_RANK.get(r[1], 9))
    for name, force, pattern in ordered:
        for m in pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end in claimed):
                continue
            kind = m.group("kind").lower().rstrip(".")
            kind = {"art": "article", "sec": "section", "para": "paragraph",
                    "§": "clause"}.get(kind, kind)
            number = normalize_clause_number(m.group("num"))
            if (kind, number) in seen:
                continue
            seen.add((kind, number))
            claimed.append((m.start(), m.end()))
            refs.append(XRef(target_kind=kind, target=number, trigger=name,
                             force=force, raw=m.group(0).strip()))

    for m in _ATTACHMENT_RE.finditer(text):
        kind = m.group("kind").lower()
        target = m.group("id").upper()
        if (kind, target) in seen:
            continue
        seen.add((kind, target))
        refs.append(XRef(target_kind=kind, target=target, trigger="attachment",
                         force="reference", raw=m.group(0).strip()))

    return sorted(refs, key=lambda r: _FORCE_RANK.get(r.force, 9))


def has_incorporation_language(text: str) -> bool:
    """True when the clause pulls an external document into the contract."""
    return bool(_INCORPORATION_RE.search(collapse_whitespace(text)))


def incorporated_attachments(text: str) -> list[str]:
    """Attachments this clause incorporates by reference (labels only).

    Only returned when incorporation language is actually present — a passing
    mention of "Exhibit A" is a pointer, whereas "Exhibit A is incorporated by
    reference" makes Exhibit A binding contract text that must be read.
    """
    text = collapse_whitespace(text)
    if not _INCORPORATION_RE.search(text):
        return []
    return [f"{m.group('kind').capitalize()} {m.group('id').upper()}"
            for m in _ATTACHMENT_RE.finditer(text)]


def unresolved(refs: list[XRef], available_keys: set[str]) -> list[str]:
    """Referenced targets that are not present in the retrieved corpus.

    These become answer Limitations: the lawyer must know that the evidence
    set is missing a document the contract makes binding.
    """
    missing = [r.label for r in refs if r.key() not in available_keys]
    return list(dict.fromkeys(missing))
