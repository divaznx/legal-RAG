"""Dataclasses for the legal object model.

Each carries its own provenance (`document_id`, `clause_id`, character span)
so that any figure surfaced in a dashboard can be traced back to the exact
words in the source. That traceability is the difference between a number a
lawyer can rely on and a number they have to re-verify by hand — at which
point the product has saved them nothing.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Provenance:
    document_id: str
    clause_id: str
    span_start: int = 0
    span_end: int = 0


@dataclass
class Obligation:
    """A duty, right, or prohibition borne by one party.

    `modality` is kept separate from `action` and never collapsed into it:
    "may suspend" and "shall suspend" differ by one word and by the entire
    allocation of risk. An obligation register that renders rights as duties
    is worse than no register.
    """
    document_id: str
    clause_id: str
    modality: str                 # obligation | right | prohibition
    action: str
    obligor: str | None = None
    obligee: str | None = None
    condition: str | None = None
    deadline_raw: str | None = None
    deadline_days: int | None = None
    penalty_hint: str | None = None
    concepts: list[str] = field(default_factory=list)
    span_start: int = 0
    span_end: int = 0
    confidence: float = 0.5
    status: str = "unreviewed"
    id: str = field(default_factory=new_id)

    @property
    def summary(self) -> str:
        return f"{self.obligor or 'Unspecified party'} {modal_phrase(self)}"


@dataclass
class KeyDate:
    """A deadline, stored as a RULE rather than a resolved date.

    "90 days before the end of the then-current term" has no fixed date until
    the term is known, and the term moves on every renewal. Persisting a
    computed date would silently go stale; persisting the rule keeps it true.
    """
    document_id: str
    clause_id: str
    kind: str                     # payment | notice | renewal | cure | term | expiry | response
    rule_type: str                # relative | absolute | duration
    raw: str
    days: int | None = None
    unit: str | None = None
    direction: str | None = None
    anchor: str | None = None
    business_days: bool = False
    absolute_date: str | None = None
    computed_date: str | None = None
    span_start: int = 0
    span_end: int = 0
    confidence: float = 0.5
    id: str = field(default_factory=new_id)

    @property
    def summary(self) -> str:
        if self.rule_type == "absolute":
            return f"{self.kind}: {self.absolute_date}"
        if self.rule_type == "duration":
            return f"{self.kind}: {self.days} {self.unit}(s)"
        anchor = (self.anchor or "the trigger event").replace("_", " ")
        qualifier = " business" if self.business_days else ""
        return f"{self.kind}: {self.days}{qualifier} {self.unit}(s) {self.direction} {anchor}"


@dataclass
class MoneyTerm:
    document_id: str
    clause_id: str
    kind: str                     # fee | cap | interest | penalty | credit
    raw: str
    amount: float | None = None
    currency: str | None = None
    multiplier: float | None = None
    basis: str | None = None
    period: str | None = None
    span_start: int = 0
    span_end: int = 0
    confidence: float = 0.5
    id: str = field(default_factory=new_id)


@dataclass
class Finding:
    """One playbook rule failing against one document.

    `rationale` carries the firm's own reason for the position, not a model's
    explanation. That is what makes a finding arguable in a negotiation
    instead of merely assertive.
    """
    document_id: str
    playbook_id: str
    rule_id: str
    finding_type: str             # missing_clause | deviation | prohibited_language | threshold_breach
    severity: str                 # blocker | high | medium | note
    title: str
    detail: str
    clause_id: str | None = None
    category: str | None = None
    rationale: str | None = None
    suggested: str | None = None
    status: str = "open"
    id: str = field(default_factory=new_id)


SEVERITY_ORDER = {"blocker": 0, "high": 1, "medium": 2, "note": 3}

_MODAL_WORD = {"obligation": "must", "right": "may", "prohibition": "must not"}
# Actions that already begin with their own operative verb. Prefixing a modal
# yields "Guarantor must unconditionally and irrevocably guarantees...".
_ALREADY_INFLECTED = re.compile(
    r"^(?:unconditionally\s+|irrevocably\s+|and\s+)*"
    r"(?:guarantees|indemnifies|undertakes|grants|agrees\s+to|covenants\s+to)\b",
    re.IGNORECASE,
)


def modal_phrase(obligation) -> str:
    """"must pay the Charges" / "guarantees the performance of..."."""
    action = obligation.action if isinstance(obligation, Obligation) else obligation["action"]
    modality = (obligation.modality if isinstance(obligation, Obligation)
                else obligation["modality"])
    if _ALREADY_INFLECTED.match(action):
        return action
    return f"{_MODAL_WORD[modality]} {action}"
