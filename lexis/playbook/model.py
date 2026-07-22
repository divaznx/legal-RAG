"""Playbook model: the firm's negotiating position, encoded.

The framing that makes contract review work is that RISK IS NOT A PROPERTY OF
A CLAUSE. A 12-month liability cap is aggressive if you are the customer and
generous if you are the vendor; a 90-day termination notice is protective for
whoever depends on the service and onerous for whoever wants out. Asking a
model "is this clause risky?" with no position supplied produces confident,
untraceable nonsense.

So review is always against a position. A playbook states, for each issue:
what must exist, what the ideal language is, which fallbacks are acceptable
and in what order, what is a walk-away, and — critically — WHY the firm takes
that position. The "why" is what turns a finding from an assertion into
something a lawyer can argue from in a negotiation.

This also happens to be the commercial moat. Once a legal team has encoded
eighty rules reflecting how they actually negotiate, the playbook is their
institutional knowledge and the switching cost is enormous. It is worth more
to the product than any model upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Deterministic checks that run against the EXTRACTED structure rather than
# against prose. Because deadlines and caps are already parsed into numbers at
# ingest, "notice must not exceed 90 days" is arithmetic, not interpretation —
# it cannot hallucinate and it returns the same verdict every time.
CHECK_TYPES = {
    "must_exist",            # a clause covering this concept is present
    "must_not_exist",        # e.g. no automatic renewal
    "max_days",              # deadline of `date_kind` <= value
    "min_days",              # deadline of `date_kind` >= value
    "cap_present",           # a liability cap exists at all
    "max_cap_multiplier",    # cap <= N x fees
    "min_cap_multiplier",    # cap >= N x fees
    "forbidden_language",    # regex patterns that must not appear
    "required_language",     # regex patterns that must appear
}

SEVERITIES = ("blocker", "high", "medium", "note")


@dataclass
class Check:
    type: str
    value: float | None = None
    date_kind: str | None = None
    patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in CHECK_TYPES:
            raise ValueError(
                f"Unknown check type {self.type!r}. Valid: {', '.join(sorted(CHECK_TYPES))}"
            )


@dataclass
class Rule:
    id: str
    title: str
    concept: str                     # maps onto lexis.legal.ontology concepts
    check: Check
    severity: str = "medium"
    category: str = "general"
    rationale: str = ""              # the firm's reason — quoted in the finding
    ideal: str = ""                  # preferred language
    fallback: list[str] = field(default_factory=list)   # ordered concessions
    guidance: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"Rule {self.id}: severity must be one of {SEVERITIES}")

    @property
    def suggested(self) -> str:
        """What to propose when the rule fails: the ideal, else the best fallback."""
        return self.ideal or (self.fallback[0] if self.fallback else "")


@dataclass
class Playbook:
    id: str
    name: str
    position: str                    # "customer" | "vendor" | "neutral"
    version: str = "1.0"
    applies_to: list[str] = field(default_factory=list)   # doc_type codes
    rules: list[Rule] = field(default_factory=list)

    def for_document(self, doc_type: str | None) -> list[Rule]:
        if not self.applies_to or doc_type is None:
            return self.rules
        return self.rules if doc_type in self.applies_to else []


def load(path: str | Path) -> Playbook:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return from_dict(data)


def from_dict(data: dict) -> Playbook:
    rules = []
    for raw in data.get("rules", []):
        check_data = dict(raw.get("check") or {"type": "must_exist"})
        rules.append(Rule(
            id=raw["id"],
            title=raw["title"],
            concept=raw["concept"],
            check=Check(**check_data),
            severity=raw.get("severity", "medium"),
            category=raw.get("category", "general"),
            rationale=raw.get("rationale", ""),
            ideal=raw.get("ideal", ""),
            fallback=raw.get("fallback", []) or [],
            guidance=raw.get("guidance", ""),
        ))
    return Playbook(
        id=data["id"], name=data["name"], position=data.get("position", "neutral"),
        version=str(data.get("version", "1.0")),
        applies_to=data.get("applies_to", []) or [], rules=rules,
    )


def builtin(name: str) -> Playbook:
    """Load a playbook shipped with the product."""
    path = Path(__file__).parent / "library" / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in path.parent.glob("*.yaml")))
        raise FileNotFoundError(f"No built-in playbook {name!r}. Available: {available}")
    return load(path)


def list_builtin() -> list[str]:
    return sorted(p.stem for p in (Path(__file__).parent / "library").glob("*.yaml"))
