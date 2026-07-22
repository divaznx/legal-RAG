"""Contract review: run a playbook against a document's extracted structure.

The orchestration is deterministic and the checks are arithmetic or set
operations. Nothing here asks a model whether a clause is acceptable.

Two consequences worth stating plainly, because they are what make the output
usable rather than merely plausible:

MISSING CLAUSES ARE FOUND BY SET DIFFERENCE. The playbook enumerates the
concepts that must be present; the document's clauses carry the concepts
found at ingest; the gap is the answer. Language models are poor at noticing
absence — asked what a contract is missing, they produce a plausible list of
clauses contracts usually have. Sets are perfect at it.

THRESHOLDS ARE ARITHMETIC. "Notice must not exceed 90 days" and "the cap must
not exceed 1x annual fees" are comparisons against numbers already parsed at
ingest. They cannot hallucinate, and they return the same verdict on every
run — which matters because a review that changes between runs on an
unchanged contract is worse than no review.
"""

from __future__ import annotations

import re

from ..store import repository
from ..store.models import Finding
from .model import Playbook, Rule

# Concepts a rule may be satisfied by beyond its own name. The ontology models
# retrieval adjacency; this models COVERAGE — whether the document addresses
# the issue at all — which is a different and looser question.
_COVERING_CONCEPTS: dict[str, tuple[str, ...]] = {
    "limitation_of_liability": ("limitation_of_liability", "damages"),
    "termination": ("termination", "effect_of_termination"),
    "cure_period": ("cure_period", "breach"),
    "fees": ("fees", "late_payment"),
    "confidentiality": ("confidentiality",),
    "data_protection": ("data_protection", "security"),
    "intellectual_property": ("intellectual_property",),
    "indemnification": ("indemnification",),
    "governing_law": ("governing_law", "jurisdiction", "dispute_resolution"),
    "suspension": ("suspension",),
    "term": ("term",),
    "amendment": ("amendment", "entire_agreement"),
    "assignment": ("assignment",),
    "force_majeure": ("force_majeure",),
    "warranty": ("warranty",),
    "service_levels": ("service_levels",),
}


def _covering(concept: str) -> tuple[str, ...]:
    return _COVERING_CONCEPTS.get(concept, (concept,))


def _clauses_for_concept(clauses: list[dict], concept: str) -> list[dict]:
    import json
    wanted = set(_covering(concept))
    return [c for c in clauses if wanted & set(json.loads(c["concepts_json"] or "[]"))]


def _finding(rule: Rule, document_id: str, playbook: Playbook, finding_type: str,
             detail: str, clause_id: str | None = None) -> Finding:
    return Finding(
        document_id=document_id, playbook_id=playbook.id, rule_id=rule.id,
        finding_type=finding_type, severity=rule.severity, category=rule.category,
        title=rule.title, detail=detail,
        # The firm's own reason, not a model's explanation. This is what makes
        # the finding arguable in a negotiation rather than merely assertive.
        rationale=rule.rationale.strip(), suggested=rule.suggested.strip(),
        clause_id=clause_id,
    )


def _check_rule(rule: Rule, playbook: Playbook, document_id: str, clauses: list[dict],
                dates: list[dict], money: list[dict], full_text: str) -> list[Finding]:
    check = rule.check
    matched = _clauses_for_concept(clauses, rule.concept)
    first_id = matched[0]["id"] if matched else None

    if check.type == "must_exist":
        if not matched:
            return [_finding(rule, document_id, playbook, "missing_clause",
                             f"No clause addressing '{rule.concept.replace('_', ' ')}' "
                             f"was found in this document.")]
        return []

    if check.type == "must_not_exist":
        if matched:
            return [_finding(rule, document_id, playbook, "deviation",
                             f"{matched[0]['section'] or 'A clause'} addresses "
                             f"'{rule.concept.replace('_', ' ')}', which this position "
                             f"does not accept.", first_id)]
        return []

    if check.type in ("forbidden_language", "required_language"):
        hits = [(p, m) for p in check.patterns
                for m in [re.search(p, full_text, re.IGNORECASE)] if m]
        if check.type == "forbidden_language" and hits:
            pattern, match = hits[0]
            return [_finding(rule, document_id, playbook, "prohibited_language",
                             f"The document contains \"{match.group(0).strip()}\".",
                             first_id)]
        if check.type == "required_language" and not hits:
            return [_finding(rule, document_id, playbook, "deviation",
                             "Required language was not found in the document.", first_id)]
        return []

    if check.type in ("max_days", "min_days"):
        relevant = [d for d in dates
                    if d["kind"] == check.date_kind and d["days"] is not None]
        if not relevant:
            return []          # nothing to measure; a must_exist rule covers absence
        if check.type == "max_days":
            breaches = [d for d in relevant if d["days"] > (check.value or 0)]
            comparison = "exceeds"
        else:
            breaches = [d for d in relevant if d["days"] < (check.value or 0)]
            comparison = "is shorter than"
        if breaches:
            worst = max(breaches, key=lambda d: abs(d["days"] - (check.value or 0)))
            return [_finding(rule, document_id, playbook, "threshold_breach",
                             f"{worst['section'] or 'A clause'} sets a {check.date_kind} "
                             f"period of {worst['days']} days, which {comparison} the "
                             f"{check.value:g}-day limit in this playbook "
                             f"(\"{worst['raw']}\").", worst["clause_id"])]
        return []

    if check.type == "cap_present":
        caps = [m for m in money if m["kind"] == "cap"]
        if not caps:
            return [_finding(rule, document_id, playbook, "missing_clause",
                             "No liability cap was found. Liability appears uncapped.",
                             first_id)]
        return []

    if check.type in ("max_cap_multiplier", "min_cap_multiplier"):
        caps = [m for m in money if m["kind"] == "cap" and m["multiplier"] is not None]
        if not caps:
            return []
        if check.type == "max_cap_multiplier":
            breaches = [m for m in caps if m["multiplier"] > (check.value or 0)]
            comparison, limit = "exceeds", "maximum"
        else:
            breaches = [m for m in caps if m["multiplier"] < (check.value or 0)]
            comparison, limit = "is below", "minimum"
        if breaches:
            worst = breaches[0]
            return [_finding(rule, document_id, playbook, "threshold_breach",
                             f"{worst['section'] or 'A clause'} sets a cap of "
                             f"{worst['multiplier']:g}x ({worst['raw']}), which "
                             f"{comparison} the {limit} of {check.value:g}x in this "
                             f"playbook.", worst["clause_id"])]
        return []

    return []


def review(document: str, playbook: Playbook, tenant_id: str = "default",
           persist: bool = True) -> list[Finding]:
    """Run every applicable rule against one document."""
    document_id = repository.document_id_for(document, tenant_id)
    if document_id is None:
        raise ValueError(
            f"{document} has not been ingested, or was ingested before the "
            f"structured store existed. Re-ingest it and try again."
        )

    clauses = repository.clauses_for(document_id)
    dates = repository.key_dates(document=document, tenant_id=tenant_id)
    money = repository.money_terms(document=document, tenant_id=tenant_id)
    full_text = "\n\n".join(c["text"] for c in clauses)

    doc_type = None
    row = repository.connect().execute(
        "SELECT doc_type FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    if row:
        doc_type = row["doc_type"]

    rules = playbook.for_document(doc_type)
    findings: list[Finding] = []
    for rule in rules:
        findings.extend(
            _check_rule(rule, playbook, document_id, clauses, dates, money, full_text)
        )

    if persist:
        repository.record_review(document_id, playbook.id, playbook.version,
                                 playbook.position, len(rules), findings, tenant_id)
    return findings
