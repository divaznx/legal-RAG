"""Obligation extraction: who owes what, to whom, when, and under what condition.

Deterministic by design. An obligation register is a system of record — a
lawyer will filter it, export it, and act on it — so it has to be stable:
the same contract must yield the same register today and next month. An LLM
pass over each clause would drift on re-run, and "the register changed but the
contract didn't" destroys trust in the whole product.

The rules below cover the drafting patterns that carry obligations in
commercial agreements: a party, a modal verb, and an action, optionally gated
by a condition and bounded by a deadline. Anything they miss is missed
visibly (nothing extracted) rather than invented, which is the right failure
direction for a register someone will rely on.
"""

from __future__ import annotations

import re

from ..legal import ontology
from ..store.models import Obligation
from .text import collapse, normalize, parse_count, sentences, strip_clause_number

# Ordered longest-first: "shall not" must win over "shall".
_MODALS: tuple[tuple[str, str], ...] = (
    (r"shall\s+not", "prohibition"),
    (r"must\s+not", "prohibition"),
    (r"may\s+not", "prohibition"),
    (r"will\s+not", "prohibition"),
    (r"is\s+not\s+(?:permitted|entitled|required)\s+to", "prohibition"),
    (r"shall\s+be\s+(?:required|obliged)\s+to", "obligation"),
    (r"is\s+(?:required|obliged)\s+to", "obligation"),
    (r"are\s+(?:required|obliged)\s+to", "obligation"),
    (r"agrees\s+to", "obligation"),
    (r"undertakes\s+to", "obligation"),
    (r"covenants\s+to", "obligation"),
    # Performative present tense. A guarantee, an indemnity and a licence
    # grant are drafted as statements, not as modal duties — "the Guarantor
    # unconditionally guarantees", "the Supplier indemnifies the Customer".
    # A modal-only extractor returns nothing for an entire guarantee article,
    # which in a three-party deal silently drops one party from the register.
    (r"unconditionally\s+and\s+irrevocably\s+guarantees", "obligation"),
    (r"irrevocably\s+guarantees", "obligation"),
    (r"guarantees", "obligation"),
    (r"indemnifies\s+and\s+holds?\s+harmless", "obligation"),
    (r"indemnifies", "obligation"),
    (r"undertakes", "obligation"),
    (r"grants", "obligation"),
    (r"shall", "obligation"),
    (r"must", "obligation"),
    (r"is\s+entitled\s+to", "right"),
    (r"may", "right"),
    (r"will", "obligation"),
)

_MODAL_RE = re.compile(
    r"(?<!\w)(?P<modal>" + "|".join(p for p, _ in _MODALS) + r")(?!\w)",
    re.IGNORECASE,
)
_MODALITY_BY_PATTERN = {p.replace(r"\s+", " "): kind for p, kind in _MODALS}

# Modals that are themselves the operative verb, rather than auxiliaries.
_PERFORMATIVE_RE = re.compile(
    r"(?:unconditionally\s+and\s+irrevocably\s+)?(?:irrevocably\s+)?guarantees|"
    r"indemnifies(?:\s+and\s+holds?\s+harmless)?|undertakes|grants|"
    r"agrees\s+to|undertakes\s+to|covenants\s+to",
    re.IGNORECASE,
)

# Party-shaped subjects. A capitalised noun phrase alone is too loose — it
# matches "This Agreement shall commence", which is not an obligation on
# anybody.
_PARTY_RE = re.compile(
    r"(?:the\s+|each\s+|either\s+|neither\s+|any\s+|no\s+|all\s+)?"
    r"(?P<party>part(?:y|ies)|provider|supplier|vendor|contractor|client|customer|"
    r"purchaser|buyer|seller|licensor|licensee|disclosing\s+party|receiving\s+party|"
    r"indemnifying\s+party|indemnified\s+party|company|employer|employee|landlord|"
    r"tenant|lessor|lessee|borrower|lender|guarantor|authorised\s+users?|"
    r"authorized\s+users?|subcontractors?)",
    re.IGNORECASE,
)

# Leading gate: "Subject to Clause 3, the Supplier grants..."
_LEADING_CONDITION_RE = re.compile(
    r"^\s*(?P<cond>(?:subject to|if|where|unless|upon|in the event(?: of| that)?|"
    r"provided that|save (?:that|for)|notwithstanding|on the occurrence of|"
    r"except (?:as|where|for))\b[^,]{0,160}),\s*",
    re.IGNORECASE,
)
# Trailing gate: "...may suspend if any invoice remains unpaid for 45 days."
_TRAILING_CONDITION_RE = re.compile(
    r"(?<!\w)(?P<cond>(?:if|where|unless|provided that|so long as|to the extent that|"
    r"in the event(?: of| that)?)\b[^.;]{4,160})",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r"(?<!\w)(?P<raw>(?:within|no later than|not later than|by no later than|"
    r"at least|not less than|after|before)\s+[^,.;]{0,60}?"
    r"(?P<unit>business\s+days?|working\s+days?|calendar\s+days?|days?|months?|"
    r"years?|hours?|weeks?))",
    re.IGNORECASE,
)

_PENALTY_RE = re.compile(
    r"(?<!\w)(?:interest|penalty|liquidated damages|service credit|late fee|"
    r"suspend|suspension|terminate|termination|indemnif\w+|forfeit\w*)",
    re.IGNORECASE,
)

# Sentence subjects that look like parties but bear no duty.
_NON_ACTOR = re.compile(
    r"^(?:this|the)\s+(?:agreement|clause|section|schedule|exhibit|annex|term)\b",
    re.IGNORECASE,
)

EXTRACTOR_VERSION = 1


def _modality(modal: str) -> str:
    normalized = collapse(modal).lower()
    for pattern, kind in _MODALS:
        if re.fullmatch(pattern, normalized, re.IGNORECASE):
            return kind
    return "obligation"


def _obligor(subject: str) -> tuple[str | None, float]:
    """Normalise the sentence subject to a party, with a confidence."""
    subject = collapse(subject).strip(" ,;")
    # "All amounts are exclusive of VAT, WHICH THE CUSTOMER shall pay" — the
    # duty belongs to the relative clause's subject, not the sentence's. Taking
    # the whole span yields "All Customer", which is visibly wrong in a
    # register a lawyer is reading.
    relative = re.search(r"\b(?:which|that|who)\b(?P<rest>.*)$", subject, re.IGNORECASE)
    if relative and relative.group("rest").strip():
        subject = relative.group("rest").strip(" ,;")
    if not subject or _NON_ACTOR.match(subject):
        return None, 0.3
    m = _PARTY_RE.search(subject)
    if m:
        party = collapse(m.group("party")).title()
        # "Each party" / "Either party" carry the quantifier's meaning.
        quantifier = re.match(r"^\s*(each|either|neither|no|all|any)\b", subject, re.IGNORECASE)
        if quantifier:
            return f"{quantifier.group(1).title()} {party}", 0.85
        return party, 0.9
    # A capitalised named entity ("Northwind Technologies Ltd shall ...")
    if re.match(r"^[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,4}$", subject):
        return subject, 0.6
    return None, 0.35


def _deadline(sentence: str) -> tuple[str | None, int | None]:
    m = _DEADLINE_RE.search(sentence)
    if not m:
        return None, None
    raw = collapse(m.group("raw"))
    count = parse_count(raw)
    if count is None:
        return raw, None
    unit = m.group("unit").lower()
    if "month" in unit:
        count *= 30
    elif "year" in unit:
        count *= 365
    elif "week" in unit:
        count *= 7
    elif "hour" in unit:
        count = max(1, round(count / 24))
    return raw, count


def extract(clause_text: str, document_id: str, clause_id: str) -> list[Obligation]:
    """Obligations, rights, and prohibitions carried by one clause."""
    found: list[Obligation] = []

    norm, sents = sentences(clause_text)
    for sentence, sentence_offset in sents:
        body, consumed = strip_clause_number(sentence)
        offset = sentence_offset + consumed

        condition: str | None = None
        lead = _LEADING_CONDITION_RE.match(body)
        if lead:
            condition = collapse(lead.group("cond"))
            offset += lead.end()
            body = body[lead.end():]

        modal = _MODAL_RE.search(body)
        if not modal:
            continue

        obligor, confidence = _obligor(body[: modal.start()])
        if obligor is None:
            continue

        # For a performative verb the modal IS the main verb, so stripping it
        # leaves "Guarantor must to the Customer the due performance..." — a
        # register entry that reads as broken English. Keep the verb.
        matched_modal = collapse(modal.group("modal"))
        performative = bool(_PERFORMATIVE_RE.fullmatch(matched_modal))
        rest = collapse(body[modal.end():]).rstrip(".;, ")
        action = f"{matched_modal} {rest}".strip() if performative else rest
        if len(action) < 3:
            continue

        if condition is None:
            trailing = _TRAILING_CONDITION_RE.search(action)
            if trailing:
                condition = collapse(trailing.group("cond"))

        deadline_raw, deadline_days = _deadline(body)
        penalty = _PENALTY_RE.search(action)

        found.append(Obligation(
            document_id=document_id,
            clause_id=clause_id,
            modality=_modality(modal.group("modal")),
            action=action[:400],
            obligor=obligor,
            condition=condition[:300] if condition else None,
            deadline_raw=deadline_raw,
            deadline_days=deadline_days,
            penalty_hint=penalty.group(0).lower() if penalty else None,
            concepts=ontology.detect_concepts(sentence)[:5],
            span_start=norm.origin(offset),
            span_end=norm.origin(offset + len(body)),
            # A named or role-matched subject is a far stronger signal than a
            # bare capitalised phrase; the register shows the difference.
            confidence=round(confidence, 2),
        ))

    return found
