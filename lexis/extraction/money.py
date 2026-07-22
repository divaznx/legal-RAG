"""Monetary term extraction, classified by legal function.

The amount alone is nearly useless. GBP 96,000 is a fee, 150% is a liability
cap, 4% is a default interest rate, and GBP 5,000 might be either a cap or a
credit — and a dashboard that lists them together as "money" tells a lawyer
nothing. What makes the figure actionable is its ROLE, so classification is
the point of this module and the amount is almost incidental.

Caps in particular are usually relative ("two times the fees paid in the
twelve months preceding the claim"), so `multiplier` and `basis` matter more
than `amount`, and a cap extractor that only understood absolute figures
would miss most real caps.
"""

from __future__ import annotations

import re

from ..store.models import MoneyTerm
from .text import collapse, normalize, sentences

EXTRACTOR_VERSION = 1

_CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR", "₹": "INR", "Rs": "INR", "Rs.": "INR"}

_AMOUNT_RE = re.compile(
    r"(?<!\w)(?P<cur>USD|EUR|GBP|INR|AUD|CAD|SGD|JPY|CHF|Rs\.?|₹|\$|€|£)\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>million|billion|lakh|crore|k|m|bn)?",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(
    r"(?<!\w)(?P<num>\d{1,4}(?:\.\d+)?)\s*(?:%|per\s*cent(?:um)?|percent)",
    re.IGNORECASE,
)

_MULTIPLE_RE = re.compile(
    r"(?<!\w)(?:(?P<num>\d+(?:\.\d+)?)\s*(?:x|times)|"
    r"(?P<word>one|two|three|four|five)\s+times)\s+(?:the\s+)?(?P<basis>[^,.;]{0,60})",
    re.IGNORECASE,
)

_WORD_MULTIPLES = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}

_SCALES = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9,
           "k": 1e3, "lakh": 1e5, "crore": 1e7}

# Ordered: the first matching role wins, so a sentence about a liability cap
# containing the word "fees" is classified as a cap, not a fee.
_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cap", r"aggregate liability|total liability|liability.{0,30}(?:limited|not exceed|"
            r"shall not exceed)|limitation of liability|cap(?:ped)? at|maximum liability|"
            r"shall not exceed"),
    ("interest", r"interest|base rate|per annum above|accrue"),
    ("penalty", r"penalty|liquidated damages|forfeit"),
    ("credit", r"service credit|credit against|rebate"),
    ("fee", r"fee|retainer|subscription|charge|price|payable|invoice|consideration"),
)

_PERIOD_RE = re.compile(
    r"per\s+(?P<p>annum|year|month|quarter|week|day)|(?P<a>annually|monthly|quarterly|weekly)",
    re.IGNORECASE,
)

# A cap carrying no number at all: "aggregate liability shall not exceed the
# fees paid in the twelve months preceding the claim". This is one of the most
# common cap formulations in commercial drafting and it contains no digit to
# match on, so an extractor built only around amounts and percentages reports
# the contract as UNCAPPED. That false negative propagates into a blocker
# finding telling a lawyer their capped contract has unlimited liability —
# the kind of error that ends a customer's trust in the whole product.
# "shall exceed" (no "not") is included deliberately: the standard formulation
# is "NEITHER PARTY'S aggregate liability shall exceed the fees paid", where
# the negation sits in the subject, not the verb. Requiring "shall not exceed"
# misses the most common cap in commercial drafting. The converse risk is
# nil — no one drafts a minimum liability.
_IMPLICIT_CAP_RE = re.compile(
    r"(?:aggregate|total|maximum|entire)?\s*liabilit(?:y|ies)[^.;]{0,80}?"
    r"(?:shall\s+not\s+exceed|will\s+not\s+exceed|not\s+exceed|shall\s+exceed|"
    r"exceeds?|(?:is|are|shall\s+be)\s+limited\s+to|capped\s+at)\s+"
    r"(?:an\s+amount\s+equal\s+to\s+)?(?:the\s+)?(?P<basis>[^.;]{3,120})",
    re.IGNORECASE,
)


def _kind(sentence: str) -> str:
    lowered = sentence.lower()
    for kind, pattern in _KIND_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return "fee"


def _period(sentence: str) -> str | None:
    m = _PERIOD_RE.search(sentence)
    if not m:
        return None
    value = (m.group("p") or m.group("a") or "").lower()
    return {"annum": "year", "annually": "year", "monthly": "month",
            "quarterly": "quarter", "weekly": "week"}.get(value, value or None)


def extract(clause_text: str, document_id: str, clause_id: str) -> list[MoneyTerm]:
    found: list[MoneyTerm] = []
    seen: set[str] = set()

    norm, sents = sentences(clause_text)
    for sentence, offset in sents:
        kind = _kind(sentence)
        period = _period(sentence)
        cap_found = False

        for m in _AMOUNT_RE.finditer(sentence):
            raw = collapse(m.group(0))
            if raw in seen:
                continue
            seen.add(raw)
            try:
                amount = float(m.group("num").replace(",", ""))
            except ValueError:
                continue
            scale = (m.group("scale") or "").lower()
            amount *= _SCALES.get(scale, 1.0)
            symbol = m.group("cur")
            currency = _CURRENCY_SYMBOLS.get(symbol, symbol.upper().rstrip("."))
            found.append(MoneyTerm(
                document_id=document_id, clause_id=clause_id, kind=kind, raw=raw,
                amount=amount, currency=currency, period=period,
                span_start=norm.origin(offset + m.start()),
                    span_end=norm.origin(offset + m.end()),
                confidence=0.9,
            ))

        # Relative caps: "150% of the Subscription Fees", "two times the fees".
        for m in _MULTIPLE_RE.finditer(sentence):
            raw = collapse(m.group(0))
            if raw in seen:
                continue
            seen.add(raw)
            value = m.group("num")
            multiplier = float(value) if value else _WORD_MULTIPLES.get(
                (m.group("word") or "").lower(), 0.0)
            if not multiplier:
                continue
            cap_found = True
            found.append(MoneyTerm(
                document_id=document_id, clause_id=clause_id,
                kind=kind if kind == "cap" else "cap",
                raw=raw, multiplier=multiplier, basis=collapse(m.group("basis"))[:120],
                span_start=norm.origin(offset + m.start()),
                    span_end=norm.origin(offset + m.end()),
                confidence=0.85,
            ))

        for m in _PERCENT_RE.finditer(sentence):
            raw = collapse(m.group(0))
            if raw in seen:
                continue
            seen.add(raw)
            value = float(m.group("num"))
            # A percentage in a liability clause is a cap expressed as a
            # multiple of fees; elsewhere it is a rate.
            if kind == "cap":
                cap_found = True
                found.append(MoneyTerm(
                    document_id=document_id, clause_id=clause_id, kind="cap", raw=raw,
                    multiplier=value / 100.0, basis=_basis_after(sentence, m.end()),
                    span_start=norm.origin(offset + m.start()),
                    span_end=norm.origin(offset + m.end()),
                    confidence=0.85,
                ))
            else:
                found.append(MoneyTerm(
                    document_id=document_id, clause_id=clause_id,
                    kind=kind if kind in ("interest", "penalty", "credit") else "interest",
                    raw=raw, amount=value, currency="%", period=period,
                    span_start=norm.origin(offset + m.start()),
                    span_end=norm.origin(offset + m.end()),
                    confidence=0.8,
                ))

        # A cap with no figure in it. Recorded as 1x its stated basis, which is
        # what "shall not exceed the fees paid" means.
        if not cap_found:
            m = _IMPLICIT_CAP_RE.search(sentence)
            if m:
                found.append(MoneyTerm(
                    document_id=document_id, clause_id=clause_id, kind="cap",
                    raw=collapse(m.group(0))[:200], multiplier=1.0,
                    basis=collapse(m.group("basis"))[:120],
                    span_start=norm.origin(offset + m.start()),
                    span_end=norm.origin(offset + m.end()),
                    confidence=0.75,
                ))

    return found


def _basis_after(sentence: str, position: int) -> str | None:
    tail = sentence[position:position + 120]
    m = re.search(r"of\s+(?:the\s+)?(?P<basis>[^,.;]{3,90})", tail, re.IGNORECASE)
    return collapse(m.group("basis")) if m else None
