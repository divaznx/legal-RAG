"""Deadline extraction — stored as rules, not as dates.

Almost no contractual deadline is a date. "Within thirty (30) days of
invoice", "at least ninety (90) days before the end of the then-current
term", "within seventy-two (72) hours of becoming aware" are all RULES whose
resolution depends on an event that has not happened yet, or that moves on
every renewal.

A calendar built by resolving those to fixed dates at ingest is wrong the
moment the anchor moves, and wrong silently. So the rule is what persists:
offset, unit, direction, and the anchor event. Resolution happens on read,
against whatever the system knows about the anchor — and where the anchor is
unknown, the entry says so instead of inventing a date.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from ..store.models import KeyDate
from .text import collapse, normalize, parse_count, sentences

EXTRACTOR_VERSION = 1

# What the clock runs from. Ordered so that specific anchors beat generic ones
# ("written notice of the breach" is a breach-notice anchor, not bare notice).
_ANCHORS: tuple[tuple[str, str], ...] = (
    ("breach_notice", r"written notice of the breach|notice specifying the breach"),
    ("awareness", r"becoming aware|discovery of|awareness of"),
    ("invoice", r"invoice(?:\s+date)?|receipt of (?:the )?invoice|date of invoice"),
    ("term_end", r"end of the (?:then-current|current|initial) term|expiry of the term|"
                 r"end of the term|then-current term"),
    ("effective_date", r"effective date|commencement date|restatement date|start date"),
    ("termination", r"termination|expiry|expiration"),
    ("notice", r"written notice|prior notice|notice"),
    ("claim", r"the claim|claim arising|event giving rise"),
    ("request", r"request|demand"),
)

# What kind of deadline it is. Drives the calendar's grouping and the
# playbook's numeric checks.
_KINDS: tuple[tuple[str, str], ...] = (
    ("cure", r"cure|remed(?:y|ied|ies)|uncured|rectif\w+"),
    ("renewal", r"renew\w*|auto-?renewal|successive|extension"),
    ("notice", r"notice"),
    # Response before payment: "if the Customer disputes an invoice it shall
    # NOTIFY the Supplier within fifteen (15) Business Days" is a dispute
    # window, not a payment term. Classifying it as payment made it look like
    # a 15-day payment term and tripped an "aggressive payment terms" finding
    # against a contract whose actual terms are a month longer.
    ("response", r"respond|notify|notification|report|escalat\w+"),
    ("payment", r"pay(?:able|ment)?|invoice|due|overdue|retainer|subscription fees"),
    ("termination", r"terminat\w+|suspend|suspension"),
    ("expiry", r"expir\w+|survive|survival"),
    ("term", r"initial term|term of\b|period of"),
)

_RELATIVE_RE = re.compile(
    # The left word boundary is load-bearing: without it "thereafter" matches
    # as "after", and because the trailing group then consumes the rest of the
    # sentence, the genuine rule that follows — "not less than ninety (90)
    # days before the end of the then-current term", the renewal deadline —
    # is never seen at all.
    r"(?<!\w)(?P<lead>within|no later than|not later than|at least|not less than|"
    r"no less than|upon|after|before|prior to|following|from)\s+"
    r"(?P<qty>[^,.;]{0,45}?)\s*"
    r"(?P<unit>business\s+days?|working\s+days?|calendar\s+days?|days?|months?|"
    r"years?|hours?|weeks?)"
    r"(?P<tail>[^.;,]{0,80})?",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"(?:for (?:an? )?(?:initial )?(?:term|period) of|continues for|"
    r"survives?(?: termination(?: or expiry)?)? for|for a period of)\s+"
    r"(?P<qty>[^,.;]{0,45}?)\s*"
    r"(?P<unit>days?|months?|years?)",
    re.IGNORECASE,
)

# Marks a duration as being the agreement's own term, and therefore measurable
# from the effective date.
_TERM_DURATION_RE = re.compile(
    r"initial term|term of (?:this|the) agreement|commences? on|"
    r"this agreement (?:continues|commences)|term of\s+\w+\s*\(?\d",
    re.IGNORECASE,
)

# Language that makes the date which follows the END of the term.
_TERM_END_RE = re.compile(
    r"continues?\s+until|continue\s+until|expires?\s+on|ends?\s+on|"
    r"terminates?\s+on|until\s*$|through\s+to",
    re.IGNORECASE,
)

# "the tenth (10th) Business Day of the month following the month of invoice"
# is a monthly billing cycle (~40 days), not a 10-day payment term. Reading it
# as 10 days understates the term by a month and trips a "payment terms are
# too aggressive" finding that is simply wrong. No number is better than the
# wrong number.
_MONTHLY_CYCLE_RE = re.compile(r"day\s+of\s+the\s+(?:month|calendar month)", re.IGNORECASE)

_ABSOLUTE_RE = re.compile(
    r"(?<!\w)(?P<raw>"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December),?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})(?!\w)",
    re.IGNORECASE,
)

_BEFORE_LEADS = {"at least", "not less than", "no less than", "before", "prior to"}

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def _classify_kind(sentence: str) -> str:
    lowered = sentence.lower()
    for kind, pattern in _KINDS:
        if re.search(pattern, lowered):
            return kind
    return "other"


def _anchor(text: str) -> str | None:
    lowered = text.lower()
    for anchor, pattern in _ANCHORS:
        if re.search(pattern, lowered):
            return anchor
    return None


def _normalize_unit(unit: str) -> tuple[str, bool]:
    unit = unit.lower()
    business = "business" in unit or "working" in unit
    if "hour" in unit:
        return "hour", business
    if "week" in unit:
        return "week", business
    if "month" in unit:
        return "month", business
    if "year" in unit:
        return "year", business
    return "day", business


def _parse_absolute(raw: str) -> str | None:
    raw = raw.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if m:
        return raw
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", raw)
    if m and m.group(2).lower() in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", raw)
    if m and m.group(1).lower() in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None


def extract(clause_text: str, document_id: str, clause_id: str) -> list[KeyDate]:
    found: list[KeyDate] = []
    seen: set[tuple] = set()

    norm, sents = sentences(clause_text)
    for sentence, offset in sents:
        kind = _classify_kind(sentence)

        for m in _ABSOLUTE_RE.finditer(sentence):
            iso = _parse_absolute(m.group("raw"))
            key = ("absolute", iso or m.group("raw"))
            if key in seen:
                continue
            seen.add(key)
            # "continues until 31 March 2028" states the term's end as a date
            # rather than a duration. Without recognising it, every rule
            # anchored to the end of the term — including the renewal notice,
            # the most consequential date in the contract — stays unresolved.
            expiry = bool(_TERM_END_RE.search(sentence[:m.start()]))
            found.append(KeyDate(
                document_id=document_id, clause_id=clause_id,
                kind="expiry" if expiry else (kind if kind != "other" else "term"),
                anchor="term_end" if expiry else None,
                rule_type="absolute", absolute_date=iso, computed_date=iso,
                raw=collapse(m.group("raw")),
                span_start=norm.origin(offset + m.start()),
                span_end=norm.origin(offset + m.end()),
                confidence=0.9 if iso else 0.5,
            ))

        for m in _DURATION_RE.finditer(sentence):
            count = parse_count(m.group("qty"))
            if count is None:
                continue
            unit, business = _normalize_unit(m.group("unit"))
            key = ("duration", count, unit)
            if key in seen:
                continue
            seen.add(key)
            # Only a duration that describes the AGREEMENT'S term runs from
            # the effective date. "A force majeure event continuing for more
            # than sixty (60) days" is also a duration, but it is measured
            # from an event that may never occur — anchoring it to signature
            # would put a confident, meaningless date in the calendar.
            anchored = bool(_TERM_DURATION_RE.search(sentence))
            found.append(KeyDate(
                document_id=document_id, clause_id=clause_id,
                kind="term" if anchored else kind,
                rule_type="duration", days=count, unit=unit, business_days=business,
                anchor="effective_date" if anchored else None,
                raw=collapse(m.group(0)),
                span_start=norm.origin(offset + m.start()),
                span_end=norm.origin(offset + m.end()),
                confidence=0.8 if anchored else 0.6,
            ))

        for m in _RELATIVE_RE.finditer(sentence):
            count = parse_count(m.group("qty"))
            if count is None:
                continue
            unit, business = _normalize_unit(m.group("unit"))
            # A monthly billing cycle, not an N-day offset. Keep the rule for
            # the calendar but drop the count, so no downstream check compares
            # against a number that means something else.
            if _MONTHLY_CYCLE_RE.search(m.group(0)):
                count = None
            lead = collapse(m.group("lead")).lower()
            direction = "before" if lead in _BEFORE_LEADS else "after"
            # The anchor usually trails the quantity ("within 30 days OF THE
            # INVOICE DATE"); fall back to the sentence when it doesn't.
            anchor = _anchor(m.group("tail") or "") or _anchor(sentence)
            key = ("relative", count, unit, direction, anchor, m.group(0)[:30])
            if key in seen:
                continue
            seen.add(key)
            found.append(KeyDate(
                document_id=document_id, clause_id=clause_id, kind=kind,
                rule_type="relative", days=count, unit=unit, direction=direction,
                anchor=anchor, business_days=business,
                raw=collapse(m.group(0))[:200],
                span_start=norm.origin(offset + m.start()),
                span_end=norm.origin(offset + m.end()),
                confidence=0.85 if anchor else 0.6,
            ))

    return found


_UNIT_DAYS = {"hour": 1 / 24, "day": 1, "week": 7}

_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return _MONTH_LENGTHS[month - 1]


def _shift_months(anchor: date, months: int) -> date:
    """Calendar-correct month arithmetic.

    A 30-day approximation puts a 36-month term two weeks early — and the
    renewal notice that hangs off it two weeks earlier still. Legal deadlines
    are exact dates or they are not deadlines.
    """
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(anchor.day, _days_in_month(year, month)))


def resolve(rule: dict, anchor_date: date) -> date | None:
    """Resolve a relative rule against a known anchor date.

    Returns None rather than a guess when the rule is not resolvable, because
    a calendar entry on the wrong day is worse than a calendar entry marked
    "depends on an event we haven't seen".
    """
    if rule.get("rule_type") != "relative" or not rule.get("days"):
        return None
    count = rule["days"]
    unit = rule.get("unit") or "day"
    backwards = rule.get("direction") == "before"

    if unit in ("month", "year"):
        months = count * (12 if unit == "year" else 1)
        return _shift_months(anchor_date, -months if backwards else months)

    days = count * _UNIT_DAYS.get(unit, 1)
    if rule.get("business_days"):
        days *= 7 / 5          # working days to calendar days
    delta = timedelta(days=round(days))
    return anchor_date - delta if backwards else anchor_date + delta
