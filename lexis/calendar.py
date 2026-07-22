"""The legal calendar: deadline rules resolved into dates a lawyer can act on.

Contract deadlines are rules, not dates, so a calendar is a JOIN between the
rules extracted at ingest and the events the system actually knows about. Two
design choices follow, and both are about not lying:

An entry whose anchor is unknown is returned WITHOUT a date and marked with
what it is waiting on. The alternative — assuming today, or the ingest date,
as the anchor — produces a calendar full of confident, wrong dates, which is
strictly worse than an empty one because someone will diary from it.

Only resolved entries are exported to .ics. Calendar software has no way to
express "this depends on an event that has not happened", and a placeholder
reminder on an arbitrary day is how a real renewal gets missed.

ICS export is deliberately dependency-free: RFC 5545 is a simple line format,
and a legal calendar that only works if a library is installed is a calendar
that will not survive an air-gapped deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .extraction.dates import resolve
from .store import repository

# Anchors resolvable from a known effective date. `term_end` is deliberately
# NOT here: it is a derived date, not the effective date, and resolving
# "90 days before the end of the term" against signature produces a confident
# date that is years wrong. It is computed separately below.
_RESOLVABLE_FROM_EFFECTIVE = {"effective_date"}

_KIND_LABEL = {
    "payment": "Payment due", "notice": "Notice period", "renewal": "Renewal decision",
    "cure": "Cure period", "termination": "Termination window", "expiry": "Expiry",
    "term": "Term", "response": "Response due", "other": "Date",
}


@dataclass
class CalendarEntry:
    document: str
    section: str | None
    kind: str
    description: str
    raw: str
    due: date | None = None
    blocked_on: str | None = None

    @property
    def title(self) -> str:
        label = _KIND_LABEL.get(self.kind, self.kind.title())
        return f"{label} - {self.document}"


def build(document: str | None = None, effective_date: str | None = None,
          tenant_id: str = "default") -> list[CalendarEntry]:
    """Resolve stored date rules against what is known."""
    anchor: date | None = None
    if effective_date:
        try:
            anchor = datetime.strptime(effective_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"effective date must be YYYY-MM-DD: {exc}") from exc

    rows = repository.key_dates(document=document, tenant_id=tenant_id)

    # Derive each document's term end from its initial-term duration. This is
    # what makes the single most valuable date in a contract computable: the
    # renewal-notice deadline, drafted as "at least 90 days before the end of
    # the then-current term". Missing that date commits the client to another
    # full term, so a calendar that cannot resolve it is not doing the job.
    term_end: dict[str, date] = {}
    # An expiry stated as a date ("continues until 31 March 2028") is the
    # stronger signal and needs no effective date at all, so it is taken
    # first; a stated initial-term duration is the fallback.
    for row in rows:
        if (row["rule_type"] == "absolute" and row["anchor"] == "term_end"
                and row["absolute_date"] and row["filename"] not in term_end):
            term_end[row["filename"]] = datetime.strptime(
                row["absolute_date"], "%Y-%m-%d").date()
    if anchor:
        for row in rows:
            if (row["rule_type"] == "duration" and row["anchor"] == "effective_date"
                    and row["filename"] not in term_end):
                end = resolve({**row, "rule_type": "relative", "direction": "after"}, anchor)
                if end:
                    term_end[row["filename"]] = end

    entries: list[CalendarEntry] = []
    for row in rows:
        due: date | None = None
        blocked: str | None = None
        document_term_end = term_end.get(row["filename"])

        if row["rule_type"] == "absolute" and row["absolute_date"]:
            due = datetime.strptime(row["absolute_date"], "%Y-%m-%d").date()
        elif row["rule_type"] == "relative":
            if row["anchor"] == "term_end" and document_term_end:
                due = resolve(row, document_term_end)
            elif anchor and row["anchor"] in _RESOLVABLE_FROM_EFFECTIVE:
                due = resolve(row, anchor)
            elif row["anchor"] == "term_end":
                blocked = "the end of the term (no initial term found)"
            else:
                blocked = (row["anchor"] or "an unidentified trigger event").replace("_", " ")
        elif row["rule_type"] == "duration":
            if anchor and row["anchor"] == "effective_date":
                due = resolve({**row, "rule_type": "relative", "direction": "after"}, anchor)
            elif row["anchor"] == "effective_date":
                blocked = "the effective date (pass --effective to resolve)"
            else:
                blocked = "the event this period runs from"

        entries.append(CalendarEntry(
            document=row["filename"], section=row["section"], kind=row["kind"],
            description=row["raw"], raw=row["raw"], due=due, blocked_on=blocked,
        ))

    entries.sort(key=lambda e: (e.due is None, e.due or date.max, e.document))
    return entries


def _escape(text: str) -> str:
    """RFC 5545 text escaping."""
    return (text.replace("\\", "\\\\").replace(";", r"\;")
                .replace(",", r"\,").replace("\n", r"\n"))


def to_ics(entries: list[CalendarEntry], path: str | Path) -> int:
    """Write resolved entries as an .ics file. Returns the number written."""
    dated = [e for e in entries if e.due is not None]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Lexis//Legal Calendar//EN",
             "CALSCALE:GREGORIAN"]
    for i, entry in enumerate(dated):
        description = f"{entry.description}"
        if entry.section:
            description += f" ({entry.section})"
        lines += [
            "BEGIN:VEVENT",
            f"UID:lexis-{stamp}-{i}@lexis.local",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{entry.due.strftime('%Y%m%d')}",
            f"SUMMARY:{_escape(entry.title)}",
            f"DESCRIPTION:{_escape(description)}",
            "BEGIN:VALARM",
            "TRIGGER:-P14D",          # a fortnight is the minimum useful warning
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_escape(entry.title)}",
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")

    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return len(dated)
