"""Defined-term extraction, powering definition-first retrieval.

In a contract, a capitalised term is a variable, not English. "Confidential
Information" may exclude anything already public; "Services" may be whatever
Exhibit A says. Reading an operative clause without its definitions produces
a fluent answer about the wrong subject matter.

So definitions are indexed as their own retrievable class at ingest time, and
the retrieval stage seeds every evidence set with the definitions of the
terms the question and the candidate clauses actually use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .entities import collapse_lines, collapse_whitespace

# '"Confidential Information" means ...' / 'Confidential Information shall mean ...'
_DEFINITION_RES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "means",
        re.compile(
            r"[\"“”'](?P<term>[A-Za-z][\w \-/&]{1,60}?)[\"“”'][ \t]*"
            r"(?:shall[ \t]+)?(?:means?|has the meaning)\b",
        ),
    ),
    (
        "means",
        re.compile(
            r"(?<![\w\"“])(?P<term>[A-Z][\w\-]*(?:[ \t]+[A-Z][\w\-]*){0,4})[ \t]+"
            r"(?:shall[ \t]+)?means?\b",
        ),
    ),
    (
        "refers_to",
        re.compile(
            r"[\"“”']?(?P<term>[A-Z][\w\-]*(?:[ \t]+[A-Z][\w\-]*){0,4})[\"“”']?[ \t]+"
            r"(?:refers to|is defined as|shall be construed as)\b",
        ),
    ),
    (
        "cross_referenced",
        re.compile(
            r"[\"“”'](?P<term>[A-Za-z][\w \-/&]{1,60}?)[\"“”']\s+has the meaning\s+"
            r"(?:set forth|given|ascribed)\b",
        ),
    ),
)

# Parenthetical labels: '... Acme Consulting LLC ("Provider")'
_PARENTHETICAL_RE = re.compile(
    r"\(\s*(?:the\s+|each\s+(?:a|an)\s+|collectively,?\s+the\s+|hereinafter\s+(?:the\s+)?)?"
    r"[\"“”'](?P<term>[A-Z][\w \-/&]{1,60}?)[\"“”']\s*\)"
)

# Sentences that look definitional even without a clean term match.
_DEFINITIONAL_HINT_RE = re.compile(
    r"(?i)\b(?:means|shall mean|is defined as|has the meaning|for (?:the )?purposes of this "
    r"(?:agreement|clause|section)|in this agreement,)\b"
)

# Terms whose "definition" would be an artefact of the sentence grammar
# ("This Agreement means ...", "The Parties means ...") rather than a real
# defined term.
_STOP_TERMS = {
    "the", "this", "that", "these", "those", "it", "he", "she", "they",
    "provider", "client", "party", "parties", "agreement", "this agreement",
    "the agreement", "the parties", "the party", "each party", "either party",
    "neither party", "no", "none", "nothing", "notwithstanding", "provided",
    "such", "any", "all", "if", "where", "when", "and", "or", "but",
}

# Party labels are legitimate defined terms when introduced parenthetically —
# '... Acme Consulting LLC ("Provider")' is precisely how a contract binds a
# role noun to a legal entity, and a lawyer asking "who is the Provider?"
# needs it. They stay excluded from the prose patterns, where the same words
# are almost always ordinary sentence subjects.
_PARENTHETICAL_STOP_TERMS = _STOP_TERMS - {
    "provider", "client", "party", "parties", "the parties", "the party",
}


@dataclass(frozen=True)
class Definition:
    term: str
    style: str      # "means" | "refers_to" | "parenthetical" | "cross_referenced"
    snippet: str    # the defining sentence, trimmed

    @property
    def key(self) -> str:
        return normalize_term(self.term)


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip(" \"'“”.,;:").lower()


def _sentence_around(text: str, position: int) -> str:
    start = text.rfind(".", 0, position) + 1
    end = text.find(".", position)
    end = len(text) if end == -1 else end + 1
    return re.sub(r"\s+", " ", text[start:end]).strip()


def extract_definitions(text: str) -> list[Definition]:
    """Terms defined in this passage.

    Line wraps are joined first (so '"Confidential\\nInformation" means' is
    still one term) but paragraph breaks are preserved, so a term can never
    run across a blank line into the next paragraph.
    """
    text = collapse_lines(text)
    found: dict[str, Definition] = {}

    for style, pattern in _DEFINITION_RES:
        for m in pattern.finditer(text):
            term = m.group("term").strip()
            key = normalize_term(term)
            if key in _STOP_TERMS or len(key) < 3 or key in found:
                continue
            found[key] = Definition(term=term, style=style,
                                    snippet=_sentence_around(text, m.start()))

    for m in _PARENTHETICAL_RE.finditer(text):
        term = m.group("term").strip()
        key = normalize_term(term)
        if key in _PARENTHETICAL_STOP_TERMS or len(key) < 3 or key in found:
            continue
        found[key] = Definition(term=term, style="parenthetical",
                                snippet=_sentence_around(text, m.start()))

    return list(found.values())


def is_definitional(text: str) -> bool:
    """Whether a passage reads as definitional — used to flag whole
    'Definitions' sections that list terms without one clean pattern each."""
    return bool(_DEFINITIONAL_HINT_RE.search(collapse_whitespace(text)))


# --- question side --------------------------------------------------------

_QUOTED_RE = re.compile(r"[\"“”'](?P<term>[\w][\w \-/&]{1,60}?)[\"“”']")
_ASKED_RE = (
    re.compile(r"(?i)\bwhat does\s+(?P<term>.{2,60}?)\s+mean\b"),
    re.compile(r"(?i)\b(?:definition|meaning)\s+of\s+(?P<term>.{2,60}?)(?:\s+in\b|[?.,]|$)"),
    re.compile(r"(?i)\bhow is\s+(?P<term>.{2,60}?)\s+defined\b"),
    re.compile(r"(?i)\bwhat is meant by\s+(?P<term>.{2,60}?)(?:[?.,]|$)"),
    re.compile(r"(?i)\bdefine\s+(?P<term>.{2,60}?)(?:[?.,]|$)"),
)
_TITLECASE_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b(?P<term>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

_QUESTION_NOISE = {
    "the term", "a term", "term", "the phrase", "the word", "the clause",
    "the agreement", "this agreement", "the contract", "the document",
    "clause", "section", "article", "paragraph", "exhibit", "schedule",
    "annex", "appendix", "version", "page",
}

# Structural references ("Clause 6", "Exhibit A", "v2.1") are addresses, not
# defined terms — Title Case matching would otherwise turn every clause
# reference into a spurious definition lookup.
_STRUCTURAL_RE = re.compile(
    r"(?i)^(?:clause|section|article|paragraph|exhibit|schedule|annex|appendix|"
    r"addendum|attachment|version|v)\b[\s.]*[\dA-Z.]*$"
)


def definition_targets(question: str, known_terms: set[str] | None = None) -> list[str]:
    """Terms the question is asking to have defined.

    Three signals, in decreasing reliability: an explicit "what does X mean"
    frame, a quoted phrase, and Title Case (contract convention for defined
    terms). `known_terms` — the defined terms actually present in the corpus —
    additionally rescues lowercase questions like "what does confidential
    information mean", which no capitalisation heuristic can catch.
    """
    # (candidate, is_strong) — strong signals are explicit "what does X mean"
    # frames and quoted phrases; Title Case alone is only suggestive.
    candidates: list[tuple[str, bool]] = []

    for pattern in _ASKED_RE:
        m = pattern.search(question)
        if m:
            candidates.append((m.group("term"), True))

    candidates.extend((m.group("term"), True) for m in _QUOTED_RE.finditer(question))
    candidates.extend((m.group("term"), False) for m in _TITLECASE_RE.finditer(question))

    known = {t.lower() for t in (known_terms or set())}
    normalized: list[str] = []
    for candidate, strong in candidates:
        key = normalize_term(candidate)
        key = re.sub(r"^(?:the|a|an)\s+", "", key)
        if len(key) < 3 or key in _STOP_TERMS or key in _QUESTION_NOISE:
            continue
        if _STRUCTURAL_RE.match(key):
            continue
        # A lone Title Case word ("Acme", "MSA") is a proper noun far more
        # often than a defined term. Accept it only when the corpus really
        # defines it; multi-word Title Case is contract convention and stands
        # on its own.
        if not strong and " " not in key and key not in known:
            continue
        normalized.append(key)

    if known:
        lowered = question.lower()
        for term in known:
            if len(term) >= 4 and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
                normalized.append(term)

    return list(dict.fromkeys(normalized))
