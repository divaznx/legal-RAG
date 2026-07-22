"""Shared text utilities for clause-level extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Normalized:
    """Whitespace-collapsed text that can still point back at the original.

    Clause text arrives hard-wrapped, so multi-word legal phrases straddle
    line breaks: "gives not\\nless than ninety (90) days" defeats any pattern
    containing "not less than", and the renewal deadline is silently never
    extracted. Collapsing fixes the matching; the index map preserves the
    character offsets, because a fact that cannot be highlighted in the source
    document is a fact a lawyer has to re-verify by hand.
    """
    text: str
    _origin: list[int]

    def origin(self, index: int) -> int:
        if not self._origin:
            return 0
        return self._origin[min(max(index, 0), len(self._origin) - 1)]


def normalize(text: str) -> Normalized:
    out: list[str] = []
    origin: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space and out:
                out.append(" ")
                origin.append(i)
            in_space = True
        else:
            out.append(ch)
            origin.append(i)
            in_space = False
    return Normalized("".join(out), origin)

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

# Contracts overwhelmingly write "thirty (30) days", so the parenthesised
# digits are the reliable signal and the words are the fallback. Reading the
# words first would also have to handle "thirty-six", "one hundred and
# twenty", and their disagreements with the digits — a class of bug where the
# extracted deadline is silently wrong rather than absent.
_PAREN_DIGITS = re.compile(r"\((\d{1,4})\)")
_BARE_DIGITS = re.compile(r"(?<![\d.])(\d{1,4})(?![\d.%])")


def parse_count(text: str) -> int | None:
    """The integer a quantity phrase denotes: 'thirty (30)' -> 30."""
    m = _PAREN_DIGITS.search(text)
    if m:
        return int(m.group(1))
    m = _BARE_DIGITS.search(text)
    if m:
        return int(m.group(1))

    lowered = text.lower()
    for compound in re.findall(r"\b([a-z]+)-([a-z]+)\b", lowered):
        tens, units = compound
        if tens in _WORD_NUMBERS and units in _WORD_NUMBERS:
            return _WORD_NUMBERS[tens] + _WORD_NUMBERS[units]
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"(?<!\w){word}(?!\w)", lowered):
            return value
    return None


# Split on sentence punctuation only when followed by whitespace and a capital
# or a sub-clause number. The negative lookbehind on digits keeps "12.3" and
# "99.5%" intact, which naive splitting shreds.
_SENTENCE_SPLIT = re.compile(r"(?<![\d])(?<!\bNo)\.(?=\s+(?:[A-Z(]|\d+\.\d))|(?<=[;:])\s+(?=[A-Z])")


def sentences(text: str) -> tuple[Normalized, list[tuple[str, int]]]:
    """Split into sentences over whitespace-collapsed text.

    Returns the `Normalized` view alongside (sentence, offset-in-normalized)
    pairs. Callers match against the normalized text — otherwise every
    multi-word pattern breaks on line wrapping — and convert the resulting
    positions back with `Normalized.origin` so spans still index the original.
    """
    norm = normalize(text)
    body = norm.text
    out: list[tuple[str, int]] = []
    start = 0
    for m in _SENTENCE_SPLIT.finditer(body):
        end = m.end()
        piece = body[start:end]
        stripped = piece.strip()
        if stripped:
            out.append((stripped, start + (len(piece) - len(piece.lstrip()))))
        start = end
    tail = body[start:]
    if tail.strip():
        out.append((tail.strip(), start + (len(tail) - len(tail.lstrip()))))
    return norm, out


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_clause_number(sentence: str) -> tuple[str, int]:
    """Remove a leading sub-clause number, returning the offset consumed."""
    m = re.match(r"^\s*\d+(?:\.\d+)*\s+", sentence)
    return (sentence[m.end():], m.end()) if m else (sentence, 0)
