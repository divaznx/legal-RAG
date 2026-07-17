"""PII redaction applied at ingest time.

Confidential values are replaced with the placeholder vocabulary the answer
model is trained on ([EMAIL], [PHONE], [SSN], ...). Redaction happens before
chunking/embedding, so raw PII never reaches the vector store or the LLM.

Regex-based redaction is deterministic but not exhaustive — in particular,
person/client *names* are only caught when keyword-anchored ("Client: John
Doe"). This limitation is surfaced in the README.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Order matters: more specific patterns run first so e.g. an SSN is not
# half-eaten by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "[PASSPORT]",
        re.compile(r"(?i)\bpassport\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z][0-9]{7,8}|[A-Z]{2}[0-9]{6,7})\b"),
    ),
    (
        "[ACCOUNT_NUMBER]",
        re.compile(r"(?i)\b(?:account|acct\.?|iban)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z]{0,4}[0-9][0-9 \-]{6,28}[0-9])\b"),
    ),
    ("[ACCOUNT_NUMBER]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),  # card-like digit runs
    (
        "[TAX_ID]",
        re.compile(r"(?i)\b(?:tax\s*id|tin|ein|pan)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9\-]{8,15})\b"),
    ),
    (
        "[CUSTOMER_ID]",
        re.compile(r"(?i)\b(?:customer|client)\s*id\s*[:\-]?\s*([A-Z0-9\-]{4,20})\b"),
    ),
    (
        "[PHONE]",
        re.compile(r"(?<![\d.])(?:\+?\d{1,3}[ .\-]?)?(?:\(\d{2,4}\)[ .\-]?)?\d{3,4}[ .\-]\d{3,4}(?:[ .\-]\d{2,4})?(?![\d.])"),
    ),
    (
        "[CLIENT_NAME]",
        re.compile(r"(?i)\b(?:client|claimant|plaintiff|defendant)[ \t]*(?:name)?[ \t]*:[ \t]*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})"),
    ),
]


@dataclass
class RedactionResult:
    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: Counter = Counter()
    for placeholder, pattern in _PATTERNS:
        def _sub(match: re.Match, _p: str = placeholder) -> str:
            counts[_p] += 1
            # Keyword-anchored patterns capture the sensitive value in group 1;
            # keep the anchor text so the sentence still reads naturally.
            if match.groups() and match.group(1) is not None:
                start = match.start(1) - match.start()
                return match.group(0)[:start] + _p
            return _p

        text = pattern.sub(_sub, text)
    return RedactionResult(text=text, counts=counts)
