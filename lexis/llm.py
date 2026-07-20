"""LLM access (OpenAI-compatible / Ollama) + the citation-verification guardrail.

The guardrail is mechanical, not model-based: every `(Source: ...)` tag in
the generated answer is parsed and checked against the (document, page) pairs
actually present in the retrieved chunks. Fabricated citations are reported;
an Answer section with zero citations fails verification outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import httpx
from openai import OpenAI

from .config import settings
from .prompts import REFUSAL, SYSTEM_PROMPT, user_message
from .vector_store import RetrievedChunk


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key)


def llm_available() -> bool:
    base = settings.ollama_base_url.rstrip("/").removesuffix("/v1")
    try:
        return httpx.get(f"{base}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    corrective_note: str | None = None,
    format_hint: str | None = None,
    max_tokens: int | None = None,
):
    """Yield answer text deltas as the model produces them.

    Streaming is the single biggest perceived-latency lever (~75% reduction
    in time-to-first-content). The system prompt is byte-identical across
    requests, so a kept-alive Ollama model reuses its KV-cache prefix and
    prefill only pays for the retrieved chunks + question.

    `corrective_note` powers the one-shot regeneration after a failed
    citation verification: it is appended to the user message, so the KV
    prefix (system prompt) is still reused.
    """
    content = user_message(question, chunks, format_hint=format_hint)
    if corrective_note:
        content += f"\n\nIMPORTANT CORRECTION:\n{corrective_note}"
    stream = client().chat.completions.create(
        model=settings.ollama_model,
        temperature=0.0,
        max_tokens=max_tokens or settings.llm_max_tokens,
        stream=True,
        extra_body={"keep_alive": settings.llm_keep_alive},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def corrective_note_for(chunks: list[RetrievedChunk], reasons: list[str] | None = None) -> str:
    """The exact valid citation strings for a regeneration attempt, prefixed
    with the specific reasons the previous answer failed."""
    valid = []
    for c in chunks:
        section = c.section or "-"
        cite = f"(Source: {c.document} | Page {c.page} | {section} | v{c.version})"
        if cite not in valid:
            valid.append(cite)
    why = f" Specifically: {'; '.join(reasons)}." if reasons else ""
    return (
        f"Your previous answer failed citation verification.{why} Regenerate it "
        "using ONLY statements supported by the retrieved chunks above, and "
        "cite ONLY these exact citations:\n" + "\n".join(f"- {v}" for v in valid) +
        "\nRemove any statement you cannot support with one of these citations."
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    return "".join(stream_answer(question, chunks)).strip()


def warm() -> None:
    """Load the model into memory (and pin it via keep_alive) off the hot path."""
    if not llm_available():
        return
    client().chat.completions.create(
        model=settings.ollama_model,
        max_tokens=1,
        extra_body={"keep_alive": settings.llm_keep_alive},
        messages=[{"role": "user", "content": "ok"}],
    )


# Primary form: (Source: doc | Page n | Clause | vX). Fallback catches the
# chunk-header style smaller models tend to echo: "Document: doc | Page: n".
# The optional <sec> group captures the clause/section segment so fabricated
# clause numbers are caught too (Feature: Answer Verification).
_CITATION_RES = [
    re.compile(
        r"\(Source:\s*(?P<doc>[^|)]+?)\s*\|\s*Page:?\s*(?P<page>\d+)\s*"
        r"(?:\|\s*(?P<sec>[^|)]+?)\s*)?(?:\|[^)]*)?\)"
    ),
    re.compile(r"Document:\s*(?P<doc>[^|\n]+?)\s*\|\s*Page:?\s*(?P<page>\d+)\b(?P<sec>)?"),
]


@dataclass
class CitationReport:
    total: int = 0
    verified: int = 0
    fabricated: list[str] = field(default_factory=list)
    clause_mismatches: list[str] = field(default_factory=list)
    uncited_answer: bool = False
    refusal: bool = False

    @property
    def passed(self) -> bool:
        if self.refusal:
            return True
        return (
            self.total > 0
            and not self.fabricated
            and not self.clause_mismatches
            and not self.uncited_answer
        )

    def failure_reasons(self) -> list[str]:
        """Named reasons verification failed — for diagnostics and the
        regeneration corrective note (Feature: Answer Verification)."""
        if self.passed:
            return []
        reasons: list[str] = []
        if self.fabricated:
            reasons.append(f"fabricated_citation x{len(self.fabricated)} (document/page not in retrieved set)")
        if self.clause_mismatches:
            reasons.append(f"clause_mismatch x{len(self.clause_mismatches)} (real page, fabricated clause number)")
        if self.uncited_answer:
            reasons.append("missing_citations (no verifiable citation in the answer)")
        return reasons


# Small models often refuse in their own words instead of echoing the exact
# refusal string; recognize those as legitimate no-evidence responses.
_SOFT_REFUSAL_RE = re.compile(
    r"(?i)\b(do(?:es)? not (?:mention|contain|address|include|state)"
    r"|no (?:retrieved )?(?:evidence|information|documents?)"
    r"|cannot (?:be )?answer)"
)


def _normalize_section(sec: str) -> str:
    normalized = " ".join(sec.strip().rstrip(".").lower().split())
    # Models sometimes echo the header field label ("Section: Item 3.03");
    # strip the label-with-colon prefix — "Section 5" (no colon) is kept.
    return re.sub(r"^(?:section|clause|article|item)\s*:\s*", "", normalized)


def _section_matches(cited: str, known: set[str]) -> bool:
    """Lenient structural match: exact, or numbering prefix either way, so
    citing "Clause 4" against a "Clause 4.2" chunk (or vice versa) passes."""
    for k in known:
        if cited == k or cited.startswith(k + ".") or k.startswith(cited + "."):
            return True
        if cited.startswith(k + " ") or k.startswith(cited + " "):
            return True
    return False


def verify_citations(answer: str, chunks: list[RetrievedChunk]) -> CitationReport:
    report = CitationReport()
    if REFUSAL in answer:
        # A refusal cites nothing by design; that is a pass.
        report.refusal = True
        return report

    valid_pairs = {(c.document.lower(), c.page) for c in chunks}
    # Sections known per (doc, page) — child sections plus their parents, so
    # a citation of either hierarchy level verifies. Pairs where any chunk
    # has no section stay lenient: the clause segment cannot be verified.
    sections: dict[tuple[str, int], set[str]] = {}
    unverifiable: set[tuple[str, int]] = set()
    for c in chunks:
        key = (c.document.lower(), c.page)
        if c.section is None:
            unverifiable.add(key)
            continue
        bucket = sections.setdefault(key, set())
        bucket.add(_normalize_section(c.section))
        if c.parent_section:
            bucket.add(_normalize_section(c.parent_section))

    seen_spans: list[tuple[int, int]] = []
    for pattern in _CITATION_RES:
        for m in pattern.finditer(answer):
            if any(m.start() >= s and m.end() <= e for s, e in seen_spans):
                continue  # already counted by a higher-priority pattern
            seen_spans.append((m.start(), m.end()))
            report.total += 1
            pair = (m.group("doc").strip().lower(), int(m.group("page")))
            if pair not in valid_pairs:
                report.fabricated.append(m.group(0).strip())
                continue
            cited_sec = (m.groupdict().get("sec") or "").strip()
            if (
                cited_sec
                and cited_sec != "-"
                and not re.fullmatch(r"v\d[\d.]*", cited_sec, re.IGNORECASE)
                and pair not in unverifiable
                and pair in sections
                and not _section_matches(_normalize_section(cited_sec), sections[pair])
            ):
                # Real document+page, but a clause number no retrieved chunk
                # of that page carries — a fabricated clause reference.
                report.clause_mismatches.append(m.group(0).strip())
                continue
            report.verified += 1

    if report.total == 0 and _SOFT_REFUSAL_RE.search(answer):
        report.refusal = True
        return report

    report.uncited_answer = report.total == 0
    return report
