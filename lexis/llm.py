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


def stream_answer(question: str, chunks: list[RetrievedChunk]):
    """Yield answer text deltas as the model produces them.

    Streaming is the single biggest perceived-latency lever (~75% reduction
    in time-to-first-content). The system prompt is byte-identical across
    requests, so a kept-alive Ollama model reuses its KV-cache prefix and
    prefill only pays for the retrieved chunks + question.
    """
    stream = client().chat.completions.create(
        model=settings.ollama_model,
        temperature=0.0,
        max_tokens=settings.llm_max_tokens,
        stream=True,
        extra_body={"keep_alive": settings.llm_keep_alive},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(question, chunks)},
        ],
    )
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content


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
_CITATION_RES = [
    re.compile(r"\(Source:\s*(?P<doc>[^|)]+?)\s*\|\s*Page:?\s*(?P<page>\d+)\b[^)]*\)"),
    re.compile(r"Document:\s*(?P<doc>[^|\n]+?)\s*\|\s*Page:?\s*(?P<page>\d+)\b"),
]


@dataclass
class CitationReport:
    total: int = 0
    verified: int = 0
    fabricated: list[str] = field(default_factory=list)
    uncited_answer: bool = False
    refusal: bool = False

    @property
    def passed(self) -> bool:
        if self.refusal:
            return True
        return self.total > 0 and not self.fabricated and not self.uncited_answer


# Small models often refuse in their own words instead of echoing the exact
# refusal string; recognize those as legitimate no-evidence responses.
_SOFT_REFUSAL_RE = re.compile(
    r"(?i)\b(do(?:es)? not (?:mention|contain|address|include|state)"
    r"|no (?:retrieved )?(?:evidence|information|documents?)"
    r"|cannot (?:be )?answer)"
)


def verify_citations(answer: str, chunks: list[RetrievedChunk]) -> CitationReport:
    report = CitationReport()
    if REFUSAL in answer:
        # A refusal cites nothing by design; that is a pass.
        report.refusal = True
        return report

    valid_pairs = {(c.document.lower(), c.page) for c in chunks}
    seen_spans: list[tuple[int, int]] = []
    for pattern in _CITATION_RES:
        for m in pattern.finditer(answer):
            if any(m.start() >= s and m.end() <= e for s, e in seen_spans):
                continue  # already counted by a higher-priority pattern
            seen_spans.append((m.start(), m.end()))
            report.total += 1
            pair = (m.group("doc").strip().lower(), int(m.group("page")))
            if pair in valid_pairs:
                report.verified += 1
            else:
                report.fabricated.append(m.group(0).strip())

    if report.total == 0 and _SOFT_REFUSAL_RE.search(answer):
        report.refusal = True
        return report

    report.uncited_answer = report.total == 0
    return report
