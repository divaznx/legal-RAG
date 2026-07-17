"""Section-aware chunking that preserves citation metadata.

Every chunk knows its document, page, nearest preceding section/clause
heading, document version, and OCR quality — exactly the fields the answer
model is required to cite, so citations can be mechanically verified later.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from .config import settings
from .parsing import Page

_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?P<kw>ARTICLE|Article|SECTION|Section|CLAUSE|Clause)\s+(?P<num>[0-9IVXLCivxlc]+(?:\.\d+)*)"
    r"|(?P<plain>\d+(?:\.\d+)+\s+[A-Z][^\n]{0,60})"
    r")"
)


def detect_heading(paragraph: str) -> str | None:
    m = _HEADING_RE.match(paragraph)
    if not m:
        return None
    if m.group("kw"):
        return f"{m.group('kw').capitalize()} {m.group('num')}"
    return m.group("plain").strip()


# A "fact block" is a run of key-value lines ("Client: ...", "Client ID: ...",
# "Effective Date: ..."). These carry the document's explicit facts, so they
# are emitted as their own atomic chunks: short, term-dense chunks rank far
# higher on BM25 and cross-encoders than the same lines diluted inside a
# 900-char prose chunk — which is what makes "Who is the client?" retrieve
# the fact line instead of generic "the undersigned client" prose.
_FACT_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9 ./#()'&-]{0,40}:\s*\S")


def is_fact_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    hits = sum(1 for line in lines if _FACT_LINE_RE.match(line))
    if hits >= 2 and hits >= len(lines) - 1:
        return True
    return hits == 1 and len(lines) == 1 and len(lines[0]) < 100


def extract_fact_keys(paragraph: str) -> list[str]:
    """Normalized keys of a fact block's key-value lines, e.g. ["client",
    "client id", "contact email"]. Indexed in the payload so the engine can
    deterministically boost fact chunks when the question names a key —
    redaction strips the values, so the keys are the retrievable signal."""
    keys = []
    for line in paragraph.splitlines():
        line = line.strip()
        if _FACT_LINE_RE.match(line):
            keys.append(line.split(":", 1)[0].strip().lower())
    return keys


@dataclass
class Chunk:
    document: str
    page: int
    section: str | None
    version: str
    ocr_confidence: float
    ocr_source: str
    text: str
    index: int = 0
    fact_keys: list[str] = field(default_factory=list)
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.document}:p{self.page}:{self.index}"
            self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def payload(self) -> dict:
        return {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "version": self.version,
            "ocr_confidence": self.ocr_confidence,
            "ocr_source": self.ocr_source,
            "text": self.text,
            "fact_keys": self.fact_keys,
        }


def _split_oversized(paragraph: str) -> list[str]:
    """Split a paragraph longer than chunk_size on sentence boundaries.

    Without this, a single huge paragraph becomes one oversized chunk; the
    embedding model truncates at 512 tokens, so everything past the cutoff
    would be stored but unsearchable.
    """
    if len(paragraph) <= settings.chunk_size:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.;:])\s+", paragraph):
        while len(sentence) > settings.chunk_size:  # pathological unbroken run
            pieces.append(sentence[: settings.chunk_size])
            sentence = sentence[settings.chunk_size:]
        if current and len(current) + len(sentence) + 1 > settings.chunk_size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages: list[Page], document: str, version: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        section: str | None = None
        buf = ""
        buf_section: str | None = None

        def flush() -> None:
            nonlocal buf, counter
            if not buf.strip():
                buf = ""
                return
            chunks.append(
                Chunk(
                    document=document,
                    page=page.number,
                    section=buf_section,
                    version=version,
                    ocr_confidence=page.ocr_confidence,
                    ocr_source=page.ocr_source,
                    text=buf.strip(),
                    index=counter,
                )
            )
            counter += 1
            # carry a tail of the flushed text into the next chunk as overlap
            buf = buf[-settings.chunk_overlap:] if settings.chunk_overlap else ""

        paragraphs: list[str] = []
        for raw in re.split(r"\n\s*\n", page.text):
            raw = raw.strip()
            if raw:
                paragraphs.extend(_split_oversized(raw))

        for para in paragraphs:
            heading = detect_heading(para)
            if heading:
                section = heading

            # Key-value fact blocks become their own atomic chunks so explicit
            # facts ("Client: ...") never drown inside surrounding prose.
            if not heading and is_fact_block(para):
                flush()
                buf = ""  # no overlap bleed into or out of a fact chunk
                chunks.append(
                    Chunk(
                        document=document,
                        page=page.number,
                        section=section or "Preamble",
                        version=version,
                        ocr_confidence=page.ocr_confidence,
                        ocr_source=page.ocr_source,
                        text=para,
                        index=counter,
                        fact_keys=extract_fact_keys(para),
                    )
                )
                counter += 1
                buf_section = None
                continue

            if buf and len(buf) + len(para) + 2 > settings.chunk_size:
                flush()
                buf_section = section
            if not buf:
                # a fresh buffer takes the section active *now* — never a
                # heading that appears later inside the chunk
                buf_section = section
            buf = f"{buf}\n\n{para}" if buf else para
        flush()
    return chunks
