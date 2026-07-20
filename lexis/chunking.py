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
    r"(?P<kw>ARTICLE|Article|SECTION|Section|CLAUSE|Clause|ITEM|Item)\s+(?P<num>[0-9IVXLCivxlc]+(?:\.\d+)*)"
    r"|(?P<akw>EXHIBIT|Exhibit|SCHEDULE|Schedule|APPENDIX|Appendix|ATTACHMENT|Attachment)\s+"
    r"(?P<aid>[A-Z](?![a-z])|\d+(?:\.\d+)*)"
    r"|(?P<plain>\d+(?:\.\d+)+\s+[A-Z][^\n]{0,60})"
    r")"
)


def detect_heading(paragraph: str) -> str | None:
    m = _HEADING_RE.match(paragraph)
    if not m:
        return None
    if m.group("kw"):
        return f"{m.group('kw').capitalize()} {m.group('num')}"
    if m.group("akw"):
        return f"{m.group('akw').capitalize()} {m.group('aid')}"
    return m.group("plain").strip()


_SECTION_NUM_RE = re.compile(
    r"^(?P<kw>Article|Section|Clause)?\s*(?P<num>\d+(?:\.\d+)*)", re.IGNORECASE
)


def parent_of(section: str | None) -> str | None:
    """One level up the numbering hierarchy (Feature: Parent-Child Retrieval).

    "Clause 4.2" -> "Clause 4"; "2.1 Payment Terms" -> "2"; top-level
    sections and roman-numeral articles have no derivable parent -> None.
    """
    if not section:
        return None
    m = _SECTION_NUM_RE.match(section.strip())
    if not m or "." not in m.group("num"):
        return None
    parent_num = m.group("num").rsplit(".", 1)[0]
    kw = m.group("kw")
    return f"{kw.capitalize()} {parent_num}" if kw else parent_num


# --- Table detection (Feature: Table Preservation) -------------------------
# A block is treated as a table when multiple lines share columnar structure:
# pipe-separated cells, or 3+ aligned multi-space column gaps. Table blocks
# are chunked atomically and never split or overlap-bled, so payment
# schedules etc. reach the LLM with rows/columns intact.
_PIPE_ROW_RE = re.compile(r"\|.*\|")
_COLUMN_GAP_RE = re.compile(r"\S(?:  +|\t)\S")


def is_table_block(paragraph: str) -> bool:
    lines = [line for line in paragraph.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    pipe_rows = sum(1 for line in lines if _PIPE_ROW_RE.search(line))
    if pipe_rows >= 2:
        return True
    columnar = sum(1 for line in lines if len(_COLUMN_GAP_RE.findall(line)) >= 2)
    return columnar >= 2 and columnar >= len(lines) - 1


# --- Defined terms (Feature: Definition-aware Retrieval) -------------------
# Terms a legal document explicitly defines: `"Confidential Information"
# means ...` or inline parentheticals `("Provider")`. Indexed per chunk so
# the engine can boost definition chunks when a question uses the term.
_DEFINED_MEANS_RE = re.compile(r"[\"“]([A-Z][A-Za-z ]{1,40})[\"”]\s+(?:shall mean|means)")
_DEFINED_PAREN_RE = re.compile(r"\(\s*(?:the\s+)?[\"“]([A-Z][A-Za-z ]{1,40})[\"”]\s*\)")


def extract_defined_terms(text: str) -> list[str]:
    terms = []
    for pattern in (_DEFINED_MEANS_RE, _DEFINED_PAREN_RE):
        for m in pattern.finditer(text):
            term = m.group(1).strip()
            if term and term not in terms:
                terms.append(term)
    return terms


# --- Cross-references (Feature: Cross-Reference Resolution) -----------------
# Any structural target a legal drafter can point at: numbered units
# (Clause/Section/Article/Item 3.03) and lettered/numbered attachments
# (Exhibit A, Schedule 2.1, Appendix 3, Attachment B). Trigger phrases like
# "incorporated by reference", "subject to", "pursuant to", "as described
# in", and "see ..." all resolve to one of these target forms, so extracting
# every target covers every trigger phrase without a fragile phrase list.
_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"(?P<kw>Clause|Section|Article|Item)\s+(?P<num>\d+(?:\.\d+)*|[IVXLC]+)"
    r"|(?P<akw>Exhibit|Schedule|Appendix|Attachment)\s+(?P<aid>[A-Z](?![a-z])|\d+(?:\.\d+)*)"
    r")\b"
)


def extract_references(text: str, own_section: str | None) -> list[str]:
    """Structural targets this chunk's text mentions, excluding its own
    section — the edges of the cross-reference graph."""
    own = (own_section or "").lower()
    refs = []
    for m in _REFERENCE_RE.finditer(text):
        if m.group("kw"):
            label = f"{m.group('kw').capitalize()} {m.group('num')}"
        else:
            label = f"{m.group('akw').capitalize()} {m.group('aid')}"
        if label.lower() != own and label not in refs:
            refs.append(label)
    return refs


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
    # Hierarchy (parent-child retrieval): one level up the section numbering.
    parent_section: str | None = None
    heading: str | None = None          # full heading line, e.g. "Clause 4. Termination"
    is_table: bool = False              # atomic table block — structure preserved
    defined_terms: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # sections mentioned in text
    # Document-level enrichment, stamped by ingest (all optional).
    doc_meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.document}:p{self.page}:{self.index}"
            self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
        if self.parent_section is None:
            self.parent_section = parent_of(self.section)

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
            "index": self.index,
            "parent_section": self.parent_section,
            "heading": self.heading,
            "is_table": self.is_table,
            "defined_terms": self.defined_terms,
            "references": self.references,
            **self.doc_meta,
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


def chunk_pages(
    pages: list[Page],
    document: str,
    version: str,
    doc_meta: dict | None = None,
) -> list[Chunk]:
    doc_meta = doc_meta or {}
    chunks: list[Chunk] = []
    counter = 0

    def make_chunk(**kwargs) -> Chunk:
        nonlocal counter
        text = kwargs.pop("text")
        chunk = Chunk(
            document=document,
            version=version,
            text=text,
            index=counter,
            defined_terms=extract_defined_terms(text),
            references=extract_references(text, kwargs.get("section")),
            doc_meta=doc_meta,
            **kwargs,
        )
        counter += 1
        return chunk

    for page in pages:
        section: str | None = None
        heading_line: str | None = None
        buf = ""
        buf_section: str | None = None
        buf_heading: str | None = None

        def flush() -> None:
            nonlocal buf
            if not buf.strip():
                buf = ""
                return
            chunks.append(
                make_chunk(
                    page=page.number,
                    section=buf_section,
                    heading=buf_heading,
                    ocr_confidence=page.ocr_confidence,
                    ocr_source=page.ocr_source,
                    text=buf.strip(),
                )
            )
            # carry a tail of the flushed text into the next chunk as overlap
            buf = buf[-settings.chunk_overlap:] if settings.chunk_overlap else ""

        # Tables are exempted from oversized-paragraph splitting: breaking a
        # table mid-row destroys exactly the structure we must preserve.
        paragraphs: list[str] = []
        for raw in re.split(r"\n\s*\n", page.text):
            raw = raw.strip()
            if not raw:
                continue
            if is_table_block(raw):
                paragraphs.append(raw)
            else:
                paragraphs.extend(_split_oversized(raw))

        for para in paragraphs:
            heading = detect_heading(para)
            if heading:
                # A new clause begins: close out the previous chunk and start
                # clean — no overlap bleed across a clause boundary, so every
                # chunk's section label is exact and no clause is split
                # across chunks unless it alone exceeds chunk_size.
                if settings.chunk_at_headings and buf.strip():
                    flush()
                    buf = ""
                section = heading
                heading_line = para.splitlines()[0].strip()

            # Tables and key-value fact blocks become their own atomic chunks:
            # explicit facts and schedules never drown inside prose, and no
            # overlap bleeds into or out of them.
            atomic_table = not heading and is_table_block(para)
            atomic_fact = not heading and not atomic_table and is_fact_block(para)
            if atomic_table or atomic_fact:
                flush()
                buf = ""
                chunks.append(
                    make_chunk(
                        page=page.number,
                        section=section or "Preamble",
                        heading=heading_line,
                        ocr_confidence=page.ocr_confidence,
                        ocr_source=page.ocr_source,
                        text=para,
                        fact_keys=extract_fact_keys(para) if atomic_fact else [],
                        is_table=atomic_table,
                    )
                )
                buf_section = None
                continue

            if buf and len(buf) + len(para) + 2 > settings.chunk_size:
                flush()
                buf_section = section
                buf_heading = heading_line
            if not buf:
                # a fresh buffer takes the section active *now* — never a
                # heading that appears later inside the chunk
                buf_section = section
                buf_heading = heading_line
            buf = f"{buf}\n\n{para}" if buf else para
        flush()
    return chunks
