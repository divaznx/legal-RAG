"""Clause-atomic chunking that preserves citation metadata.

Every chunk knows its document, page, clause/section heading, clause number,
document version, and OCR quality — exactly the fields the answer model is
required to cite, so citations can be mechanically verified later.

THE CHUNK BOUNDARY IS A LEGAL BOUNDARY. A chunk never spans a heading, and
the overlap carried between chunks never crosses one either. Both rules exist
because a chunk carries exactly one section label into the citation: text
packed in from the next clause, or an overlap tail carried in from the
previous one, gets cited under the wrong clause number. For a lawyer that is
not a ranking nuisance, it is a wrong answer with a confident-looking cite —
"Clause 5" printed against the text of Clause 4.

Chunks additionally carry the structure the legal intelligence layer needs:
the terms they define, the clauses and exhibits they reference, and the legal
concepts they cover.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace

from . import security
from .config import settings
from .legal import definitions as legal_definitions
from .legal import ontology, xref
from .legal.entities import normalize_clause_number
from .parsing import Page

_KEYWORD_HEADING_RE = re.compile(
    r"^\s*(?P<kw>ARTICLE|Article|SECTION|Section|CLAUSE|Clause|PARAGRAPH|Paragraph|§)\s*"
    r"(?P<num>\d+(?:\.\d+)*[A-Za-z]?|[IVXLC]+(?:\.\d+)*)"
    r"\s*(?:[.:\-–—)]\s*)?(?P<title>[^\n]*)?"
)

_MAX_TITLE_CHARS = 60
_MAX_TITLE_WORDS = 8


def _clean_title(raw: str | None) -> str:
    """The heading's title, or "" when the text after the number is body prose.

    Sub-clauses usually run straight from the number into the substance —
    '12.1 Nothing in this Agreement limits either party's liability...' — and
    treating that opening line as a title produces citations like "Clause 12.1.
    Nothing in this Agreement limits either party's liability for death or".
    A real heading is a short noun phrase, not a sentence.
    """
    title = (raw or "").strip()
    if ". " in title:  # a sentence continues; not a heading
        return ""
    title = title.strip(" .:-–—")
    if not title or len(title) > _MAX_TITLE_CHARS or len(title.split()) > _MAX_TITLE_WORDS:
        return ""
    return title

# "2.1 Payment Terms" — dotted decimal numbering, near-universal in contracts
# and almost never how an ordinary sentence starts. The title may open with a
# quotation mark because that is exactly how definition sub-clauses are
# written: '1.2 "Confidential Information" means ...'.
_DOTTED_HEADING_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)+)\s*(?:[.:\-–—)]\s*)?(?P<title>[\"“”'A-Z][^\n]*)"
)

# "4. TERMINATION" on a line of its own. Constrained to a whole short line so
# that "30. days after notice" or a numbered list item in running prose cannot
# be mistaken for a clause heading.
_BARE_HEADING_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s*[.):\-–—]\s+(?P<title>[A-Z][A-Za-z][A-Za-z ,/&'\-]{2,60})\s*$"
)


@dataclass(frozen=True)
class Heading:
    kind: str          # "clause" | "section" | "article" | "paragraph"
    number: str        # normalized for lookup: "4", "4.2", roman "IX" -> "9"
    raw_number: str    # as the document writes it: "IX"
    title: str
    raw: str

    @property
    def label(self) -> str:
        # Citations quote the document's own numbering. A lawyer checking a
        # cite for "Article 9" against a contract that says "ARTICLE IX" has
        # to stop and reconcile it; the normalized form exists only so that
        # lookups by either spelling resolve to the same clause.
        return f"{self.kind.capitalize()} {self.raw_number}"

    @property
    def display(self) -> str:
        return f"{self.label}. {self.title}".rstrip(". ") if self.title else self.label


def _first_line(paragraph: str) -> str:
    return paragraph.split("\n", 1)[0]


def parse_heading(paragraph: str) -> Heading | None:
    """Structured heading of a paragraph, or None if it isn't one."""
    line = _first_line(paragraph)

    m = _KEYWORD_HEADING_RE.match(line)
    if m:
        after = (m.group("title") or "").strip()
        # A heading's title is a noun phrase and is capitalised. A lowercase
        # continuation means this is running prose that merely happens to
        # begin with a clause reference — "Clause 6 or Clause 8.", "Clause 12.3
        # does not apply to..." — and treating it as a heading would file that
        # sentence under the wrong clause number.
        if after and not after[0].isupper() and after[0] not in "\"“”'":
            return None
        kw = m.group("kw").lower()
        kind = "clause" if kw == "§" else kw
        return Heading(kind=kind, number=normalize_clause_number(m.group("num")),
                       raw_number=m.group("num").strip(), title=_clean_title(after),
                       raw=line.strip())

    m = _DOTTED_HEADING_RE.match(line)
    if m:
        return Heading(kind="section", number=normalize_clause_number(m.group("num")),
                       raw_number=m.group("num").strip(),
                       title=_clean_title(m.group("title")), raw=line.strip())

    m = _BARE_HEADING_RE.match(line)
    if m:
        return Heading(kind="clause", number=normalize_clause_number(m.group("num")),
                       raw_number=m.group("num").strip(),
                       title=_clean_title(m.group("title")), raw=line.strip())

    return None


def detect_heading(paragraph: str) -> str | None:
    """Citation label of a paragraph's heading ("Clause 4"), or None."""
    heading = parse_heading(paragraph)
    return heading.label if heading else None


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
    # --- legal structure ---
    clause_number: str | None = None          # "4", "4.2" — exact-lookup key
    heading: str | None = None                # "Clause 4. Termination"
    parent_section: str | None = None         # enclosing Article/parent clause
    defined_terms: list[str] = field(default_factory=list)      # normalized keys
    defined_term_labels: list[str] = field(default_factory=list)
    xrefs: list[str] = field(default_factory=list)              # "clause:6", "exhibit:A"
    xref_labels: list[str] = field(default_factory=list)
    incorporates: list[str] = field(default_factory=list)       # binding attachments
    concepts: list[str] = field(default_factory=list)
    is_definition: bool = False
    # text that addresses the AI system rather than stating contractual terms
    is_suspect: bool = False
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
            "clause_number": self.clause_number,
            "heading": self.heading,
            "parent_section": self.parent_section,
            "defined_terms": self.defined_terms,
            "defined_term_labels": self.defined_term_labels,
            "xrefs": self.xrefs,
            "xref_labels": self.xref_labels,
            "incorporates": self.incorporates,
            "concepts": self.concepts,
            "is_definition": self.is_definition,
            "is_suspect": self.is_suspect,
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


@dataclass
class _Section:
    heading: Heading | None
    parent: str | None
    paragraphs: list[str] = field(default_factory=list)


# A heading line inside a block: PDF text extraction frequently drops the
# blank lines between clauses, so splitting on "\n\n" alone leaves an entire
# contract as one paragraph with no detectable clause structure. Length-capped
# because a wrapped prose line that merely begins "Clause 5 shall survive..."
# must not be mistaken for a heading.
_MAX_HEADING_LINE = 100


def _split_on_headings(block: str) -> list[str]:
    """Split a block wherever an internal line is itself a clause heading."""
    lines = block.split("\n")
    pieces: list[str] = []
    current: list[str] = []
    for i, line in enumerate(lines):
        is_heading = (
            i > 0
            and len(line.strip()) <= _MAX_HEADING_LINE
            and parse_heading(line) is not None
        )
        if is_heading and current:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _sectionize(page_text: str) -> list[_Section]:
    """Group a page's paragraphs into heading-delimited sections.

    A new section starts at every heading, so no section — and therefore no
    chunk — ever straddles a clause boundary.
    """
    sections: list[_Section] = [_Section(heading=None, parent=None)]
    current_article: str | None = None
    current_top: Heading | None = None   # last un-numbered-suffix heading, e.g. "Clause 12"

    blocks = [b for raw in re.split(r"\n\s*\n", page_text)
              for b in _split_on_headings(raw.strip()) if b.strip()]

    for raw in blocks:
        pieces = _split_oversized(raw)
        for position, para in enumerate(pieces):
            # Only the first piece of a split paragraph can carry its heading;
            # the continuations belong to the section the heading opened.
            heading = parse_heading(para) if position == 0 else None
            if heading is not None:
                parent = current_article
                if heading.kind == "article":
                    current_article = heading.display
                    current_top = None
                    parent = None
                elif "." not in heading.number:
                    current_top = heading
                else:
                    # A sub-clause inherits its parent's full heading ("Clause
                    # 12. Limitation of Liability") when we have seen it, so
                    # the evidence shows 12.3 under the heading that governs
                    # how it reads.
                    prefix = heading.number.rsplit(".", 1)[0]
                    if current_top is not None and current_top.number == prefix:
                        parent = current_top.display
                        # ...and its NOUN. A contract that numbers its
                        # provisions "Clause 8" calls 8.3 a clause, not a
                        # section; citing "Section 8.3" sends the reader
                        # looking for a part of the document that isn't there.
                        heading = replace(heading, kind=current_top.kind)
                    else:
                        parent = current_article or f"Clause {prefix}"
                sections.append(_Section(heading=heading, parent=parent, paragraphs=[para]))
            else:
                sections[-1].paragraphs.append(para)

    return [s for s in sections if any(p.strip() for p in s.paragraphs)]


def _analyze(text: str) -> dict:
    """Legal structure of a chunk's text, indexed for retrieval."""
    found = legal_definitions.extract_definitions(text)
    refs = xref.extract(text)
    return {
        "defined_terms": [d.key for d in found],
        "defined_term_labels": [d.term for d in found],
        "xrefs": list(dict.fromkeys(r.key() for r in refs)),
        "xref_labels": list(dict.fromkeys(r.label for r in refs)),
        "incorporates": xref.incorporated_attachments(text),
        "concepts": ontology.detect_concepts(text)[:8],
        "is_definition": bool(found) or legal_definitions.is_definitional(text),
        "is_suspect": security.is_suspect(text),
    }


def chunk_pages(pages: list[Page], document: str, version: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0

    def emit(text: str, page: Page, section: _Section, fact_keys: list[str] | None = None) -> None:
        nonlocal counter
        text = text.strip()
        if not text:
            return
        heading = section.heading
        # Text before the first heading is the recitals/preamble — label it as
        # such rather than citing it with a bare "-".
        section_title = heading.label if heading else "Preamble"
        is_definitions_section = bool(
            heading and re.search(r"(?i)definition|interpretation", heading.title or "")
        )
        analysis = _analyze(text)
        chunks.append(
            Chunk(
                document=document,
                page=page.number,
                section=section_title,
                version=version,
                ocr_confidence=page.ocr_confidence,
                ocr_source=page.ocr_source,
                text=text,
                index=counter,
                fact_keys=fact_keys or [],
                clause_number=heading.number if heading else None,
                heading=heading.display if heading else None,
                parent_section=section.parent,
                defined_terms=analysis["defined_terms"],
                defined_term_labels=analysis["defined_term_labels"],
                xrefs=analysis["xrefs"],
                xref_labels=analysis["xref_labels"],
                incorporates=analysis["incorporates"],
                concepts=analysis["concepts"],
                is_definition=analysis["is_definition"] or is_definitions_section,
                is_suspect=analysis["is_suspect"],
            )
        )
        counter += 1

    for page in pages:
        for section in _sectionize(page.text):
            # A section whose only content is its own heading line ("Clause 10.
            # Suspension", with the substance in 10.1 and 10.2) carries no
            # terms. Indexing it spends an evidence slot on a title, and those
            # slots displace the sub-clauses that answer the question. The
            # number stays reachable: clause lookups expand a parent reference
            # to its sub-clauses.
            body = "\n\n".join(section.paragraphs).strip()
            if section.heading and body == section.heading.raw.strip():
                continue

            buf = ""
            for para in section.paragraphs:
                # Key-value fact blocks become their own atomic chunks so
                # explicit facts ("Client: ...") never drown inside prose.
                if is_fact_block(para) and parse_heading(para) is None:
                    emit(buf, page, section)
                    buf = ""
                    emit(para, page, section, fact_keys=extract_fact_keys(para))
                    continue

                if buf and len(buf) + len(para) + 2 > settings.chunk_size:
                    emit(buf, page, section)
                    # Overlap is carried only WITHIN this section, so a chunk
                    # can never inherit text belonging to another clause.
                    buf = buf[-settings.chunk_overlap:] if settings.chunk_overlap else ""
                buf = f"{buf}\n\n{para}" if buf else para
            emit(buf, page, section)

    return chunks
