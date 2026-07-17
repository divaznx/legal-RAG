"""Document parsing: PDF / DOCX / TXT -> pages with provenance metadata.

Each page carries an OCR/extraction-quality heuristic: a PDF page that yields
almost no text is very likely a scan without a text layer, so it is flagged
`possible-scan` with low confidence — downstream, that flag propagates into
chunk metadata and ultimately into the answer's Limitations section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .config import settings


@dataclass
class Page:
    number: int
    text: str
    ocr_confidence: float  # heuristic, 0.0 - 1.0
    ocr_source: str        # "printed" | "possible-scan" | "text"


_VERSION_RE = re.compile(r"[_\-]v(\d+(?:\.\d+)*)", re.IGNORECASE)


def detect_version(filename: str) -> str:
    m = _VERSION_RE.search(Path(filename).stem)
    return m.group(1) if m else "1.0"


def parse_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            # a corrupt page shouldn't kill the whole document — index it as
            # an unreadable (possible-scan) page and let Limitations surface it
            text = ""
        if len(text) < settings.low_text_chars_per_page:
            pages.append(Page(i, text, ocr_confidence=0.3, ocr_source="possible-scan"))
        else:
            pages.append(Page(i, text, ocr_confidence=0.95, ocr_source="printed"))
    return pages


def parse_docx(path: Path) -> list[Page]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    # .docx has no fixed page boundaries; index as a single page 1.
    return [Page(1, text, ocr_confidence=1.0, ocr_source="text")]


def parse_txt(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Page(1, text, ocr_confidence=1.0, ocr_source="text")]


_PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt, ".md": parse_txt}

SUPPORTED_EXTENSIONS = tuple(_PARSERS)


def parse(path: Path) -> list[Page]:
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {path.suffix} (supported: {', '.join(_PARSERS)})")
    pages = parser(path)
    return [p for p in pages if p.text.strip()] or pages[:1]
