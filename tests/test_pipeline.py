"""Unit tests for the pure-logic pipeline stages (no LLM, no vector DB).

Run:  python -m pytest tests/ -q     (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.chunking import Chunk, chunk_pages, detect_heading, extract_fact_keys, is_fact_block, _split_oversized
from lexis.config import settings
from lexis.llm import CitationReport, verify_citations
from lexis.parsing import Page, detect_version
from lexis.prompts import REFUSAL
from lexis.redaction import redact
from lexis.vector_store import RetrievedChunk


def _chunks(text: str) -> list[Chunk]:
    return chunk_pages([Page(1, text, 1.0, "text")], document="t.txt", version="1.0")


def test_redaction():
    r = redact("Contact email: a.b@example.com and SSN 123-45-6789, call +1 415 555-0142.")
    assert "[EMAIL]" in r.text and "[SSN]" in r.text and "[PHONE]" in r.text
    assert "a.b@example.com" not in r.text and "123-45-6789" not in r.text
    # keyword-anchored: anchor text survives, value doesn't
    r = redact("Client: Jane Doe\nClient ID: XY-1234")
    assert "Client: [CLIENT_NAME]" in r.text and "Client ID: [CUSTOMER_ID]" in r.text
    # name matching never crosses a line boundary
    r = redact("Client: Jane Doe\nAgreement Terms Follow")
    assert "Agreement Terms Follow" in r.text


def test_heading_detection():
    assert detect_heading("Clause 4. Termination") == "Clause 4"
    assert detect_heading("ARTICLE IX — Liability") == "Article IX"
    assert detect_heading("2.1 Payment Terms shall apply") is not None
    assert detect_heading("The parties agree as follows.") is None


def test_fact_block():
    block = "Client: [CLIENT_NAME]\nClient ID: [CUSTOMER_ID]\nContact email: [EMAIL]"
    assert is_fact_block(block)
    assert extract_fact_keys(block) == ["client", "client id", "contact email"]
    assert not is_fact_block("The Provider shall deliver the services described in Exhibit A.")


def test_fact_block_atomic_chunk():
    text = (
        "This Agreement is entered into by the undersigned client.\n\n"
        "Client: [CLIENT_NAME]\nClient ID: [CUSTOMER_ID]\n\n"
        "Clause 1. Services\nThe Provider shall deliver services."
    )
    chunks = _chunks(text)
    fact = [c for c in chunks if c.fact_keys]
    assert len(fact) == 1
    assert fact[0].text.startswith("Client:")
    assert fact[0].section == "Preamble"
    assert fact[0].fact_keys == ["client", "client id"]


def test_section_not_inherited_from_later_heading():
    text = "Preamble text before any clause.\n\nClause 1. Services\nBody of clause one."
    chunks = _chunks(text)
    # the chunk starts in the preamble: its section must not claim 'Clause 1'
    # unless the buffer started at/after that heading
    assert chunks[0].section is None


def test_oversized_paragraph_split():
    sentence = "This is a sentence about obligations. "
    huge = sentence * 60  # ~2300 chars, no blank lines
    pieces = _split_oversized(huge.strip())
    assert len(pieces) > 1
    assert all(len(p) <= settings.chunk_size for p in pieces)
    chunks = _chunks(huge)
    assert all(len(c.text) <= settings.chunk_size + settings.chunk_overlap + 2 for c in chunks)


def test_version_detection():
    assert detect_version("MSA_Acme_v2.1.txt") == "2.1"
    assert detect_version("contract-v3.pdf") == "3"
    assert detect_version("plain.docx") == "1.0"


def _rc(document: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        score=0.5, dense_score=0.7, document=document, page=page, section=None,
        version="1.0", ocr_confidence=1.0, ocr_source="text", text="text",
    )


def test_citation_verification_pass():
    answer = "The fee is USD 100. (Source: MSA_v1.txt | Page 3 | Clause 2 | v1.0)"
    report = verify_citations(answer, [_rc("MSA_v1.txt", 3)])
    assert report.passed and report.verified == 1 and not report.fabricated


def test_citation_verification_catches_fabrication():
    answer = (
        "A real claim. (Source: MSA_v1.txt | Page 3 | Clause 2 | v1.0)\n"
        "A fabricated one. (Source: Ghost_Agreement.pdf | Page 99 | Clause 1 | v9.0)"
    )
    report = verify_citations(answer, [_rc("MSA_v1.txt", 3)])
    assert not report.passed
    assert report.verified == 1 and len(report.fabricated) == 1
    assert "Ghost_Agreement.pdf" in report.fabricated[0]


def test_citation_verification_uncited_answer_fails():
    report = verify_citations("The fee is definitely USD 100, trust me.", [_rc("a.txt", 1)])
    assert not report.passed and report.uncited_answer


def test_refusals_pass_verification():
    assert verify_citations(REFUSAL, [_rc("a.txt", 1)]).passed
    soft = "The retrieved documents do not mention GDPR at all."
    report = verify_citations(soft, [_rc("a.txt", 1)])
    assert report.passed and report.refusal


def test_chunk_header_echo_counts_as_citation():
    answer = "Per Document: MSA_v1.txt | Page: 3 the fee is USD 100."
    report = verify_citations(answer, [_rc("MSA_v1.txt", 3)])
    assert report.passed and report.verified == 1


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
