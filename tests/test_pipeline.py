"""Unit tests for the pure-logic pipeline stages (no LLM, no vector DB).

Run:  python -m pytest tests/ -q     (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.chunking import (
    Chunk, chunk_pages, detect_heading, extract_defined_terms, extract_fact_keys,
    extract_references, is_fact_block, is_table_block, parent_of, _split_oversized,
)
from lexis.config import settings
from lexis.evaluation import (
    evaluate_case_retrieval, GoldenCase, mrr, ndcg_at_k, precision_at_k, recall_at_k,
)
from lexis.graph import build_document_graph
from lexis.llm import CitationReport, verify_citations
from lexis.metadata import extract_doc_meta
from lexis.packing import pack, _dedupe, _merge_adjacent, _strip_repeated_headings
from lexis.parsing import Page, detect_version
from lexis.prompts import REFUSAL, format_context
from lexis.query import (
    QueryClass, classify, decompose, defined_terms_in, rewrite, strategy_for,
)
from lexis.redaction import redact
from lexis.retrieval import base_document, select_final, _version_tuple
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


# --- Parent-child hierarchy (Feature 1) -------------------------------------

def test_parent_of():
    assert parent_of("Clause 4.2") == "Clause 4"
    assert parent_of("Section 2.1.3") == "Section 2.1"
    assert parent_of("2.1 Payment Terms") == "2"
    assert parent_of("Clause 4") is None
    assert parent_of("Article IX") is None
    assert parent_of(None) is None


def test_chunks_never_span_clause_boundaries():
    text = (
        "Clause 1. Services\nThe Provider shall deliver services.\n\n"
        "Clause 2. Fees\nThe Client shall pay a monthly retainer of USD 12,000.\n\n"
        "Clause 3. Term\nTwelve months, renewing automatically."
    )
    chunks = _chunks(text)
    sections = [c.section for c in chunks]
    assert sections == ["Clause 1", "Clause 2", "Clause 3"]
    # each chunk holds exactly its own clause — no overlap bleed across headings
    assert chunks[1].text.startswith("Clause 2.")
    assert "Clause 1" not in chunks[1].text


def test_chunk_gets_parent_section():
    text = "Clause 4.2 Cure Period\nA breach must be cured within fifteen days."
    chunks = _chunks(text)
    assert chunks[0].section == "Clause 4.2"
    assert chunks[0].parent_section == "Clause 4"


# --- Table preservation (Feature 15) ----------------------------------------

def test_table_detection():
    table = "Item | Amount | Due\nRetainer | 12000 | Net 30\nSupport | 2000 | Net 45"
    assert is_table_block(table)
    assert not is_table_block("The Provider shall deliver the services.")
    columnar = "Milestone one    2026-01-01    USD 5,000\nMilestone two    2026-03-01    USD 7,500"
    assert is_table_block(columnar)


def test_table_chunked_atomically():
    text = (
        "Clause 2. Fees\nThe payment schedule is set out below.\n\n"
        "Item | Amount | Due\nRetainer | 12000 | Net 30\nSupport | 2000 | Net 45\n\n"
        "All amounts are in USD."
    )
    chunks = _chunks(text)
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    assert tables[0].text.splitlines()[0] == "Item | Amount | Due"
    assert tables[0].section == "Clause 2"


# --- Defined terms & references (Features 6 / 14) ---------------------------

def test_defined_terms_extraction():
    text = ('"Confidential Information" means any non-public information. '
            'This agreement is between Acme Consulting LLC ("Provider") and you.')
    terms = extract_defined_terms(text)
    assert "Confidential Information" in terms and "Provider" in terms


def test_reference_extraction():
    text = "except for breaches of Clause 5 (Confidentiality), see also Section 2.1."
    refs = extract_references(text, own_section="Clause 6")
    assert refs == ["Clause 5", "Section 2.1"]
    assert extract_references("as stated in Clause 6.", "Clause 6") == []


def test_graph_build():
    chunks = _chunks(
        "Clause 6. Liability\nLiability is capped, except for breaches of Clause 5."
    )
    edges = build_document_graph(chunks)
    assert edges == {"Clause 6": ["Clause 5"]}


# --- Metadata enrichment (Feature 7) ----------------------------------------

def test_doc_meta_extraction():
    text = (
        "MASTER SERVICES AGREEMENT\n\nThis Agreement has an Effective Date: "
        "January 1, 2026.\n\nClause 7. Governing Law\nThis Agreement is governed "
        "by the laws of the State of Delaware."
    )
    meta = extract_doc_meta("MSA_Acme_v1.0.txt", text)
    assert meta.document_type == "txt"
    assert meta.agreement_type == "MSA"
    assert meta.jurisdiction == "State of Delaware"
    assert meta.effective_date == "January 1, 2026"
    assert meta.language == "en"
    assert meta.tenant is None and meta.permissions == []


def test_doc_meta_stamped_on_chunks():
    pages = [Page(1, "Clause 1. Services\nThe Provider shall deliver services.", 1.0, "text")]
    chunks = chunk_pages(pages, "t.txt", "1.0", doc_meta={"agreement_type": "MSA", "tenant": None})
    assert chunks[0].payload()["agreement_type"] == "MSA"


# --- Query classification (Feature 3) ---------------------------------------

def test_query_classification():
    assert classify("Who is the client?") == QueryClass.ENTITY
    assert classify("What is the monthly retainer?") == QueryClass.PAYMENT
    assert classify("What is the termination notice period?") == QueryClass.TERMINATION
    assert classify("What changed between version 1.0 and version 2.1?") == QueryClass.COMPARISON
    # a document noun routes summarization to whole-document overview…
    assert classify("Summarize the key terms of the agreement") == QueryClass.OVERVIEW
    # …while topic summaries without one stay SUMMARIZATION
    assert classify("Summarize the key points on liability") == QueryClass.SUMMARIZATION
    assert classify("How is Confidential Information defined?") == QueryClass.DEFINITION
    assert classify("What does Clause 4 say?") == QueryClass.CLAUSE
    assert classify("Tell me about the weather") == QueryClass.GENERAL


def test_strategy_adjustments():
    assert strategy_for(QueryClass.ENTITY).bm25_weight > strategy_for(QueryClass.GENERAL).bm25_weight
    assert strategy_for(QueryClass.COMPARISON).per_document
    assert strategy_for(QueryClass.DEFINITION).definitions_first
    summ = strategy_for(QueryClass.SUMMARIZATION)
    assert summ.resolved_candidate_k() > settings.candidate_k
    assert summ.resolved_final_k() > settings.final_k


# --- Query rewriting & decomposition (Features 4 / 5) -----------------------

def test_query_rewrite_bounded_and_internal():
    variants = rewrite("Who owns the IP?")
    assert variants  # IP triggers intellectual-property expansions
    assert len(variants) <= settings.query_rewrite_max_variants
    assert all(v != "Who owns the IP?" for v in variants)


def test_query_decomposition():
    parts = decompose("Who is the client and what are the payment terms?")
    assert len(parts) == 2
    assert "client" in parts[0].lower() and "payment" in parts[1].lower()
    # conjunctions that are not multi-part questions never split
    assert decompose("What is the difference between v1 and v2?") == [
        "What is the difference between v1 and v2?"
    ]
    assert decompose("What are the terms and conditions?") == ["What are the terms and conditions?"]


def test_defined_terms_in_question():
    terms = defined_terms_in("How long does Confidential Information stay protected?")
    assert "Confidential Information" in terms


# --- Context packing (Feature 2) --------------------------------------------

def _pc(document: str, page: int, index: int, text: str, section=None, heading=None) -> RetrievedChunk:
    return RetrievedChunk(
        score=0.5, dense_score=0.7, document=document, page=page, section=section,
        version="1.0", ocr_confidence=1.0, ocr_source="text", text=text,
        index=index, heading=heading,
    )


def test_packing_dedupes_and_orders():
    a = _pc("a.txt", 1, 0, "The fee is USD 100.")
    dup = _pc("a.txt", 1, 0, "The fee is USD 100.")
    b = _pc("a.txt", 1, 5, "Confidentiality survives three years.")
    packed = pack([b, a, dup])
    assert len(packed) == 2
    assert packed[0].index == 0 and packed[1].index == 5  # document order restored


def test_packing_merges_overlapping_adjacent_chunks():
    shared = "OVERLAP ZONE 1234567890 ABCDEFG"
    a = _pc("a.txt", 1, 0, "First chunk body text ends with " + shared)
    b = _pc("a.txt", 1, 1, shared + " and the second chunk continues here.")
    merged = _merge_adjacent([a, b])
    assert len(merged) == 1
    assert merged[0].text.count(shared) == 1  # overlap stitched out, not repeated


def test_packing_strips_repeated_headings():
    h = "Clause 4. Termination"
    a = _pc("a.txt", 1, 0, h + "\nEither party may terminate.", section="Clause 4", heading=h)
    b = _pc("a.txt", 1, 3, h + "\nNotice must be written.", section="Clause 4", heading=h)
    out = _strip_repeated_headings([a, b])
    assert out[0].text.startswith(h)
    assert not out[1].text.startswith(h)
    assert "Notice must be written." in out[1].text


def test_contained_duplicate_removed():
    small = _pc("a.txt", 1, 2, "Confidentiality survives three years.")
    big = _pc("a.txt", 1, 1, "Clause 5 says: Confidentiality survives three years. More text.")
    assert len(_dedupe([big, small])) == 1


# --- Parent context in prompt (Feature 1) -----------------------------------

def test_format_context_renders_parent_and_table():
    c = _pc("a.txt", 1, 0, "A breach must be cured in fifteen days.", section="Clause 4.2")
    c.parent_section = "Clause 4"
    c.parent_context = "Clause 4. Termination — either party may terminate."
    t = _pc("a.txt", 1, 1, "Item | Amount\nRetainer | 12000", section="Clause 2")
    t.is_table = True
    rendered = format_context([c, t])
    assert "[Parent context for Chunk 1 — Clause 4" in rendered
    assert "TABLE (structure preserved" in rendered


# --- Answer verification: clause fabrication (Feature 10) -------------------

def _rc_sec(document: str, page: int, section: str | None) -> RetrievedChunk:
    return RetrievedChunk(
        score=0.5, dense_score=0.7, document=document, page=page, section=section,
        version="1.0", ocr_confidence=1.0, ocr_source="text", text="text",
    )


def test_citation_clause_verified():
    answer = "The fee is USD 100. (Source: MSA_v1.txt | Page 1 | Clause 2 | v1.0)"
    report = verify_citations(answer, [_rc_sec("MSA_v1.txt", 1, "Clause 2")])
    assert report.passed and report.verified == 1 and not report.clause_mismatches


def test_citation_fabricated_clause_caught():
    answer = "The fee is USD 100. (Source: MSA_v1.txt | Page 1 | Clause 9 | v1.0)"
    report = verify_citations(answer, [_rc_sec("MSA_v1.txt", 1, "Clause 2")])
    assert not report.passed
    assert len(report.clause_mismatches) == 1 and report.verified == 0


def test_citation_subclause_prefix_lenient():
    answer = "Cure period is fifteen days. (Source: MSA_v1.txt | Page 1 | Clause 4 | v1.0)"
    report = verify_citations(answer, [_rc_sec("MSA_v1.txt", 1, "Clause 4.2")])
    assert report.passed and report.verified == 1


def test_citation_clause_lenient_when_section_unknown():
    # a sectionless chunk on the same page makes the clause unverifiable -> lenient
    answer = "The fee is USD 100. (Source: MSA_v1.txt | Page 1 | Clause 2 | v1.0)"
    report = verify_citations(answer, [_rc_sec("MSA_v1.txt", 1, None)])
    assert report.passed and report.verified == 1


# --- Version awareness (Feature 11) -----------------------------------------

def test_version_base_and_ordering():
    assert base_document("MSA_Acme_v2.1.txt") == "MSA_Acme"
    assert base_document("MSA_Acme_v1.0.txt") == "MSA_Acme"
    assert base_document("plain.docx") == "plain"
    assert _version_tuple("2.1") > _version_tuple("1.0")
    assert _version_tuple("10.0") > _version_tuple("9.9")


# --- Comparison round-robin selection (Feature 3) ---------------------------

def test_select_final_round_robin_across_documents():
    chunks = [_pc("A.txt", 1, i, f"a{i}") for i in range(3)] + \
             [_pc("B.txt", 1, i, f"b{i}") for i in range(3)]
    strategy = strategy_for(QueryClass.COMPARISON)
    out = select_final(chunks, strategy, per_document=True)
    docs_in_top2 = {c.document for c in out[:2]}
    assert docs_in_top2 == {"A.txt", "B.txt"}  # both documents represented


# --- Evaluation metrics (Feature 9) -----------------------------------------

def test_eval_metric_math():
    assert recall_at_k([True, False]) == 0.5
    assert precision_at_k([True, False, False, True]) == 0.5
    assert mrr([False, True, False]) == 0.5
    assert mrr([False, False]) == 0.0
    ndcg = ndcg_at_k([True, False, True], 2)
    assert 0.91 < ndcg < 0.93  # dcg=1.5, idcg=1.6309


def test_eval_case_matching():
    case = GoldenCase(
        id="t", question="q",
        relevant=[{"document": "MSA_Acme_v1.0.txt", "section": "Clause 2"}],
    )
    hit = _rc_sec("MSA_Acme_v1.0.txt", 1, "Clause 2")
    hit.dense_score = 0.8
    miss = _rc_sec("Other.txt", 1, "Clause 9")
    row = evaluate_case_retrieval(case, [hit, miss], k=2)
    assert row["recall"] == 1.0 and row["mrr"] == 1.0 and row["success"]
    assert row["gate_passed"] and row["gate_correct"]


# --- Clause-aware retrieval + ambiguity detection ---------------------------

def test_clause_reference_extraction():
    from lexis.query import clause_references_in, requested_version_in

    assert clause_references_in("What does Clause 5 say?") == ["5"]
    assert clause_references_in("Compare clause 7.2 and Section 8") == ["7.2", "8"]
    assert clause_references_in("see Article 12 and § 4") == ["12", "4"]
    assert clause_references_in("the clause states nothing specific") == []
    assert requested_version_in("Clause 2 of MSA v1.0") == "1.0"
    assert requested_version_in("in version 2.1 of the agreement") == "2.1"
    assert requested_version_in("What is the notice period?") is None


def test_section_number_matching():
    from lexis.vector_store import number_matches, section_number

    assert section_number("Clause 7.2 Termination") == "7.2"
    assert section_number("2.1 Payment Terms") == "2.1"
    assert section_number(None) is None
    assert number_matches("7", "7.2") and number_matches("7.2", "7")
    assert number_matches("5", "5")
    assert not number_matches("7", "72") and not number_matches("1", "11")


def test_clause_boost_actual_over_reference():
    from lexis import engine

    actual = _rc("MSA_v1.txt", 1)
    actual.section, actual.rerank_score = "Clause 5", 0.1
    referencing = _rc("MSA_v1.txt", 2)
    referencing.section, referencing.rerank_score = "Clause 9", 0.9
    referencing.text = "as described in Clause 5 above"
    engine._boost_clause_chunks("What does Clause 5 require?", [actual, referencing])
    assert actual.rerank_score == 0.995 and "clause-exact" in actual.matched_on
    assert referencing.rerank_score == 0.9  # merely references the clause: no boost
    assert actual.rerank_score > referencing.rerank_score


def test_clause_boost_respects_requested_version():
    from lexis import engine

    old = _rc("MSA_Acme_v1.0.txt", 1)
    old.section, old.version, old.rerank_score = "Clause 2", "1.0", 0.2
    new = _rc("MSA_Acme_v2.1.txt", 1)
    new.section, new.version, new.rerank_score = "Clause 2", "2.1", 0.2
    engine._boost_clause_chunks("What does Clause 2 of MSA v1.0 say?", [old, new])
    assert old.rerank_score == 0.995
    assert new.rerank_score == 0.2  # user pinned v1.0 — latest must not hijack


def test_ambiguity_multi_document_clarification():
    from lexis import engine

    a = _rc("MSA_Acme_v1.0.txt", 1)
    a.section, a.matched_on = "Clause 5", ["clause-exact"]
    b = _rc("NDA_Beta_v1.0.txt", 1)
    b.section, b.matched_on = "Clause 5", ["clause-exact"]
    docs, extra, notes = engine._clause_ambiguity("What does Clause 5 say?", [a, b], [a])
    assert docs == ["MSA_Acme_v1.0.txt", "NDA_Beta_v1.0.txt"]
    # naming the document resolves the ambiguity
    docs2, _, _ = engine._clause_ambiguity("What does Clause 5 of MSA Acme say?", [a, b], [a])
    assert docs2 is None


def test_ambiguity_multi_version_representation():
    from lexis import engine

    v1 = _rc("MSA_Acme_v1.0.txt", 1)
    v1.section, v1.version, v1.matched_on = "Clause 4", "1.0", ["clause-exact"]
    v2 = _rc("MSA_Acme_v2.1.txt", 1)
    v2.section, v2.version, v2.matched_on = "Clause 4", "2.1", ["clause-exact"]
    docs, extra, notes = engine._clause_ambiguity("What does Clause 4 say?", [v2, v1], [v2])
    assert docs is None
    assert extra == [v1]  # missing version pulled in for comparison
    assert notes and "versions" in notes[0]
    # an explicit version request means nothing is ambiguous
    docs3, extra3, notes3 = engine._clause_ambiguity(
        "What does Clause 4 of v2.1 say?", [v2, v1], [v2]
    )
    assert docs3 is None and extra3 == [] and notes3 == []


def test_citation_failure_reasons():
    report = CitationReport(total=2, verified=1, fabricated=["(Source: Ghost.pdf | Page 9)"])
    assert any("fabricated_citation" in r for r in report.failure_reasons())
    assert CitationReport(refusal=True).failure_reasons() == []
    uncited = CitationReport(total=0, uncited_answer=True)
    assert any("missing_citations" in r for r in uncited.failure_reasons())
    mismatch = CitationReport(total=1, clause_mismatches=["(Source: a.txt | Page 1 | Clause 99 | v1)"])
    assert any("clause_mismatch" in r for r in mismatch.failure_reasons())


# --- Intent-aware retrieval: overview + clause exact-only + label hygiene ---

def test_overview_classification():
    from lexis.query import QueryClass, classify

    assert classify("Explain the merger agreement of Twitter and X.") == QueryClass.OVERVIEW
    assert classify("Summarize this contract") == QueryClass.OVERVIEW
    assert classify("Give me an overview") == QueryClass.OVERVIEW
    assert classify("Tell me about this agreement") == QueryClass.OVERVIEW
    # a clause/topic question with no document noun must NOT become OVERVIEW
    assert classify("Explain the termination notice requirements") != QueryClass.OVERVIEW
    assert classify("What does Clause 5 say?") == QueryClass.CLAUSE


def _patched_manifest(fn):
    import lexis.ingest as ingest_mod

    fake = {
        "MSA_Acme_v1.0.txt": {"version": "1.0"},
        "MSA_Acme_v2.1.txt": {"version": "2.1"},
        "NDA_Beta_v1.0.txt": {"version": "1.0"},
        # partial-match distractor: shares "Acme" but not "MSA"
        "Form8K_Acme_v1.0.txt": {"version": "1.0"},
    }
    original = ingest_mod.load_manifest
    ingest_mod.load_manifest = lambda: fake
    try:
        fn()
    finally:
        ingest_mod.load_manifest = original


def test_overview_target_resolution():
    from lexis.retrieval import resolve_overview_target, target_documents

    def check():
        assert target_documents("Explain the MSA Acme agreement") == [
            "MSA_Acme_v1.0.txt", "MSA_Acme_v2.1.txt"
        ]
        # named family resolves to its latest version
        target, alts = resolve_overview_target("Explain the MSA Acme agreement")
        assert target == "MSA_Acme_v2.1.txt" and alts == []
        # explicit version pins the older document
        target, _ = resolve_overview_target("Overview of MSA Acme v1.0")
        assert target == "MSA_Acme_v1.0.txt"
        # a partial-match distractor (Form8K_Acme) must not create ambiguity
        target, alts = resolve_overview_target("Overview of the Form8K Acme filing")
        assert target == "Form8K_Acme_v1.0.txt"
        # bare "this contract" with multiple families -> ambiguous
        target, alts = resolve_overview_target("Summarize this contract")
        assert target is None and len(alts) == 4

    _patched_manifest(check)


def test_internal_label_scrub():
    from lexis.engine import _scrub_internal_labels

    raw = (
        "[Chunk 1] The fee is USD 100. (Source: a.txt | Page 1 | Clause 2 | v1.0)\n"
        "As stated in Chunk 2, the term renews. [Parent context for Chunk 2 — Clause 3]"
    )
    clean = _scrub_internal_labels(raw)
    assert "Chunk 1" not in clean and "Chunk 2" not in clean
    assert "(Source: a.txt | Page 1 | Clause 2 | v1.0)" in clean  # citations untouched
    assert "the retrieved evidence" in clean


# --- Cross-reference resolution ----------------------------------------------

def test_reference_extraction_broad():
    text = (
        "The information in Item 1.01 is incorporated by reference into this "
        "Item 3.03, subject to Clause 5 and pursuant to Section 8.1. "
        "See Exhibit A, Schedule 2.1, and Appendix 3. Attachment B applies."
    )
    refs = extract_references(text, own_section="Item 3.03")
    assert "Item 1.01" in refs and "Item 3.03" not in refs  # own section excluded
    assert "Clause 5" in refs and "Section 8.1" in refs
    assert "Exhibit A" in refs and "Schedule 2.1" in refs
    assert "Appendix 3" in refs and "Attachment B" in refs


def test_item_and_exhibit_headings():
    assert detect_heading("Item 1.01 Entry into a Material Definitive Agreement") == "Item 1.01"
    assert detect_heading("Exhibit A") == "Exhibit A"
    assert detect_heading("SCHEDULE 2.1 Pricing") == "Schedule 2.1"
    assert detect_heading("Appendix 3 — Data Terms") == "Appendix 3"
    # ordinary prose starting with these words must not become headings
    assert detect_heading("Exhibit and schedule details follow.") is None


def test_item_clause_query_parsing():
    from lexis.query import QueryClass, classify, clause_references_in

    assert classify("What does Item 3.03 report?") == QueryClass.CLAUSE
    assert clause_references_in("What does Item 3.03 report?") == ["3.03"]


def test_citation_with_echoed_field_label_verifies():
    chunk = _rc("Form8K.txt", 1)
    chunk.section = "Item 3.03"
    answer = ('The info is incorporated by reference. '
              '(Source: Form8K.txt | Page 1 | Section: Item 3.03 | v1.0)')
    report = verify_citations(answer, [chunk])
    assert report.passed and report.verified == 1 and not report.clause_mismatches


def test_item_never_matches_clause_numbering():
    from lexis import engine
    from lexis.query import clause_reference_pairs
    from lexis.vector_store import keyword_compatible, section_keyword

    assert clause_reference_pairs("What does Item 3.03 report?") == [("item", "3.03")]
    assert section_keyword("Clause 3") == "clause" and section_keyword("Item 3.03") == "item"
    assert not keyword_compatible("item", "clause")
    assert keyword_compatible("section", "clause")  # drafting synonyms

    clause3 = _rc("MSA.txt", 1)
    clause3.section, clause3.rerank_score = "Clause 3", 0.1
    item = _rc("8K.txt", 1)
    item.section, item.rerank_score = "Item 3.03", 0.1
    engine._boost_clause_chunks("What does Item 3.03 report?", [clause3, item])
    assert item.rerank_score == 0.995
    assert clause3.rerank_score == 0.1  # different numbering system: untouched


def test_chained_reference_expansion_with_cycle():
    from lexis import graph as graph_mod
    from lexis import packing, vector_store as vs

    def chunk_for(section):
        c = _rc("8K.txt", 1)
        c.section = section
        return c

    # A -> B -> C -> A (cycle); expansion must terminate and follow the chain.
    edges = {"Item 3.03": ["Item 1.01"], "Item 1.01": ["Item 9.01"], "Item 9.01": ["Item 3.03"]}
    orig_refs, orig_fetch = graph_mod.referenced_sections, vs.fetch_section_chunk
    graph_mod.referenced_sections = lambda doc, sec: edges.get(sec or "", [])
    vs.fetch_section_chunk = lambda doc, sec: chunk_for(sec)
    was_enabled, was_depth = settings.graph_enabled, settings.graph_max_depth
    settings.graph_enabled, settings.graph_max_depth = True, 3
    try:
        out = packing.expand_references([chunk_for("Item 3.03")])
    finally:
        graph_mod.referenced_sections, vs.fetch_section_chunk = orig_refs, orig_fetch
        settings.graph_enabled, settings.graph_max_depth = was_enabled, was_depth

    sections = [c.section for c in out]
    assert sections == ["Item 3.03", "Item 1.01", "Item 9.01"]  # cycle back to 3.03 blocked
    assert out[1].matched_on == ["graph-reference:Item 3.03->Item 1.01"]
    assert out[2].matched_on == ["graph-reference:Item 1.01->Item 9.01"]


def test_expansion_respects_cap():
    from lexis import graph as graph_mod
    from lexis import packing, vector_store as vs

    edges = {"A": ["B", "C", "D", "E", "F"]}
    orig_refs, orig_fetch = graph_mod.referenced_sections, vs.fetch_section_chunk
    graph_mod.referenced_sections = lambda doc, sec: edges.get(sec or "", [])

    def chunk_for(doc, sec):
        c = _rc(doc, 1)
        c.section = sec
        return c

    vs.fetch_section_chunk = chunk_for
    was_enabled = settings.graph_enabled
    settings.graph_enabled = True
    try:
        seed = _rc("d.txt", 1)
        seed.section = "A"
        out = packing.expand_references([seed])
    finally:
        graph_mod.referenced_sections, vs.fetch_section_chunk = orig_refs, orig_fetch
        settings.graph_enabled = was_enabled
    assert len(out) == 1 + settings.graph_max_expansion


# --- Consequence-aware retrieval ---------------------------------------------

def test_consequence_classification():
    from lexis.query import QueryClass, classify

    assert classify("What if I break Clause 3?") == QueryClass.CONSEQUENCE
    assert classify("What happens if the client breaches the agreement?") == QueryClass.CONSEQUENCE
    assert classify("What are the consequences of violating Clause 5?") == QueryClass.CONSEQUENCE
    assert classify("Is there a penalty for late payment?") == QueryClass.CONSEQUENCE
    # consequence outranks the clause/termination rules its phrasing touches
    assert classify("What happens if I terminate early?") == QueryClass.CONSEQUENCE
    # plain lookups must NOT become consequence analysis
    assert classify("What does Clause 3 say?") == QueryClass.CLAUSE
    assert classify("What is the termination notice period?") == QueryClass.TERMINATION


def test_consequence_boost_pins_provisions():
    from lexis.engine import _boost_consequence_chunks

    termination = _rc("MSA.txt", 1)
    termination.section, termination.text = "Clause 4", "Termination. Either party may terminate..."
    preamble = _rc("MSA.txt", 1)
    preamble.section, preamble.text = "Preamble", "Client: [CLIENT_NAME]\nClient ID: [CUSTOMER_ID]"
    termination.rerank_score = preamble.rerank_score = 0.05
    _boost_consequence_chunks("What if the client breaches the contract?", [termination, preamble])
    assert termination.rerank_score == 0.9 and "consequence-boost" in termination.matched_on
    assert preamble.rerank_score == 0.05  # parties block is not a consequence provision


def test_consequence_strategy_widens_context():
    from lexis.query import CONSEQUENCE_PROBES, QueryClass, strategy_for

    strategy = strategy_for(QueryClass.CONSEQUENCE)
    assert strategy.resolved_final_k() == settings.consequence_final_k > settings.final_k
    assert len(CONSEQUENCE_PROBES) == 3  # bounded probe fan-out


# --- Legal intelligence layer: concept graph + jurisdiction -------------------

def test_concept_probes():
    from lexis.query import concept_probes

    probes = concept_probes("How long do confidentiality obligations survive?")
    assert probes and any("survival" in p for p in probes)
    probes = concept_probes("What are the Client's obligations?")
    assert probes and any("deliverables" in p or "duties" in p for p in probes)
    assert concept_probes("Who is the client?") == []  # no concept, no probes
    assert len(concept_probes("obligations to pay confidential IP on termination")) <= 2


def test_obligation_subject_and_pattern():
    from lexis.query import QueryClass, classify, obligation_pattern, obligation_subject

    q = "What are the Client's obligations under the MSA Acme agreement?"
    assert classify(q) == QueryClass.OBLIGATION  # duties, not identity
    assert classify("Who is the client?") == QueryClass.ENTITY  # identity stays ENTITY
    assert obligation_subject(q) == "client"
    duty = obligation_pattern("client")
    assert duty.search("The Client shall pay a monthly retainer of USD 15,000")
    assert duty.search("Each party shall protect the other party's Confidential Information")
    assert not duty.search("The Provider shall deliver the consulting services")


def test_jurisdiction_boost():
    from lexis.engine import _boost_jurisdiction_chunks

    delaware = _rc("Merger.pdf", 1)
    delaware.jurisdiction, delaware.rerank_score = "State of Delaware", 0.5
    newyork = _rc("MSA.txt", 1)
    newyork.jurisdiction, newyork.rerank_score = "State of New York", 0.5
    _boost_jurisdiction_chunks("What are the closing conditions under Delaware law?",
                               [delaware, newyork])
    assert delaware.rerank_score > 0.5 and "jurisdiction-match" in delaware.matched_on
    assert newyork.rerank_score == 0.5
    # question naming no jurisdiction: no-op
    plain = _rc("MSA.txt", 1)
    plain.jurisdiction, plain.rerank_score = "State of New York", 0.5
    _boost_jurisdiction_chunks("What is the retainer?", [plain])
    assert plain.rerank_score == 0.5


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
