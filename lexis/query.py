"""Query understanding: classification, rewriting, decomposition.

All of it is deterministic rule-based logic — no extra model calls, so the
query-understanding stage costs microseconds and is fully unit-testable.

- classify()        -> QueryClass driving a per-class RetrievalStrategy
  (Feature: Query Classification)
- rewrite()         -> internal synonym-expanded retrieval variants
  (Feature: Query Rewriting — variants are never shown to the user)
- decompose()       -> split multi-part questions into retrieval sub-tasks
  (Feature: Query Decomposition)
- defined_terms_in()-> capitalized defined terms present in the question
  (Feature: Definition-aware Retrieval)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .config import settings


class QueryClass(str, Enum):
    ENTITY = "entity_lookup"
    CLAUSE = "clause_lookup"
    DEFINITION = "definition_lookup"
    DATE = "date_lookup"
    PAYMENT = "payment_terms"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    OBLIGATION = "obligation_extraction"
    CONSEQUENCE = "consequence_analysis"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    OVERVIEW = "document_overview"
    GENERAL = "general_search"


# Ordered — first match wins. Comparison outranks everything (a "compare the
# termination clauses" question must route per-document, not to TERMINATION).
_CLASS_RULES: list[tuple[QueryClass, re.Pattern]] = [
    # Consequence/breach analysis outranks everything: "What if I break
    # Clause 3?" must not fall into CLAUSE (which would narrow context to
    # the clause alone) or TERMINATION (one consequence bucket of several).
    (QueryClass.CONSEQUENCE, re.compile(
        r"(?i)(?:\bwhat\s+(?:happens|occurs)\b|\bwhat\s+if\b|\bconsequences?\b"
        r"|\bpenalt\w+\b|\bbreach\w*\b|\bviolat\w*\b|\bin\s+the\s+event\s+of\s+default\b"
        r"|\bbreak\w*\b(?=[^.?!]{0,40}\b(?:clause|section|article|agreement|contract|terms)\b))")),
    (QueryClass.COMPARISON, re.compile(
        r"(?i)\b(compare|comparison|difference|differs?|changed|what changed|versus|vs\.?)\b"
        r"|\bbetween\b.+\band\b", re.IGNORECASE)),
    # Whole-document requests ("explain the merger agreement", "tell me about
    # this contract") route to stratified overview retrieval — a document
    # noun must be present so "explain the termination clause" stays CLAUSE.
    (QueryClass.OVERVIEW, re.compile(
        r"(?i)(?:\b(?:overview of|explain|tell me about|describe|walk me through|what is|summar\w*)\b"
        r"[^.?!]{0,80}\b(?:agreement|contract|document|deal|merger|msa|nda)\b"
        r"|\bgive me an overview\b)")),
    (QueryClass.SUMMARIZATION, re.compile(
        r"(?i)\b(summar|overview|key (?:points|terms)|main (?:points|terms)|at a glance)\b")),
    (QueryClass.DEFINITION, re.compile(
        r"(?i)\b(what does .{1,60} mean|meaning of|definition of|how is .{1,60} defined|define\b)")),
    (QueryClass.PAYMENT, re.compile(
        r"(?i)\b(pay(?:ment)?s?|fees?|retainer|invoice|pricing|price|cost|interest|late charge)\b")),
    (QueryClass.TERMINATION, re.compile(
        r"(?i)\b(terminat|cancel|non-?renewal|end (?:the|this) agreement|notice period)\b")),
    (QueryClass.CONFIDENTIALITY, re.compile(
        r"(?i)\b(confidential|non-?disclosure|nda|trade secret)\b")),
    (QueryClass.DATE, re.compile(
        r"(?i)\b(when\b|what date|deadline|effective date|expir|duration|how long|term length)\b")),
    (QueryClass.CLAUSE, re.compile(
        r"(?i)\b(clause|section|article|item)\s*\d")),
    # OBLIGATION before ENTITY: "the Client's obligations" names the client
    # but asks for duties, not identity.
    (QueryClass.OBLIGATION, re.compile(
        r"(?i)\b(obligat\w*|shall|must|required to|responsib\w*|dut(?:y|ies)|deliverables?)\b")),
    (QueryClass.ENTITY, re.compile(
        r"(?i)\b(who is|who are|name of|which (?:company|party|parties)|the client\b|the provider\b|the parties\b)")),
]


@dataclass
class RetrievalStrategy:
    """Per-class retrieval knobs; defaults reproduce the unclassified path."""
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    candidate_k: int = 0          # 0 -> settings.candidate_k
    final_k: int = 0              # 0 -> settings.final_k
    per_document: bool = False    # retrieve independently per document (comparison)
    definitions_first: bool = False
    overview: bool = False        # stratified whole-document retrieval

    def resolved_candidate_k(self) -> int:
        return self.candidate_k or settings.candidate_k

    def resolved_final_k(self) -> int:
        return self.final_k or settings.final_k


def classify(question: str) -> QueryClass:
    if not settings.query_classification:
        return QueryClass.GENERAL
    for qclass, pattern in _CLASS_RULES:
        if pattern.search(question):
            return qclass
    return QueryClass.GENERAL


def strategy_for(qclass: QueryClass) -> RetrievalStrategy:
    base = RetrievalStrategy(
        dense_weight=settings.dense_weight, bm25_weight=settings.bm25_weight
    )
    if qclass in (QueryClass.ENTITY, QueryClass.CLAUSE):
        # Exact identifiers ("Clause 9.1", names, IDs): keyword leg up.
        base.bm25_weight *= 1.5
    elif qclass == QueryClass.SUMMARIZATION:
        # Broader semantic context: wider fan-out, more chunks to the LLM.
        base.dense_weight *= 1.2
        base.candidate_k = settings.candidate_k * 2
        base.final_k = settings.final_k + 2
    elif qclass == QueryClass.COMPARISON:
        base.per_document = True
    elif qclass == QueryClass.DEFINITION:
        base.definitions_first = True
    elif qclass == QueryClass.OVERVIEW and settings.overview_enabled:
        base.overview = True
        base.final_k = settings.overview_final_k
    elif qclass == QueryClass.CONSEQUENCE:
        # The referenced clause AND the consequence provisions must both fit.
        base.final_k = settings.consequence_final_k
    return base


# Consequence-aware retrieval (Feature: Consequence Analysis): fixed probe
# queries retrieved alongside the user's question and RRF-fused, so breach
# questions surface the provisions that define what actually happens —
# termination, remedies, liability, default, indemnification, enforcement.
CONSEQUENCE_PROBES = [
    "termination default breach remedies cure period",
    "limitation of liability damages indemnification",
    "governing law dispute resolution enforcement survival",
]

# Sections that define what happens on breach — shared by the retrieval-side
# provision injection and the engine-side ranking floor.
CONSEQUENCE_SECTION_RE = re.compile(
    r"(?i)\b(terminat|liabilit|remed|default|indemn|damag|surviv|governing law|dispute|enforc|breach)"
)


# --- Legal concept graph (Feature: Legal Intelligence Layer) ----------------
# Provisions that materially affect each other: a confidentiality question is
# also governed by survival and injunctive relief; a payment question by late
# interest and suspension on default. One related-provision probe per
# detected concept is retrieved alongside the question and RRF-fused.
LEGAL_CONCEPT_GRAPH: dict[str, list[str]] = {
    "confidentiality": ["survival injunctive relief permitted disclosure exceptions"],
    "payment": ["late payment interest suspension default non-payment"],
    "termination": ["effect of termination return of property survival"],
    "intellectual property": ["ownership assignment work product license rights"],
    "obligation": [
        "shall pay deliver provide responsibilities deliverables",
        "confidentiality payment obligations duties",
    ],
}

_CONCEPT_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("confidentiality", re.compile(r"(?i)\b(confidential|non-?disclosure|nda|trade secret)")),
    ("payment", re.compile(r"(?i)\b(pay|fees?|invoice|retainer|pricing)")),
    ("termination", re.compile(r"(?i)\b(terminat|cancel|rescis)")),
    ("intellectual property", re.compile(r"(?i)\b(intellectual property|\bip\b|work product)")),
    ("obligation", re.compile(r"(?i)\b(obligat|responsib|dut(?:y|ies)|deliverables?)")),
]


# Whose obligations? "The Client's obligations" = sentences where the Client
# (or both parties) shall/must — legally precise, not keyword similarity.
_OBLIGOR_RE = re.compile(
    r"(?i)\b(client|customer|purchaser|provider|vendor|supplier|contractor|employee|employer)\b"
)


def obligation_subject(question: str) -> str | None:
    m = _OBLIGOR_RE.search(question)
    return m.group(1).lower() if m else None


def obligation_pattern(subject: str) -> re.Pattern:
    """Sentences imposing duties on `subject` (or on both parties)."""
    return re.compile(
        rf"(?i)\b(?:{subject}|each party|either party|both parties)\b"
        r"[^.]{0,60}\b(?:shall|must|will|agrees?\s+to)\b"
    )


def concept_probes(question: str, limit: int = 2) -> list[str]:
    """Related-provision probe queries for the legal concepts a question
    touches. Bounded: each probe costs an embed + two index searches."""
    probes: list[str] = []
    for concept, pattern in _CONCEPT_TRIGGERS:
        if pattern.search(question):
            for probe in LEGAL_CONCEPT_GRAPH.get(concept, []):
                if probe not in probes:
                    probes.append(probe)
    return probes[:limit]


# --- Query rewriting (internal only) ---------------------------------------
# Legal-domain synonym expansions: trigger phrase -> retrieval variants.
_EXPANSIONS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"(?i)\b(ip|intellectual property)\b"),
     ["intellectual property rights ownership",
      "assignment of intellectual property",
      "work product license rights"]),
    (re.compile(r"(?i)\bterminat|cancel"),
     ["termination for convenience notice period",
      "termination material breach cure period"]),
    (re.compile(r"(?i)\bpay|fee|retainer|invoice|cost|price"),
     ["fees and payment monthly retainer",
      "invoice due days late payment interest"]),
    (re.compile(r"(?i)\bconfidential|non-?disclosure|nda|secret"),
     ["confidential information obligations care",
      "confidentiality survival termination years"]),
    (re.compile(r"(?i)\bliabilit|damages|indemnif"),
     ["limitation of liability aggregate cap",
      "indirect incidental consequential damages"]),
    (re.compile(r"(?i)\bgoverning law|jurisdiction|governed"),
     ["governing law laws of the state",
      "jurisdiction venue disputes"]),
    (re.compile(r"(?i)\brenew|term\b|duration|how long"),
     ["initial term renewal automatic",
      "term months commencement effective date"]),
    # Entity synonyms: legal drafts alternate freely between these labels.
    (re.compile(r"(?i)\b(client|customer|purchaser)\b"),
     ["client customer party receiving services"]),
    (re.compile(r"(?i)\b(vendor|supplier|provider|contractor)\b"),
     ["provider vendor supplier contractor services"]),
    (re.compile(r"(?i)\bwho is|client\b|provider\b|part(?:y|ies)\b"),
     ["client name client id parties",
      "entered into by and between provider"]),
]


def rewrite(question: str) -> list[str]:
    """Internal retrieval variants for a vague/underspecified legal question.

    Returns up to settings.query_rewrite_max_variants NEW queries (the
    original is always retrieved as well by the orchestrator). Never exposed
    to the user."""
    if not settings.query_rewrite:
        return []
    variants: list[str] = []
    for pattern, expansions in _EXPANSIONS:
        if pattern.search(question):
            for expansion in expansions:
                if expansion not in variants:
                    variants.append(expansion)
    # A quoted/capitalized defined term is worth a definition-shaped probe.
    for term in defined_terms_in(question):
        probe = f'"{term}" means definition'
        if probe not in variants:
            variants.append(probe)
    return variants[: settings.query_rewrite_max_variants]


# --- Query decomposition ----------------------------------------------------
_INTERROGATIVE_RE = re.compile(r"(?i)\b(who|what|when|where|which|how|why|do(?:es)?|is|are)\b")
_SPLIT_RE = re.compile(r"(?i)\s+and\s+(?=(?:also\s+)?(?:who|what|when|where|which|how|why)\b)")


def decompose(question: str) -> list[str]:
    """Split a multi-part question into independent retrieval sub-queries.

    "Who is the client and what are the payment terms?" ->
    ["Who is the client", "what are the payment terms?"]

    Conservative: only splits where a fresh interrogative starts the second
    part, so "between v1 and v2" or "terms and conditions" never split."""
    if not settings.query_decomposition:
        return [question]
    parts = [p.strip(" ?") + "?" for p in _SPLIT_RE.split(question) if p.strip(" ?")]
    if len(parts) < 2:
        return [question]
    parts = [p for p in parts if _INTERROGATIVE_RE.search(p) and len(p.split()) >= 3]
    if len(parts) < 2:
        return [question]
    return parts[: settings.query_decomposition_max_parts]


# --- Clause reference extraction (Feature: Clause-aware Retrieval) ----------
# "Clause 5", "clause 7.2", "Section 8", "Article 12", "§ 4" — the identifier
# the user explicitly asked about, normalized to its number ("5", "7.2").
# \b cannot precede the non-word "§", so the symbol is its own alternative.
_CLAUSE_REF_RE = re.compile(
    r"(?i)(?:\b(?P<kw>clause|section|article|item|art\.?|sec\.?)|§)\s*(?P<num>\d+(?:\.\d+)*)\b"
)

_REQUESTED_VERSION_RE = re.compile(r"(?i)\bv(?:ersion)?\s*\.?\s*(\d+(?:\.\d+)*)\b")


def clause_reference_pairs(question: str) -> list[tuple[str | None, str]]:
    """(keyword, number) pairs explicitly named in the question, e.g.
    [("item", "3.03")]. The keyword matters: "Item 3.03" must never match a
    "Clause 3" chunk — Items and Clauses are different numbering systems."""
    pairs: list[tuple[str | None, str]] = []
    for m in _CLAUSE_REF_RE.finditer(question):
        kw = (m.group("kw") or "").rstrip(".").lower() or None
        pair = (kw, m.group("num"))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def clause_references_in(question: str) -> list[str]:
    """Normalized clause/section numbers explicitly named in the question."""
    seen: list[str] = []
    for _kw, num in clause_reference_pairs(question):
        if num not in seen:
            seen.append(num)
    return seen


def requested_version_in(question: str) -> str | None:
    """An explicit document version the user asked about ("v1.0", "version 2"),
    or None. When present, version preference must yield to the user's choice."""
    m = _REQUESTED_VERSION_RE.search(question)
    return m.group(1) if m else None


# --- Defined-term detection -------------------------------------------------
# Core terms virtually every commercial agreement defines, plus any
# Title-Case phrase in the question (e.g. "Confidential Information").
_CORE_DEFINED_TERMS = (
    "Confidential Information", "Effective Date", "Intellectual Property",
    "Agreement", "Affiliate", "Provider", "Client", "Contractor", "Party",
    "Parties", "Notice", "Services", "Change Order",
)
_TITLECASE_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def defined_terms_in(question: str) -> list[str]:
    found: list[str] = []
    for term in _CORE_DEFINED_TERMS:
        if re.search(rf"(?i)\b{re.escape(term)}\b", question) and term not in found:
            found.append(term)
    for m in _TITLECASE_PHRASE_RE.finditer(question):
        phrase = m.group(1)
        if phrase not in found and not _INTERROGATIVE_RE.match(phrase.split()[0]):
            found.append(phrase)
    return found
