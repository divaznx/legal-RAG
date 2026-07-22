"""The legal concept graph — what makes retrieval legal-aware instead of
purely lexical.

Embeddings alone cannot answer "what happens if I breach Clause 8?", because
the *answer* lives in clauses that never use the word "breach": Limitation of
Liability, Indemnification, Governing Law, Survival. Cosine similarity has no
notion that default triggers remedies. This module encodes that knowledge
explicitly.

Three relations are modelled per concept:

- `surface_forms` — how the concept is actually written in contracts and in
  lawyer questions. Drives detection and synonym expansion (feature: legal
  synonym expansion).
- `related` — concepts that are legally adjacent and should be co-retrieved
  for context (feature: legal concept expansion).
- `consequences` — concepts that describe what LEGALLY FOLLOWS. Walked only
  for consequence/risk questions, where the user is asking about effect
  rather than content (feature: consequence-aware retrieval).

The graph is deliberately hand-curated and auditable rather than learned: a
lawyer must be able to read it and see why a clause was pulled in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Concept:
    name: str
    surface_forms: tuple[str, ...]
    related: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()


# Ordered roughly by how often each shows up in commercial contract review.
_CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "breach",
        ("breach", "breaches", "breached", "material breach", "default", "defaults",
         "violation", "violate", "violates", "non-performance", "nonperformance",
         "fail to perform", "fails to perform", "failure to perform", "non-compliance"),
        related=("cure_period", "notice", "termination", "remedies"),
        consequences=("cure_period", "notice", "termination", "remedies", "damages",
                      "limitation_of_liability", "indemnification", "suspension",
                      "dispute_resolution", "governing_law", "survival"),
    ),
    Concept(
        "cure_period",
        ("cure", "cure period", "uncured", "remedy the breach", "right to cure",
         "grace period", "remediation period"),
        related=("breach", "notice", "termination"),
        consequences=("termination", "remedies", "damages"),
    ),
    Concept(
        "termination",
        ("termination", "terminate", "terminates", "terminated", "cancellation",
         "cancel", "rescind", "rescission", "wind down", "wind-down", "exit",
         "termination for convenience", "termination for cause"),
        related=("notice", "term", "breach", "survival", "effect_of_termination"),
        consequences=("notice", "survival", "effect_of_termination", "fees",
                      "remedies", "limitation_of_liability"),
    ),
    Concept(
        "effect_of_termination",
        ("effect of termination", "upon termination", "consequences of termination",
         "post-termination", "return of materials", "transition assistance"),
        related=("termination", "survival", "confidentiality"),
        consequences=("survival", "fees", "confidentiality"),
    ),
    Concept(
        "notice",
        ("notice", "notices", "written notice", "prior written notice", "notify",
         "notification", "notice period"),
        related=("termination", "breach", "cure_period"),
        consequences=("termination", "cure_period"),
    ),
    Concept(
        "remedies",
        ("remedy", "remedies", "specific performance", "injunctive relief",
         "injunction", "equitable relief", "cumulative remedies", "self-help"),
        related=("breach", "damages", "dispute_resolution"),
        consequences=("damages", "limitation_of_liability", "dispute_resolution"),
    ),
    Concept(
        "damages",
        ("damages", "losses", "liquidated damages", "consequential damages",
         "indirect damages", "incidental damages", "punitive damages",
         "direct damages", "loss of profits", "lost profits"),
        related=("limitation_of_liability", "remedies", "indemnification"),
        consequences=("limitation_of_liability", "indemnification", "insurance"),
    ),
    Concept(
        "limitation_of_liability",
        ("limitation of liability", "liability cap", "cap on liability",
         "aggregate liability", "liable", "liability", "exclusion of liability",
         "limitation on damages", "maximum liability"),
        related=("damages", "indemnification", "insurance", "warranty"),
        consequences=("damages", "indemnification"),
    ),
    Concept(
        "indemnification",
        ("indemnification", "indemnify", "indemnity", "indemnities", "hold harmless",
         "defend and hold harmless", "indemnified party", "indemnifying party"),
        related=("limitation_of_liability", "damages", "insurance", "third_party_claims"),
        consequences=("limitation_of_liability", "insurance", "dispute_resolution"),
    ),
    Concept(
        "third_party_claims",
        ("third party claim", "third-party claim", "claims by third parties"),
        related=("indemnification", "limitation_of_liability"),
        consequences=("indemnification", "limitation_of_liability"),
    ),
    Concept(
        "governing_law",
        ("governing law", "governed by", "choice of law", "applicable law",
         "laws of the state", "laws of", "construed in accordance"),
        related=("dispute_resolution", "jurisdiction", "venue"),
        consequences=("dispute_resolution", "jurisdiction", "venue"),
    ),
    Concept(
        "jurisdiction",
        ("jurisdiction", "exclusive jurisdiction", "submit to the jurisdiction",
         "courts of", "court", "forum"),
        related=("governing_law", "venue", "dispute_resolution"),
        consequences=("dispute_resolution", "venue"),
    ),
    Concept(
        "venue",
        ("venue", "forum selection", "place of arbitration", "seat of arbitration"),
        related=("jurisdiction", "governing_law", "dispute_resolution"),
    ),
    Concept(
        "dispute_resolution",
        ("dispute resolution", "dispute", "disputes", "arbitration", "arbitral",
         "mediation", "escalation", "litigation", "class action waiver",
         "jury trial waiver"),
        related=("governing_law", "jurisdiction", "venue", "remedies"),
        consequences=("governing_law", "jurisdiction", "venue"),
    ),
    Concept(
        "survival",
        ("survival", "survive", "survives", "surviving", "shall survive termination"),
        related=("termination", "confidentiality", "limitation_of_liability"),
    ),
    Concept(
        "confidentiality",
        ("confidentiality", "confidential information", "non-disclosure",
         "nondisclosure", "nda", "proprietary information", "trade secret",
         "trade secrets", "keep confidential"),
        related=("survival", "data_protection", "intellectual_property", "remedies"),
        consequences=("remedies", "damages", "survival", "limitation_of_liability",
                      "indemnification"),
    ),
    Concept(
        "data_protection",
        ("data protection", "personal data", "gdpr", "ccpa", "privacy",
         "data processing", "data subject", "data breach", "processor",
         "controller", "hipaa"),
        related=("confidentiality", "security", "indemnification"),
        consequences=("indemnification", "damages", "notice"),
    ),
    Concept(
        "security",
        ("information security", "security measures", "safeguards",
         "security incident", "penetration test"),
        related=("data_protection", "confidentiality", "audit"),
    ),
    Concept(
        "fees",
        ("fees", "fee", "payment", "payments", "retainer", "invoice", "invoices",
         "compensation", "charges", "price", "pricing", "rates", "consideration",
         "amounts payable", "remuneration"),
        related=("late_payment", "taxes", "expenses", "suspension"),
        consequences=("late_payment", "suspension", "termination", "breach", "remedies"),
    ),
    Concept(
        "late_payment",
        ("late payment", "late payments", "overdue", "past due", "interest",
         "default interest", "unpaid", "arrears"),
        related=("fees", "suspension", "breach"),
        consequences=("suspension", "termination", "breach", "remedies", "damages"),
    ),
    Concept(
        "suspension",
        ("suspend", "suspension", "suspend services", "stop work", "withhold services"),
        related=("late_payment", "breach", "termination"),
        consequences=("termination", "remedies"),
    ),
    Concept(
        "taxes",
        ("tax", "taxes", "vat", "gst", "withholding", "sales tax"),
        related=("fees",),
    ),
    Concept(
        "expenses",
        ("expenses", "out-of-pocket", "reimbursement", "reimbursable", "travel costs"),
        related=("fees",),
    ),
    Concept(
        "term",
        ("term", "initial term", "duration", "commencement", "effective date",
         "expiration", "expiry", "renewal", "renew", "auto-renewal",
         "automatic renewal", "extension", "renewal term"),
        related=("termination", "notice"),
        consequences=("termination", "notice", "fees"),
    ),
    Concept(
        "services",
        ("services", "scope of services", "scope of work", "deliverables",
         "statement of work", "sow", "work product", "performance", "obligations"),
        related=("acceptance", "service_levels", "change_control", "fees"),
        consequences=("acceptance", "breach", "remedies"),
    ),
    Concept(
        "acceptance",
        ("acceptance", "accept", "acceptance criteria", "rejection", "sign-off",
         "deemed accepted"),
        related=("services", "service_levels", "warranty"),
        consequences=("remedies", "fees"),
    ),
    Concept(
        "service_levels",
        ("service level", "service levels", "sla", "uptime", "availability",
         "response time", "service credits", "performance standards"),
        related=("services", "remedies", "fees"),
        consequences=("remedies", "fees", "termination"),
    ),
    Concept(
        "change_control",
        ("change order", "change control", "change request", "variation",
         "amendment to scope"),
        related=("services", "amendment", "fees"),
    ),
    Concept(
        "warranty",
        ("warranty", "warranties", "warrants", "representation", "representations",
         "represents and warrants", "disclaimer", "as is", "fitness for a particular purpose",
         "merchantability"),
        related=("limitation_of_liability", "remedies", "indemnification"),
        consequences=("remedies", "damages", "limitation_of_liability", "indemnification"),
    ),
    Concept(
        "intellectual_property",
        ("intellectual property", "ip", "copyright", "patent", "trademark",
         "ownership", "license", "licence", "work made for hire", "moral rights",
         "background ip", "foreground ip"),
        related=("confidentiality", "indemnification", "warranty"),
        consequences=("indemnification", "remedies", "damages"),
    ),
    Concept(
        "assignment",
        ("assignment", "assign", "assigns", "transfer of rights", "novation",
         "change of control", "successors and assigns"),
        related=("subcontracting", "termination"),
        consequences=("termination", "breach"),
    ),
    Concept(
        "subcontracting",
        ("subcontract", "subcontractor", "subcontracting", "delegate", "delegation"),
        related=("assignment", "services", "indemnification"),
    ),
    Concept(
        "force_majeure",
        ("force majeure", "act of god", "acts of god", "beyond reasonable control",
         "unforeseeable event", "pandemic", "epidemic"),
        related=("termination", "suspension", "notice"),
        consequences=("suspension", "termination", "notice"),
    ),
    Concept(
        "insurance",
        ("insurance", "insured", "policy limits", "certificate of insurance",
         "coverage", "liability insurance"),
        related=("indemnification", "limitation_of_liability"),
    ),
    Concept(
        "audit",
        ("audit", "audit rights", "inspect", "inspection", "records", "books and records"),
        related=("compliance", "fees", "security"),
    ),
    Concept(
        "compliance",
        ("compliance", "comply with laws", "anti-bribery", "anti-corruption",
         "fcpa", "sanctions", "export control", "modern slavery", "code of conduct"),
        related=("audit", "indemnification", "termination"),
        consequences=("termination", "indemnification", "breach"),
    ),
    Concept(
        "non_compete",
        ("non-compete", "noncompete", "non-competition", "non-solicit",
         "nonsolicitation", "non-solicitation", "restrictive covenant", "no-hire"),
        related=("confidentiality", "remedies", "term"),
        consequences=("remedies", "damages", "injunctive relief"),
    ),
    Concept(
        "amendment",
        ("amendment", "amend", "amended", "modification", "modify", "variation",
         "written amendment", "restatement", "amended and restated", "supersede",
         "supersedes", "superseded"),
        related=("entire_agreement", "waiver", "change_control"),
    ),
    Concept(
        "entire_agreement",
        ("entire agreement", "integration clause", "merger clause",
         "supersedes all prior"),
        related=("amendment", "order_of_precedence"),
    ),
    Concept(
        "order_of_precedence",
        ("order of precedence", "conflict between", "in the event of a conflict",
         "priority of documents", "prevail", "prevails"),
        related=("entire_agreement", "incorporation_by_reference"),
    ),
    Concept(
        "waiver",
        ("waiver", "waive", "waived", "no waiver", "failure to enforce"),
        related=("amendment", "remedies"),
    ),
    Concept(
        "severability",
        ("severability", "severable", "unenforceable provision", "invalid provision"),
        related=("entire_agreement",),
    ),
    Concept(
        "incorporation_by_reference",
        ("incorporated by reference", "incorporation by reference", "attached hereto",
         "annexed hereto", "forms part of this agreement", "exhibit", "schedule",
         "annex", "appendix", "addendum"),
        related=("order_of_precedence", "services"),
    ),
    Concept(
        "notices_address",
        ("notices shall be sent", "address for notices", "attention:", "care of"),
        related=("notice",),
    ),
    Concept(
        "counterparts",
        ("counterparts", "electronic signature", "execution", "signed", "signature page"),
        related=("entire_agreement",),
    ),
    Concept(
        "parties",
        ("party", "parties", "provider", "supplier", "vendor", "contractor",
         "client", "customer", "purchaser", "buyer", "seller", "licensor",
         "licensee", "disclosing party", "receiving party", "company", "counterparty"),
        related=("assignment", "services"),
    ),
    Concept(
        "definitions",
        ("definition", "definitions", "defined term", "defined terms", "means",
         "shall mean", "interpretation", "construction"),
        related=("entire_agreement",),
    ),
)

CONCEPTS: dict[str, Concept] = {c.name: c for c in _CONCEPTS}


# --- synonym groups -------------------------------------------------------
# Bidirectional equivalences used to rewrite the query so the BM25 leg fires
# on the term the *document* uses, not the one the lawyer typed. Concept
# surface forms already cover most of this; these are the abbreviation and
# role-noun swaps that must expand in both directions.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("nda", "non-disclosure agreement", "nondisclosure agreement", "confidentiality agreement"),
    ("msa", "master services agreement", "master service agreement"),
    ("sow", "statement of work", "scope of work", "work order"),
    ("sla", "service level agreement", "service levels"),
    ("dpa", "data processing agreement", "data protection agreement"),
    ("loi", "letter of intent"),
    ("mou", "memorandum of understanding"),
    ("po", "purchase order"),
    ("t&c", "terms and conditions", "terms of service"),
    ("eula", "end user license agreement"),
    ("vendor", "supplier", "provider", "contractor", "service provider"),
    ("client", "customer", "purchaser", "buyer"),
    ("breach", "default", "violation"),
    ("terminate", "cancel", "end the agreement"),
    ("fees", "charges", "price", "consideration"),
    ("indemnify", "hold harmless"),
    ("governing law", "applicable law", "choice of law"),
    ("liability cap", "limitation of liability", "aggregate liability"),
    ("confidential information", "proprietary information"),
    ("effective date", "commencement date", "start date"),
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


@lru_cache(maxsize=1)
def _concept_matchers() -> tuple[tuple[str, re.Pattern], ...]:
    """One compiled alternation per concept.

    Longest-first alternation so "limitation of liability" wins over the bare
    "liability" that is also one of its surface forms.
    """
    matchers = []
    for concept in _CONCEPTS:
        forms = sorted(concept.surface_forms, key=len, reverse=True)
        pattern = "|".join(re.escape(f) for f in forms)
        matchers.append((concept.name, re.compile(rf"(?<!\w)(?:{pattern})(?!\w)")))
    return tuple(matchers)


@lru_cache(maxsize=1)
def _synonym_index() -> dict[str, tuple[str, ...]]:
    index: dict[str, tuple[str, ...]] = {}
    for group in SYNONYM_GROUPS:
        for term in group:
            index[term] = tuple(t for t in group if t != term)
    return index


def detect_concepts(text: str) -> list[str]:
    """Concept names present in a question or clause, most-mentioned first."""
    normalized = _norm(text)
    hits: dict[str, int] = {}
    for name, pattern in _concept_matchers():
        found = len(pattern.findall(normalized))
        if found:
            hits[name] = found
    return sorted(hits, key=lambda n: (-hits[n], n))


def expand_related(concepts: list[str], depth: int = 1) -> list[str]:
    """Breadth-first walk over `related` edges (legal concept expansion)."""
    return _walk(concepts, depth, edge="related")


def expand_consequences(concepts: list[str], depth: int = 2) -> list[str]:
    """Walk `consequences` edges: breach -> remedies -> damages -> liability cap.

    Depth 2 is the sweet spot measured on contract corpora — depth 1 misses
    the liability cap that limits the remedy, depth 3 starts pulling in
    unrelated boilerplate.
    """
    return _walk(concepts, depth, edge="consequences")


def _walk(seeds: list[str], depth: int, edge: str) -> list[str]:
    # dict.fromkeys, not a set: `list(set(...))` iterates in hash order, which
    # Python randomises per process. The walk's output order decides which
    # concepts get a retrieval probe within MAX_CONCEPT_PROBES, so a set here
    # makes the same question probe different concepts on different runs — and
    # therefore return different evidence. Reproducibility is not optional in
    # a system whose answers get relied on.
    ordered: list[str] = list(dict.fromkeys(s for s in seeds if s in CONCEPTS))
    seen = set(ordered)
    frontier = list(ordered)
    for _ in range(max(depth, 0)):
        # Neighbours are interleaved ACROSS the frontier, not appended one
        # source at a time. The probe budget is small, so draining the first
        # concept's neighbourhood before touching the second means a question
        # detecting {services, survival, termination} spends every probe on
        # things adjacent to "services" and never probes survival at all.
        branches = [
            [n for n in getattr(CONCEPTS[name], edge) if n in CONCEPTS and n not in seen]
            for name in frontier if name in CONCEPTS
        ]
        nxt: list[str] = []
        for rank in range(max((len(b) for b in branches), default=0)):
            for branch in branches:
                if rank < len(branch) and branch[rank] not in seen:
                    seen.add(branch[rank])
                    nxt.append(branch[rank])
                    ordered.append(branch[rank])
        if not nxt:
            break
        frontier = nxt
    return ordered


def concept_terms(concepts: list[str], per_concept: int = 3) -> list[str]:
    """Representative surface forms for concepts — the text appended to the
    query so the sparse/BM25 leg can actually match the target clauses."""
    terms: list[str] = []
    for name in concepts:
        concept = CONCEPTS.get(name)
        if concept is None:
            continue
        terms.extend(concept.surface_forms[:per_concept])
    return list(dict.fromkeys(terms))


def synonyms_for(text: str) -> list[str]:
    """Synonym/abbreviation swaps for terms appearing in the text."""
    normalized = _norm(text)
    out: list[str] = []
    for term, alternatives in _synonym_index().items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            out.extend(alternatives)
    return list(dict.fromkeys(out))
