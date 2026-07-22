"""Legal intent classification and the retrieval policy each intent implies.

Different legal questions need structurally different evidence sets. "What
does Clause 6 say?" wants one clause verbatim. "What happens if I breach
Clause 8?" wants eight clauses spread across the contract, none of which is
Clause 8. Retrieving four semantically-similar chunks serves the first and
silently fails the second — which is why intent is resolved *before*
retrieval and drives the retrieval plan rather than only the prompt.

Classification is weighted-pattern based, not model based: it runs in
microseconds, is unit-testable, and a lawyer can be told exactly why a
question was treated as a comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalPolicy:
    """How the pipeline should gather evidence for this intent."""

    final_k: int = 6
    candidate_k: int = 40
    definitions_first: bool = True     # retrieve defined terms before operative clauses
    related_depth: int = 1             # legal-concept graph expansion depth
    consequence_depth: int = 0         # consequence-edge expansion depth
    follow_xrefs: bool = True          # resolve "see Clause X" / "subject to"
    parent_context: bool = True        # attach parent heading / sibling context
    multi_document: bool = False       # may evidence span more than one agreement?
    prefer_latest_version: bool = True
    exact_clause_priority: bool = False  # pin the literally-referenced clause first


@dataclass(frozen=True)
class Intent:
    name: str
    description: str
    policy: RetrievalPolicy


INTENTS: dict[str, Intent] = {
    "clause_lookup": Intent(
        "clause_lookup",
        "Retrieve the text of a specific clause, section, or article.",
        RetrievalPolicy(final_k=5, related_depth=0, exact_clause_priority=True),
    ),
    "definition_lookup": Intent(
        "definition_lookup",
        "Retrieve how a term is defined in the agreement.",
        RetrievalPolicy(final_k=5, related_depth=0, follow_xrefs=True),
    ),
    "contract_overview": Intent(
        "contract_overview",
        "Summarise the agreement as a whole.",
        RetrievalPolicy(final_k=10, candidate_k=60, related_depth=0, parent_context=False),
    ),
    "comparison": Intent(
        "comparison",
        "Compare clauses across versions or across agreements.",
        RetrievalPolicy(final_k=10, candidate_k=60, related_depth=1,
                        multi_document=True, prefer_latest_version=False),
    ),
    "consequence_analysis": Intent(
        "consequence_analysis",
        "Explain what follows from an event, breach, or trigger.",
        # The widest evidence set of any intent: the chain runs trigger ->
        # notice -> cure -> termination -> remedies -> damages -> cap ->
        # survival, and each link is a different clause. Answering from four
        # chunks means silently dropping most of the chain.
        RetrievalPolicy(final_k=12, candidate_k=60, related_depth=1, consequence_depth=2),
    ),
    "risk_analysis": Intent(
        "risk_analysis",
        "Identify exposure, unfavourable terms, and unlimited liabilities.",
        RetrievalPolicy(final_k=10, candidate_k=60, related_depth=1, consequence_depth=1),
    ),
    "timeline": Intent(
        "timeline",
        "Extract dates, deadlines, notice periods, and durations.",
        RetrievalPolicy(final_k=8, candidate_k=50, related_depth=1),
    ),
    "obligation_lookup": Intent(
        "obligation_lookup",
        "Identify what a named party must, may, or must not do.",
        RetrievalPolicy(final_k=8, candidate_k=50, related_depth=1),
    ),
    "party_lookup": Intent(
        "party_lookup",
        "Identify the parties, signatories, and their roles.",
        RetrievalPolicy(final_k=5, related_depth=0, parent_context=False),
    ),
    "amount_lookup": Intent(
        "amount_lookup",
        "Extract monetary amounts, caps, rates, and thresholds.",
        # Roomier than a bare figure lookup needs, because the exceptions to a
        # cap live in neighbouring sub-clauses and a cap quoted without them
        # is wrong.
        RetrievalPolicy(final_k=8, related_depth=1),
    ),
    "jurisdiction_lookup": Intent(
        "jurisdiction_lookup",
        "Identify governing law, jurisdiction, venue, and dispute forum.",
        RetrievalPolicy(final_k=6, related_depth=1),
    ),
    "legal_interpretation": Intent(
        "legal_interpretation",
        "Interpret whether the agreement permits, requires, or prohibits something.",
        RetrievalPolicy(final_k=8, candidate_k=50, related_depth=1, consequence_depth=1),
    ),
}

DEFAULT_INTENT = "legal_interpretation"


# (intent, weight, pattern). Weights matter more than order: "what happens if
# I breach Clause 8" matches both a clause reference and a consequence
# trigger, and the consequence reading must win.
_RULES: tuple[tuple[str, float, re.Pattern], ...] = tuple(
    (name, weight, re.compile(pattern, re.IGNORECASE))
    for name, weight, pattern in (
        # --- consequence -------------------------------------------------
        ("consequence_analysis", 5.0, r"\bwhat (?:happens|occurs|follows)\b"),
        ("consequence_analysis", 5.0, r"\bif (?:i|we|they|the \w+) (?:breach|default|violate|fail|miss|terminate|don'?t|do not)\b"),
        ("consequence_analysis", 4.0, r"\bconsequences?\b|\brepercussions?\b|\bfallout\b"),
        ("consequence_analysis", 3.5, r"\bwhat (?:can|could|will|would) happen\b"),
        ("consequence_analysis", 3.0, r"\bremed(?:y|ies) (?:for|available)\b"),
        ("consequence_analysis", 3.0, r"\bin the event of (?:a )?(?:breach|default|non-?payment)\b"),
        ("consequence_analysis", 2.5, r"\bexposed? to\b|\bon the hook\b"),
        # --- comparison ---------------------------------------------------
        ("comparison", 5.0, r"\bcompare\b|\bcomparison\b|\bdiff(?:erence|erences|er)\b"),
        ("comparison", 4.5, r"\bwhat changed\b|\bchanges? between\b|\bversus\b|\bvs\.?\b"),
        ("comparison", 4.0, r"\bbetween (?:the )?(?:two|both|v?\d[\d.]*\s*and)\b"),
        ("comparison", 3.5, r"\b(?:v\d[\d.]*)\s+and\s+(?:v\d[\d.]*)\b"),
        ("comparison", 3.0, r"\bwhich (?:one )?is (?:more|less|better|worse|stricter|broader)\b"),
        # --- definition ---------------------------------------------------
        ("definition_lookup", 5.0, r"\bhow is\s+.{2,60}\s+defined\b|\bdefinition of\b"),
        ("definition_lookup", 4.5, r"\bwhat does\s+.{2,60}\s+mean\b|\bwhat is meant by\b"),
        ("definition_lookup", 4.0, r"\bdefined term\b|\bdefine[sd]?\b"),
        ("definition_lookup", 3.0, r"\bmeaning of\b"),
        # --- overview -------------------------------------------------------
        ("contract_overview", 5.0, r"\bsummar(?:ise|ize|y)\b|\boverview\b|\btl;?dr\b"),
        ("contract_overview", 4.0, r"\bwhat is this (?:agreement|contract|document)\b"),
        ("contract_overview", 3.5, r"\bkey (?:terms|points|provisions)\b|\bmain (?:terms|points)\b"),
        ("contract_overview", 3.0, r"\bwalk me through\b|\bbrief me\b"),
        # --- risk ------------------------------------------------------------
        ("risk_analysis", 5.0, r"\brisks?\b|\brisky\b|\bred flags?\b"),
        ("risk_analysis", 4.0, r"\bexposure\b|\bunfavou?rable\b|\bone-?sided\b|\bonerous\b"),
        ("risk_analysis", 3.5, r"\bshould i (?:worry|be concerned)\b|\bconcerns?\b"),
        ("risk_analysis", 3.0, r"\bunlimited liability\b|\buncapped\b"),
        # --- timeline ---------------------------------------------------------
        ("timeline", 5.0, r"\btimeline\b|\bkey dates\b|\bdeadlines?\b"),
        ("timeline", 4.0, r"\bwhen (?:does|do|will|must|is)\b"),
        ("timeline", 3.5, r"\bhow (?:long|many days|much notice)\b"),
        ("timeline", 3.0, r"\bnotice period\b|\bexpir(?:e|es|y|ation)\b|\brenewal date\b"),
        # --- obligations -------------------------------------------------------
        ("obligation_lookup", 4.5, r"\bobligations?\b|\bduties\b|\bresponsibilit(?:y|ies)\b"),
        ("obligation_lookup", 4.0, r"\bwhat (?:must|shall|should|is required to|has to|does)\s+(?:the\s+)?\w+\s+(?:do|deliver|provide|perform|pay)\b"),
        # "What services must the Provider deliver" — the object sits between
        # the interrogative and the modal, so the pattern above cannot see it.
        ("obligation_lookup", 4.0, r"\b(?:must|shall|is required to|has to)\s+(?:the\s+)?\w+\s+(?:deliver|provide|perform|pay|supply|do|furnish|maintain)\b"),
        ("obligation_lookup", 3.5, r"\bwho is responsible\b|\brequired to\b"),
        # --- party --------------------------------------------------------------
        ("party_lookup", 5.0, r"\bwho (?:are|is) the (?:part(?:y|ies)|signator(?:y|ies)|client|customer|provider|supplier|vendor)\b"),
        ("party_lookup", 4.0, r"\bbetween whom\b|\bwho signed\b|\bcounterpart(?:y|ies)\b"),
        ("party_lookup", 3.0, r"\bname of the (?:client|provider|vendor|supplier)\b"),
        # --- amount --------------------------------------------------------------
        ("amount_lookup", 4.5, r"\bhow much\b|\bwhat (?:is|are) the (?:fee|fees|price|rate|cost|retainer|cap)\b"),
        ("amount_lookup", 4.0, r"\bliability cap\b|\bcapped at\b|\bmonetary (?:limit|value)\b"),
        ("amount_lookup", 3.0, r"\bamount\b|\bpayable\b"),
        # --- jurisdiction ---------------------------------------------------------
        ("jurisdiction_lookup", 5.0, r"\bgoverning law\b|\bwhich law\b|\bwhat law governs\b"),
        ("jurisdiction_lookup", 4.5, r"\bjurisdiction\b|\bvenue\b|\bwhich court\b|\bwhere .{0,25}(?:sue|dispute|litigat)\w*\b"),
        ("jurisdiction_lookup", 4.0, r"\barbitrat\w+\b|\bdispute resolution\b|\bforum\b"),
        # --- clause lookup ----------------------------------------------------------
        ("clause_lookup", 4.0, r"\b(?:clause|section|article|paragraph|§)\s*\d+(?:\.\d+)*\b"),
        ("clause_lookup", 3.5, r"\bwhat does\s+(?:clause|section|article)\b|\bshow me\b|\bquote\b|\bverbatim\b|\bfull text\b"),
        ("clause_lookup", 3.0, r"\b(?:termination|confidentiality|indemnit\w+|liability|payment|governing law)\s+(?:clause|section|provision)\b"),
        # --- interpretation ------------------------------------------------------------
        ("legal_interpretation", 3.5, r"\b(?:can|may|am i allowed to|are we permitted to|is it permissible)\b"),
        ("legal_interpretation", 3.0, r"\bdoes (?:the|this) (?:agreement|contract|clause) (?:allow|permit|cover|require|prohibit)\b"),
        ("legal_interpretation", 2.5, r"\bwould .{0,40}\bbreach\b|\bam i in breach\b"),
    )
)


@dataclass
class IntentResult:
    name: str
    confidence: float
    scores: dict[str, float]
    matched: list[str]

    @property
    def intent(self) -> Intent:
        return INTENTS[self.name]

    @property
    def policy(self) -> RetrievalPolicy:
        return self.intent.policy


def classify(question: str) -> IntentResult:
    scores: dict[str, float] = {}
    matched: list[str] = []
    for name, weight, pattern in _RULES:
        hit = pattern.search(question)
        if hit:
            scores[name] = scores.get(name, 0.0) + weight
            matched.append(f"{name}: '{hit.group(0).strip()}'")

    if not scores:
        return IntentResult(DEFAULT_INTENT, 0.0, {}, [])

    best = max(scores, key=lambda n: scores[n])
    total = sum(scores.values())
    return IntentResult(best, round(scores[best] / total, 3), scores, matched)
