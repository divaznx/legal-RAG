"""Query planning — turns a lawyer's sentence into a retrieval plan.

This is the "legal intelligence layer" proper: everything that must be
decided *before* the vector store is touched.

    question
      -> document resolution   (which agreement?)
      -> intent classification (what kind of legal answer?)
      -> entity recognition    (clauses, parties, money, dates, law)
      -> concept detection     (breach, liability, survival, ...)
      -> concept expansion     (related + consequence edges)
      -> sub-queries           (one probe per legal concept)

The sub-query design is the load-bearing part. A single embedding of "what
happens if I breach Clause 8?" sits nowhere near the Limitation of Liability
clause in vector space, so one query can never retrieve it — no matter how
good the embedding model is. Issuing a separate probe per expanded concept
and fusing the results is what makes consequence-aware retrieval work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import definitions as definitions_mod
from . import entities as entities_mod
from . import intent as intent_mod
from . import ontology, resolution
from .entities import ClauseRef, LegalEntities
from .profile import DocumentProfile
from .resolution import DocumentResolution

# Probe budget. Each sub-query costs one embedding + one hybrid search;
# beyond ~8 the fused set is dominated by weak concept matches.
MAX_CONCEPT_PROBES = 8

_PRIMARY_WEIGHT = 1.0
_DEFINITION_WEIGHT = 0.8
_RELATED_WEIGHT = 0.55
_CONSEQUENCE_WEIGHT = 0.7


@dataclass
class SubQuery:
    text: str
    weight: float
    purpose: str


@dataclass
class QueryPlan:
    question: str
    intent: intent_mod.IntentResult
    resolution: DocumentResolution
    entities: LegalEntities
    concepts: list[str] = field(default_factory=list)
    expanded_concepts: list[str] = field(default_factory=list)
    definition_targets: list[str] = field(default_factory=list)
    subqueries: list[SubQuery] = field(default_factory=list)
    sparse_query: str = ""
    clause_targets: list[ClauseRef] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    known_terms: list[str] = field(default_factory=list)   # defined terms in scope

    @property
    def policy(self) -> intent_mod.RetrievalPolicy:
        return self.intent.policy

    def explain(self) -> dict:
        """Human-readable trace — surfaced in the UI/API so a lawyer can audit
        why particular clauses were pulled in."""
        return {
            "intent": self.intent.name,
            "intent_confidence": self.intent.confidence,
            "intent_signals": self.intent.matched,
            "documents": self.resolution.documents,
            "document_reason": self.resolution.reason,
            "superseded": self.resolution.superseded,
            "concepts": self.concepts,
            "expanded_concepts": [c for c in self.expanded_concepts if c not in self.concepts],
            "definition_targets": self.definition_targets,
            "clause_targets": [c.label for c in self.clause_targets],
            "synonyms": self.synonyms,
            "entities": self.entities.as_dict(),
            "subqueries": [{"text": s.text, "purpose": s.purpose, "weight": s.weight}
                           for s in self.subqueries],
        }


def _known_terms(profiles: list[DocumentProfile], documents: list[str]) -> set[str]:
    scope = set(documents)
    terms: set[str] = set()
    for p in profiles:
        if not scope or p.document in scope:
            terms.update(t.lower() for t in p.defined_terms)
    return terms


def plan(
    question: str,
    profiles: list[DocumentProfile],
    allow_clarification: bool = True,
) -> QueryPlan:
    intent = intent_mod.classify(question)
    resolved = resolution.resolve(question, profiles, intent,
                                  allow_clarification=allow_clarification)
    entities = entities_mod.extract(question)
    policy = intent.policy

    concepts = ontology.detect_concepts(question)

    expanded = list(concepts)
    if policy.consequence_depth:
        expanded = _merge(expanded, ontology.expand_consequences(concepts, policy.consequence_depth))
    if policy.related_depth:
        expanded = _merge(expanded, ontology.expand_related(concepts, policy.related_depth))

    synonyms = ontology.synonyms_for(question)
    known_terms = _known_terms(profiles, resolved.documents)
    targets = definitions_mod.definition_targets(
        question, known_terms
    ) if policy.definitions_first else []

    subqueries = _build_subqueries(question, concepts, expanded, targets, synonyms, policy)
    sparse_query = _build_sparse_query(question, expanded, synonyms, targets)

    return QueryPlan(
        question=question,
        intent=intent,
        resolution=resolved,
        entities=entities,
        concepts=concepts,
        expanded_concepts=expanded,
        definition_targets=targets,
        subqueries=subqueries,
        sparse_query=sparse_query,
        clause_targets=entities.clause_refs,
        synonyms=synonyms,
        known_terms=sorted(known_terms),
    )


def _merge(base: list[str], extra: list[str]) -> list[str]:
    return list(dict.fromkeys([*base, *extra]))


def _concept_probe(name: str) -> str:
    """A short natural-language probe for one concept.

    Phrased as contract language rather than as a keyword list, because the
    dense model was trained on prose: "limitation of liability, liability cap,
    aggregate liability" embeds far closer to the real clause than the bare
    concept name would.
    """
    concept = ontology.CONCEPTS.get(name)
    if concept is None:
        return name.replace("_", " ")
    return ", ".join(concept.surface_forms[:4])


def _build_subqueries(
    question: str,
    concepts: list[str],
    expanded: list[str],
    definition_targets: list[str],
    synonyms: list[str],
    policy: intent_mod.RetrievalPolicy,
) -> list[SubQuery]:
    subqueries = [SubQuery(question, _PRIMARY_WEIGHT, "primary")]

    # Synonym-rewritten primary: fires the dense leg on the vocabulary the
    # contract uses ("Provider") when the lawyer typed "vendor".
    if synonyms:
        subqueries.append(
            SubQuery(f"{question} {' '.join(synonyms[:6])}", 0.75, "primary:synonyms")
        )

    for term in definition_targets[:3]:
        subqueries.append(
            SubQuery(f'"{term}" means definition of {term}', _DEFINITION_WEIGHT,
                     f"definition:{term}")
        )

    seeds = set(concepts)
    budget = MAX_CONCEPT_PROBES
    for name in expanded:
        if budget <= 0:
            break
        if name in seeds and len(concepts) > 1:
            continue  # already carried by the primary query
        weight = _CONSEQUENCE_WEIGHT if policy.consequence_depth else _RELATED_WEIGHT
        subqueries.append(SubQuery(_concept_probe(name), weight, f"concept:{name}"))
        budget -= 1

    return subqueries


def _build_sparse_query(
    question: str,
    expanded: list[str],
    synonyms: list[str],
    definition_targets: list[str],
) -> str:
    """BM25 leg gets everything — unlike a dense vector, a bag of terms does
    not get diluted by adding more of them."""
    terms = [question, *synonyms, *definition_targets, *ontology.concept_terms(expanded, 2)]
    return " ".join(dict.fromkeys(terms))
