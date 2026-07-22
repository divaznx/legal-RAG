"""Legal intelligence layer.

Everything in this package makes the retrieval pipeline legal-aware *before*
it touches the vector store, and legally complete *after* it does.

    ontology     concept graph: relations + consequence edges + synonyms
    intent       legal intent classification -> per-intent retrieval policy
    entities     parties, clauses, courts, statutes, money, dates, jurisdictions
    definitions  defined-term extraction; definition-first retrieval
    xref         "see Clause 4" / "subject to" / incorporated-by-reference
    profile      document-level legal profile built at ingest
    resolution   Document Resolution Layer (which agreement? which version?)
    planner      composes all of the above into a retrieval plan

Every stage is deterministic and inspectable: `QueryPlan.explain()` returns
the full trace, so a lawyer can audit why each clause reached the answer.
"""

from . import definitions, entities, intent, ontology, planner, profile, resolution, xref

__all__ = ["definitions", "entities", "intent", "ontology", "planner", "profile",
           "resolution", "xref"]
