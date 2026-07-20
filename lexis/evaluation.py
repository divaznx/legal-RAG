"""Evaluation harness (Feature: Evaluation Harness).

Benchmarks retrieval and (optionally) generation against a golden dataset:

    python cli.py eval                 # retrieval metrics only (no LLM needed)
    python cli.py eval --generate      # + citation/faithfulness/hallucination
    python cli.py eval --k 6 --golden eval/golden.json

Golden dataset format (JSON list):

    {
      "id": "payment-terms",
      "question": "What is the monthly retainer?",
      "relevant": [{"document": "MSA_Acme_v2.1.txt", "section": "Clause 2"}],
      "expect_refusal": false
    }

A retrieved chunk matches a relevance spec when the document matches and,
if given, the page matches and the section matches by numbering prefix.

Metrics:
- Recall@K      fraction of relevance specs covered by the top-K chunks
- Precision@K   fraction of top-K chunks matching any spec
- MRR           mean reciprocal rank of the first relevant chunk
- nDCG@K        binary-gain nDCG
- Retrieval success rate   any relevant chunk in the top final_k
- Gate accuracy  refusal-expected cases must fail the dense gate
- (--generate) Citation accuracy, faithfulness, hallucination rate,
  refusal correctness, latency

The metric functions are pure and unit-tested; every retrieval change is
measurable by re-running this command.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .vector_store import RetrievedChunk


# --- Pure metric functions --------------------------------------------------

def recall_at_k(spec_covered: list[bool]) -> float:
    """spec_covered[i] = spec i matched by at least one top-K chunk."""
    return sum(spec_covered) / len(spec_covered) if spec_covered else 0.0

def precision_at_k(chunk_relevant: list[bool]) -> float:
    """chunk_relevant[i] = top-K chunk i matches any spec."""
    return sum(chunk_relevant) / len(chunk_relevant) if chunk_relevant else 0.0

def mrr(chunk_relevant: list[bool]) -> float:
    for i, rel in enumerate(chunk_relevant, start=1):
        if rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(chunk_relevant: list[bool], n_relevant_specs: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rel in enumerate(chunk_relevant, start=1) if rel)
    ideal_hits = min(len(chunk_relevant), max(n_relevant_specs, 1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# --- Golden dataset ---------------------------------------------------------

@dataclass
class GoldenCase:
    id: str
    question: str
    relevant: list[dict] = field(default_factory=list)
    expect_refusal: bool = False
    k: int = 0  # per-case K override (overview cases evaluate wider)


def load_golden(path: str | Path | None = None) -> list[GoldenCase]:
    p = Path(path or settings.golden_dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {p} — create it or pass --golden"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [GoldenCase(**case) for case in raw]


def _matches(chunk: RetrievedChunk, spec: dict) -> bool:
    if chunk.document.lower() != str(spec.get("document", "")).lower():
        return False
    if "page" in spec and chunk.page != int(spec["page"]):
        return False
    if "section" in spec:
        want = " ".join(str(spec["section"]).lower().split())
        have = " ".join((chunk.section or "").lower().split())
        label_match = (
            have == want or have.startswith(want + ".") or want.startswith(have + ".")
            or have.startswith(want + " ")
        )
        # Fallback: a chunk whose *text* contains the section's heading line
        # still holds that section's content (large-chunk configurations
        # aggregate several clauses under the first clause's label).
        text_match = bool(
            re.search(rf"(?mi)^\s*{re.escape(str(spec['section']))}\b", chunk.text)
        )
        if not (label_match or text_match):
            return False
    return True


# --- Runner -----------------------------------------------------------------

def evaluate_case_retrieval(case: GoldenCase, chunks: list[RetrievedChunk], k: int) -> dict:
    top = chunks[:k]
    chunk_relevant = [any(_matches(c, s) for s in case.relevant) for c in top]
    spec_covered = [any(_matches(c, s) for c in top) for s in case.relevant]
    gate_passed = any(c.dense_score >= settings.min_dense_score for c in chunks)
    return {
        "id": case.id,
        "recall": recall_at_k(spec_covered),
        "precision": precision_at_k(chunk_relevant),
        "mrr": mrr(chunk_relevant),
        "ndcg": ndcg_at_k(chunk_relevant, len(case.relevant)),
        "success": any(chunk_relevant),
        "gate_passed": gate_passed,
        "gate_correct": (not gate_passed) if case.expect_refusal else gate_passed,
    }


def run_eval(golden_path: str | None = None, k: int | None = None, generate: bool = False) -> dict:
    from . import engine  # deferred: pulls in embedding/reranker models

    cases = load_golden(golden_path)
    k = k or settings.eval_k

    # The semantic cache would fake latency and mask retrieval regressions.
    cache_was = settings.cache_enabled
    settings.cache_enabled = False
    try:
        retrieval_rows, generation_rows = [], []
        for case in cases:
            chunks, _debug, timings = engine.retrieve_only(case.question)
            row = evaluate_case_retrieval(case, chunks, case.k or k)
            row["latency_ms"] = timings.get("total", 0.0)
            retrieval_rows.append(row)

            if generate:
                result = engine.ask(case.question)
                cit = result.citations
                generation_rows.append({
                    "id": case.id,
                    "refused": result.refused,
                    "refusal_correct": result.refused == case.expect_refusal,
                    "citations_total": cit.total,
                    "citations_verified": cit.verified,
                    "faithful": cit.passed,
                    "hallucinated": bool(cit.fabricated or cit.clause_mismatches
                                         or (cit.uncited_answer and not cit.refusal)),
                    "latency_ms": result.timings_ms.get("total", 0.0),
                })
    finally:
        settings.cache_enabled = cache_was

    answerable = [r for r, c in zip(retrieval_rows, cases) if not c.expect_refusal]
    report: dict = {
        "k": k,
        "cases": len(cases),
        "retrieval": {
            "recall_at_k": _avg([r["recall"] for r in answerable]),
            "precision_at_k": _avg([r["precision"] for r in answerable]),
            "mrr": _avg([r["mrr"] for r in answerable]),
            "ndcg_at_k": _avg([r["ndcg"] for r in answerable]),
            "success_rate": _avg([1.0 if r["success"] else 0.0 for r in answerable]),
            "gate_accuracy": _avg([1.0 if r["gate_correct"] else 0.0 for r in retrieval_rows]),
            "avg_latency_ms": _avg([r["latency_ms"] for r in retrieval_rows]),
        },
        "per_case": retrieval_rows,
    }
    if generate and generation_rows:
        total_cites = sum(g["citations_total"] for g in generation_rows)
        verified_cites = sum(g["citations_verified"] for g in generation_rows)
        answered = [g for g in generation_rows if not g["refused"]]
        report["generation"] = {
            "citation_accuracy": (verified_cites / total_cites) if total_cites else None,
            "faithfulness": _avg([1.0 if g["faithful"] else 0.0 for g in generation_rows]),
            "hallucination_rate": _avg([1.0 if g["hallucinated"] else 0.0 for g in answered]),
            "refusal_correctness": _avg([1.0 if g["refusal_correct"] else 0.0 for g in generation_rows]),
            "avg_latency_ms": _avg([g["latency_ms"] for g in generation_rows]),
        }
        report["per_case_generation"] = generation_rows
    return report


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def print_report(report: dict) -> None:
    r = report["retrieval"]
    print(f"Golden dataset: {report['cases']} case(s), K={report['k']}")
    print("\nRetrieval metrics (answerable cases):")
    print(f"  Recall@{report['k']:<2}       {r['recall_at_k']}")
    print(f"  Precision@{report['k']:<2}    {r['precision_at_k']}")
    print(f"  MRR             {r['mrr']}")
    print(f"  nDCG@{report['k']:<2}         {r['ndcg_at_k']}")
    print(f"  Success rate    {r['success_rate']}")
    print(f"  Gate accuracy   {r['gate_accuracy']}  (refusal cases must fail the dense gate)")
    print(f"  Avg latency     {r['avg_latency_ms']} ms (embed+retrieve+rerank)")
    print("\nPer case:")
    for row in report["per_case"]:
        print(f"  {row['id']:<28} recall={row['recall']:.2f} mrr={row['mrr']:.2f} "
              f"ndcg={row['ndcg']:.2f} gate={'ok' if row['gate_correct'] else 'WRONG'} "
              f"{row['latency_ms']:.0f}ms")
    if "generation" in report:
        g = report["generation"]
        print("\nGeneration metrics:")
        print(f"  Citation accuracy    {g['citation_accuracy']}")
        print(f"  Faithfulness         {g['faithfulness']}")
        print(f"  Hallucination rate   {g['hallucination_rate']}")
        print(f"  Refusal correctness  {g['refusal_correctness']}")
        print(f"  Avg latency          {g['avg_latency_ms']} ms")
