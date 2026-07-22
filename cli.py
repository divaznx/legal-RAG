"""Lexis Enterprise CLI.

  python cli.py ingest <file> [<file> ...]   parse+redact+index documents
  python cli.py ask "question"               grounded, cited answer
  python cli.py docs                         list ingested documents
  python cli.py delete <document-name>       remove a document + its vectors
"""

from __future__ import annotations

import argparse
import sys


def cmd_ingest(paths: list[str]) -> None:
    from lexis import ingest

    for path in paths:
        report = ingest.ingest_file(path)
        redactions = ", ".join(f"{k}x{v}" for k, v in report.redactions.items()) or "none"
        print(f"[ok] {report.document} (v{report.version}): "
              f"{report.pages} page(s), {report.chunks} chunk(s), redactions: {redactions}")
        if report.low_ocr_pages:
            print(f"     warning: possible scanned pages (low OCR confidence): {report.low_ocr_pages}")
        if report.injection.get("flagged"):
            print(f"     SECURITY WARNING: {len(report.injection['findings'])} passage(s) attempt "
                  f"to instruct the AI system ({', '.join(report.injection['categories'])}).")
            for finding in report.injection["findings"][:3]:
                print(f"       - \"{finding['excerpt']}\"")
            print("     The document was indexed; this text is treated as content, never "
                  "as instructions. Review it before relying on answers from this document.")


def cmd_ask(question: str) -> None:
    from lexis import engine, llm

    if not llm.llm_available():
        print("error: LLM endpoint is not reachable "
              "(is Ollama running? `ollama serve` / check OLLAMA_BASE_URL)", file=sys.stderr)
        sys.exit(1)

    result = None
    for kind, payload in engine.ask_stream(question):
        if kind == "delta":
            print(payload, end="", flush=True)
        else:
            result = payload
    print()

    print("\n" + "=" * 60)
    verdict = "VERIFIED" if result.citations.passed else "UNVERIFIED"
    cached = " (semantic cache hit)" if result.cached else ""
    print(f"System verification: {verdict} "
          f"({result.citations.verified}/{result.citations.total} citations matched retrieved chunks)")
    print(f"System confidence:   {result.confidence}{cached}")
    for limitation in result.limitations:
        print(f"Limitation:          {limitation}")
    stages = " | ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items())
    print(f"Latency:             {stages}")


_SEVERITY_MARK = {"blocker": "[BLOCKER]", "high": "[HIGH]   ",
                  "medium": "[MEDIUM] ", "note": "[NOTE]   "}


def cmd_review(document: str, playbook_name: str, playbook_file: str | None) -> None:
    from lexis import playbook as pb

    book = pb.load(playbook_file) if playbook_file else pb.builtin(playbook_name)
    try:
        findings = pb.review(document, book)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    rules = len(book.for_document(None))
    print(f"\n{book.name}  (position: {book.position}, v{book.version})")
    print(f"Reviewed {document} against {rules} rules -> {len(findings)} finding(s)\n")

    if not findings:
        print("  No findings. The document is consistent with this playbook.")
        return

    order = {"blocker": 0, "high": 1, "medium": 2, "note": 3}
    for f in sorted(findings, key=lambda f: order[f.severity]):
        print(f"{_SEVERITY_MARK[f.severity]} {f.title}  ({f.category})")
        print(f"           {f.detail}")
        if f.rationale:
            print(f"           Why: {' '.join(f.rationale.split())}")
        if f.suggested:
            print(f"           Ask for: {' '.join(f.suggested.split())}")
        print()

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    print("Summary: " + ", ".join(f"{n} {sev}" for sev, n in
                                  sorted(counts.items(), key=lambda kv: order[kv[0]])))


def cmd_obligations(document: str | None, obligor: str | None, modality: str | None) -> None:
    from lexis.store import repository

    rows = repository.obligations(document=document, obligor=obligor, modality=modality)
    if not rows:
        print("No obligations recorded. Ingest a document first.")
        return
    from lexis.store.models import modal_phrase

    print(f"\n{len(rows)} obligation(s)\n")
    for r in rows:
        deadline = f"  [{r['deadline_raw']}]" if r["deadline_raw"] else ""
        print(f"  {r['filename']} {r['section'] or '-':<14} "
              f"{r['obligor'] or '?'} {modal_phrase(r)[:92]}{deadline}")
        if r["condition"]:
            print(f"     condition: {r['condition'][:100]}")


def cmd_calendar(document: str | None, effective: str | None, ics: str | None) -> None:
    from lexis.calendar import build, to_ics

    entries = build(document=document, effective_date=effective)
    if not entries:
        print("No dates recorded. Ingest a document first.")
        return

    print(f"\n{len(entries)} date rule(s)\n")
    for e in entries:
        when = e.due.isoformat() if e.due else "unresolved"
        print(f"  {when:<12} {e.kind:<12} {e.document} {e.section or '-':<14} {e.description[:70]}")
        if e.due is None:
            print(f"     depends on: {e.blocked_on}")

    if ics:
        count = to_ics(entries, ics)
        print(f"\n[ok] wrote {count} dated event(s) to {ics}")
        print("     Import into Outlook or Google Calendar.")


def cmd_portfolio() -> None:
    from lexis.store import repository

    s = repository.portfolio_summary()
    print(f"\nDocuments    {s['documents']}")
    print(f"Clauses      {s['clauses']}")
    print(f"Obligations  {s['obligations']}")
    print(f"Key dates    {s['key_dates']}")
    by_sev = s["findings_by_severity"]
    if by_sev:
        print("\nOpen findings by severity")
        for sev in ("blocker", "high", "medium", "note"):
            if sev in by_sev:
                print(f"  {sev:<9} {by_sev[sev]}")
    else:
        print("\nNo review has been run yet (try: python cli.py review <doc>)")


def cmd_playbooks() -> None:
    from lexis import playbook as pb

    for name in pb.list_builtin():
        book = pb.builtin(name)
        print(f"  {name:<20} {book.name} "
              f"(position: {book.position}, {len(book.rules)} rules)")


def cmd_warm() -> None:
    from lexis import engine

    engine.warmup()
    print("[ok] embedder, sparse encoder, reranker, vector store, and LLM are warm")


def cmd_docs() -> None:
    from lexis import ingest

    manifest = ingest.load_manifest()
    if not manifest:
        print("No documents ingested yet.")
        return
    for name, info in sorted(manifest.items()):
        print(f"- {name} (v{info['version']}): {info['pages']} page(s), "
              f"{info['chunks']} chunk(s), ingested {info['ingested_at']}")


def cmd_delete(document: str) -> None:
    from lexis import ingest

    if ingest.delete_document(document):
        print(f"[ok] deleted {document}")
    else:
        print(f"error: no such document: {document}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexis Enterprise — grounded legal RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest one or more documents")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("ask", help="ask a question over the ingested documents")
    p.add_argument("question")

    sub.add_parser("docs", help="list ingested documents")

    p = sub.add_parser("delete", help="delete a document")
    p.add_argument("document")

    sub.add_parser("warm", help="preload models + LLM (do this once after boot)")

    p = sub.add_parser("review", help="review a document against a playbook")
    p.add_argument("document")
    p.add_argument("--playbook", default="saas_customer", help="built-in playbook name")
    p.add_argument("--playbook-file", help="path to a custom playbook YAML")

    p = sub.add_parser("obligations", help="obligation register")
    p.add_argument("--document")
    p.add_argument("--obligor", help="filter by party, e.g. Customer")
    p.add_argument("--modality", choices=["obligation", "right", "prohibition"])

    p = sub.add_parser("calendar", help="deadlines and renewal dates")
    p.add_argument("--document")
    p.add_argument("--effective", help="effective date (YYYY-MM-DD) to resolve rules against")
    p.add_argument("--ics", help="write an .ics calendar file")

    sub.add_parser("portfolio", help="corpus-wide obligation and risk summary")
    sub.add_parser("playbooks", help="list built-in playbooks")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args.paths)
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "docs":
        cmd_docs()
    elif args.command == "delete":
        cmd_delete(args.document)
    elif args.command == "warm":
        cmd_warm()
    elif args.command == "review":
        cmd_review(args.document, args.playbook, args.playbook_file)
    elif args.command == "obligations":
        cmd_obligations(args.document, args.obligor, args.modality)
    elif args.command == "calendar":
        cmd_calendar(args.document, args.effective, args.ics)
    elif args.command == "portfolio":
        cmd_portfolio()
    elif args.command == "playbooks":
        cmd_playbooks()


if __name__ == "__main__":
    main()
