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


if __name__ == "__main__":
    main()
