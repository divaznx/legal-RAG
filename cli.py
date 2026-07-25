"""Lexis Enterprise CLI — operator and administrator surface.

Documents (all accept --tenant, default from DEFAULT_TENANT):

  python cli.py ingest <file> [<file> ...]   parse+redact+index documents
  python cli.py ask "question"               grounded, cited answer
  python cli.py docs                         list ingested documents
  python cli.py delete <document-name>       remove a document + its vectors
  python cli.py warm                         preload models + LLM

Administration:

  python cli.py tenants                      tenants with data on this box
  python cli.py keys add --label L --tenant T [--role admin]
  python cli.py keys list [--all]
  python cli.py keys revoke <key-id>
  python cli.py audit tail [--tenant T] [--limit N] [--action ask]
  python cli.py audit verify                 recompute the whole hash chain
  python cli.py migrate                      adopt a pre-tenancy corpus

Key management lives here rather than on the HTTP API on purpose: issuing
credentials over the network is a far larger blast radius than answering
questions, and an on-prem appliance has a console.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys


def _principal(tenant: str):
    """The local operator, as the audit log sees them.

    Shell access already implies full control of the box, so the role is
    `admin`; the value of the record is the attribution and the timestamp,
    not an access decision.
    """
    from lexis.auth import Principal

    return Principal(key_id="cli", label=f"cli:{getpass.getuser()}", tenant=tenant, role="admin")


def cmd_ingest(paths: list[str], tenant: str) -> None:
    from lexis import audit, ingest

    for path in paths:
        try:
            report = ingest.ingest_file(path)
        except (ValueError, OSError) as exc:
            audit.record("ingest", outcome="error", principal=_principal(tenant), client="cli",
                         detail={"path": path, "error": str(exc)})
            print(f"error: {path}: {exc}", file=sys.stderr)
            continue

        audit.record(
            "ingest", principal=_principal(tenant), client="cli",
            documents=[report.document],
            detail={"version": report.version, "pages": report.pages, "chunks": report.chunks,
                    "redactions": report.redactions, "low_ocr_pages": report.low_ocr_pages,
                    "injection_flagged": bool(report.injection.get("flagged"))},
        )
        redactions = ", ".join(f"{k}x{v}" for k, v in report.redactions.items()) or "none"
        print(f"[ok] {report.document} (v{report.version}) -> {tenant}: "
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


def cmd_ask(question: str, tenant: str) -> None:
    from lexis import audit, engine, llm

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

    audit.answer_event(result, principal=_principal(tenant), client="cli")

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


def cmd_docs(tenant: str) -> None:
    from lexis import ingest

    manifest = ingest.load_manifest()
    if not manifest:
        print(f"No documents ingested yet for tenant '{tenant}'.")
        return
    for name, info in sorted(manifest.items()):
        print(f"- {name} (v{info['version']}): {info['pages']} page(s), "
              f"{info['chunks']} chunk(s), ingested {info['ingested_at']}")


def cmd_delete(document: str, tenant: str) -> None:
    from lexis import audit, ingest

    deleted = ingest.delete_document(document)
    audit.record("delete", outcome="ok" if deleted else "error",
                 principal=_principal(tenant), client="cli", documents=[document],
                 detail={} if deleted else {"error": "not found"})
    if deleted:
        print(f"[ok] deleted {document} from tenant '{tenant}'")
    else:
        print(f"error: no such document in tenant '{tenant}': {document}", file=sys.stderr)
        sys.exit(1)


def cmd_tenants() -> None:
    from lexis import auth, tenancy

    on_disk = set(tenancy.known_tenants())
    with_keys: dict[str, int] = {}
    for key in auth.list_keys():
        with_keys[key["tenant"]] = with_keys.get(key["tenant"], 0) + 1

    names = sorted(on_disk | set(with_keys))
    if not names:
        print("No tenants yet.")
        return
    for name in names:
        marks = []
        if name in on_disk:
            marks.append("data")
        if name in with_keys:
            marks.append(f"{with_keys[name]} active key(s)")
        else:
            marks.append("NO KEYS - unreachable over the API")
        print(f"- {name}: {', '.join(marks)}")


def cmd_keys_add(label: str, tenant: str, role: str) -> None:
    from lexis import audit, auth

    principal, secret = auth.issue(label, tenant, role)
    audit.record("key_issued", principal=_principal(principal.tenant), client="cli",
                 detail={"key_id": principal.key_id, "label": label,
                         "tenant": principal.tenant, "role": role})
    print(f"[ok] issued {role} key {principal.key_id} for tenant '{principal.tenant}' ({label})")
    print("\n  Copy it now - it is hashed on disk and cannot be shown again:\n")
    print(f"      {secret}\n")
    print("  Use it as:  Authorization: Bearer <key>   (or X-API-Key: <key>)")


def cmd_keys_list(include_revoked: bool) -> None:
    from lexis import auth

    keys = auth.list_keys(include_revoked=include_revoked)
    if not keys:
        print("No API keys issued yet.")
        return
    print(f"{'ID':10} {'TENANT':16} {'ROLE':8} {'CREATED':22} LABEL")
    for key in keys:
        status = f"  (revoked {key['revoked_at']})" if key.get("revoked_at") else ""
        print(f"{key['id']:10} {key['tenant']:16} {key['role']:8} "
              f"{key['created_at']:22} {key['label']}{status}")


def cmd_keys_revoke(key_id: str) -> None:
    from lexis import audit, auth

    if auth.revoke(key_id):
        audit.record("key_revoked", principal=_principal("default"), client="cli",
                     detail={"key_id": key_id})
        print(f"[ok] revoked {key_id} - it stops working on the next request")
    else:
        print(f"error: no active key with id {key_id}", file=sys.stderr)
        sys.exit(1)


def cmd_audit_tail(tenant: str | None, limit: int, action: str | None) -> None:
    from lexis import audit

    events = audit.query(tenant=tenant, limit=limit, action=action)
    if not events:
        print("No audit events recorded.")
        return
    for event in reversed(events):  # oldest first reads better in a terminal
        who = event["principal_label"] or event["principal_id"] or "-"
        line = (f"{event['seq']:>6}  {event['ts']}  {event['tenant']:12} {who:20} "
                f"{event['action']:12} {event['outcome']}")
        print(line)
        if event["question"]:
            print(f"        Q: {event['question'][:100]}")
        if event["confidence"]:
            cites = f"{event['citations_verified']}/{event['citations_total']}"
            print(f"        -> {event['confidence']} confidence, {cites} citations verified, "
                  f"docs={json.loads(event['documents'] or '[]')}")
        if event["detail"] and event["detail"] != "{}":
            print(f"        {event['detail']}")


def cmd_audit_verify() -> None:
    from lexis import audit

    ok, checked, first_bad = audit.verify_chain()
    if ok:
        print(f"[ok] audit chain intact across {checked} event(s)")
        print(f"     head hash: {audit.head() or '(empty log)'}")
        print("     Record that hash off this machine - the chain detects modified and\n"
              "     deleted rows on its own, but only an external copy of the head detects\n"
              "     the newest rows being truncated.")
    else:
        print(f"[FAIL] audit chain broken at event seq={first_bad} "
              f"(verified {checked} event(s) before it)", file=sys.stderr)
        sys.exit(1)


def cmd_migrate(tenant: str) -> None:
    """Adopt a corpus ingested before tenancy existed.

    Pre-tenancy state lived at data/manifest.json and in a collection with no
    tenant suffix. Both are moved rather than re-ingested, so the embeddings
    (the expensive part) survive.
    """
    import shutil

    from lexis import tenancy, vector_store
    from lexis.config import settings

    slug = tenancy.normalize(tenant)
    moved: list[str] = []

    with tenancy.using(slug):
        target_dir = tenancy.data_path()
        for name in ("manifest.json", "answer_cache.json"):
            legacy = settings.data_path / name
            if legacy.exists() and not (target_dir / name).exists():
                shutil.move(str(legacy), str(target_dir / name))
                moved.append(name)
        legacy_uploads = settings.data_path / "uploads"
        if legacy_uploads.is_dir() and not (target_dir / "uploads").exists():
            shutil.move(str(legacy_uploads), str(target_dir / "uploads"))
            moved.append("uploads/")

        client = vector_store.client()
        legacy_collection = settings.qdrant_collection  # no suffix = pre-tenancy
        copied = already = total = 0
        if client.collection_exists(legacy_collection):
            target = vector_store.ensure_collection(tenancy.collection())
            already = client.count(target).count
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=legacy_collection, limit=256, offset=offset,
                    with_payload=True, with_vectors=True,
                )
                if points:
                    # Upsert is keyed by point id, so re-running this command
                    # replaces rather than duplicates. `copied` therefore
                    # counts points moved, not points that were new.
                    client.upsert(collection_name=target, points=points)
                    copied += len(points)
                if offset is None:
                    break
            total = client.count(target).count

    if not moved and not copied:
        print("Nothing to migrate - no pre-tenancy state found.")
        return
    print(f"[ok] migrated to tenant '{slug}': {copied} vector(s)"
          + (f", files: {', '.join(moved)}" if moved else ""))
    if copied and already:
        print(f"     Note: the tenant already held {already} vector(s), so this looks like a "
              f"re-run.\n     Points are replaced by id, never duplicated - {total} in the "
              f"collection now.")
    if copied:
        print(f"     The old collection '{settings.qdrant_collection}' was copied, not deleted.\n"
              f"     Verify with `python cli.py docs --tenant {slug}`, then drop it yourself.")


def main() -> None:
    # --tenant is accepted both before and after the subcommand, because
    # `cli.py ingest x.txt --tenant acme` is what anyone actually types. The
    # subcommand copy defaults to SUPPRESS rather than None: an unset argparse
    # default on a subparser *overwrites* the value the top-level parser
    # already stored, which would silently send the work to the wrong tenant.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tenant", default=argparse.SUPPRESS,
                        help="tenant to operate on (default: DEFAULT_TENANT setting)")

    parser = argparse.ArgumentParser(description="Lexis Enterprise — grounded legal RAG",
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest one or more documents", parents=[common])
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("ask", help="ask a question over the ingested documents", parents=[common])
    p.add_argument("question")

    sub.add_parser("docs", help="list ingested documents", parents=[common])

    p = sub.add_parser("delete", help="delete a document", parents=[common])
    p.add_argument("document")

    sub.add_parser("warm", help="preload models + LLM (do this once after boot)")
    sub.add_parser("tenants", help="list tenants and whether they have keys")
    sub.add_parser("migrate", help="adopt a corpus ingested before tenancy existed",
                   parents=[common])

    keys = sub.add_parser("keys", help="API key management").add_subparsers(
        dest="keys_command", required=True)
    p = keys.add_parser("add", help="issue a key", parents=[common])
    p.add_argument("--label", required=True, help="who or what this key is for")
    p.add_argument("--role", default="analyst", choices=("analyst", "admin"))
    p = keys.add_parser("list", help="list keys (never shows the secrets)")
    p.add_argument("--all", action="store_true", help="include revoked keys")
    p = keys.add_parser("revoke", help="revoke a key by id")
    p.add_argument("key_id")

    audit_cmd = sub.add_parser("audit", help="read and verify the audit log").add_subparsers(
        dest="audit_command", required=True)
    p = audit_cmd.add_parser("tail", help="recent events", parents=[common])
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--action", default=None, help="filter by action (ask, ingest, delete, auth…)")
    p.add_argument("--all-tenants", action="store_true")
    audit_cmd.add_parser("verify", help="recompute the hash chain")

    args = parser.parse_args()

    from lexis import tenancy
    from lexis.config import settings

    try:
        tenant = tenancy.set_current(getattr(args, "tenant", None) or settings.default_tenant)
    except tenancy.InvalidTenant as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.command == "ingest":
        cmd_ingest(args.paths, tenant)
    elif args.command == "ask":
        cmd_ask(args.question, tenant)
    elif args.command == "docs":
        cmd_docs(tenant)
    elif args.command == "delete":
        cmd_delete(args.document, tenant)
    elif args.command == "warm":
        cmd_warm()
    elif args.command == "tenants":
        cmd_tenants()
    elif args.command == "migrate":
        cmd_migrate(tenant)
    elif args.command == "keys":
        if args.keys_command == "add":
            cmd_keys_add(args.label, tenant, args.role)
        elif args.keys_command == "list":
            cmd_keys_list(args.all)
        elif args.keys_command == "revoke":
            cmd_keys_revoke(args.key_id)
    elif args.command == "audit":
        if args.audit_command == "tail":
            cmd_audit_tail(None if args.all_tenants else tenant, args.limit, args.action)
        elif args.audit_command == "verify":
            cmd_audit_verify()


if __name__ == "__main__":
    main()
