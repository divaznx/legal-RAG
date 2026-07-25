"""Append-only audit log.

What a client needs from a legal AI system, months later and possibly in
front of someone hostile, is not "the system is accurate" — it is *who asked
what, on which date, what the system answered, and which clauses it was
looking at when it did*. That record has to exist before the question is
asked, and it has to be hard to quietly improve afterwards.

Two mechanisms, because they fail differently:

1. **SQLite triggers** reject UPDATE and DELETE on the events table. This
   makes the log append-only for anything reaching it through SQL —
   including this application, a support script, and an operator poking at
   the file with the `sqlite3` CLI.

2. **A SHA-256 hash chain** across rows. Each row stores the previous row's
   hash and its own, so an altered or removed row breaks every hash after
   it. This is what covers the case the triggers cannot: someone with
   filesystem access rebuilding the database from scratch.

Being precise about the limit, because "tamper-proof" would be a lie: the
chain detects modification and mid-log deletion. It cannot by itself detect
*truncation* of the newest rows, since a shortened chain is internally
consistent. `head()` returns the current tip hash for exactly this reason —
copy it somewhere the deployment cannot reach (a ticket, a monitoring
system, a client's own inbox) and truncation becomes detectable too.

Writes take `BEGIN IMMEDIATE` so the read-the-tip/append-a-row sequence is
serialized across processes, not just threads — the CLI and the API write to
the same log.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator

from . import tenancy
from .config import settings

if TYPE_CHECKING:  # avoids an import cycle at runtime
    from .engine import AskResult

GENESIS = "0" * 64

_lock = threading.Lock()
_initialized: set[str] = set()

# Order matters: it is part of the hash preimage. Appending a new field is
# safe (old rows verify against the columns they were written with); removing
# or reordering one invalidates every existing chain.
_HASHED_FIELDS = (
    "ts", "tenant", "principal_id", "principal_label", "action", "outcome",
    "question", "answer", "confidence", "cached", "citations_verified",
    "citations_total", "citations_passed", "documents", "evidence",
    "limitations", "latency_ms", "client", "detail",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    tenant             TEXT NOT NULL,
    principal_id       TEXT,
    principal_label    TEXT,
    action             TEXT NOT NULL,
    outcome            TEXT NOT NULL,
    question           TEXT,
    answer             TEXT,
    confidence         TEXT,
    cached             INTEGER,
    citations_verified INTEGER,
    citations_total    INTEGER,
    citations_passed   INTEGER,
    documents          TEXT,
    evidence           TEXT,
    limitations        TEXT,
    latency_ms         REAL,
    client             TEXT,
    detail             TEXT,
    prev_hash          TEXT NOT NULL,
    row_hash           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_tenant_seq ON events (tenant, seq);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def _row_hash(prev_hash: str, row: dict) -> str:
    preimage = prev_hash + "|" + json.dumps(
        [row.get(f) for f in _HASHED_FIELDS], separators=(",", ":"), sort_keys=True,
        default=str,
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = settings.audit_db_path
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        if str(path) not in _initialized:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            _initialized.add(str(path))
        yield conn
    finally:
        conn.close()


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def record(
    action: str,
    *,
    principal: Any = None,
    tenant: str | None = None,
    outcome: str = "ok",
    client: str | None = None,
    question: str | None = None,
    answer: str | None = None,
    confidence: str | None = None,
    cached: bool | None = None,
    citations: tuple[int, int, bool] | None = None,
    documents: list[str] | None = None,
    evidence: list[dict] | None = None,
    limitations: list[str] | None = None,
    latency_ms: float | None = None,
    detail: dict | None = None,
) -> int | None:
    """Append one event. Returns its sequence number, or None if disabled.

    Never raises into the caller's path: an answer that was delivered but not
    logged is a gap in the record, while an exception here would turn a
    logging fault into a failed request. The failure is surfaced as a warning
    instead, and `verify_chain` will still show the log as internally
    consistent — so the gap has to be caught by noticing the warning, which
    is why it goes to stderr rather than being swallowed.
    """
    if not settings.audit_enabled:
        return None

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "tenant": tenant or (getattr(principal, "tenant", None)) or tenancy.current(),
        "principal_id": getattr(principal, "key_id", None),
        "principal_label": getattr(principal, "label", None) or ("cli" if principal is None else None),
        "action": action,
        "outcome": outcome,
        "question": question,
        "answer": answer if settings.audit_store_answers else None,
        "confidence": confidence,
        "cached": None if cached is None else int(cached),
        "citations_verified": citations[0] if citations else None,
        "citations_total": citations[1] if citations else None,
        "citations_passed": int(citations[2]) if citations else None,
        "documents": _json(documents),
        "evidence": _json(evidence),
        "limitations": _json(limitations),
        "latency_ms": latency_ms,
        "client": client,
        "detail": _json(detail),
    }

    try:
        with _lock, _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                tip = conn.execute(
                    "SELECT row_hash FROM events ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev = tip["row_hash"] if tip else GENESIS
                row["prev_hash"] = prev
                row["row_hash"] = _row_hash(prev, row)
                columns = ", ".join(row)
                placeholders = ", ".join(f":{c}" for c in row)
                cursor = conn.execute(
                    f"INSERT INTO events ({columns}) VALUES ({placeholders})", row
                )
                conn.execute("COMMIT")
                return int(cursor.lastrowid)
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except Exception as exc:  # pragma: no cover - depends on filesystem state
        print(f"[audit] FAILED to record {action!r}: {exc}", file=sys.stderr)
        return None


def answer_event(
    result: "AskResult",
    *,
    principal: Any = None,
    client: str | None = None,
    action: str = "ask",
) -> int | None:
    """Record a completed answer, including the evidence it stood on.

    The evidence set is stored as citations (document/page/section/clause),
    not as chunk text: it is enough to reproduce what the model was shown
    without turning the audit log into a second, less protected copy of the
    corpus.
    """
    evidence = [
        {
            "document": c.document,
            "page": c.page,
            "section": c.section,
            "clause": c.clause_number,
            "version": c.version,
            "reason": c.retrieval_reason,
        }
        for c in result.chunks
    ]
    # The scope the resolution layer *chose*, which is not the same as the
    # documents that happened to produce evidence — a question resolved to
    # v2.1 that returned nothing is a different event from one resolved to
    # both versions, and only the former is a retrieval failure. Falls back
    # to the evidence when the legal layer is disabled or short-circuited.
    legal = result.legal or {}
    documents = legal.get("documents") or sorted({c.document for c in result.chunks})

    return record(
        action,
        principal=principal,
        client=client,
        outcome="refused" if result.refused else "ok",
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        cached=result.cached,
        citations=(result.citations.verified, result.citations.total, result.citations.passed),
        documents=documents,
        evidence=evidence,
        limitations=result.limitations,
        latency_ms=result.timings_ms.get("total"),
        detail={
            "needs_clarification": result.needs_clarification,
            "intent": legal.get("intent"),
            # Why this agreement and not the other one - the question a
            # disputed answer actually turns on.
            "document_reason": legal.get("document_reason"),
            "superseded": legal.get("superseded"),
        },
    )


def query(
    tenant: str | None = None,
    limit: int = 50,
    action: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Most recent events first."""
    clauses, params = [], []
    if tenant:
        clauses.append("tenant = ?")
        params.append(tenancy.normalize(tenant))
    if action:
        clauses.append("action = ?")
        params.append(action)
    if since:
        clauses.append("ts >= ?")
        params.append(since)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM events {where} ORDER BY seq DESC LIMIT ?", (*params, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def head() -> str | None:
    """Hash of the newest event — the anchor to record off-box.

    Without an external copy of this value, a log can be truncated at the
    tail and still verify; with one, truncation is a mismatch.
    """
    with _connect() as conn:
        row = conn.execute("SELECT row_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    return row["row_hash"] if row else None


def verify_chain() -> tuple[bool, int, int | None]:
    """Recompute every hash. Returns (ok, rows_checked, first_bad_seq)."""
    prev = GENESIS
    checked = 0
    with _connect() as conn:
        for row in conn.execute("SELECT * FROM events ORDER BY seq ASC"):
            data = dict(row)
            if data["prev_hash"] != prev or _row_hash(prev, data) != data["row_hash"]:
                return False, checked, int(data["seq"])
            prev = data["row_hash"]
            checked += 1
    return True, checked, None
