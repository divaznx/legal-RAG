"""Reads and writes over the legal object model.

The write side is idempotent per (document, extractor version): re-extracting
a document replaces its rows rather than appending, so improving an extractor
and re-running is safe. The read side is the product surface — obligation
register, renewal calendar, findings by severity — and every one of these is
a plain indexed query, which is the entire argument for having a structured
store at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .db import connect
from .models import Finding, KeyDate, MoneyTerm, Obligation, new_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- documents ------------------------------------------------------------

def upsert_document(
    filename: str,
    profile: dict,
    page_count: int,
    clause_count: int,
    injection_flagged: bool,
    tenant_id: str = "default",
    matter_id: str | None = None,
) -> str:
    conn = connect()
    row = conn.execute(
        "SELECT id FROM documents WHERE tenant_id = ? AND filename = ?",
        (tenant_id, filename),
    ).fetchone()
    document_id = row["id"] if row else new_id()

    conn.execute(
        """
        INSERT INTO documents (id, tenant_id, matter_id, filename, doc_type,
            doc_type_label, version, family, title, governing_law, parties_json,
            organizations_json, is_amendment, supersedes_version, page_count,
            clause_count, injection_flagged, ingested_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            doc_type=excluded.doc_type, doc_type_label=excluded.doc_type_label,
            version=excluded.version, family=excluded.family, title=excluded.title,
            governing_law=excluded.governing_law, parties_json=excluded.parties_json,
            organizations_json=excluded.organizations_json,
            is_amendment=excluded.is_amendment,
            supersedes_version=excluded.supersedes_version,
            page_count=excluded.page_count, clause_count=excluded.clause_count,
            injection_flagged=excluded.injection_flagged,
            ingested_at=excluded.ingested_at
        """,
        (
            document_id, tenant_id, matter_id, filename,
            profile.get("doc_type"), profile.get("doc_type_label"),
            profile.get("version"), profile.get("family"), profile.get("title"),
            profile.get("governing_law"),
            json.dumps(profile.get("parties") or []),
            json.dumps(profile.get("organizations") or []),
            int(bool(profile.get("is_amendment"))),
            profile.get("supersedes_version"),
            page_count, clause_count, int(injection_flagged), _now(),
        ),
    )
    conn.commit()
    return document_id


def replace_clauses(document_id: str, chunks, tenant_id: str = "default") -> dict[str, str]:
    """Write the document's clauses; return chunk-id -> clause-id."""
    conn = connect()
    conn.execute("DELETE FROM clauses WHERE document_id = ?", (document_id,))
    mapping: dict[str, str] = {}
    rows = []
    for ordinal, chunk in enumerate(chunks):
        clause_id = new_id()
        mapping[chunk.id] = clause_id
        rows.append((
            clause_id, document_id, tenant_id, chunk.clause_number, chunk.section,
            chunk.heading, chunk.parent_section, chunk.page, ordinal, chunk.text,
            json.dumps(chunk.concepts_all or chunk.concepts), int(chunk.is_definition),
            int(chunk.is_suspect),
        ))
    conn.executemany(
        """INSERT INTO clauses (id, document_id, tenant_id, number, section, heading,
               parent_section, page, ordinal, text, concepts_json, is_definition, is_suspect)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return mapping


def mark_extracted(document_id: str, extractor_version: int) -> None:
    conn = connect()
    conn.execute(
        "UPDATE documents SET extractor_version = ?, extracted_at = ? WHERE id = ?",
        (extractor_version, _now(), document_id),
    )
    conn.commit()


def stale_documents(extractor_version: int, tenant_id: str = "default") -> list[str]:
    """Documents extracted by an older extractor.

    The whole point of versioning: improving an extractor should trigger a
    targeted re-run, not a choice between a stale corpus and reprocessing
    everything.
    """
    conn = connect()
    rows = conn.execute(
        "SELECT filename FROM documents WHERE tenant_id = ? AND extractor_version < ?",
        (tenant_id, extractor_version),
    ).fetchall()
    return [r["filename"] for r in rows]


def delete_document(filename: str, tenant_id: str = "default") -> None:
    conn = connect()
    conn.execute("DELETE FROM documents WHERE tenant_id = ? AND filename = ?",
                 (tenant_id, filename))
    conn.commit()


def document_id_for(filename: str, tenant_id: str = "default") -> str | None:
    conn = connect()
    row = conn.execute(
        "SELECT id FROM documents WHERE tenant_id = ? AND filename = ?",
        (tenant_id, filename),
    ).fetchone()
    return row["id"] if row else None


# --- extracted facts ------------------------------------------------------

def replace_obligations(document_id: str, items: list[Obligation],
                        extractor_version: int, tenant_id: str = "default") -> None:
    conn = connect()
    conn.execute("DELETE FROM obligations WHERE document_id = ?", (document_id,))
    conn.executemany(
        """INSERT INTO obligations (id, document_id, clause_id, tenant_id, obligor,
               obligee, modality, action, condition, deadline_raw, deadline_days,
               penalty_hint, concepts_json, span_start, span_end, confidence, status,
               extractor_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(o.id, o.document_id, o.clause_id, tenant_id, o.obligor, o.obligee,
          o.modality, o.action, o.condition, o.deadline_raw, o.deadline_days,
          o.penalty_hint, json.dumps(o.concepts), o.span_start, o.span_end,
          o.confidence, o.status, extractor_version) for o in items],
    )
    conn.commit()


def replace_key_dates(document_id: str, items: list[KeyDate],
                      extractor_version: int, tenant_id: str = "default") -> None:
    conn = connect()
    conn.execute("DELETE FROM key_dates WHERE document_id = ?", (document_id,))
    conn.executemany(
        """INSERT INTO key_dates (id, document_id, clause_id, tenant_id, kind,
               rule_type, days, unit, direction, anchor, business_days, absolute_date,
               computed_date, raw, span_start, span_end, confidence, extractor_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(d.id, d.document_id, d.clause_id, tenant_id, d.kind, d.rule_type, d.days,
          d.unit, d.direction, d.anchor, int(d.business_days), d.absolute_date,
          d.computed_date, d.raw, d.span_start, d.span_end, d.confidence,
          extractor_version) for d in items],
    )
    conn.commit()


def replace_money_terms(document_id: str, items: list[MoneyTerm],
                        extractor_version: int, tenant_id: str = "default") -> None:
    conn = connect()
    conn.execute("DELETE FROM money_terms WHERE document_id = ?", (document_id,))
    conn.executemany(
        """INSERT INTO money_terms (id, document_id, clause_id, tenant_id, kind,
               amount, currency, multiplier, basis, period, raw, span_start, span_end,
               confidence, extractor_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(m.id, m.document_id, m.clause_id, tenant_id, m.kind, m.amount, m.currency,
          m.multiplier, m.basis, m.period, m.raw, m.span_start, m.span_end,
          m.confidence, extractor_version) for m in items],
    )
    conn.commit()


# --- product reads --------------------------------------------------------

def obligations(document: str | None = None, obligor: str | None = None,
                modality: str | None = None, tenant_id: str = "default") -> list[dict]:
    sql = """SELECT o.*, c.section, c.number, d.filename, d.version
             FROM obligations o
             JOIN clauses c   ON c.id = o.clause_id
             JOIN documents d ON d.id = o.document_id
             WHERE o.tenant_id = ?"""
    params: list = [tenant_id]
    if document:
        sql += " AND d.filename = ?"
        params.append(document)
    if obligor:
        sql += " AND lower(o.obligor) = lower(?)"
        params.append(obligor)
    if modality:
        sql += " AND o.modality = ?"
        params.append(modality)
    sql += " ORDER BY d.filename, c.ordinal"
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def key_dates(document: str | None = None, kind: str | None = None,
              tenant_id: str = "default") -> list[dict]:
    sql = """SELECT k.*, c.section, c.number, d.filename, d.version
             FROM key_dates k
             JOIN clauses c   ON c.id = k.clause_id
             JOIN documents d ON d.id = k.document_id
             WHERE k.tenant_id = ?"""
    params: list = [tenant_id]
    if document:
        sql += " AND d.filename = ?"
        params.append(document)
    if kind:
        sql += " AND k.kind = ?"
        params.append(kind)
    sql += " ORDER BY d.filename, c.ordinal"
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def money_terms(document: str | None = None, kind: str | None = None,
                tenant_id: str = "default") -> list[dict]:
    sql = """SELECT m.*, c.section, c.number, d.filename
             FROM money_terms m
             JOIN clauses c   ON c.id = m.clause_id
             JOIN documents d ON d.id = m.document_id
             WHERE m.tenant_id = ?"""
    params: list = [tenant_id]
    if document:
        sql += " AND d.filename = ?"
        params.append(document)
    if kind:
        sql += " AND m.kind = ?"
        params.append(kind)
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def clauses_for(document_id: str) -> list[dict]:
    return [dict(r) for r in connect().execute(
        "SELECT * FROM clauses WHERE document_id = ? ORDER BY ordinal", (document_id,)
    ).fetchall()]


# --- findings -------------------------------------------------------------

def record_review(document_id: str, playbook_id: str, playbook_version: str,
                  position: str, rules_run: int, items: list[Finding],
                  tenant_id: str = "default") -> str:
    conn = connect()
    conn.execute(
        "DELETE FROM findings WHERE document_id = ? AND playbook_id = ?",
        (document_id, playbook_id),
    )
    conn.executemany(
        """INSERT INTO findings (id, document_id, clause_id, tenant_id, playbook_id,
               rule_id, finding_type, severity, category, title, detail, rationale,
               suggested, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(f.id, f.document_id, f.clause_id, tenant_id, f.playbook_id, f.rule_id,
          f.finding_type, f.severity, f.category, f.title, f.detail, f.rationale,
          f.suggested, f.status, _now()) for f in items],
    )
    run_id = new_id()
    conn.execute(
        """INSERT INTO review_runs (id, document_id, tenant_id, playbook_id,
               playbook_version, position, rules_run, findings, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, document_id, tenant_id, playbook_id, playbook_version, position,
         rules_run, len(items), _now()),
    )
    conn.commit()
    return run_id


def findings(document: str | None = None, severity: str | None = None,
             status: str = "open", tenant_id: str = "default") -> list[dict]:
    sql = """SELECT f.*, c.section, d.filename
             FROM findings f
             LEFT JOIN clauses c ON c.id = f.clause_id
             JOIN documents d    ON d.id = f.document_id
             WHERE f.tenant_id = ? AND f.status = ?"""
    params: list = [tenant_id, status]
    if document:
        sql += " AND d.filename = ?"
        params.append(document)
    if severity:
        sql += " AND f.severity = ?"
        params.append(severity)
    sql += """ ORDER BY CASE f.severity WHEN 'blocker' THEN 0 WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2 ELSE 3 END, d.filename"""
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def portfolio_summary(tenant_id: str = "default") -> dict:
    """The view a GC actually opens: exposure across every contract at once.

    This is the query that is impossible against a vector store, and the
    reason the structured model exists.
    """
    conn = connect()
    counts = conn.execute(
        """SELECT severity, COUNT(*) AS n FROM findings
           WHERE tenant_id = ? AND status = 'open' GROUP BY severity""",
        (tenant_id,),
    ).fetchall()
    return {
        "documents": conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()["n"],
        "clauses": conn.execute(
            "SELECT COUNT(*) AS n FROM clauses WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()["n"],
        "obligations": conn.execute(
            "SELECT COUNT(*) AS n FROM obligations WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()["n"],
        "key_dates": conn.execute(
            "SELECT COUNT(*) AS n FROM key_dates WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()["n"],
        "findings_by_severity": {r["severity"]: r["n"] for r in counts},
    }
