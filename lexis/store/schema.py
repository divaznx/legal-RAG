"""The legal object model — DDL for the structured store.

Why this exists: the retrieval stack answers questions, and a question is not
a product. An obligation register, a renewal calendar, a risk view, "every
contract where we carry uncapped indemnity" — none of these are queries you
can put to a vector store. They are reads over structured, pre-extracted
facts. So documents are decomposed once at ingest into rows here, and
retrieval becomes one consumer of the model rather than the whole system.

Design rules, all of which are load-bearing in this domain:

1. PROVENANCE ON EVERY ROW. Each extracted fact carries document, clause, and
   character span. A figure a lawyer cannot trace back to text on a page is
   worthless — worse than absent, because it looks authoritative.
2. EXTRACTOR VERSION ON EVERY DOCUMENT. Extraction logic changes weekly. With
   a version stamp you re-run only what is stale; without one, every
   improvement forces a choice between a stale corpus and re-processing
   everything.
3. CONFIDENCE AND REVIEW STATE. Extraction is probabilistic. An unreviewed
   low-confidence obligation and an attorney-confirmed one must never render
   identically.
4. NOTHING IS DELETED ON RE-EXTRACTION except rows from the same extractor
   for the same document, so a re-run is idempotent.

SQLite is the engine (the product must install with zero infrastructure to
keep the air-gapped deployment story), but the DDL stays ANSI-compatible so
the move to Postgres for multi-tenant hosting is a driver swap, not a
rewrite. `tenant_id` is present from day one for the same reason: retrofitting
tenancy onto a live legal corpus is not something anyone should attempt.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    matter_id         TEXT,
    filename          TEXT NOT NULL,
    doc_type          TEXT,
    doc_type_label    TEXT,
    version           TEXT,
    family            TEXT,
    title             TEXT,
    governing_law     TEXT,
    parties_json      TEXT NOT NULL DEFAULT '[]',
    organizations_json TEXT NOT NULL DEFAULT '[]',
    is_amendment      INTEGER NOT NULL DEFAULT 0,
    supersedes_version TEXT,
    effective_date    TEXT,
    page_count        INTEGER NOT NULL DEFAULT 0,
    clause_count      INTEGER NOT NULL DEFAULT 0,
    injection_flagged INTEGER NOT NULL DEFAULT 0,
    extractor_version INTEGER NOT NULL DEFAULT 0,
    ingested_at       TEXT NOT NULL,
    extracted_at      TEXT,
    UNIQUE (tenant_id, filename)
);

CREATE TABLE IF NOT EXISTS clauses (
    id             TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'default',
    number         TEXT,
    section        TEXT,
    heading        TEXT,
    parent_section TEXT,
    page           INTEGER NOT NULL DEFAULT 1,
    ordinal        INTEGER NOT NULL DEFAULT 0,
    text           TEXT NOT NULL,
    concepts_json  TEXT NOT NULL DEFAULT '[]',
    is_definition  INTEGER NOT NULL DEFAULT 0,
    is_suspect     INTEGER NOT NULL DEFAULT 0
);

-- Every extracted fact below repeats (document_id, clause_id, span_start,
-- span_end). Denormalised deliberately: every dashboard read filters by
-- document, and a join to reach it on each row is wasted work.

CREATE TABLE IF NOT EXISTS obligations (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id     TEXT NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    obligor       TEXT,
    obligee       TEXT,
    modality      TEXT NOT NULL,          -- obligation | right | prohibition
    action        TEXT NOT NULL,
    condition     TEXT,
    deadline_raw  TEXT,
    deadline_days INTEGER,
    penalty_hint  TEXT,
    concepts_json TEXT NOT NULL DEFAULT '[]',
    span_start    INTEGER NOT NULL DEFAULT 0,
    span_end      INTEGER NOT NULL DEFAULT 0,
    confidence    REAL NOT NULL DEFAULT 0.5,
    status        TEXT NOT NULL DEFAULT 'unreviewed',
    extractor_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS key_dates (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id     TEXT NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    kind          TEXT NOT NULL,          -- payment | notice | renewal | cure | term | expiry | response
    -- Most legal dates are RULES, not dates: "90 days before the end of the
    -- then-current term". Storing a resolved date would be wrong the moment
    -- the anchor event moves, so the rule is stored and resolved on read.
    rule_type     TEXT NOT NULL,          -- relative | absolute | duration
    days          INTEGER,
    unit          TEXT,                   -- day | month | year | hour
    direction     TEXT,                   -- after | before
    anchor        TEXT,                   -- invoice | notice | termination | effective_date | term_end | awareness
    business_days INTEGER NOT NULL DEFAULT 0,
    absolute_date TEXT,
    computed_date TEXT,
    raw           TEXT NOT NULL,
    span_start    INTEGER NOT NULL DEFAULT 0,
    span_end      INTEGER NOT NULL DEFAULT 0,
    confidence    REAL NOT NULL DEFAULT 0.5,
    extractor_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS money_terms (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id     TEXT NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    kind          TEXT NOT NULL,          -- fee | cap | interest | penalty | credit
    amount        REAL,
    currency      TEXT,
    multiplier    REAL,                   -- "2x the fees", "150%"
    basis         TEXT,                   -- what the multiplier applies to
    period        TEXT,                   -- month | year | one_off
    raw           TEXT NOT NULL,
    span_start    INTEGER NOT NULL DEFAULT 0,
    span_end      INTEGER NOT NULL DEFAULT 0,
    confidence    REAL NOT NULL DEFAULT 0.5,
    extractor_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS findings (
    id             TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id      TEXT REFERENCES clauses(id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'default',
    playbook_id    TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    finding_type   TEXT NOT NULL,         -- missing_clause | deviation | prohibited_language | threshold_breach
    severity       TEXT NOT NULL,         -- blocker | high | medium | note
    category       TEXT,
    title          TEXT NOT NULL,
    detail         TEXT NOT NULL,
    rationale      TEXT,
    suggested      TEXT,
    status         TEXT NOT NULL DEFAULT 'open',   -- open | accepted | waived
    reviewed_by    TEXT,
    reviewed_at    TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_runs (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    playbook_id  TEXT NOT NULL,
    playbook_version TEXT,
    position     TEXT,
    rules_run    INTEGER NOT NULL DEFAULT 0,
    findings     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_clauses_doc       ON clauses(document_id);
CREATE INDEX IF NOT EXISTS ix_obligations_doc   ON obligations(document_id);
CREATE INDEX IF NOT EXISTS ix_obligations_party ON obligations(tenant_id, obligor);
CREATE INDEX IF NOT EXISTS ix_dates_doc         ON key_dates(document_id);
CREATE INDEX IF NOT EXISTS ix_dates_kind        ON key_dates(tenant_id, kind);
CREATE INDEX IF NOT EXISTS ix_money_doc         ON money_terms(document_id);
CREATE INDEX IF NOT EXISTS ix_findings_doc      ON findings(document_id);
CREATE INDEX IF NOT EXISTS ix_findings_severity ON findings(tenant_id, severity, status);
"""
