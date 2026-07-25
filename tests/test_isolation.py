"""Tenant isolation, API-key authentication, and audit-log integrity.

These are the tests that have to fail loudly, because every one of them
guards a promise made to a paying client rather than an answer-quality
property: that another client cannot see their contracts, that a revoked key
stops working, and that the record of who asked what cannot be quietly
edited afterwards.

No LLM and no network: answers are constructed directly as `AskResult`
values, so the assertions are about plumbing, not about generation.

Run:  python -m pytest tests/test_isolation.py -q
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis import audit, auth, cache, ingest, tenancy, vector_store
from lexis.config import settings
from lexis.engine import AskResult
from lexis.llm import CitationReport
from lexis.vector_store import RetrievedChunk

DOC_A = b"""Clause 1. Term
This Agreement runs for twenty-four (24) months from the Effective Date.

Clause 2. Termination
Either party may terminate on thirty (30) days written notice.
"""

DOC_B = b"""Clause 1. Confidentiality
The Receiving Party shall not disclose the Disclosing Party's Confidential
Information for a period of five (5) years.
"""


@pytest.fixture(scope="module", autouse=True)
def deployment(tmp_path_factory):
    """A throwaway deployment: own data dir, own embedded vector store."""
    root = tmp_path_factory.mktemp("lexis-deployment")
    saved = {k: getattr(settings, k) for k in
             ("data_dir", "qdrant_url", "qdrant_path", "auth_enabled", "cache_enabled")}

    settings.data_dir = str(root / "data")
    settings.qdrant_url = None          # .env may point at a server; use the embedded store
    settings.qdrant_path = str(root / "qdrant")
    settings.auth_enabled = True
    settings.cache_enabled = True

    # Module-level singletons captured settings at first use; reset them so
    # this deployment is genuinely fresh.
    vector_store._client = None
    vector_store._ensured.clear()
    auth._cache = None
    audit._initialized.clear()

    yield root

    for key, value in saved.items():
        setattr(settings, key, value)
    vector_store._client = None
    vector_store._ensured.clear()
    auth._cache = None
    audit._initialized.clear()


@pytest.fixture(autouse=True)
def reset_tenant_binding():
    """Leave every test on the default tenant.

    Without this, one test's binding decides where the next test's data
    lands, and the resulting failure points at the wrong test entirely.
    """
    yield
    tenancy._current.set(settings.default_tenant)


def _write(tmp_path: Path, name: str, body: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _answer(question: str, document: str, confidence: str = "High") -> AskResult:
    chunk = RetrievedChunk(
        score=0.9, dense_score=0.8, document=document, page=1, section="Clause 2",
        version="1.0", ocr_confidence=1.0, ocr_source="text",
        text="Either party may terminate on thirty (30) days written notice.",
        clause_number="2",
    )
    return AskResult(
        question=question,
        answer=f"## Answer\nThirty days. (Source: {document} | Page 1 | Clause 2 | v1.0)\n",
        chunks=[chunk],
        citations=CitationReport(total=1, verified=1),
        confidence=confidence,
        timings_ms={"total": 42.0},
    )


# --------------------------------------------------------------- tenancy


def test_tenant_ids_are_validated():
    """A tenant id becomes a directory name and a collection name, so an
    unvalidated one is a path traversal, not just a typo."""
    assert tenancy.normalize("Acme-Legal") == "acme-legal"  # canonicalised, not rejected
    for bad in ("../../etc", "a/b", "", "  ", "-leading", "x" * 64, "acme legal"):
        with pytest.raises(tenancy.InvalidTenant):
            tenancy.normalize(bad)


def test_tenant_binding_is_restored_after_the_block():
    with tenancy.using("alpha"):
        assert tenancy.current() == "alpha"
        with tenancy.using("beta"):
            assert tenancy.current() == "beta"
        assert tenancy.current() == "alpha"
    assert tenancy.current() == settings.default_tenant


def test_scoped_rebinds_the_tenant_on_every_step():
    """A streaming response is pulled through `iterate_in_threadpool`, which
    runs each `__next__` in a fresh copy of the context — so a binding made
    on one step is gone by the next. `scoped` must re-bind every time, not
    once at the top."""
    def produce():
        for _ in range(3):
            yield tenancy.current()

    seen = []
    for value in tenancy.scoped("firm-one", produce()):
        seen.append(value)
        # Stand in for the server discarding the per-step context.
        tenancy._current.set(settings.default_tenant)

    assert seen == ["firm-one"] * 3


def test_using_does_not_crash_when_it_exits_in_another_context():
    """ContextVar.reset() raises if the token came from a different Context.
    That is a no-op situation, not a failure, and it must not surface as a
    500 halfway through a response."""
    import contextvars

    manager = tenancy.using("firm-one")
    contextvars.copy_context().run(manager.__enter__)  # token minted in a child context
    manager.__exit__(None, None, None)                 # …reset attempted out here


def test_documents_are_isolated_by_tenant(tmp_path):
    with tenancy.using("firm-one"):
        ingest.ingest_file(_write(tmp_path, "MSA_Acme_v1.0.txt", DOC_A))
    with tenancy.using("firm-two"):
        ingest.ingest_file(_write(tmp_path, "NDA_Globex_v1.0.txt", DOC_B))

    with tenancy.using("firm-one"):
        assert list(ingest.load_manifest()) == ["MSA_Acme_v1.0.txt"]
    with tenancy.using("firm-two"):
        assert list(ingest.load_manifest()) == ["NDA_Globex_v1.0.txt"]

    # …and the vectors really are in separate collections, so isolation does
    # not depend on any query remembering to filter.
    client = vector_store.client()
    one, two = tenancy.collection("firm-one"), tenancy.collection("firm-two")
    assert one != two
    assert client.count(one).count > 0
    assert client.count(two).count > 0

    payloads = client.scroll(collection_name=one, limit=100, with_payload=True)[0]
    assert {p.payload["document"] for p in payloads} == {"MSA_Acme_v1.0.txt"}


def test_deleting_a_document_leaves_other_tenants_untouched(tmp_path):
    with tenancy.using("firm-three"):
        ingest.ingest_file(_write(tmp_path, "Shared_Name.txt", DOC_A))
    with tenancy.using("firm-four"):
        ingest.ingest_file(_write(tmp_path, "Shared_Name.txt", DOC_B))
        assert ingest.delete_document("Shared_Name.txt")
        assert ingest.load_manifest() == {}

    with tenancy.using("firm-three"):
        assert "Shared_Name.txt" in ingest.load_manifest()


def test_answer_cache_is_per_tenant():
    """Two tenants asking the same question of different corpora must not
    share an entry — a cache hit across the boundary would disclose the other
    client's answer verbatim."""
    embedding = [0.1] * 8
    payload = {"answer": "Thirty days.", "chunks": [], "citations": {},
               "confidence": "High", "limitations": [], "legal": {}}

    with tenancy.using("firm-one"):
        cache.store("notice period?", embedding, payload, ["MSA_Acme_v1.0.txt"])
        assert cache.lookup("notice period?", embedding, ["MSA_Acme_v1.0.txt"]) is not None
    with tenancy.using("firm-two"):
        assert cache.lookup("notice period?", embedding, ["MSA_Acme_v1.0.txt"]) is None


# ------------------------------------------------------------------ auth


def test_roles_are_ranked_not_compared_by_equality():
    analyst, _ = auth.issue("Reader", "firm-one", "analyst")
    admin, _ = auth.issue("Operator", "firm-one", "admin")
    assert analyst.can("analyst") and not analyst.can("admin")
    assert admin.can("analyst") and admin.can("admin")


def test_keys_are_not_stored_in_plaintext():
    principal, secret = auth.issue("Jane", "firm-one")
    stored = settings.api_keys_path.read_text(encoding="utf-8")
    assert secret not in stored
    assert auth.verify(secret) == principal
    assert all("hash" not in key for key in auth.list_keys())


def test_unknown_and_revoked_keys_are_rejected():
    principal, secret = auth.issue("Temp", "firm-one")
    assert auth.verify(secret) is not None
    assert auth.verify("lxs_not-a-real-key") is None
    assert auth.verify(None) is None
    assert auth.verify("") is None

    assert auth.revoke(principal.key_id)
    # Reloaded from disk on change, so revocation applies to the very next
    # request rather than the next restart.
    assert auth.verify(secret) is None
    assert auth.revoke(principal.key_id) is False


def test_a_key_carries_its_tenant():
    _, secret = auth.issue("Firm Two reviewer", "firm-two")
    principal = auth.verify(secret)
    assert principal.tenant == "firm-two"


# ----------------------------------------------------------------- audit


def test_audit_log_rejects_updates_and_deletes():
    seq = audit.record("ask", tenant="firm-one", question="q", answer="a")
    assert seq is not None

    conn = sqlite3.connect(settings.audit_db_path)
    try:
        for statement in ("UPDATE events SET answer = 'edited' WHERE seq = ?",
                          "DELETE FROM events WHERE seq = ?"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement, (seq,))
    finally:
        conn.close()


def test_audit_chain_detects_a_row_rewritten_behind_the_triggers():
    """The triggers stop SQL; the hash chain is what catches someone with
    filesystem access rebuilding the log."""
    first = audit.record("ask", tenant="firm-one", question="original question")
    audit.record("ask", tenant="firm-one", question="later question")
    assert audit.verify_chain()[0] is True

    conn = sqlite3.connect(settings.audit_db_path)
    try:
        conn.execute("DROP TRIGGER events_no_update")
        conn.execute("UPDATE events SET question = 'rewritten' WHERE seq = ?", (first,))
        conn.commit()
    finally:
        conn.close()

    ok, checked, first_bad = audit.verify_chain()
    assert ok is False
    assert first_bad == first
    assert checked == first - 1  # everything before the edit still verifies

    # Restore the trigger so later tests see an intact deployment.
    conn = sqlite3.connect(settings.audit_db_path)
    try:
        conn.execute("UPDATE events SET question = 'original question' WHERE seq = ?", (first,))
        conn.executescript(
            "CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events "
            "BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;"
        )
        conn.commit()
    finally:
        conn.close()
    assert audit.verify_chain()[0] is True


def test_audit_records_the_evidence_behind_an_answer():
    result = _answer("What is the notice period?", "MSA_Acme_v1.0.txt")
    seq = audit.answer_event(result, client="test")

    event = next(e for e in audit.query(tenant=settings.default_tenant, limit=20)
                 if e["seq"] == seq)
    assert event["question"] == "What is the notice period?"
    assert event["confidence"] == "High"
    assert event["citations_verified"] == 1 and event["citations_passed"] == 1
    assert "MSA_Acme_v1.0.txt" in event["documents"]
    # The clause the answer stood on is recoverable months later.
    assert '"clause": "2"' in event["evidence"]


def test_audit_queries_are_scoped_by_tenant():
    audit.record("ask", tenant="firm-one", question="one")
    audit.record("ask", tenant="firm-two", question="two")
    for name in ("firm-one", "firm-two"):
        rows = audit.query(tenant=name, limit=100)
        assert rows and all(r["tenant"] == name for r in rows)


def test_refusals_are_logged_as_refusals():
    from lexis.prompts import REFUSAL

    refused = AskResult(question="unrelated", answer=REFUSAL,
                        citations=CitationReport(refusal=True), confidence="Low")
    seq = audit.answer_event(refused, client="test")
    event = next(e for e in audit.query(limit=20) if e["seq"] == seq)
    assert event["outcome"] == "refused"


# ------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def api_client(deployment):
    from fastapi.testclient import TestClient

    import api

    with TestClient(api.app) as client:
        yield client


@pytest.fixture(scope="module")
def keys(deployment):
    _, admin_a = auth.issue("A admin", "client-a", "admin")
    _, analyst_a = auth.issue("A analyst", "client-a", "analyst")
    _, admin_b = auth.issue("B admin", "client-b", "admin")
    return {
        "admin_a": {"Authorization": f"Bearer {admin_a}"},
        "analyst_a": {"X-API-Key": analyst_a},
        "admin_b": {"Authorization": f"Bearer {admin_b}"},
    }


def test_health_is_open_but_everything_else_is_not(api_client):
    assert api_client.get("/health").status_code == 200
    for method, path in (("get", "/me"), ("get", "/documents"), ("get", "/v1/models"),
                         ("get", "/audit"), ("delete", "/documents/x.txt")):
        response = getattr(api_client, method)(path)
        assert response.status_code == 401, path
        assert "WWW-Authenticate" in response.headers


def test_a_garbage_key_is_rejected(api_client):
    assert api_client.get("/me", headers={"X-API-Key": "lxs_wrong"}).status_code == 401
    assert api_client.get("/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_the_key_decides_the_tenant(api_client, keys):
    assert api_client.get("/me", headers=keys["admin_a"]).json()["tenant"] == "client-a"
    assert api_client.get("/me", headers=keys["admin_b"]).json()["tenant"] == "client-b"


def test_ingestion_and_deletion_require_admin(api_client, keys):
    upload = {"file": ("Analyst_Upload.txt", DOC_A, "text/plain")}
    assert api_client.post("/documents", headers=keys["analyst_a"], files=upload).status_code == 403
    assert api_client.delete("/documents/x.txt", headers=keys["analyst_a"]).status_code == 403
    # …but reading is allowed for the same key.
    assert api_client.get("/documents", headers=keys["analyst_a"]).status_code == 200


def test_one_clients_upload_is_invisible_to_another(api_client, keys):
    upload = {"file": ("ClientA_MSA_v1.0.txt", DOC_A, "text/plain")}
    assert api_client.post("/documents", headers=keys["admin_a"], files=upload).status_code == 200

    assert "ClientA_MSA_v1.0.txt" in api_client.get("/documents", headers=keys["admin_a"]).json()
    assert api_client.get("/documents", headers=keys["admin_b"]).json() == {}
    # Client B cannot delete what it cannot see, even knowing the filename.
    assert api_client.delete("/documents/ClientA_MSA_v1.0.txt",
                             headers=keys["admin_b"]).status_code == 404
    assert "ClientA_MSA_v1.0.txt" in api_client.get("/documents", headers=keys["admin_a"]).json()


def test_audit_endpoint_shows_only_the_callers_tenant(api_client, keys):
    body = api_client.get("/audit", headers=keys["admin_a"]).json()
    assert body["tenant"] == "client-a"
    assert body["chain"]["intact"] is True
    assert all(event["tenant"] == "client-a" for event in body["events"])
    assert any(event["action"] == "ingest" for event in body["events"])

    other = api_client.get("/audit", headers=keys["admin_b"]).json()
    assert all(event["tenant"] == "client-b" for event in other["events"])


def test_rejected_requests_are_themselves_recorded(api_client, keys):
    api_client.get("/me", headers={"X-API-Key": "lxs_intruder"})
    denied = [e for e in audit.query(limit=200, action="auth") if e["outcome"] == "denied"]
    assert denied, "a rejected key must leave a trace"
    assert any("/me" in (e["detail"] or "") for e in denied)


def test_streaming_answers_stay_on_the_callers_tenant(api_client, keys, monkeypatch):
    """Exercises /ask/stream without an LLM. The generator is iterated after
    the handler returns, one step per threadpool hop, and every step — including
    the audit write at the end — has to still be on the caller's tenant."""
    observed: list[str] = []

    def fake_stream(question, history=None):
        for index in range(4):
            observed.append(tenancy.current())
            yield ("delta", f"chunk{index} ")
        observed.append(tenancy.current())
        yield ("result", _answer(question, "ClientA_MSA_v1.0.txt"))

    monkeypatch.setattr("api.engine.ask_stream", fake_stream)
    monkeypatch.setattr("api.llm.llm_available", lambda: True)

    with api_client.stream("POST", "/ask/stream", headers=keys["analyst_a"],
                           json={"question": "streamed question"}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert observed == ["client-a"] * 5, observed
    assert body.count("event: delta") == 4
    assert "event: result" in body

    # …and the answer written from inside the generator landed on client-a.
    rows = audit.query(tenant="client-a", limit=50)
    assert any(r["question"] == "streamed question" for r in rows)
    assert not any(r["question"] == "streamed question"
                   for r in audit.query(tenant="client-b", limit=50))


def test_revocation_takes_effect_without_a_restart(api_client):
    principal, secret = auth.issue("Contractor", "client-a", "analyst")
    headers = {"Authorization": f"Bearer {secret}"}
    assert api_client.get("/me", headers=headers).status_code == 200
    auth.revoke(principal.key_id)
    assert api_client.get("/me", headers=headers).status_code == 401


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
