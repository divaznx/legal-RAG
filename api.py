"""Lexis Enterprise HTTP API.

Run:  uvicorn api:app --reload --port 8000

Every endpoint except /health requires an API key
(`Authorization: Bearer lxs_...` or `X-API-Key: lxs_...`). The key names the
tenant; the tenant is never read from a header, parameter, or body, so a
caller cannot reach another client's corpus by editing a request. Ingestion
and deletion additionally require the `admin` role.

Every answer, ingestion, deletion, and rejected request is written to the
append-only audit log before the response is returned.

Latency: models + LLM are warmed in a background thread at startup;
POST /ask/stream streams the answer as Server-Sent Events (event: delta
per token batch, then event: result with the verified metadata).

Note: the embedded Qdrant store is single-process — do not run the API and
the Streamlit UI at the same time against the same QDRANT_PATH (use a Qdrant
server via QDRANT_URL if you need both).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lexis import audit, auth, engine, ingest, llm, tenancy
from lexis.config import settings
from lexis.parsing import SUPPORTED_EXTENSIONS
from lexis.vector_store import StoreUnavailable


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auth_enabled:
        minted = auth.bootstrap()
        if minted is not None:
            principal, secret = minted
            # Shown once, never stored in plaintext. Losing it is recoverable
            # (`python cli.py keys add --role admin`); leaking it is not.
            print(
                "\n" + "=" * 72
                + f"\n  No API keys existed, so a bootstrap admin key was created for\n"
                  f"  tenant '{principal.tenant}'. This is the only time it is shown:\n\n"
                  f"      {secret}\n\n"
                  f"  Revoke it once real keys are issued: "
                  f"python cli.py keys revoke {principal.key_id}\n"
                + "=" * 72 + "\n",
                file=sys.stderr,
            )
    else:
        print(
            "[lexis] WARNING: AUTH_ENABLED=false - every caller who can reach this "
            "port has admin access to the default tenant's documents. Development only.",
            file=sys.stderr,
        )
    threading.Thread(target=engine.warmup, daemon=True).start()
    yield


app = FastAPI(title="Lexis Enterprise", version="0.3.0", lifespan=lifespan)


@app.exception_handler(StoreUnavailable)
async def _store_unavailable(request: Request, exc: StoreUnavailable):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(exc)}, status_code=503)


# ---------------------------------------------------------------- auth
# Key management is deliberately CLI-only. A network endpoint that mints
# credentials is a much larger blast radius than one that answers questions,
# and nothing about this product needs keys issued remotely.


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


async def _authenticate(request: Request) -> auth.Principal:
    """Resolve the caller and bind their tenant for the rest of the request.

    This must stay `async`: FastAPI runs sync dependencies in a worker
    thread, and a ContextVar set inside a worker thread does not propagate
    back to the request context — the tenant binding would silently be lost
    and every caller would land on the default tenant.
    """
    if not settings.auth_enabled:
        tenancy.set_current(settings.default_tenant)
        return auth.Principal(
            key_id="-", label="anonymous (auth disabled)",
            tenant=settings.default_tenant, role="admin",
        )

    presented = _presented_key(request)
    principal = auth.verify(presented)
    if principal is None:
        audit.record(
            "auth", outcome="denied", tenant=settings.default_tenant,
            client=_client_ip(request),
            detail={"path": request.url.path, "key_presented": bool(presented)},
        )
        raise HTTPException(
            401,
            "Missing or invalid API key. Send `Authorization: Bearer lxs_...` "
            "or `X-API-Key: lxs_...`.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    tenancy.set_current(principal.tenant)
    return principal


def _requires(role: str) -> Callable:
    async def dependency(
        request: Request,
        principal: Annotated[auth.Principal, Depends(_authenticate)],
    ) -> auth.Principal:
        if not principal.can(role):
            audit.record(
                "authz", outcome="denied", principal=principal,
                client=_client_ip(request),
                detail={"path": request.url.path, "required_role": role},
            )
            raise HTTPException(
                403, f"This operation requires the '{role}' role; "
                     f"your key has '{principal.role}'.",
            )
        return principal

    return dependency


Analyst = Annotated[auth.Principal, Depends(_requires("analyst"))]
Admin = Annotated[auth.Principal, Depends(_requires("admin"))]


class AskRequest(BaseModel):
    question: str
    # Prior user questions in this conversation, oldest first. Optional: used
    # only by document resolution to keep the "active agreement" across
    # follow-ups; it never reaches the LLM.
    history: list[str] | None = None


def _serialize(result: engine.AskResult) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "refused": result.refused,
        "confidence": result.confidence,
        "limitations": result.limitations,
        "cached": result.cached,
        "needs_clarification": result.needs_clarification,
        # Full trace of the legal intelligence layer: resolved document and
        # why, intent, entities, concept expansion, sub-queries. Exposed so a
        # reviewer can audit the scope of an answer without re-running it.
        "legal": result.legal,
        "timings_ms": result.timings_ms,
        "citation_verification": {
            "passed": result.citations.passed,
            "total": result.citations.total,
            "verified": result.citations.verified,
            "fabricated": result.citations.fabricated,
        },
        "retrieved_chunks": [asdict(c) for c in result.chunks],
    }


@app.get("/health")
def health() -> dict:
    """Unauthenticated on purpose — load balancers and container health
    checks need it, and it reveals nothing about the corpus."""
    return {"status": "ok", "llm_available": llm.llm_available()}


@app.get("/me")
def whoami(principal: Analyst) -> dict:
    """Who this key is and what it can do — the first call to make when a
    client reports a 403."""
    return principal.as_dict()


@app.get("/documents")
def list_documents(principal: Analyst) -> dict:
    return ingest.load_manifest()


@app.post("/documents")
async def upload_document(request: Request, file: UploadFile, principal: Admin) -> dict:
    name = Path(file.filename or "upload").name
    if not name.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(415, f"Unsupported file type; supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            f"File is {len(payload) // (1024 * 1024)} MB; the limit is "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB.",
        )
    if not payload:
        raise HTTPException(422, "Uploaded file is empty.")

    # Per-tenant upload directory: the originals are the un-redacted source
    # documents, so two clients' files must not share a folder even though
    # only the redacted chunks ever reach the vector store.
    uploads = tenancy.data_path() / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / name
    dest.write_bytes(payload)
    try:
        # Off the event loop: parsing, redaction, and embedding a large PDF
        # is seconds of blocking CPU, and this handler has to be `async` for
        # `await file.read()`. Left inline it stalls every concurrent
        # request, including /health, for the duration of an ingest.
        # run_in_threadpool copies the context, so the tenant binding
        # established by the auth dependency travels with it.
        report = await run_in_threadpool(ingest.ingest_file, dest)
    except ValueError as exc:
        audit.record(
            "ingest", outcome="error", principal=principal, client=_client_ip(request),
            detail={"document": name, "error": str(exc)},
        )
        raise HTTPException(422, str(exc)) from exc

    audit.record(
        "ingest", principal=principal, client=_client_ip(request),
        documents=[report.document],
        detail={
            "version": report.version,
            "pages": report.pages,
            "chunks": report.chunks,
            "bytes": len(payload),
            "redactions": report.redactions,
            "low_ocr_pages": report.low_ocr_pages,
            # An injection finding recorded at ingest is the record that the
            # uploader was warned, which matters if an answer is later
            # disputed.
            "injection_flagged": bool(report.injection.get("flagged")),
        },
    )
    return asdict(report)


@app.delete("/documents/{document}")
def delete_document(request: Request, document: str, principal: Admin) -> dict:
    deleted = ingest.delete_document(document)
    audit.record(
        "delete", outcome="ok" if deleted else "error", principal=principal,
        client=_client_ip(request), documents=[document],
        detail={} if deleted else {"error": "not found"},
    )
    if not deleted:
        raise HTTPException(404, f"No such document: {document}")
    return {"deleted": document}


@app.get("/audit")
def read_audit(principal: Admin, limit: int = 50, action: str | None = None) -> dict:
    """This tenant's audit trail.

    Scoped to the caller's own tenant with no override: an admin key is an
    admin *of one client*, and the log is the one place where seeing another
    tenant's rows would disclose their questions verbatim.
    """
    limit = max(1, min(limit, 500))
    ok, checked, first_bad = audit.verify_chain()
    return {
        "tenant": principal.tenant,
        "chain": {"intact": ok, "rows_checked": checked, "first_bad_seq": first_bad,
                  "head": audit.head()},
        "events": audit.query(tenant=principal.tenant, limit=limit, action=action),
    }


def _require_llm() -> None:
    if not llm.llm_available():
        raise HTTPException(503, "LLM endpoint unreachable (is Ollama running?)")


@app.post("/ask")
def ask(http_request: Request, request: AskRequest, principal: Analyst) -> dict:
    _require_llm()
    result = engine.ask(request.question, request.history)
    audit.answer_event(result, principal=principal, client=_client_ip(http_request))
    return _serialize(result)


@app.post("/ask/stream")
def ask_stream(http_request: Request, request: AskRequest, principal: Analyst) -> StreamingResponse:
    _require_llm()
    # Captured here because the generator below runs after this handler has
    # returned, and each of its steps executes in a fresh copy of the
    # context — see tenancy.scoped, which re-binds the tenant around every
    # step rather than relying on a binding surviving between them.
    tenant, client_ip = principal.tenant, _client_ip(http_request)

    def events():
        for kind, payload in engine.ask_stream(request.question, request.history):
            if kind == "delta":
                yield f"event: delta\ndata: {json.dumps(payload)}\n\n"
            else:
                audit.answer_event(payload, principal=principal, client=client_ip)
                yield f"event: result\ndata: {json.dumps(_serialize(payload))}\n\n"

    return StreamingResponse(tenancy.scoped(tenant, events()), media_type="text/event-stream")


# --------------------------------------------------------------------------
# OpenAI-compatible surface, so chat frontends (Open WebUI, LibreChat, ...)
# can use Lexis as a "model": point them at http://localhost:8000/v1 with any
# API key. Each chat turn runs the full RAG pipeline on the latest user
# message; prior user turns are passed to document resolution only (active
# agreement memory) and never reach the LLM. The verified metadata is
# appended to the answer as a footer.
# --------------------------------------------------------------------------

OPENAI_MODEL_ID = "lexis"


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = OPENAI_MODEL_ID
    stream: bool = False


def _message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):  # multimodal part list
        return " ".join(
            p.get("text", "") for p in message.content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _message_text(message).strip()
    return ""


def _prior_user_messages(messages: list[ChatMessage]) -> list[str]:
    """User turns before the latest one — the resolution layer's conversation
    context. Capped to the most recent few: the active agreement is whatever
    was discussed recently, not in a 50-turn-old question."""
    prior = [_message_text(m).strip() for m in messages if m.role == "user"]
    return [q for q in prior[:-1] if q][-6:]


def _verification_footer(result: engine.AskResult) -> str:
    cv = result.citations
    verdict = "VERIFIED" if cv.passed else "UNVERIFIED"
    parts = [f"Confidence: {result.confidence}", f"Citations: {cv.verified}/{cv.total} {verdict}"]
    if result.cached:
        parts.append("cached")
    footer = f"\n\n---\n`{' · '.join(parts)}`"
    for limitation in result.limitations:
        footer += f"\n- {limitation}"
    return footer


def _chunk_payload(completion_id: str, created: int, *, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": OPENAI_MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _is_meta_task(question: str) -> bool:
    # Open WebUI generates titles/tags/follow-ups by sending "### Task:"
    # prompts to the same model; those must not go through the RAG pipeline.
    return question.lstrip().startswith("### Task:")


def _passthrough(request: ChatCompletionRequest, completion_id: str, created: int):
    """Answer a frontend meta task (title/tag generation) with the raw LLM."""
    messages = [{"role": m.role, "content": _message_text(m)} for m in request.messages]
    text = (
        llm.client()
        .chat.completions.create(
            model=settings.ollama_model,
            temperature=0.0,
            max_tokens=200,
            messages=messages,
        )
        .choices[0]
        .message.content
        or ""
    )
    if not request.stream:
        return _completion_response(completion_id, created, text)

    def events():
        yield _chunk_payload(completion_id, created, delta={"role": "assistant", "content": text})
        yield _chunk_payload(completion_id, created, delta={}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _completion_response(completion_id: str, created: int, text: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": OPENAI_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
def openai_models(principal: Analyst) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": OPENAI_MODEL_ID, "object": "model", "created": 0, "owned_by": "lexis"}
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(
    http_request: Request, request: ChatCompletionRequest, principal: Analyst
):
    # Chat frontends already send `Authorization: Bearer <key>`, so the
    # per-tenant key goes in the connection settings that were previously
    # filled with a dummy value — Open WebUI needs no changes beyond that.
    _require_llm()
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    tenant, client_ip = principal.tenant, _client_ip(http_request)

    question = _last_user_message(request.messages)
    if not question:
        raise HTTPException(422, "No user message found in `messages`.")
    if _is_meta_task(question):
        # Title/tag generation carries no evidence and is not an answer about
        # the corpus; logging it would bury the real questions.
        return _passthrough(request, completion_id, created)

    history = _prior_user_messages(request.messages)

    if not request.stream:
        result = engine.ask(question, history)
        audit.answer_event(result, principal=principal, client=client_ip, action="ask:openai")
        return _completion_response(completion_id, created, result.answer + _verification_footer(result))

    def events():
        yield _chunk_payload(completion_id, created, delta={"role": "assistant", "content": ""})
        for kind, payload in engine.ask_stream(question, history):
            if kind == "delta":
                yield _chunk_payload(completion_id, created, delta={"content": payload})
            else:
                audit.answer_event(
                    payload, principal=principal, client=client_ip, action="ask:openai"
                )
                yield _chunk_payload(
                    completion_id, created, delta={"content": _verification_footer(payload)}
                )
        yield _chunk_payload(completion_id, created, delta={}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(tenancy.scoped(tenant, events()), media_type="text/event-stream")
