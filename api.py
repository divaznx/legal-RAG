"""Lexis Enterprise HTTP API.

Run:  uvicorn api:app --reload --port 8000

Latency: models + LLM are warmed in a background thread at startup;
POST /ask/stream streams the answer as Server-Sent Events (event: delta
per token batch, then event: result with the verified metadata).

Note: the embedded Qdrant store is single-process — do not run the API and
the Streamlit UI at the same time against the same QDRANT_PATH (use a Qdrant
server via QDRANT_URL if you need both).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lexis import engine, ingest, llm
from lexis.config import settings
from lexis.parsing import SUPPORTED_EXTENSIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=engine.warmup, daemon=True).start()
    yield


app = FastAPI(title="Lexis Enterprise", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


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
    return {"status": "ok", "llm_available": llm.llm_available()}


@app.get("/documents")
def list_documents() -> dict:
    return ingest.load_manifest()


@app.post("/documents")
async def upload_document(file: UploadFile) -> dict:
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

    uploads = settings.data_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / name
    dest.write_bytes(payload)
    try:
        report = ingest.ingest_file(dest)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return asdict(report)


@app.delete("/documents/{document}")
def delete_document(document: str) -> dict:
    if not ingest.delete_document(document):
        raise HTTPException(404, f"No such document: {document}")
    return {"deleted": document}


def _require_llm() -> None:
    if not llm.llm_available():
        raise HTTPException(503, "LLM endpoint unreachable (is Ollama running?)")


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    _require_llm()
    return _serialize(engine.ask(request.question))


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    _require_llm()

    def events():
        for kind, payload in engine.ask_stream(request.question):
            if kind == "delta":
                yield f"event: delta\ndata: {json.dumps(payload)}\n\n"
            else:
                yield f"event: result\ndata: {json.dumps(_serialize(payload))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# OpenAI-compatible surface, so chat frontends (Open WebUI, LibreChat, ...)
# can use Lexis as a "model": point them at http://localhost:8000/v1 with any
# API key. Each chat turn runs the full RAG pipeline on the latest user
# message (the engine is single-turn by design — prior turns are ignored),
# and the verified metadata is appended to the answer as a footer.
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
def openai_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": OPENAI_MODEL_ID, "object": "model", "created": 0, "owned_by": "lexis"}
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(request: ChatCompletionRequest):
    _require_llm()
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    question = _last_user_message(request.messages)
    if not question:
        raise HTTPException(422, "No user message found in `messages`.")
    if _is_meta_task(question):
        return _passthrough(request, completion_id, created)

    if not request.stream:
        result = engine.ask(question)
        return _completion_response(completion_id, created, result.answer + _verification_footer(result))

    def events():
        yield _chunk_payload(completion_id, created, delta={"role": "assistant", "content": ""})
        for kind, payload in engine.ask_stream(question):
            if kind == "delta":
                yield _chunk_payload(completion_id, created, delta={"content": payload})
            else:
                yield _chunk_payload(
                    completion_id, created, delta={"content": _verification_footer(payload)}
                )
        yield _chunk_payload(completion_id, created, delta={}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
