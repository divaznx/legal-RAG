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
    debug: bool = False  # include retrieval diagnostics (developer use only)


def _serialize(result: engine.AskResult) -> dict:
    payload = {
        "question": result.question,
        "answer": result.answer,
        "refused": result.refused,
        "confidence": result.confidence,
        "limitations": result.limitations,
        "cached": result.cached,
        "query_class": result.query_class,
        "regenerated": result.regenerated,
        "timings_ms": result.timings_ms,
        "citation_verification": {
            "passed": result.citations.passed,
            "total": result.citations.total,
            "verified": result.citations.verified,
            "fabricated": result.citations.fabricated,
            "clause_mismatches": result.citations.clause_mismatches,
        },
        "retrieved_chunks": [asdict(c) for c in result.chunks],
    }
    if result.debug is not None:
        payload["retrieval_debug"] = result.debug
    return payload


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
    uploads = settings.data_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / name
    dest.write_bytes(await file.read())
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
    return _serialize(engine.ask(request.question, debug=request.debug))


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    _require_llm()

    def events():
        for kind, payload in engine.ask_stream(request.question, debug=request.debug):
            if kind == "delta":
                yield f"event: delta\ndata: {json.dumps(payload)}\n\n"
            else:
                yield f"event: result\ndata: {json.dumps(_serialize(payload))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
