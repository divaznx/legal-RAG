"""Lexis Enterprise — Streamlit UI.

Run:  streamlit run ui/app.py

Imports the engine directly (no API hop). Note the embedded Qdrant store is
single-process: stop the FastAPI server first, or point both at a Qdrant
server via QDRANT_URL.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from lexis import engine, ingest, llm

st.set_page_config(page_title="Lexis Enterprise", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Warming models (first run only)…")
def _warm() -> bool:
    engine.warmup()
    return True


_warm()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚖️ Lexis Enterprise")
    st.caption("Grounded legal retrieval — answers only from your documents, every citation verified.")

    if not llm.llm_available():
        st.error("LLM endpoint unreachable. Start Ollama (`ollama serve`) or set OLLAMA_BASE_URL.")

    st.subheader("Upload documents")
    st.caption("Files are indexed the moment you add them; removing a file "
               "here deletes its chunks from the vector store too.")

    # Names this session ingested via the uploader (and ones that failed, so
    # a failing file isn't retried on every rerun).
    st.session_state.setdefault("session_uploads", set())
    st.session_state.setdefault("session_failed", set())

    uploads = st.file_uploader(
        "PDF / DOCX / TXT / MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    current = {Path(u.name).name for u in uploads} if uploads else set()

    # Sync: uploader -> index. New files ingest immediately…
    for upload in uploads or []:
        name = Path(upload.name).name
        if name in st.session_state.session_uploads or name in st.session_state.session_failed:
            continue
        with st.spinner(f"Ingesting {name}…"):
            tmp = Path(tempfile.mkdtemp()) / name
            tmp.write_bytes(upload.getvalue())
            try:
                report = ingest.ingest_file(tmp)
            except ValueError as exc:
                st.session_state.session_failed.add(name)
                st.error(f"{name}: {exc}")
                continue
        st.session_state.session_uploads.add(name)
        redactions = ", ".join(f"{k}×{v}" for k, v in report.redactions.items()) or "none"
        st.success(f"{report.document}: {report.chunks} chunks, redactions: {redactions}")
        if report.low_ocr_pages:
            st.warning(f"Possible scanned pages (low OCR confidence): {report.low_ocr_pages}")
        if report.injection.get("flagged"):
            st.error(
                f"**Security warning — {name}**: "
                f"{len(report.injection['findings'])} passage(s) attempt to instruct the AI "
                f"system rather than state contractual terms "
                f"({', '.join(report.injection['categories'])}). The document was indexed and "
                f"this text is treated as content, never as instructions — but review it "
                f"before relying on answers from this document."
            )
            for finding in report.injection["findings"][:3]:
                st.caption(f"› {finding['excerpt']}")

    # …and files removed from the uploader are deleted from the vector store.
    for name in st.session_state.session_uploads - current:
        ingest.delete_document(name)
        st.session_state.session_uploads.discard(name)
        st.toast(f"Removed {name} from the index", icon="🗑")
    st.session_state.session_failed &= current

    st.subheader("Indexed documents")
    manifest = ingest.load_manifest()
    if not manifest:
        st.caption("None — the assistant can only refuse until a document is indexed.")
    for name, info in sorted(manifest.items()):
        col_name, col_del = st.columns([5, 1])
        col_name.markdown(f"**{name}** · v{info['version']} · {info['chunks']} chunks")
        if col_del.button("🗑", key=f"del-{name}", help=f"Delete {name}"):
            # keep the name in session_uploads: if its chip is still in the
            # uploader, forgetting it would auto-re-ingest on the next rerun
            ingest.delete_document(name)
            st.rerun()
    if manifest and st.button("Remove all indexed documents", use_container_width=True):
        for name in list(manifest):
            ingest.delete_document(name)
        st.rerun()

# ---------------------------------------------------------------- chat
st.header("Ask the retrieved record")

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("meta"):
            st.caption(entry["meta"])

question = st.chat_input("Ask a question about the ingested documents…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not manifest:
            st.warning("No documents ingested yet — upload one in the sidebar.")
            st.stop()
        if not llm.llm_available():
            st.error("LLM endpoint unreachable — start Ollama (`ollama serve`) and retry.")
            st.stop()

        holder: dict = {}
        # Prior user questions give document resolution its "active agreement"
        # memory; they never reach the LLM. The current question is already
        # appended to the history above, so exclude it.
        prior_questions = [
            e["content"] for e in st.session_state.history if e["role"] == "user"
        ][:-1][-6:]

        def _deltas():
            for kind, payload in engine.ask_stream(question, prior_questions):
                if kind == "delta":
                    yield payload
                else:
                    holder["result"] = payload

        st.write_stream(_deltas())
        result = holder["result"]

        badge = "✅ VERIFIED" if result.citations.passed else "⚠️ UNVERIFIED"
        cached = " · ⚡ semantic cache hit" if result.cached else ""
        stages = " · ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items())
        meta = (
            f"{badge} — {result.citations.verified}/{result.citations.total} citations matched "
            f"retrieved chunks · System confidence: **{result.confidence}**{cached}  \n"
            f"⏱ {stages}"
        )
        st.caption(meta)
        for limitation in result.limitations:
            st.warning(limitation)

        legal = result.legal or {}
        if legal:
            with st.expander("Legal reasoning (how this answer was scoped)"):
                st.markdown(
                    f"**Question type:** {legal.get('intent', '?').replace('_', ' ')} "
                    f"(confidence {legal.get('intent_confidence', 0)})"
                )
                st.markdown(
                    f"**Answering from:** {', '.join(legal.get('documents') or ['—'])}  \n"
                    f"{legal.get('document_reason', '')}"
                )
                if legal.get("superseded"):
                    st.markdown(f"**Excluded as superseded:** {', '.join(legal['superseded'])}")
                if legal.get("clause_targets"):
                    st.markdown(f"**Clause(s) referenced:** {', '.join(legal['clause_targets'])}")
                if legal.get("definition_targets"):
                    st.markdown(f"**Defined terms:** {', '.join(legal['definition_targets'])}")
                if legal.get("concepts"):
                    st.markdown(f"**Legal concepts:** {', '.join(legal['concepts'])}")
                if legal.get("expanded_concepts"):
                    st.markdown(
                        "**Expanded to (legal chain):** "
                        + ", ".join(c.replace("_", " ") for c in legal["expanded_concepts"])
                    )

        with st.expander(f"Retrieved evidence ({len(result.chunks)} chunks)"):
            for i, chunk in enumerate(result.chunks, start=1):
                rr = f" · rerank {chunk.rerank_score:.2f}" if chunk.rerank_score is not None else ""
                parent = f" · under {chunk.parent_section}" if chunk.parent_section else ""
                st.markdown(
                    f"**[Chunk {i}]** `{chunk.document}` · Page {chunk.page} · "
                    f"{chunk.section or '—'}{parent} · v{chunk.version} · "
                    f"OCR {chunk.ocr_source} ({chunk.ocr_confidence:.2f}) · fused {chunk.score:.3f}{rr}"
                )
                st.caption(f"Why retrieved: {chunk.retrieval_reason}")
                st.text(chunk.text[:1200])

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "meta": meta}
    )
