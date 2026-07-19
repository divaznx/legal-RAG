# Lexis Enterprise — Grounded Legal RAG

A legal Retrieval-Augmented Generation system that implements the "Lexis
Enterprise" contract: answers come **only** from retrieved evidence, every
paragraph is cited, citations are **mechanically verified** after
generation, PII is redacted to placeholders **before** it ever reaches the
vector store or the LLM, and confidence is computed by the system — not
self-reported by the model.

> Not a legal advisor. Answers describe what the retrieved documents state.

## Stack (fully local, open source)

| Layer      | Technology                                              |
|------------|---------------------------------------------------------|
| Vector DB  | Qdrant (embedded, on-disk — no server needed)           |
| Embeddings | fastembed / ONNX (`BAAI/bge-small-en-v1.5`, no torch)   |
| LLM        | Ollama via the OpenAI SDK (any OpenAI-compatible URL)   |
| API        | FastAPI                                                 |
| UI         | Streamlit                                               |
| CLI        | `cli.py`                                                |

## Latency engineering (the production legal-RAG playbook)

The retrieval/serving pipeline mirrors what production legal AI products
(Harvey, CoCounsel, Lexis+ AI) and the RAG-at-scale literature converge on:

1. **Hybrid retrieval** — BM25 sparse + dense vectors as two server-side
   Qdrant prefetches fused with Reciprocal Rank Fusion. Legal text is full
   of exact identifiers ("Clause 9.1", case cites) where keyword search
   beats embeddings; hybrid beats either alone.
2. **Cross-encoder reranking** — retrieve `CANDIDATE_K=20` wide, rerank
   with `ms-marco-MiniLM-L-6` (ONNX, ~20ms) down to `FINAL_K=4`. Fewer,
   better chunks = better grounding **and** a smaller prompt, so LLM
   prefill (the dominant local-inference cost) drops too.
3. **Token streaming** — answers stream in the CLI, the API (SSE
   `/ask/stream`), and the UI; time-to-first-token replaces
   time-to-full-answer as the felt latency (~75% perceived reduction).
4. **Semantic answer cache** — query-embedding similarity cache
   (`CACHE_SIMILARITY=0.95`); repeated or paraphrased questions return in
   milliseconds. Keyed to a corpus+model fingerprint, so any ingest/delete
   invalidates it — a hit can never serve stale evidence. Only verified,
   non-refusal answers are cached.
5. **Warm everything** — embedder, sparse encoder, reranker, and the vector
   store are process-lifetime singletons; `python cli.py warm` (and API/UI
   startup) preloads them and pins the Ollama model in memory
   (`LLM_KEEP_ALIVE=30m`). The system prompt is byte-identical across
   requests so Ollama reuses its KV-cache prefix.
6. **Bounded generation** — `LLM_MAX_TOKENS=700` caps the decode phase.
7. **Per-stage timing** — every answer reports
   `embed_query / cache_lookup / hybrid_retrieve / rerank / generate` ms,
   so regressions are visible instead of vibes.

Measured on this machine (CPU-only `llama3:8b`): retrieval + rerank ≈ 1s
cold-process (~100ms in a warm API/UI process); generation 129s → 49s via
prompt-verbosity cuts + KV-prefix caching; semantic cache hits ≈ 0.4s
(~30ms warm). The remaining floor is CPU token decode — the next levers
are a smaller model (`ollama pull llama3.2:3b`, set `OLLAMA_MODEL`), a
GPU, or any hosted OpenAI-compatible endpoint via `OLLAMA_BASE_URL`.

### Score calibration: who gets to gate refusals

Three scores exist per chunk and only one is calibrated:

| Score | Range behavior | Used for |
|---|---|---|
| Dense cosine | answerable ≈ 0.69–0.84, junk ≈ 0.38–0.40 (measured) | **refusal gate** (`MIN_DENSE_SCORE=0.5`) + confidence |
| RRF fused | rank-based, always ~0.5–1.0 for someone | candidate selection only |
| Cross-encoder | erratic absolutes (right chunk 0.14, sourdough junk 0.32 — measured on two models) | **ordering only**, never gating |

Gating refusals on reranker scores caused false refusals of legitimate
questions ("Compare the liability caps…") and was removed after
measurement. In-domain-but-unanswerable questions (e.g. GDPR against an
MSA corpus) pass the dense gate deliberately — the grounded LLM refusal
plus the citation verifier are the correct guardrail for those.

### Fact-aware retrieval

Key-value fact blocks ("Client: …", "Client ID: …", "Effective Date: …")
are chunked **atomically** (section `Preamble` when no clause governs
them) and their keys are indexed as `fact_keys` payload metadata. At query
time, a fact chunk whose key appears in the question is deterministically
boosted above generic prose after reranking. This exists because redaction
replaces fact *values* with placeholders, and cross-encoders score
placeholder-only blocks near zero against fluent prose (measured 0.0035 vs
0.98 for "Who is the client?") — the ranking signal must come from indexed
structure, not text similarity.

## How the system prompt became architecture

| Prompt rule                        | Implementation |
|------------------------------------|----------------|
| Redacted placeholders (`[EMAIL]`…) | `lexis/redaction.py` replaces PII at ingest — raw values never reach Qdrant or the LLM |
| Citation metadata (doc/page/clause/version) | `lexis/chunking.py` attaches document, page, nearest section/clause heading, and filename-derived version to every chunk |
| OCR confidence                     | `lexis/parsing.py` flags near-empty PDF pages as `possible-scan` (confidence 0.3); the flag propagates into answers' Limitations |
| Zero hallucination / refusal       | `lexis/engine.py` refuses before calling the LLM when nothing retrieves above `MIN_SCORE`; the prompt mandates the exact refusal string |
| Citation accuracy                  | `lexis/llm.py::verify_citations` parses every `(Source: …)` tag and checks it against the retrieved (document, page) set — fabricated citations are reported |
| Confidence (High/Medium/Low)       | `lexis/engine.py::_confidence` computes it from retrieval score, OCR quality, and citation verification — never from model certainty |
| Conflicts                          | prompt instructs presenting each conflicting source; the two sample MSAs (v1.0 vs v2.1) exercise this |
| Output format                      | Answer / Evidence Used / Limitations / Confidence, enforced by the prompt; system verdict appended by the engine |

## Prerequisites

1. Python 3.11+
2. [Ollama](https://ollama.com) running locally with a model pulled:
   ```bash
   ollama pull llama3:8b
   ```
   (Any OpenAI-compatible endpoint works — set `OLLAMA_BASE_URL` / `OLLAMA_MODEL`.)

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell   (source .venv/Scripts/activate for Git Bash)
pip install -r requirements.txt
copy .env.example .env            # adjust as needed
```

## Use

**CLI** (fastest way to try it):

```bash
python cli.py ingest sample_docs/MSA_Acme_v1.0.txt sample_docs/MSA_Acme_v2.1.txt
python cli.py ask "What is the termination notice period, and do the versions agree?"
python cli.py docs
```

**Streamlit UI**:

```bash
streamlit run ui/app.py
```

**API**:

```bash
uvicorn api:app --reload --port 8000
# POST /documents (multipart upload) · GET /documents · DELETE /documents/{name} · POST /ask
```

⚠️ Embedded Qdrant is single-process: run the CLI, the API, **or** the UI at
a time — or set `QDRANT_URL` to a real Qdrant server to share it.

## Answer anatomy

Every answer carries two verdicts:

1. **The model's own sections** (mandated by the system prompt): Answer with
   inline `(Source: doc | Page n | Clause | vX)` citations, Evidence Used,
   Limitations, Confidence.
2. **The system verdict** (computed): `VERIFIED`/`UNVERIFIED` citation
   check, fabricated-citation list, OCR warnings, and the authoritative
   High/Medium/Low confidence.

## Limitations (honest ones)

- Redaction is regex-based: emails, phones, SSNs, card/account numbers, and
  keyword-anchored IDs/passports/names are caught; free-standing person
  names are **not** (would need an NER model — e.g. Presidio — as a drop-in
  upgrade inside `redaction.py`).
- OCR "confidence" is a text-extraction heuristic, not real OCR. Scanned
  PDFs without a text layer are flagged, not OCR'd — plug Tesseract into
  `parsing.py` to actually read them.
- Citation verification checks that (document, page) pairs exist in the
  retrieved set — it catches fabricated references and uncited answers, but
  cannot prove the cited text semantically supports the sentence.
- `.docx`/`.txt` have no page boundaries; they index as page 1.

## Layout

```
lexis/
  config.py        .env-driven settings (pydantic-settings)
  redaction.py     PII -> [PLACEHOLDER] at ingest
  parsing.py       PDF/DOCX/TXT -> pages + OCR heuristic + version detection
  chunking.py      section-aware chunks carrying full citation metadata
  embeddings.py    fastembed (ONNX) wrapper
  vector_store.py  Qdrant wrapper (embedded or server)
  ingest.py        parse -> redact -> chunk -> embed -> upsert (+ manifest)
  prompts.py       the Lexis Enterprise system prompt + context/citation contract
  llm.py           Ollama call + mechanical citation verification
  engine.py        retrieve -> generate -> verify -> computed confidence
cli.py             ingest / ask / docs / delete
api.py             FastAPI endpoints
ui/app.py          Streamlit chat with verification badge + evidence panel
sample_docs/       two MSA versions with PII and deliberate clause conflicts
```
More updates on the way.
