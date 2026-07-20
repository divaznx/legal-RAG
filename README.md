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

## Production upgrade pack (15 features, all measurable)

Incremental upgrades layered onto the stable pipeline — nothing was
redesigned, every default preserves prior behavior or is config-gated
(`lexis/config.py`):

| # | Feature | Where | Notes |
|---|---------|-------|-------|
| 1 | Parent-child retrieval | `chunking.parent_of`, `packing.attach_parents` | "Clause 4.2" chunks attach one level of "Clause 4" context; skipped if the parent is already retrieved |
| 2 | Context validation & packing | `lexis/packing.py` | dedupe, overlap-merge, repeated-heading strip, document order — LLM sees clean context; verification uses originals |
| 3 | Query classification | `lexis/query.py::classify` | 11 rule-based classes; entity/clause lookups weight BM25 1.5x, summarization widens k, comparison routes per-document |
| 4 | Query rewriting | `query.py::rewrite` | legal synonym expansion, internal only, capped at `QUERY_REWRITE_MAX_VARIANTS` |
| 5 | Query decomposition | `query.py::decompose` | "Who is the client and what are the payment terms?" retrieves each part independently, evidence merged via RRF |
| 6 | Definition-aware retrieval | `chunking.extract_defined_terms` + engine boost | chunks *defining* a term used in the question boost to 0.98 — definitions precede ordinary language |
| 7 | Metadata enrichment | `lexis/metadata.py` | document_type, agreement_type, jurisdiction, effective_date, confidentiality_level, language stamped per chunk |
| 8 | Retrieval explainability | `RetrievedChunk.diagnostics()` | BM25/dense/RRF/rerank scores, before/after ranks, match reasons; `cli.py ask --debug` or API `"debug": true` — never in normal answers |
| 9 | Evaluation harness | `lexis/evaluation.py` + `eval/golden.json` | `python cli.py eval [--generate]` — Recall@K, Precision@K, MRR, nDCG, gate accuracy, citation accuracy, hallucination rate, latency |
| 10 | Answer verification | `llm.verify_citations` + engine | fabricated clause numbers caught (lenient when unverifiable); one corrective regeneration on failure; `STRICT_VERIFICATION=true` refuses instead |
| 11 | Version-aware retrieval | `retrieval.py` | latest version of a document family gets +0.05 post-rerank; comparison queries retrieve every version per-document |
| 12 | Access control ready | payload `tenant/client/matter/permissions` | filter applied pre-retrieval only when `TENANT` is set — single-tenant deployments untouched |
| 13 | Agentic retrieval | `retrieval.py::_agentic_attempts` | on gate miss: definitions search, then keyword broadening — max `AGENTIC_MAX_RETRIES`, then refuse |
| 14 | Clause relationship graph | `lexis/graph.py` -> `data/clause_graph.json` | "Clause 6 references Clause 5" edges built at ingest; retrieval expansion opt-in via `GRAPH_ENABLED` |
| 15 | Table preservation | `chunking.is_table_block` | pipe/columnar tables chunked atomically, never split or overlap-bled, marked `TABLE` in the prompt |
| 16 | Clause-aware retrieval | `query.clause_references_in` + `retrieval` injection + engine boost | "Clause 7.2 / Section 8 / § 4" extracted from the query; the actual clause chunk is injected into candidates if hybrid search missed it and boosted to the top tier (0.995) — a chunk merely *referencing* the clause never outranks it. An explicit "v1.0" in the question pins the version and suspends latest-version preference |
| 17 | Ambiguity detection | `engine._clause_ambiguity` | a clause matching multiple unrelated agreements returns a clarification request (pre-LLM, ~1.3s) instead of guessing; multiple versions of one agreement are all represented so differences are compared, never silently dropped |
| 18 | Citation-failure taxonomy | `CitationReport.failure_reasons` | named reasons (fabricated_citation, clause_mismatch, missing_citations) logged to debug diagnostics and fed into the regeneration corrective note |

| 19 | Intent: document overview | `query.QueryClass.OVERVIEW` + `retrieval.overview_chunks` | "explain/summarize the <agreement>" retrieves stratified evidence (parties, definitions, financial, termination, closing, …) across the target document instead of top-k similarity; answers render under structured headings; measured Recall@8 = 1.0 at 299ms retrieval on the 9-chunk MSA. Trade-off: more chunks + `OVERVIEW_MAX_TOKENS=1100` = longer generation |
| 20 | Clause exact-only context | `CLAUSE_EXACT_ONLY` | when exact clause matches exist, only they (plus parent context) reach the LLM — unrelated clauses excluded from packing; document-scoped, so "Clause 5 of the MSA" never pulls Section 5.7 of an unrelated agreement |
| 21 | Answer label hygiene | `engine._scrub_internal_labels` + prompt rule | "[Chunk N]" and bare "Chunk N" retrieval internals stripped from final answers; evidence referenced only by document/version/clause + (Source: …) citations |
| 22 | Pipeline-versioned cache | `cache.corpus_fingerprint` includes `lexis.__version__` | answers generated by an older pipeline are never served after an upgrade — bump `__version__` on retrieval/prompt changes |
| 23 | Cross-reference resolution | `chunking.extract_references` → `graph.py` → `packing.expand_references` | "incorporated by reference / subject to / pursuant to / see …" targets (Clause, Section, Article, Item 3.03, Exhibit A, Schedule 2.1, Appendix, Attachment) become graph edges at ingest; retrieval follows them breadth-first up to `GRAPH_MAX_DEPTH=2` chained hops, cycle-safe via a visited set, capped at `GRAPH_MAX_EXPANSION=3` extra chunks; each added chunk's `matched_on` records its path ("graph-reference:Item 3.03->Item 1.01") in debug mode. Keyword-aware matching: "Item 3.03" never cross-matches "Clause 3" — clause/section/article are synonyms, Item is a distinct numbering system |
| 24 | Consequence-aware retrieval | `QueryClass.CONSEQUENCE` + probes + injection + boost | "What if I break Clause 3?" retrieves the clause PLUS the provisions defining consequences: three fixed probe queries (termination/remedies, liability/indemnification, governing-law/enforcement) RRF-fused with the question; named documents get their provision sections injected deterministically (fusion over a mixed corpus can crowd them out); a 0.9 ranking floor pins provisions — scoped to named documents so a 95-page merger agreement can't flood the tier; `final_k` widened to `CONSEQUENCE_FINAL_K=6`; clause exact-only narrowing is bypassed (the clause alone is exactly the wrong context); prompt contract: if the evidence doesn't define consequences, say so — never speculate. Measured: golden `consequence-breach` recall 0.50→1.00, MRR 0.17→1.00, nDCG 0.22→1.00 |
| 25 | Legal concept graph | `query.LEGAL_CONCEPT_GRAPH` + `concept_probes` | related-provision probes retrieved alongside the question (confidentiality→survival/injunctive relief, payment→late interest/suspension, termination→return of property/survival, IP→ownership/assignment); bounded at 2 probes (`CONCEPT_EXPANSION`) |
| 26 | Obligation duty retrieval | `query.obligation_subject/obligation_pattern` + retrieval injection | "the Client's obligations" = sentences where the Client (or both parties) shall/must — duty chunks fetched deterministically from named documents and floor-pinned at 0.9; identity boosts suppressed for this class. Measured: golden `obligations-client` recall 0.50→1.00, MRR 0.17→1.00 |
| 27 | Jurisdiction preference | `engine._boost_jurisdiction_chunks` | a question naming a jurisdiction ("under Delaware law") bumps chunks whose jurisdiction metadata matches (`JURISDICTION_BOOST=0.05`) — legal systems are never silently mixed |
| 28 | Named-document preference | engine post-rerank bump | chunks from documents the question names get +0.05 — the operative agreement outranks documents that merely describe it (an 8-K summarizing the MSA cannot beat the MSA itself); also lifted `comparison-versions` nDCG 0.88→1.00 |

### Measured: clause-aware retrieval (features 16-17)

Golden-set benchmark before/after (K=4, other 10 cases unchanged at 1.0):

| Case | Metric | Before | After |
|------|--------|--------|-------|
| clause-lookup-versioned ("Clause 2 of MSA v1.0") | MRR | 0.50 | **1.00** |
| clause-lookup-versioned | nDCG@4 | 0.63 | **1.00** |
| overall (11 answerable cases) | MRR | 0.9545 | **1.0** |
| overall | nDCG@4 | 0.9368 | **0.985** |

The before-failure was real: asking about v1.0 ranked the v2.1 clause first
because latest-version preference overrode the explicit version request.

### Measured: clause-boundary chunking

The eval harness (feature 9) immediately paid for itself: chunks used to
aggregate 3–4 clauses under the first clause's section label, and overlap
tails bled across clause boundaries. Flushing chunks at every heading
(`CHUNK_AT_HEADINGS=true`, the new default) moved the golden-set metrics:

| Metric | before | after |
|---|---|---|
| Recall@4 | 0.39 | **1.00** |
| MRR | 0.59 | **1.00** |
| nDCG@4 | 0.44 | **0.97** |
| Retrieval success | 0.67 | **1.00** |
| Gate accuracy | 1.00 | 1.00 |

Re-ingest after changing chunking settings — chunk boundaries are decided
at ingest time.

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
# POST /ask with {"question": "...", "debug": true} adds retrieval diagnostics
```

**Benchmarks** (golden dataset in `eval/golden.json`):

```bash
python cli.py eval               # retrieval metrics — no LLM needed
python cli.py eval --generate    # + citation accuracy / faithfulness / hallucination rate
python cli.py ask "..." --debug  # per-chunk scores, ranks, and match reasons
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
  retrieved set and that cited clause numbers match retrieved sections — it
  catches fabricated references, fabricated clause numbers, and uncited
  answers, but cannot prove the cited text semantically supports the
  sentence.
- `.docx`/`.txt` have no page boundaries; they index as page 1.
- Query classification/rewriting/decomposition are rule-based (regex +
  synonym tables): deterministic, free, and unit-tested, but they won't
  catch phrasings outside their patterns — extend the tables in
  `lexis/query.py` as the corpus grows.
- Metadata enrichment (jurisdiction, effective date, agreement type) is
  heuristic regex over the redacted text; fields it can't find stay None
  rather than being guessed.
- The dense refusal gate was calibrated on 900-char chunks; clause-level
  chunks shift the score distribution, and terse keyword-only junk queries
  ("sourdough hydration schedule") can pass it. The grounded LLM refusal +
  citation verifier remain the backstop, as designed.

## Layout

```
lexis/
  config.py        .env-driven settings (pydantic-settings) — every feature has a knob
  redaction.py     PII -> [PLACEHOLDER] at ingest
  parsing.py       PDF/DOCX/TXT -> pages + OCR heuristic + version detection
  metadata.py      document-level enrichment (agreement type, jurisdiction, ...)
  chunking.py      clause-boundary chunks: hierarchy, tables, defined terms, references
  graph.py         clause relationship graph (ingest-built, retrieval opt-in)
  embeddings.py    fastembed (ONNX) wrapper
  vector_store.py  Qdrant wrapper: weighted hybrid legs, filters, diagnostics
  query.py         classification / rewriting / decomposition (rule-based)
  retrieval.py     multi-query RRF fusion, comparison routing, versions, agentic ladder
  packing.py       context dedupe/merge/order + parent-context attachment
  ingest.py        parse -> redact -> enrich -> chunk -> embed -> upsert (+ manifest, graph)
  prompts.py       the Lexis Enterprise system prompt + context/citation contract
  llm.py           Ollama call + mechanical citation & clause verification
  engine.py        retrieve -> rerank+boost -> pack -> generate -> verify (-> regenerate)
  evaluation.py    golden-dataset benchmark: Recall@K, MRR, nDCG, hallucination rate
cli.py             ingest / ask [--debug] / docs / delete / warm / eval
api.py             FastAPI endpoints (debug flag supported)
ui/app.py          Streamlit chat with verification badge + evidence panel
sample_docs/       two MSA versions with PII and deliberate clause conflicts
eval/golden.json   golden dataset for the benchmark suite
```
