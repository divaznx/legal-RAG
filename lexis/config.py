from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    data_dir: str = "./data"
    qdrant_path: str = "./data/qdrant"
    qdrant_url: Optional[str] = None
    qdrant_collection: str = "lexis_chunks_v2"  # v2: named dense + sparse (hybrid) schema

    # Embeddings (fastembed / ONNX)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # Reranking — cross-encoder over the hybrid candidate set.
    # NOTE: reranker scores ORDER candidates only; they are uncalibrated and
    # must never gate refusals (measured: correct chunks 0.14, junk 0.32).
    use_reranker: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Refusal gate — dense cosine of the best candidate. Measured on this
    # corpus: answerable questions 0.69-0.84, off-domain junk 0.38-0.40.
    min_dense_score: float = 0.5

    # LLM — any OpenAI-compatible endpoint (Ollama by default)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3:8b"
    ollama_api_key: str = "ollama"
    llm_max_tokens: int = 700     # generation cap — direct latency lever
    llm_keep_alive: str = "30m"   # keep the model resident between requests (Ollama)

    # Chunking / retrieval
    chunk_size: int = 900
    chunk_overlap: int = 150
    # Start a new chunk at every clause/section heading so chunks never span
    # clause boundaries and section labels are exact (spec: "always chunk
    # strictly at natural clause boundaries"). Measured on the golden set:
    # recall@4 0.39 -> see README. Disable to restore size-only chunking.
    chunk_at_headings: bool = True
    candidate_k: int = 20  # hybrid retrieval fan-out (reranked down to final_k)
    final_k: int = 4       # chunks that reach the LLM — smaller prompt = faster prefill

    # Semantic answer cache
    cache_enabled: bool = True
    cache_similarity: float = 0.95
    cache_max_entries: int = 200

    # --- Query understanding (classification / rewriting / decomposition) ---
    query_classification: bool = True
    query_rewrite: bool = True
    query_rewrite_max_variants: int = 3   # variants beyond the original query
    query_decomposition: bool = True
    query_decomposition_max_parts: int = 3

    # --- Retrieval strategy ---
    # Per-leg RRF weights; query classification may override per class.
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    comparison_per_doc_k: int = 8  # candidates per document for comparison queries

    # --- Parent-child retrieval ---
    # Attach one level of parent-section context to retrieved child chunks.
    parent_context_enabled: bool = True
    parent_context_max_chars: int = 600

    # --- Context validation & packing ---
    context_packing: bool = True

    # --- Definition-aware retrieval ---
    definition_boost: bool = True

    # --- Clause-aware retrieval ---
    # Explicit references ("Clause 7.2") are extracted from the query; the
    # actual clause chunk is injected into candidates if hybrid search missed
    # it, and exact section-number matches outrank chunks that merely
    # reference the clause.
    clause_boost: bool = True
    clause_injection: bool = True

    # --- Clause lookups: exact-only context ---
    # When exact clause matches exist, only they (plus attached parent
    # context) reach the LLM — unrelated clauses are excluded from packing.
    clause_exact_only: bool = True

    # --- Document overview / summarization ---
    # Overview queries retrieve stratified evidence across the target
    # document (parties, definitions, financial terms, termination, ...)
    # instead of relying on top-k semantic similarity alone.
    overview_enabled: bool = True
    overview_final_k: int = 8       # more sections = longer prompt; latency trade-off
    overview_max_tokens: int = 1100  # structured summaries need more room than answers

    # --- Legal concept graph (related-provision retrieval expansion) ---
    concept_expansion: bool = True

    # --- Jurisdiction preference ---
    # A question naming a jurisdiction bumps chunks whose jurisdiction
    # metadata matches; different legal systems are never silently mixed.
    jurisdiction_preference: bool = True
    jurisdiction_boost: float = 0.05

    # --- Consequence-aware retrieval ---
    # "What if I break Clause 3?" retrieves the clause plus the provisions
    # that define consequences (termination, remedies, liability, default).
    consequence_final_k: int = 6

    # --- Ambiguity detection ---
    # A clause lookup matching multiple unrelated documents returns a
    # clarification request instead of silently picking one; multiple
    # versions of the same agreement are both surfaced for comparison.
    ambiguity_detection: bool = True

    # --- Version-aware retrieval ---
    prefer_latest_version: bool = True
    version_boost: float = 0.05  # rerank-score bump for latest-version chunks

    # --- Agentic retrieval (bounded retry ladder before refusing) ---
    agentic_retrieval: bool = True
    agentic_max_retries: int = 2

    # --- Answer verification ---
    # On citation-verification failure, regenerate once with a corrective note.
    regenerate_on_citation_failure: bool = True
    # If the regenerated answer still fails verification, refuse instead of
    # returning a flagged answer. Off by default (previous behavior).
    strict_verification: bool = False

    # --- Clause relationship graph (optional retrieval expansion) ---
    # Cross-reference resolution: sections referenced by retrieved chunks
    # ("incorporated by reference", "subject to", "see Exhibit A") are pulled
    # into the context, chained up to graph_max_depth hops, cycle-safe.
    graph_enabled: bool = True
    graph_max_expansion: int = 3  # extra chunks pulled in via references, max
    graph_max_depth: int = 2      # chained-reference hops (A->B->C), loop-bounded

    # --- Access control (enterprise multi-tenancy readiness) ---
    # None = single-tenant mode: no filter is applied, legacy points match.
    tenant: Optional[str] = None

    # --- Retrieval explainability ---
    debug_retrieval: bool = False  # include diagnostics in results by default

    # --- Evaluation harness ---
    golden_dataset: str = "./eval/golden.json"
    eval_k: int = 4

    # A PDF page yielding fewer extracted characters than this is treated as
    # a possible scan (low OCR confidence).
    low_text_chars_per_page: int = 200

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
