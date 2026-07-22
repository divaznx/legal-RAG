from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    data_dir: str = "./data"
    qdrant_path: str = "./data/qdrant"
    qdrant_url: Optional[str] = None
    # v3: adds the indexed legal-structure payload (clause_number, defined_terms,
    # concepts, xrefs). The bump also forces a re-ingest, which is deliberate —
    # v2 chunks were written by the pre-fix chunker and carry section labels
    # that can be off by one clause.
    qdrant_collection: str = "lexis_chunks_v3"

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
    candidate_k: int = 20  # hybrid retrieval fan-out (reranked down to final_k)
    final_k: int = 4       # fallback when no intent policy applies

    # --- Legal intelligence layer ---
    # Master switch. Off = plain hybrid RAG (the v0.2 behaviour).
    legal_intelligence: bool = True
    # Ask which agreement is meant instead of guessing across unrelated
    # contracts. Turning this off makes the system answer from all matching
    # documents, which risks blending clauses — only do it for demos.
    allow_clarification: bool = True
    # Evidence slots reserved for definitions of terms the question uses, and
    # for clauses pulled in by cross-reference. Budgeted separately so a long
    # consequence chain can never crowd out the definitions it depends on.
    definition_budget: int = 2
    xref_budget: int = 3
    # Sibling sub-clauses of the top evidence (12.3 -> 12.1, 12.2, 12.4).
    # Sub-clauses of one provision qualify each other constantly, so this is
    # reserved rather than left to compete on relevance. Sized to return a
    # whole ordinary provision: half a provision is worse than none, because
    # it reads complete.
    sibling_budget: int = 6
    # Slots reserved for the expanded legal chain — one clause per link
    # (cure, termination, remedies, damages, liability cap). Halved for
    # questions that only need related context rather than a consequence.
    concept_budget: int = 4
    # Cap on total evidence chunks regardless of intent policy — the prompt
    # still has to fit and prefill still dominates latency.
    max_evidence_chunks: int = 14

    # Input validation. The upper bound is a cost/latency guard: the question
    # is embedded, expanded into the sparse query, and prepended to every
    # prompt, so an unbounded one is a denial-of-service vector.
    min_question_chars: int = 3
    max_question_chars: int = 2000
    # Upload ceiling for the HTTP API (bytes).
    max_upload_bytes: int = 25 * 1024 * 1024

    # Semantic answer cache
    cache_enabled: bool = True
    cache_similarity: float = 0.95
    cache_max_entries: int = 200

    # A PDF page yielding fewer extracted characters than this is treated as
    # a possible scan (low OCR confidence).
    low_text_chars_per_page: int = 200

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
