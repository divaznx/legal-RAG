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
    candidate_k: int = 20  # hybrid retrieval fan-out (reranked down to final_k)
    final_k: int = 4       # chunks that reach the LLM — smaller prompt = faster prefill

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
