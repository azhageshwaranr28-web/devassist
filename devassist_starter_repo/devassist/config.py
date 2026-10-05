"""Central environment-driven configuration for DevAssist."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parent
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "devassist")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "220"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    rerank_min_score: float = float(os.getenv("RERANK_MIN_SCORE", "-4.0"))
    hybrid_search: bool = _bool("HYBRID_SEARCH", True)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")

    @property
    def raw_questions(self) -> Path:
        return self.root / "data/raw/questions.jsonl"

    @property
    def raw_answers(self) -> Path:
        return self.root / "data/raw/answers.jsonl"

    @property
    def documents_file(self) -> Path:
        return self.root / "data/processed/documents.jsonl"

    @property
    def chunks_file(self) -> Path:
        return self.root / "data/processed/chunks.jsonl"


settings = Settings()
