"""Cross-encoder reranking of a small first-stage candidate set."""
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from config import settings


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model)

    def rank(self, query: str, candidates: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]:
        if not candidates:
            return []
        top_n = top_n or settings.rerank_top_n
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        reranked = [dict(candidate, reranker_score=float(score)) for candidate, score in zip(candidates, scores)]
        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        return reranked[:top_n]
