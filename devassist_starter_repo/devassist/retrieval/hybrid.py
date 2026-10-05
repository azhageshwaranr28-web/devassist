"""Semantic retrieval plus BM25, fused by Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#./:@-]+")


def technical_tokenize(text: str) -> list[str]:
    """Keep symbols useful in identifiers, flags, versions, URLs and error names."""
    return TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(self) -> None:
        if not settings.chunks_file.exists():
            raise FileNotFoundError("No chunks found. Run: python -m ingestion.build_index")
        self.model = SentenceTransformer(settings.embedding_model)
        qpath = str((settings.root / settings.qdrant_path).resolve()) if not Path(settings.qdrant_path).is_absolute() else settings.qdrant_path
        self.client = QdrantClient(path=qpath)
        with settings.chunks_file.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        self.by_id = {int(c["chunk_id"]): c for c in chunks}
        self.ordered_ids = [int(c["chunk_id"]) for c in chunks]
        self.bm25 = BM25Okapi([technical_tokenize(c["text"]) for c in chunks])

    @staticmethod
    def process_query(query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Please describe a technical problem before searching.")
        return cleaned[:8000]

    def _vector_ids(self, query: str, top_k: int) -> tuple[list[int], dict[int, float]]:
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        points = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [int(p.id) for p in points], {int(p.id): float(p.score) for p in points}

    def _bm25_ids(self, query: str, top_k: int) -> tuple[list[int], dict[int, float]]:
        raw_scores = self.bm25.get_scores(technical_tokenize(query))
        order = np.argsort(raw_scores)[::-1][:top_k]
        ids = [self.ordered_ids[int(i)] for i in order if raw_scores[int(i)] > 0]
        return ids, {self.ordered_ids[int(i)]: float(raw_scores[int(i)]) for i in order}

    def search(self, query: str, top_k: int | None = None, hybrid: bool | None = None) -> list[dict[str, Any]]:
        query = self.process_query(query)
        top_k = top_k or settings.retrieval_top_k
        hybrid = settings.hybrid_search if hybrid is None else hybrid
        wide_k = max(top_k * 2, 40)
        vector_ids, vector_scores = self._vector_ids(query, wide_k)
        ranked_lists = [vector_ids]
        bm25_scores: dict[int, float] = {}
        if hybrid:
            bm25_ids, bm25_scores = self._bm25_ids(query, wide_k)
            ranked_lists.append(bm25_ids)

        fused = reciprocal_rank_fusion(ranked_lists)
        output: list[dict[str, Any]] = []
        per_question: dict[int, int] = {}
        for chunk_id, rrf_score in fused:
            chunk = self.by_id.get(int(chunk_id))
            if not chunk:
                continue
            qid = int(chunk["metadata"]["question_id"])
            if per_question.get(qid, 0) >= 2:
                continue
            per_question[qid] = per_question.get(qid, 0) + 1
            output.append(
                {
                    **chunk,
                    "retrieval_score": rrf_score,
                    "vector_score": vector_scores.get(chunk_id),
                    "bm25_score": bm25_scores.get(chunk_id),
                }
            )
            if len(output) >= top_k:
                break
        return output
