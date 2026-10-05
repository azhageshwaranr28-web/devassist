"""End-to-end DevAssist query pipeline."""
from __future__ import annotations

import time
from typing import Any

from config import settings
from generation.citations import sources_from_answer
from generation.llm import GroundedGenerator
from reranking.reranker import Reranker
from retrieval.hybrid import HybridRetriever


class DevAssistPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = GroundedGenerator()

    def answer(self, raw_query: str) -> dict[str, Any]:
        started = time.perf_counter()
        query = self.retriever.process_query(raw_query)

        t0 = time.perf_counter()
        candidates = self.retriever.search(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        contexts = self.reranker.rank(query, candidates)
        rerank_ms = (time.perf_counter() - t0) * 1000

        if not contexts or float(contexts[0]["reranker_score"]) < settings.rerank_min_score:
            return {
                "query": query,
                "status": "insufficient",
                "answer": (
                    "I do not have enough relevant evidence in the indexed Stack Overflow subset to answer "
                    "this safely. Try adding the exact error message, library version, and a minimal code example."
                ),
                "sources": [],
                "contexts": contexts,
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 1),
                    "reranking": round(rerank_ms, 1),
                    "total": round((time.perf_counter() - started) * 1000, 1),
                },
            }

        t0 = time.perf_counter()
        answer = self.generator.generate(query, contexts)
        generation_ms = (time.perf_counter() - t0) * 1000
        return {
            "query": query,
            "status": "answered",
            "answer": answer,
            "sources": sources_from_answer(answer, contexts),
            "contexts": contexts,
            "timings_ms": {
                "retrieval": round(retrieval_ms, 1),
                "reranking": round(rerank_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
        }
