"""Retrieval metrics counted at Stack Overflow question-ID level."""
from __future__ import annotations


def unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(v) for v in values))


def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = unique(retrieved)[:k]
    return len(set(top) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = unique(retrieved)[:k]
    return len(set(top) & set(relevant)) / k


def reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
    relevant_set = set(relevant)
    for rank, qid in enumerate(unique(retrieved), start=1):
        if qid in relevant_set:
            return 1.0 / rank
    return 0.0
