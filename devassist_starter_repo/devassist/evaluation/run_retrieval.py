"""Run repeatable retrieval evaluation and compare vector/hybrid/reranked variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from reranking.reranker import Reranker
from retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "evaluation/test_set.jsonl"
RESULTS = ROOT / "evaluation/results/retrieval_results.csv"


def evaluate_ids(ids: list[int], relevant: list[int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(ids, relevant, 5),
        "precision@5": precision_at_k(ids, relevant, 5),
        "mrr": reciprocal_rank(ids, relevant),
    }


def main() -> None:
    cases = [json.loads(line) for line in TEST_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [case for case in cases if case.get("answerable", True) and case.get("relevant_question_ids")]
    retriever = HybridRetriever()
    reranker = Reranker()
    rows = []
    for case in cases:
        for name, hybrid in [("vector", False), ("hybrid", True)]:
            candidates = retriever.search(case["query"], top_k=20, hybrid=hybrid)
            ids = [int(c["metadata"]["question_id"]) for c in candidates]
            rows.append({"case_id": case["id"], "configuration": name, **evaluate_ids(ids, case["relevant_question_ids"])})
            reranked = reranker.rank(case["query"], candidates, top_n=5)
            reranked_ids = [int(c["metadata"]["question_id"]) for c in reranked]
            rows.append({"case_id": case["id"], "configuration": f"{name}+reranker", **evaluate_ids(reranked_ids, case["relevant_question_ids"])})

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for configuration in sorted({row["configuration"] for row in rows}):
        selected = [row for row in rows if row["configuration"] == configuration]
        print(configuration, {metric: round(mean(r[metric] for r in selected), 3) for metric in ["recall@5", "precision@5", "mrr"]})
    print(f"Detailed rows: {RESULTS}")


if __name__ == "__main__":
    main()
