from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_metrics_deduplicate_questions():
    retrieved = [1, 1, 2, 3, 4]
    relevant = [2, 4]
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
