from __future__ import annotations

import math

from rag_eval.metrics import (
    dedupe_doc_ids,
    f1_at_k,
    average_precision_at_k,
    first_hit_rank_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
)


def test_hit_precision_recall_mrr_examples() -> None:
    gold = ["d1", "d3"]
    retrieved = ["d2", "d1", "d3"]
    assert hit_at_k(retrieved, gold, 1) == 0
    assert recall_at_k(retrieved, gold, 1) == 0
    assert hit_at_k(retrieved, gold, 2) == 1
    assert precision_at_k(retrieved, gold, 2) == 1 / 2
    assert recall_at_k(retrieved, gold, 2) == 1 / 2
    assert mrr_at_k(retrieved, gold, 2) == 1 / 2
    assert first_hit_rank_at_k(retrieved, gold, 2) == 2
    assert hit_at_k(retrieved, gold, 3) == 1
    assert precision_at_k(retrieved, gold, 3) == 2 / 3
    assert recall_at_k(retrieved, gold, 3) == 1
    assert math.isclose(f1_at_k(retrieved, gold, 3), 0.8)
    assert math.isclose(average_precision_at_k(retrieved, gold, 3), ((1 / 2) + (2 / 3)) / 2)
    assert r_precision(retrieved, gold) == 1 / 2


def test_ndcg_at_k() -> None:
    qrels = {"d1": 3, "d2": 2, "d3": 1}
    assert math.isclose(ndcg_at_k(["d1", "d2", "d3"], qrels, 3), 1.0)
    assert 0 < ndcg_at_k(["d3", "d2", "d1"], qrels, 3) < 1
    assert ndcg_at_k(["x"], {}, 3) == 0


def test_dedupe_logic() -> None:
    assert dedupe_doc_ids(["d1", "d1", "", None, "d2", "d1"]) == ["d1", "d2"]


def test_empty_gold_behavior() -> None:
    retrieved = ["d1", "d2"]
    assert hit_at_k(retrieved, [], 2) == 0
    assert recall_at_k(retrieved, [], 2) == 0
    assert mrr_at_k(retrieved, [], 2) == 0
    assert precision_at_k(retrieved, [], 2) == 0
    assert f1_at_k(retrieved, [], 2) == 0


def test_retrieved_shorter_than_k_denominators() -> None:
    retrieved = ["d1"]
    gold = ["d1", "d2"]
    assert precision_at_k(retrieved, gold, 3, denominator="fixed_k") == 1 / 3
    assert precision_at_k(retrieved, gold, 3, denominator="retrieved_count") == 1
