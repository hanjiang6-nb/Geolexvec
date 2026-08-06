from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def dedupe_doc_ids(doc_ids: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id is None:
            continue
        value = str(doc_id).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _top(retrieved: Sequence[str], k: int) -> list[str]:
    return list(retrieved[: max(k, 0)])


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    return 1.0 if set(_top(retrieved, k)) & gold_set else 0.0


def precision_at_k(
    retrieved: Sequence[str],
    gold: Iterable[str],
    k: int,
    denominator: str = "fixed_k",
) -> float:
    if k <= 0:
        return 0.0
    pred = _top(retrieved, k)
    if denominator == "retrieved_count":
        denom = len(pred)
    elif denominator == "fixed_k":
        denom = k
    else:
        raise ValueError("denominator must be fixed_k or retrieved_count")
    if denom == 0:
        return 0.0
    return len(set(pred) & set(gold)) / denom


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    return len(set(_top(retrieved, k)) & gold_set) / len(gold_set)


def f1_at_k(
    retrieved: Sequence[str],
    gold: Iterable[str],
    k: int,
    denominator: str = "fixed_k",
) -> float:
    p = precision_at_k(retrieved, gold, k, denominator=denominator)
    r = recall_at_k(retrieved, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    for idx, doc_id in enumerate(_top(retrieved, k), start=1):
        if doc_id in gold_set:
            return 1.0 / idx
    return 0.0


def first_hit_rank_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    for idx, doc_id in enumerate(_top(retrieved, k), start=1):
        if doc_id in gold_set:
            return float(idx)
    return 0.0


def average_precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for idx, doc_id in enumerate(_top(retrieved, k), start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in gold_set:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(len(gold_set), k)


def r_precision(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return precision_at_k(retrieved, gold_set, len(gold_set), denominator="fixed_k")


def dcg(rels: Sequence[int | float]) -> float:
    return sum((2**float(rel) - 1.0) / math.log2(idx + 1) for idx, rel in enumerate(rels, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevance_judgments: Mapping[str, int | float], k: int) -> float:
    if k <= 0 or not relevance_judgments:
        return 0.0
    rels = [float(relevance_judgments.get(doc_id, 0.0)) for doc_id in _top(retrieved, k)]
    ideal = sorted((float(v) for v in relevance_judgments.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(rels) / idcg


def compute_binary_metrics(
    retrieved: Sequence[str],
    gold: Iterable[str],
    k: int,
    prefix: str,
    precision_denominator: str = "fixed_k",
) -> dict[str, float]:
    return {
        f"{prefix}-Hit@{k}": hit_at_k(retrieved, gold, k),
        f"{prefix}-Precision@{k}": precision_at_k(
            retrieved, gold, k, denominator=precision_denominator
        ),
        f"{prefix}-Recall@{k}": recall_at_k(retrieved, gold, k),
        f"{prefix}-F1@{k}": f1_at_k(retrieved, gold, k, denominator=precision_denominator),
        f"{prefix}-MRR@{k}": mrr_at_k(retrieved, gold, k),
        f"{prefix}-AP@{k}": average_precision_at_k(retrieved, gold, k),
        f"{prefix}-FirstHitRank@{k}": first_hit_rank_at_k(retrieved, gold, k),
    }


def unanswerable_metrics(
    retrieved: Sequence[str],
    context_gold: Iterable[str],
    relevance_judgments: Mapping[str, int | float],
    k: int,
) -> dict[str, float]:
    pred = _top(retrieved, k)
    context_set = set(context_gold)
    has_strong = any(float(relevance_judgments.get(doc_id, 0.0)) >= 2.0 for doc_id in pred)
    return {
        f"RetrievedAny@{k}": 1.0 if pred else 0.0,
        f"RetrievedGoldContext@{k}": 1.0 if set(pred) & context_set else 0.0,
        f"FalseSupport@{k}": 1.0 if has_strong else 0.0,
        f"SafeNoStrongEvidence@{k}": 0.0 if has_strong else 1.0,
    }
