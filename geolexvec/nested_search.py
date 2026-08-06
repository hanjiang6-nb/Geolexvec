from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np


FEATURE_NAMES = ("text_score", "joint_entity_score", "phrase_score")


def simplex_weights(units: int) -> Iterable[np.ndarray]:
    for text in range(units + 1):
        for entity in range(units - text + 1):
            phrase = units - text - entity
            yield np.asarray([text, entity, phrase], dtype=np.float64) / units


def refined_weights(center: np.ndarray) -> np.ndarray:
    center_units = np.rint(np.asarray(center) * 50).astype(int)
    lower = np.maximum(0, center_units - 5)
    upper = np.minimum(50, center_units + 5)
    rows = []
    for text in range(int(lower[0]), int(upper[0]) + 1):
        for entity in range(int(lower[1]), int(upper[1]) + 1):
            phrase = 50 - text - entity
            if int(lower[2]) <= phrase <= int(upper[2]):
                rows.append([text / 50, entity / 50, phrase / 50])
    return np.unique(np.asarray(rows, dtype=np.float64), axis=0)


def reciprocal_ranks(
    features: np.ndarray,
    relevance: np.ndarray,
    query_indices: Sequence[int],
    weights: np.ndarray,
) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    output = np.zeros((len(weights), len(query_indices)), dtype=np.float32)
    document_positions = np.arange(features.shape[1])[:, None]
    for start in range(0, len(weights), 128):
        stop = min(start + 128, len(weights))
        batch = weights[start:stop]
        for column, raw_index in enumerate(query_indices):
            index = int(raw_index)
            scores = features[index] @ batch.T
            strict = np.flatnonzero(relevance[index] >= 3)
            ranks = np.full(stop - start, features.shape[1] + 1, dtype=np.int32)
            for strict_index in strict:
                gold_scores = scores[int(strict_index)][None, :]
                candidate_ranks = 1 + np.sum(scores > gold_scores, axis=0)
                candidate_ranks += np.sum(
                    (scores == gold_scores)
                    & (document_positions < int(strict_index)),
                    axis=0,
                )
                ranks = np.minimum(ranks, candidate_ranks)
            output[start:stop, column] = np.where(
                ranks <= 10, 1.0 / ranks, 0.0
            )
    return output


def choose_best(weights: np.ndarray, scores: np.ndarray, target: np.ndarray) -> int:
    best = float(scores.max())
    candidates = np.flatnonzero(np.isclose(scores, best, atol=1e-12, rtol=0.0))
    distances = np.sum((weights[candidates] - target) ** 2, axis=1)
    return int(candidates[int(np.argmin(distances))])


def nested_search(
    features: np.ndarray,
    relevance: np.ndarray,
    train_indices: np.ndarray,
    inner_fold_ids: np.ndarray,
    constraint: Callable[[np.ndarray], np.ndarray],
    target: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    coarse = np.asarray(list(simplex_weights(10)))
    coarse = coarse[constraint(coarse)]
    coarse_rr = reciprocal_ranks(features, relevance, train_indices, coarse)
    coarse_scores = np.mean(
        np.stack(
            [coarse_rr[:, inner_fold_ids == fold].mean(axis=1) for fold in range(4)]
        ),
        axis=0,
    )
    coarse_best = choose_best(coarse, coarse_scores, target)
    fine = refined_weights(coarse[coarse_best])
    fine = fine[constraint(fine)]
    fine_rr = reciprocal_ranks(features, relevance, train_indices, fine)
    fine_scores = np.mean(
        np.stack(
            [fine_rr[:, inner_fold_ids == fold].mean(axis=1) for fold in range(4)]
        ),
        axis=0,
    )
    fine_best = choose_best(fine, fine_scores, target)
    for stage, candidates, scores in (
        ("coarse_0.10", coarse, coarse_scores),
        ("refined_0.02", fine, fine_scores),
    ):
        for values, score in zip(candidates, scores):
            history.append(
                {
                    "stage": stage,
                    **{
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, values)
                    },
                    "mean_inner_mrr_at_10": float(score),
                }
            )
    return fine[fine_best], history


def build_run(
    qids: Sequence[str],
    indices: Sequence[int],
    features: np.ndarray,
    evidence_ids: Sequence[str],
    weights: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    run: dict[str, list[dict[str, Any]]] = {}
    doc_positions = np.arange(len(evidence_ids))
    for raw_index in indices:
        index = int(raw_index)
        scores = features[index] @ np.asarray(weights, dtype=np.float32)
        top = np.lexsort((doc_positions, -scores))[:10]
        run[qids[index]] = [
            {
                "doc_id": evidence_ids[int(doc_index)],
                "rank": rank,
                "score": float(scores[int(doc_index)]),
            }
            for rank, doc_index in enumerate(top, start=1)
        ]
    return run
