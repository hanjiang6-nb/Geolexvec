from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def sentence_cosine_matrix(
    question_vectors: np.ndarray, evidence_vectors: np.ndarray
) -> np.ndarray:
    questions = normalize_rows(question_vectors)
    evidence = normalize_rows(evidence_vectors)
    if questions.ndim != 2 or evidence.ndim != 2:
        raise ValueError("sentence-vector inputs must be two-dimensional")
    if questions.shape[1] != evidence.shape[1]:
        raise ValueError("question and evidence sentence-vector dimensions differ")
    return questions @ evidence.T


def contextual_entity_similarity(
    query_vectors: np.ndarray, evidence_vectors: np.ndarray
) -> float:
    """Average maximum non-negative cosine over all entity pairs.

    Canonical links are used when learning the occurrence-vector adapter, but
    retrieval compares every query entity with every evidence entity. No hard
    alias score, same-canonical filter, or entity-type filter is applied here.
    """
    query_matrix = normalize_rows(query_vectors)
    evidence_matrix = normalize_rows(evidence_vectors)
    if query_matrix.size == 0 or evidence_matrix.size == 0:
        return 0.0
    if query_matrix.ndim != 2 or evidence_matrix.ndim != 2:
        raise ValueError("entity-vector inputs must be two-dimensional")
    if query_matrix.shape[1] != evidence_matrix.shape[1]:
        raise ValueError("query and evidence entity-vector dimensions differ")
    cosine = query_matrix @ evidence_matrix.T
    return float(np.maximum(cosine.max(axis=1), 0.0).mean())


def phrase_group_score(query_groups: Sequence[Sequence[str]], evidence_text: str) -> float:
    if not query_groups:
        return 0.0
    hits = sum(
        1
        for group in query_groups
        if any(
            normalized in evidence_text
            for keyword in group
            if (normalized := str(keyword).strip())
        )
    )
    return hits / len(query_groups)


def fuse_scores(
    sentence_score: np.ndarray | float,
    entity_score: np.ndarray | float,
    phrase_score: np.ndarray | float,
    weights: Sequence[float],
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (3,) or np.any(values < 0):
        raise ValueError("GeoLexVec requires three non-negative module weights")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"module weights must sum to one, found {total}")
    return (
        values[0] * np.asarray(sentence_score)
        + values[1] * np.asarray(entity_score)
        + values[2] * np.asarray(phrase_score)
    )
