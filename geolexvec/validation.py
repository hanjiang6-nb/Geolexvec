from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SEED = 20260725


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balanced_group_folds(
    groups: Sequence[str],
    question_types: Sequence[str],
    topics: Sequence[str],
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    group_names = sorted(set(map(str, groups)))
    if len(group_names) < n_splits:
        raise ValueError(f"{len(group_names)} groups cannot form {n_splits} folds")
    labels = sorted(
        {f"type={value}" for value in question_types}
        | {f"topic={value}" for value in topics}
    )
    label_index = {label: index for index, label in enumerate(labels)}
    group_indices: dict[str, list[int]] = {name: [] for name in group_names}
    vectors = {
        name: np.zeros(len(labels) + 1, dtype=np.float64) for name in group_names
    }
    for index, (group, question_type, topic) in enumerate(
        zip(groups, question_types, topics)
    ):
        name = str(group)
        group_indices[name].append(index)
        vectors[name][0] += 1.0
        vectors[name][1 + label_index[f"type={question_type}"]] += 1.0
        vectors[name][1 + label_index[f"topic={topic}"]] += 1.0

    total = sum(vectors.values(), np.zeros(len(labels) + 1, dtype=np.float64))
    target = total / n_splits
    scale = np.maximum(target, 1.0)
    rng = np.random.default_rng(seed)
    jitter = {name: float(rng.random()) for name in group_names}
    ordered = sorted(
        group_names,
        key=lambda name: (
            -vectors[name][0],
            -float(np.max(vectors[name][1:])),
            jitter[name],
        ),
    )
    fold_vectors = np.zeros((n_splits, len(labels) + 1), dtype=np.float64)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    for position, name in enumerate(ordered):
        if position < n_splits:
            chosen = position
        else:
            costs = []
            for fold in range(n_splits):
                trial = fold_vectors.copy()
                trial[fold] += vectors[name]
                imbalance = float(np.sum(((trial - target) / scale) ** 2))
                costs.append(
                    (imbalance, fold_vectors[fold, 0], len(fold_groups[fold]), fold)
                )
            chosen = min(costs)[-1]
        fold_vectors[chosen] += vectors[name]
        fold_groups[chosen].append(name)

    folds = [
        np.asarray(
            sorted(index for name in names for index in group_indices[name]),
            dtype=np.int32,
        )
        for names in fold_groups
    ]
    all_indices = np.concatenate(folds)
    if len(all_indices) != len(groups) or len(set(map(int, all_indices))) != len(groups):
        raise AssertionError("group-fold assignment is not a complete partition")
    return folds


def cluster_inference(
    frame: pd.DataFrame,
    groups_by_qid: dict[str, str],
    iterations: int,
    baseline: str,
    comparisons: Sequence[str],
    metrics: Sequence[str],
    seed: int = SEED + 7001,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        pivot = frame.pivot(index="question_id", columns="model", values=metric)
        for model in comparisons:
            paired = pivot[[baseline, model]].dropna()
            differences = paired[baseline] - paired[model]
            group_names = np.asarray(
                [groups_by_qid[str(question_id)] for question_id in paired.index]
            )
            unique_groups = np.unique(group_names)
            sums = np.asarray(
                [differences[group_names == group].sum() for group in unique_groups]
            )
            counts = np.asarray(
                [np.sum(group_names == group) for group in unique_groups]
            )
            bootstrap_means = np.empty(iterations, dtype=np.float64)
            null_means = np.empty(iterations, dtype=np.float64)
            for start in range(0, iterations, 1000):
                stop = min(start + 1000, iterations)
                selected = rng.integers(
                    0,
                    len(unique_groups),
                    size=(stop - start, len(unique_groups)),
                )
                bootstrap_means[start:stop] = (
                    sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
                )
                signs = rng.choice(
                    np.asarray([-1.0, 1.0]),
                    size=(stop - start, len(unique_groups)),
                )
                null_means[start:stop] = np.abs(
                    (signs * sums).sum(axis=1) / counts.sum()
                )
            observed = float(differences.mean())
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            p_value = float(
                (np.sum(null_means >= abs(observed)) + 1) / (iterations + 1)
            )
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "comparison": model,
                    "n_questions": len(paired),
                    "n_source_groups": len(unique_groups),
                    "manual_mean": float(paired[baseline].mean()),
                    "comparison_mean": float(paired[model].mean()),
                    "mean_difference": observed,
                    "cluster_bootstrap_ci95_low": float(low),
                    "cluster_bootstrap_ci95_high": float(high),
                    "group_sign_permutation_p": p_value,
                }
            )

    p_values = np.asarray([row["group_sign_permutation_p"] for row in rows])
    order = np.argsort(p_values)
    adjusted = np.empty(len(rows), dtype=np.float64)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * len(order) / rank)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["p_fdr_bh"] = float(value)
        row["significant_0.05"] = bool(value < 0.05)
    return rows
