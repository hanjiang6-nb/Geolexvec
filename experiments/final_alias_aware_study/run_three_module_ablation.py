from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from geolexvec.validation import (  # noqa: E402
    balanced_group_folds,
    cluster_inference,
    write_csv,
)
from geolexvec.nested_search import (  # noqa: E402
    build_run,
    nested_search,
)
from rag_eval.evaluator import evaluate  # noqa: E402
from rag_eval.io_utils import load_gold  # noqa: E402


TOP_K = tuple(range(1, 11))
MODEL_ORDER = (
    "AliasAwareFusion",
    "WithoutText",
    "WithoutEntity",
    "WithoutPhrase",
)
SUMMARY_METRICS = (
    "Strict-Hit@1",
    "Strict-Hit@5",
    "Strict-Hit@10",
    "Strict-MRR@10",
    "Strict-Recall@10",
    "nDCG@10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trend_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    subsets = {
        "all": np.ones(len(frame), dtype=bool),
        "changed": frame["question_changed"].astype(bool).to_numpy(),
        "unchanged": ~frame["question_changed"].astype(bool).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for subset, mask in subsets.items():
        selected = frame.loc[mask]
        for model, model_rows in selected.groupby("model", sort=False):
            for k in TOP_K:
                for metric in ("Strict-Hit", "Strict-Recall", "Strict-MRR", "nDCG"):
                    column = f"{metric}@{k}"
                    rows.append(
                        {
                            "subset": subset,
                            "model": model,
                            "k": k,
                            "metric": metric,
                            "value": float(model_rows[column].mean()),
                            "n_questions": len(model_rows),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested three-module ablation for Alias-Aware Fusion."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vector-seed", type=int, required=True)
    parser.add_argument("--fold-seed", type=int, default=20260725)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.gold)
    with np.load(args.cache, allow_pickle=False) as cached:
        qids = [str(value) for value in cached["qids"]]
        evidence_ids = [str(value) for value in cached["evidence_ids"]]
        groups = [str(value) for value in cached["groups"]]
        text_features = np.asarray(cached["text_features"], dtype=np.float32)
        phrase_features = np.asarray(cached["phrase_features"], dtype=np.float32)
        if "context_entity_features" in cached.files:
            entity_features = np.asarray(
                cached["context_entity_features"], dtype=np.float32
            )
        else:
            variants = [str(value) for value in cached["variant_names"]]
            all_entity_features = np.asarray(
                cached["entity_features"], dtype=np.float32
            )
            if "AliasAwareFusion" not in variants:
                raise RuntimeError(
                    "AliasAwareFusion is missing from the entity variant cache"
                )
            entity_features = all_entity_features[:, :, variants.index("AliasAwareFusion")]
        relevance = np.asarray(cached["relevance"], dtype=np.uint8)
        formula_version = str(cached["formula_version"].item())
        cache_fingerprint = str(cached["fingerprint"].item())

    if len(qids) != 2034 or len(evidence_ids) != 1413 or set(qids) != set(gold):
        raise RuntimeError("cache does not match the 2,034-question, 1,413-evidence benchmark")
    expected_shape = (len(qids), len(evidence_ids))
    if text_features.shape != expected_shape or phrase_features.shape != expected_shape:
        raise RuntimeError("text or phrase feature matrix has an unexpected shape")

    features = np.stack(
        [text_features, entity_features, phrase_features], axis=2
    ).astype(np.float32, copy=False)

    question_types = [str(gold[qid].get("question_type", "")) for qid in qids]
    topics = [str(gold[qid].get("geological_topic", "")) for qid in qids]
    outer_folds = balanced_group_folds(
        groups, question_types, topics, 5, args.fold_seed
    )
    configurations = {
        "AliasAwareFusion": (
            lambda weights: np.ones(len(weights), dtype=bool),
            np.full(3, 1.0 / 3.0),
        ),
        "WithoutText": (
            lambda weights: np.isclose(weights[:, 0], 0.0),
            np.asarray([0.0, 0.5, 0.5]),
        ),
        "WithoutEntity": (
            lambda weights: np.isclose(weights[:, 1], 0.0),
            np.asarray([0.5, 0.0, 0.5]),
        ),
        "WithoutPhrase": (
            lambda weights: np.isclose(weights[:, 2], 0.0),
            np.asarray([0.5, 0.5, 0.0]),
        ),
    }

    runs = {model: {} for model in MODEL_ORDER}
    selected_rows: list[dict[str, Any]] = []
    fold_by_qid: dict[str, int] = {}
    for fold_number, test_indices in enumerate(outer_folds, start=1):
        test_set = set(map(int, test_indices))
        train_indices = np.asarray(
            [index for index in range(len(qids)) if index not in test_set],
            dtype=np.int32,
        )
        inner_folds = balanced_group_folds(
            [groups[int(index)] for index in train_indices],
            [question_types[int(index)] for index in train_indices],
            [topics[int(index)] for index in train_indices],
            4,
            args.fold_seed + fold_number * 101,
        )
        inner_fold_ids = np.empty(len(train_indices), dtype=np.int8)
        for inner_fold, local_indices in enumerate(inner_folds):
            inner_fold_ids[local_indices] = inner_fold

        for model in MODEL_ORDER:
            constraint, target = configurations[model]
            print(
                f"vector seed {args.vector_seed}, fold {fold_number}/5: {model}",
                flush=True,
            )
            weights, history = nested_search(
                features,
                relevance,
                train_indices,
                inner_fold_ids,
                constraint,
                target,
            )
            write_csv(
                args.out_dir / "grid_history" / f"fold_{fold_number}_{model}.csv",
                history,
            )
            runs[model].update(
                build_run(qids, test_indices, features, evidence_ids, weights)
            )
            selected_rows.append(
                {
                    "vector_seed": args.vector_seed,
                    "fold": fold_number,
                    "model": model,
                    "text_score": float(weights[0]),
                    "entity_score": float(weights[1]),
                    "phrase_score": float(weights[2]),
                }
            )
        for index in test_indices:
            fold_by_qid[qids[int(index)]] = fold_number

    if any(len(run) != len(qids) for run in runs.values()):
        raise AssertionError("one or more out-of-fold runs are incomplete")

    evaluation = evaluate(gold, runs, list(TOP_K), args.out_dir / "evaluation")
    per_question = pd.DataFrame(evaluation["per_question"])
    per_question["outer_fold"] = per_question["question_id"].map(fold_by_qid)
    per_question["question_changed"] = per_question["question_id"].map(
        {qid: bool(gold[qid].get("question_changed", False)) for qid in qids}
    )
    per_question["vector_seed"] = args.vector_seed
    per_question.to_csv(
        args.out_dir / "per_question_metrics.csv", index=False, encoding="utf-8-sig"
    )

    fold_metrics = (
        per_question.groupby(["outer_fold", "model"], as_index=False)[
            list(SUMMARY_METRICS)
        ]
        .mean()
        .rename(columns={"outer_fold": "fold"})
    )
    fold_metrics.to_csv(
        args.out_dir / "five_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = fold_metrics.groupby("model", as_index=False).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in SUMMARY_METRICS
            for stat in ("mean", "std")
        }
    )
    summary.to_csv(
        args.out_dir / "mean_std_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(trend_rows(per_question)).to_csv(
        args.out_dir / "topk_trend_long.csv", index=False, encoding="utf-8-sig"
    )

    significance = cluster_inference(
        per_question,
        {qid: groups[index] for index, qid in enumerate(qids)},
        args.iterations,
        baseline="AliasAwareFusion",
        comparisons=[model for model in MODEL_ORDER if model != "AliasAwareFusion"],
        metrics=list(SUMMARY_METRICS),
        seed=args.fold_seed + 7001,
    )
    write_csv(args.out_dir / "cluster_significance.csv", significance)
    write_csv(args.out_dir / "selected_fold_weights.csv", selected_rows)
    (args.out_dir / "oof_runs.json").write_text(
        json.dumps(
            {
                "vector_seed": args.vector_seed,
                "fold_seed": args.fold_seed,
                "fold_by_qid": fold_by_qid,
                "runs": runs,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    protocol = {
        "scope": "final three-module leave-one-module-out ablation",
        "models": list(MODEL_ORDER),
        "feature_order": ["text", "contextual_entity", "phrase"],
        "entity_formula_version": formula_version,
        "entity_definition": (
            "each query entity uses the maximum non-negative learned cosine over all "
            "evidence entities; scores are averaged by query entity count"
        ),
        "vector_definition": (
            "L2-normalized original vector plus rank-8 low-rank residual, followed by "
            "gated canonical prototype fusion and a final L2 normalization"
        ),
        "weight_selection": (
            "weights are reselected for every model and outer fold using four grouped "
            "inner folds; removed-module weight is constrained to zero"
        ),
        "outer_validation": "five source-document-disjoint folds",
        "top_k": list(TOP_K),
        "vector_seed": args.vector_seed,
        "fold_seed": args.fold_seed,
        "question_count": len(qids),
        "evidence_count": len(evidence_ids),
        "cache_fingerprint": cache_fingerprint,
        "input_sha256": {"gold": sha256(args.gold), "cache": sha256(args.cache)},
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.sort_values("Strict-MRR@10_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
