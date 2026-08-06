from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolexvec.validation import cluster_inference  # noqa: E402
from rag_eval.io_utils import load_gold  # noqa: E402


SEEDS = (20260725, 20260726, 20260727)
MODELS = ("AliasAwareFusion", "WithoutText", "WithoutEntity", "WithoutPhrase")
METRICS = (
    "Strict-Hit@1",
    "Strict-Hit@5",
    "Strict-Hit@10",
    "Strict-MRR@10",
    "Strict-Recall@10",
    "nDCG@10",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the GeoLexVec main experiment.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.gold)
    frames = []
    weight_frames = []
    trend_frames = []
    seed_rows = []
    for seed in SEEDS:
        seed_dir = args.ablation_root / f"seed_{seed}"
        frame = pd.read_csv(seed_dir / "per_question_metrics.csv")
        if len(frame) != 2034 * len(MODELS) or set(frame["model"]) != set(MODELS):
            raise RuntimeError(f"incomplete seed output: {seed}")
        frame["vector_seed"] = seed
        frames.append(frame)
        weights = pd.read_csv(seed_dir / "selected_fold_weights.csv")
        weights["vector_seed"] = seed
        weight_frames.append(weights)
        trend = pd.read_csv(seed_dir / "topk_trend_long.csv")
        trend["vector_seed"] = seed
        trend_frames.append(trend)
        selected = frame.loc[frame["model"] == "AliasAwareFusion"]
        seed_rows.append(
            {
                "vector_seed": seed,
                **{metric: float(selected[metric].mean()) for metric in METRICS},
            }
        )

    all_results = pd.concat(frames, ignore_index=True)
    averaged = all_results.groupby(["question_id", "model"], as_index=False)[list(METRICS)].mean()
    significance = cluster_inference(
        averaged,
        {str(qid): str(row.get("source_doc_id") or "") for qid, row in gold.items()},
        args.iterations,
        baseline="AliasAwareFusion",
        comparisons=["WithoutText", "WithoutEntity", "WithoutPhrase"],
        metrics=list(METRICS),
    )
    pd.DataFrame(significance).to_csv(
        args.out_dir / "three_module_significance.csv", index=False, encoding="utf-8-sig"
    )

    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(args.out_dir / "seed_stability.csv", index=False, encoding="utf-8-sig")
    trend = pd.concat(trend_frames, ignore_index=True)
    summary = (
        trend.groupby(["subset", "model", "k", "metric"], as_index=False)
        .agg(value_mean=("value", "mean"), value_std=("value", "std"))
        .fillna({"value_std": 0.0})
    )
    summary.to_csv(args.out_dir / "topk_summary.csv", index=False, encoding="utf-8-sig")
    main_metrics = summary.loc[
        (summary["subset"] == "all") & (summary["model"] == "AliasAwareFusion")
    ]
    main_metrics.to_csv(args.out_dir / "geolexvec_main_metrics.csv", index=False, encoding="utf-8-sig")

    ablation = summary.loc[
        (summary["subset"] == "all")
        & (summary["metric"] == "Strict-MRR")
        & (summary["k"] == 10)
    ].copy()
    full = float(ablation.loc[ablation["model"] == "AliasAwareFusion", "value_mean"].iloc[0])
    ablation["full_minus_variant"] = full - ablation["value_mean"]
    ablation.to_csv(
        args.out_dir / "three_module_ablation_mrr_at_10.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weights = pd.concat(weight_frames, ignore_index=True)
    weights.to_csv(args.out_dir / "selected_weights.csv", index=False, encoding="utf-8-sig")
    weight_summary = weights.groupby("model", as_index=False).agg(
        text_mean=("text_score", "mean"),
        text_std=("text_score", "std"),
        entity_mean=("entity_score", "mean"),
        entity_std=("entity_score", "std"),
        phrase_mean=("phrase_score", "mean"),
        phrase_std=("phrase_score", "std"),
    )
    weight_summary.to_csv(args.out_dir / "weight_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "questions": len(gold),
        "evidence_candidates": 1413,
        "vector_seeds": list(SEEDS),
        "outer_folds": 5,
        "inner_folds": 4,
        "cluster_iterations": args.iterations,
        "final_entity_model": "AliasAwareFusion",
        "text_module": "BAAI/bge-small-zh-v1.5 sentence-vector cosine",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(ablation[["model", "value_mean", "full_minus_variant"]].to_string(index=False))


if __name__ == "__main__":
    main()
