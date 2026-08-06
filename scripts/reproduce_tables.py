from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolexvec.validation import cluster_inference  # noqa: E402
from rag_eval.evaluator import evaluate  # noqa: E402
from rag_eval.io_utils import load_gold  # noqa: E402


SEEDS = (20260725, 20260726, 20260727)
METRICS = (
    "Strict-Hit@1",
    "Strict-Hit@5",
    "Strict-Hit@10",
    "Strict-MRR@10",
    "nDCG@10",
)


def aggregate_seed_predictions(results_root: Path, out_dir: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = results_root / f"seed_{seed}" / "per_question_metrics.csv"
        frame = pd.read_csv(path)
        frame["vector_seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    averaged = combined.groupby(["question_id", "model"], as_index=False)[list(METRICS)].mean()
    averaged.to_csv(out_dir / "three_module_per_question.csv", index=False, encoding="utf-8-sig")
    summary = averaged.groupby("model", as_index=False)[list(METRICS)].mean()
    summary.to_csv(out_dir / "table5_three_module.csv", index=False, encoding="utf-8-sig")
    return averaged


def baseline_runs(path: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    runs: dict[str, dict[str, list[dict[str, object]]]] = {}
    basic = json.loads((path / "basic_runs.json").read_text(encoding="utf-8"))
    runs.update(basic)
    neural_dir = path / "neural"
    neural_names = {
        "cross_encoder_runs.json": "Cross-Encoder",
        "learned_sparse_runs.json": "BGE-M3 learned sparse",
        "multivector_runs.json": "BGE-M3 late interaction",
    }
    for filename, model_name in neural_names.items():
        payload = json.loads((neural_dir / filename).read_text(encoding="utf-8"))
        runs[model_name] = payload
    return runs


def build_table4(gold_path: Path, baseline_dir: Path, out_dir: Path) -> pd.DataFrame:
    gold = load_gold(gold_path)
    runs = baseline_runs(baseline_dir)
    evaluation = evaluate(gold, runs, list(range(1, 11)), out_dir / "baseline_evaluation")
    rows = []
    display_names = {
        "BgeBaseZhV15": "BGE-base",
        "BgeBaseBm25RRF": "BGE-base + BM25 RRF",
    }
    for row in evaluation["overall"]:
        name = str(row["model"])
        rows.append(
            {
                "model": display_names.get(name, name),
                **{metric: float(row[metric]) for metric in METRICS},
                "source": "precomputed external baseline run",
            }
        )
    return pd.DataFrame(rows)


def build_table5(
    three_module: pd.DataFrame,
    internal_path: Path,
    out_dir: Path,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_csv(internal_path)
    internal_summary = internal.groupby("model", as_index=False)[list(METRICS)].mean()
    internal_summary.to_csv(out_dir / "table5_entity_internal.csv", index=False, encoding="utf-8-sig")

    gold = load_gold(ROOT / "data/evaluation_gold.jsonl")
    groups = {qid: str(row.get("source_doc_id") or "") for qid, row in gold.items()}
    significance_rows = cluster_inference(
        three_module,
        groups,
        iterations,
        baseline="AliasAwareFusion",
        comparisons=["WithoutText", "WithoutEntity", "WithoutPhrase"],
        metrics=list(METRICS),
    )
    internal_for_test = internal[["question_id", "model", *METRICS]].copy()
    significance_rows.extend(
        cluster_inference(
            internal_for_test,
            groups,
            iterations,
            baseline="Full adaptation",
            comparisons=["Prototype only", "Residual only", "Raw contextual vector"],
            metrics=list(METRICS),
        )
    )
    significance = pd.DataFrame(significance_rows)
    significance.to_csv(out_dir / "table5_significance.csv", index=False, encoding="utf-8-sig")
    return internal_summary, significance


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate manuscript Tables 4 and 5 from locked runs.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reproduced_tables")
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    three_module = aggregate_seed_predictions(
        ROOT / "results/three_module_seeds", args.out_dir
    )
    table4 = build_table4(
        ROOT / "data/evaluation_gold.jsonl",
        ROOT / "data/baselines",
        args.out_dir,
    )
    main_model = three_module[three_module["model"] == "AliasAwareFusion"].groupby(
        "question_id", as_index=False
    )[list(METRICS)].mean()
    main_model.insert(1, "model", "GeoLexVec")
    main_row = main_model.groupby("model", as_index=False)[list(METRICS)].mean()
    main_row["source"] = "three-seed out-of-fold prediction"
    table4 = pd.concat([main_row, table4], ignore_index=True)
    table4 = table4.sort_values("Strict-MRR@10", ascending=False, ignore_index=True)
    table4.to_csv(args.out_dir / "table4_external_comparison.csv", index=False, encoding="utf-8-sig")
    _, significance = build_table5(
        three_module,
        ROOT / "results/entity_internal_ablation/per_question_metrics.csv",
        args.out_dir,
        args.iterations,
    )
    print("Table 4")
    print(table4.to_string(index=False))
    print("\nTable 5: three-module ablation")
    print(pd.read_csv(args.out_dir / "table5_three_module.csv").to_string(index=False))
    print("\nTable 5: entity-internal ablation")
    print(pd.read_csv(args.out_dir / "table5_entity_internal.csv").to_string(index=False))
    print(f"\nWrote reproducibility tables and significance tests to {args.out_dir}")


if __name__ == "__main__":
    main()
