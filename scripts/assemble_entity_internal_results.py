from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEEDS = (20260725, 20260726, 20260727)
METRICS = (
    "Strict-Hit@1",
    "Strict-Hit@5",
    "Strict-Hit@10",
    "Strict-MRR@10",
    "nDCG@10",
)


def averaged_variant(root: Path, display_name: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "per_question_metrics.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["model"] == "AliasAwareFusion", ["question_id", *METRICS]]
        if len(frame) != 2034 or frame["question_id"].nunique() != 2034:
            raise ValueError(f"incomplete entity-internal run: {path}")
        frame["vector_seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    averaged = combined.groupby("question_id", as_index=False)[list(METRICS)].mean()
    averaged["model"] = display_name
    return averaged[["question_id", *METRICS, "model"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the four entity-adapter variants from three-seed runs."
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_source)
    raw = raw.loc[
        raw["model"] == "Raw contextual vector", ["question_id", *METRICS, "model"]
    ]
    if len(raw) != 2034 or raw["question_id"].nunique() != 2034:
        raise ValueError("raw contextual-vector rows are incomplete")
    frames = [
        averaged_variant(args.full_root, "Full adaptation"),
        averaged_variant(args.prototype_root, "Prototype only"),
        averaged_variant(args.residual_root, "Residual only"),
        raw,
    ]
    expected_ids = set(frames[0]["question_id"])
    if any(set(frame["question_id"]) != expected_ids for frame in frames[1:]):
        raise ValueError("entity-internal variants use different question IDs")
    output = pd.concat(frames, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        args.out_dir / "per_question_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = output.groupby("model", as_index=False)[list(METRICS)].mean()
    summary.to_csv(
        args.out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
