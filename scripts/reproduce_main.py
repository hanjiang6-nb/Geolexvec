from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


SEEDS = (20260725, 20260726, 20260727)


def run(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the GeoLexVec main experiment.")
    parser.add_argument("--out-dir", type=Path, default=Path("reproduced"))
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    gold = root / "data/evaluation_gold.jsonl"

    def run_seed(seed: int) -> None:
        cache = root / f"data/feature_caches/seed_{seed}_sentence_variants.npz"
        if not cache.is_file() or cache.read_bytes()[:32].startswith(b"version https://git-lfs"):
            raise FileNotFoundError(f"missing Git LFS object: {cache}")
        run(
            [
                sys.executable,
                "experiments/final_alias_aware_study/run_three_module_ablation.py",
                "--gold",
                str(gold),
                "--cache",
                str(cache),
                "--out-dir",
                str(out_dir / "seeds" / f"seed_{seed}"),
                "--vector-seed",
                str(seed),
                "--fold-seed",
                "20260725",
            ],
            root,
            out_dir / "logs" / f"seed_{seed}.log",
        )

    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 3))) as executor:
        list(executor.map(run_seed, SEEDS))

    run(
        [
            sys.executable,
            "experiments/analyze_main_experiment.py",
            "--gold",
            str(gold),
            "--ablation-root",
            str(out_dir / "seeds"),
            "--out-dir",
            str(out_dir / "analysis"),
            "--iterations",
            str(args.iterations),
        ],
        root,
        out_dir / "logs" / "analysis.log",
    )

    expected = json.loads((root / "configs/expected_results.json").read_text(encoding="utf-8"))
    tolerance = float(expected["cross_platform_tolerance"])
    actual = pd.read_csv(out_dir / "analysis/three_module_ablation_mrr_at_10.csv")
    differences = {}
    for model, target in expected["strict_mrr_at_10"].items():
        value = float(actual.loc[actual["model"] == model, "value_mean"].iloc[0])
        differences[model] = abs(value - float(target))
        if differences[model] > tolerance:
            raise RuntimeError(
                f"{model} differs by {differences[model]}, tolerance={tolerance}"
            )
    report = {
        "status": "passed",
        "iterations": args.iterations,
        "tolerance": tolerance,
        "absolute_differences": differences,
    }
    (out_dir / "reproduction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
