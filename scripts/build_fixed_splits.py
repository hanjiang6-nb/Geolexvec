from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geolexvec.validation import balanced_group_folds  # noqa: E402
from rag_eval.io_utils import load_gold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the locked 5x4 grouped splits.")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/evaluation_gold.jsonl")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/feature_caches/seed_20260725_sentence_variants.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/splits/source_grouped_5x4.json",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    gold = load_gold(args.gold)
    with np.load(args.cache, allow_pickle=False) as cached:
        qids = [str(value) for value in cached["qids"]]
        groups = [str(value) for value in cached["groups"]]
    if len(qids) != 2034 or set(qids) != set(gold):
        raise RuntimeError("split inputs do not match the locked 2,034-query benchmark")

    question_types = [str(gold[qid].get("question_type", "")) for qid in qids]
    topics = [str(gold[qid].get("geological_topic", "")) for qid in qids]
    outer = balanced_group_folds(groups, question_types, topics, 5, args.seed)
    rows = []
    all_indices = np.arange(len(qids), dtype=np.int32)
    for outer_number, test_indices in enumerate(outer, start=1):
        test_set = set(map(int, test_indices))
        train_indices = np.asarray(
            [index for index in all_indices if int(index) not in test_set],
            dtype=np.int32,
        )
        inner = balanced_group_folds(
            [groups[int(index)] for index in train_indices],
            [question_types[int(index)] for index in train_indices],
            [topics[int(index)] for index in train_indices],
            4,
            args.seed + outer_number * 101,
        )
        test_groups = sorted({groups[int(index)] for index in test_indices})
        train_groups = sorted({groups[int(index)] for index in train_indices})
        if set(test_groups) & set(train_groups):
            raise AssertionError("a source document crosses an outer-fold boundary")
        rows.append(
            {
                "outer_fold": outer_number,
                "test_question_ids": [qids[int(index)] for index in test_indices],
                "test_source_doc_ids": test_groups,
                "train_question_ids": [qids[int(index)] for index in train_indices],
                "train_source_doc_ids": train_groups,
                "inner_validation_question_ids": [
                    [qids[int(train_indices[int(local)])] for local in local_indices]
                    for local_indices in inner
                ],
                "inner_validation_source_doc_ids": [
                    sorted(
                        {
                            groups[int(train_indices[int(local)])]
                            for local in local_indices
                        }
                    )
                    for local_indices in inner
                ],
            }
        )

    payload = {
        "protocol": "source-document-grouped nested validation",
        "fold_seed": args.seed,
        "question_count": len(qids),
        "source_document_count": len(set(groups)),
        "outer_fold_count": 5,
        "inner_fold_count": 4,
        "folds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote locked 5x4 splits for {len(qids)} queries to {args.out}")


if __name__ == "__main__":
    main()
