from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .metrics import compute_binary_metrics, ndcg_at_k, r_precision, unanswerable_metrics
from .qrels import write_qrels, write_run_files


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str], metric_cols: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(k, "") for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda x: tuple(str(v) for v in x[0])):
        item = {k: v for k, v in zip(group_keys, key)}
        item["n_questions"] = len(group)
        for col in metric_cols:
            item[col] = mean([float(r.get(col, 0.0) or 0.0) for r in group])
        out.append(item)
    return out


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    gold_rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, list[dict[str, Any]]]],
    k_list: list[int],
    out_dir: str | Path,
    precision_denominator: str = "fixed_k",
    write_aggregates: bool = True,
) -> dict[str, Any]:
    if not gold_rows:
        raise ValueError("gold rows are empty")
    if not runs:
        raise ValueError("retrieval runs are empty")
    gold_ids = set(gold_rows)
    for model, by_qid in runs.items():
        run_ids = set(by_qid)
        missing = gold_ids - run_ids
        extra = run_ids - gold_ids
        if missing or extra:
            raise ValueError(
                f"{model} query coverage mismatch: missing={len(missing)}, extra={len(extra)}"
            )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_question: list[dict[str, Any]] = []
    for model, by_qid in sorted(runs.items()):
        for qid, gold in sorted(gold_rows.items()):
            items = by_qid.get(qid, [])
            retrieved = [item["doc_id"] for item in items]
            row: dict[str, Any] = {
                "question_id": qid,
                "question": gold.get("question", ""),
                "question_type": gold.get("question_type", ""),
                "geological_topic": gold.get("geological_topic", ""),
                "difficulty": gold.get("difficulty", ""),
                "answerable": bool(gold.get("answerable", True)),
                "model": model,
                "retrieved_doc_ids": json.dumps(retrieved, ensure_ascii=False),
                "gold_strict": json.dumps(gold.get("_strict_gold", []), ensure_ascii=False),
                "gold_context": json.dumps(gold.get("_context_gold", []), ensure_ascii=False),
                "retrieved_count": len(retrieved),
                "Strict-RPrecision": r_precision(retrieved, gold.get("_strict_gold", [])),
                "Context-RPrecision": r_precision(retrieved, gold.get("_context_gold", [])),
            }
            for k in k_list:
                row.update(
                    compute_binary_metrics(
                        retrieved,
                        gold.get("_strict_gold", []),
                        k,
                        "Strict",
                        precision_denominator,
                    )
                )
                row.update(
                    compute_binary_metrics(
                        retrieved,
                        gold.get("_context_gold", []),
                        k,
                        "Context",
                        precision_denominator,
                    )
                )
                row[f"nDCG@{k}"] = ndcg_at_k(retrieved, gold.get("_relevance_judgments", {}), k)
                if not bool(gold.get("answerable", True)):
                    row.update(
                        unanswerable_metrics(
                            retrieved, gold.get("_context_gold", []), gold.get("_relevance_judgments", {}), k
                        )
                    )
                else:
                    row.update(
                        {
                            f"RetrievedAny@{k}": 0.0,
                            f"RetrievedGoldContext@{k}": 0.0,
                            f"FalseSupport@{k}": 0.0,
                            f"SafeNoStrongEvidence@{k}": 0.0,
                        }
                    )
            per_question.append(row)

    write_csv(out / "per_question_retrieval_metrics.csv", per_question)
    write_qrels(gold_rows, out)
    write_run_files(runs, out / "runs")
    if not write_aggregates:
        return {
            "per_question": per_question,
            "overall": [],
            "by_question_type": [],
            "by_geological_topic": [],
            "by_difficulty": [],
            "by_answerable": [],
        }

    metric_cols = [
        col
        for col in per_question[0].keys()
        if any(col.endswith(f"@{k}") for k in k_list)
        or col in {"retrieved_count", "Strict-RPrecision", "Context-RPrecision"}
    ] if per_question else []
    overall = _aggregate(per_question, ["model"], metric_cols)
    by_type = _aggregate(per_question, ["model", "question_type"], metric_cols)
    by_topic = _aggregate(per_question, ["model", "geological_topic"], metric_cols)
    by_difficulty = _aggregate(per_question, ["model", "difficulty"], metric_cols)
    by_answerable = _aggregate(per_question, ["model", "answerable"], metric_cols)

    write_csv(out / "overall_retrieval_metrics.csv", overall)
    write_csv(out / "retrieval_metrics_by_question_type.csv", by_type)
    write_csv(out / "retrieval_metrics_by_geological_topic.csv", by_topic)
    write_csv(out / "retrieval_metrics_by_difficulty.csv", by_difficulty)
    write_csv(out / "retrieval_metrics_by_answerability.csv", by_answerable)
    return {
        "per_question": per_question,
        "overall": overall,
        "by_question_type": by_type,
        "by_geological_topic": by_topic,
        "by_difficulty": by_difficulty,
        "by_answerable": by_answerable,
    }


def inspect_gold(gold_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qids = list(gold_rows.keys())
    source_ids = [str(r.get("source_sentence_id", "")).strip() for r in gold_rows.values() if r.get("source_sentence_id")]
    return {
        "total_questions": len(gold_rows),
        "answerable": Counter(bool(r.get("answerable", True)) for r in gold_rows.values()),
        "question_type": Counter(str(r.get("question_type", "")) for r in gold_rows.values()),
        "geological_topic": Counter(str(r.get("geological_topic", "")) for r in gold_rows.values()),
        "difficulty": Counter(str(r.get("difficulty", "")) for r in gold_rows.values()),
        "empty_strict_gold": sum(1 for r in gold_rows.values() if not r.get("_strict_gold")),
        "empty_context_gold": sum(1 for r in gold_rows.values() if not r.get("_context_gold")),
        "empty_relevance_judgments": sum(1 for r in gold_rows.values() if not r.get("_relevance_judgments")),
        "duplicate_question_ids": [qid for qid, c in Counter(qids).items() if c > 1],
        "duplicate_source_sentence_ids": [sid for sid, c in Counter(source_ids).items() if c > 1],
    }
