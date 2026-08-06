from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_file_stem(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum():
            chars.append(ch.lower())
        elif ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "model"


def write_qrels(gold_rows: dict[str, dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strict_path = out / "qrels_strict.txt"
    graded_path = out / "qrels_graded.txt"
    strict_lines: list[str] = []
    graded_lines: list[str] = []
    for qid, row in sorted(gold_rows.items()):
        for doc_id in row.get("_strict_gold", []):
            strict_lines.append(f"{qid} 0 {doc_id} 3")
        judgments = row.get("_relevance_judgments", {})
        for doc_id, rel in sorted(judgments.items()):
            graded_lines.append(f"{qid} 0 {doc_id} {int(float(rel))}")
    strict_path.write_text("\n".join(strict_lines) + ("\n" if strict_lines else ""), encoding="utf-8")
    graded_path.write_text("\n".join(graded_lines) + ("\n" if graded_lines else ""), encoding="utf-8")
    return strict_path, graded_path


def write_run_files(
    runs: dict[str, dict[str, list[dict[str, Any]]]],
    out_dir: str | Path,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model, by_qid in sorted(runs.items()):
        safe_model = safe_file_stem(model)
        path = out / f"{safe_model}.run.txt"
        lines: list[str] = []
        for qid, items in sorted(by_qid.items()):
            for rank, item in enumerate(items, start=1):
                lines.append(f"{qid} Q0 {item['doc_id']} {rank} {item.get('score', 0.0)} {safe_model}")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        paths.append(path)
    return paths
