from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", "", text).strip("。.;；")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def repair(path: Path, evidence: list[dict[str, Any]], write: bool) -> dict[str, Any]:
    before_hash = sha256(path)
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list) or len(rows) != len(evidence):
        raise ValueError(f"{path}: entity index and benchmark have different lengths")
    keys_before = [set(row) for row in rows]
    unknown_before = 0
    changed = 0
    for position, (row, benchmark_row) in enumerate(zip(rows, evidence)):
        if normalized_text(row.get("text", "")) != normalized_text(
            benchmark_row.get("text", "")
        ):
            raise ValueError(f"{path}: text mismatch at candidate position {position}")
        if "unknown_doc_" in str(row.get("doc_id", "")):
            unknown_before += 1
        expected_doc_id = str(benchmark_row["id"])
        expected_article_id = str(benchmark_row["article_id"])
        if row.get("doc_id") != expected_doc_id or row.get("article_id") != expected_article_id:
            changed += 1
        row["doc_id"] = expected_doc_id
        row["article_id"] = expected_article_id
    if any(set(row) != keys for row, keys in zip(rows, keys_before)):
        raise AssertionError("entity-index schema changed during ID repair")
    if write:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(rows),
        "changed_ids": changed,
        "unknown_ids_before": unknown_before,
        "unknown_ids_after": sum(
            "unknown_doc_" in str(row.get("doc_id", "")) for row in rows
        ),
        "sha256_before": before_hash,
        "sha256_after": sha256(path) if write else before_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align entity-index document IDs with the released evidence IDs."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    evidence = load_jsonl(args.evidence)
    reports = [repair(path, evidence, args.write) for path in args.index]
    payload = {
        "schema": "geolexvec.entity-index-id-repair.v1",
        "matching_rule": "candidate-order plus normalized exact sentence text",
        "evidence_rows": len(evidence),
        "write_applied": args.write,
        "indexes": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
