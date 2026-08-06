from __future__ import annotations

import hashlib
import json
import pickle
import re
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (20260725, 20260726, 20260727)
EXCLUDED_DIRS = {
    ".git", ".venv", ".pytest_cache", ".pytest_tmp", "__pycache__",
    "reproduced", "reproduced_tables", "ci_reproduced_tables",
}
TEXT_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitattributes", ".gitignore"}:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in root.rglob("*.json"):
        if EXCLUDED_DIRS.intersection(path.relative_to(root).parts):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON: {path.relative_to(root)}: {exc}")

    for path in root.rglob("*.jsonl"):
        if EXCLUDED_DIRS.intersection(path.relative_to(root).parts):
            continue
        try:
            rows = load_jsonl(path)
            if not all(isinstance(row, dict) for row in rows):
                failures.append(f"non-object JSONL row: {path.relative_to(root)}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSONL: {path.relative_to(root)}: {exc}")

    for path in root.rglob("*.py"):
        if EXCLUDED_DIRS.intersection(path.relative_to(root).parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"Python compile: {path.relative_to(root)}: {exc}")

    checksum_path = root / "manifest/SHA256SUMS.txt"
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"checksum: {relative}")

    expected_jsonl = {
        "data/evaluation_gold.jsonl": 2034,
        "data/raw/benchmark/questions_2034.jsonl": 2034,
        "data/raw/benchmark/evidence_1413.jsonl": 1413,
        "data/raw/benchmark/articles_356.jsonl": 356,
    }
    for relative, expected in expected_jsonl.items():
        path = root / relative
        if not path.is_file() or jsonl_count(path) != expected:
            failures.append(f"row count: {relative}")

    evidence_rows = load_jsonl(root / "data/raw/benchmark/evidence_1413.jsonl")
    article_rows = load_jsonl(root / "data/raw/benchmark/articles_356.jsonl")
    evidence_ids = [str(row.get("id") or "") for row in evidence_rows]
    article_ids = {str(row.get("article_id") or "") for row in article_rows}
    expected_evidence_schema = {"id", "source_id", "article_id", "text"}
    if len(set(evidence_ids)) != 1413:
        failures.append("unique evidence IDs")
    if any(set(row) != expected_evidence_schema for row in evidence_rows):
        failures.append("evidence four-field schema")
    if any(not str(row.get("text") or "").strip() for row in evidence_rows):
        failures.append("empty evidence text")
    if any(str(row.get("article_id") or "") not in article_ids for row in evidence_rows):
        failures.append("unresolved evidence article link")

    for relative in (
        "data/raw/entity_training/indexed_docs.pkl",
        "data/raw/entity_training/contextual_index.pkl",
    ):
        path = root / relative
        with path.open("rb") as handle:
            index_rows = pickle.load(handle)
        if not isinstance(index_rows, list) or len(index_rows) != 1413:
            failures.append(f"entity index row count: {relative}")
            continue
        for position, (index_row, evidence_row) in enumerate(
            zip(index_rows, evidence_rows)
        ):
            if str(index_row.get("doc_id") or "") != str(evidence_row["id"]):
                failures.append(f"entity index evidence ID: {relative}:{position}")
                break
            if str(index_row.get("article_id") or "") != str(evidence_row["article_id"]):
                failures.append(f"entity index article ID: {relative}:{position}")
                break
            if str(index_row.get("text") or "") != str(evidence_row["text"]):
                failures.append(f"entity index text order: {relative}:{position}")
                break

    repair_report = json.loads(
        (root / "data/raw/entity_training/index_id_repair_report.json").read_text(
            encoding="utf-8"
        )
    )
    if repair_report.get("schema") != "geolexvec.entity-index-id-repair.v1":
        failures.append("entity index repair report schema")
    if any(row.get("unknown_ids_after") != 0 for row in repair_report.get("indexes", [])):
        failures.append("entity index repair retained placeholder IDs")

    aliases = json.loads(
        (root / "data/raw/entity_training/en_zh_equivalence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    accepted = [row for row in aliases.get("mappings", []) if row.get("accepted")]
    if aliases.get("accepted_mapping_count") != 62 or len(accepted) != 62:
        failures.append("accepted alias count")

    phrase_groups = json.loads(
        (root / "data/config/keyword_groups.json").read_text(encoding="utf-8-sig")
    )
    if not phrase_groups or not all(isinstance(group, list) and group for group in phrase_groups):
        failures.append("geological phrase groups")
    if any(keyword != keyword.strip() for group in phrase_groups for keyword in group):
        failures.append("phrase keyword has leading/trailing whitespace")

    splits = json.loads(
        (root / "data/splits/source_grouped_5x4.json").read_text(encoding="utf-8")
    )
    if splits.get("question_count") != 2034 or len(splits.get("folds", [])) != 5:
        failures.append("fixed 5x4 split manifest")
    outer_test_ids = [qid for fold in splits.get("folds", []) for qid in fold["test_question_ids"]]
    if len(outer_test_ids) != 2034 or len(set(outer_test_ids)) != 2034:
        failures.append("outer-fold query partition")

    for seed in SEEDS:
        path = root / f"data/feature_caches/seed_{seed}_sentence_variants.npz"
        protocol_path = root / f"data/feature_caches/seed_{seed}_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if path.read_bytes()[:32].startswith(b"version https://git-lfs"):
            failures.append(f"missing Git LFS object: {seed}")
            continue
        if sha256(path) != protocol["output_sha256"]:
            failures.append(f"feature-cache hash: {seed}")
        with np.load(path, allow_pickle=False) as cache:
            if [str(value) for value in cache["variant_names"]] != ["AliasAwareFusion"]:
                failures.append(f"entity variant: {seed}")
            for key in ("text_features", "context_entity_features", "phrase_features", "relevance"):
                if cache[key].shape != (2034, 1413):
                    failures.append(f"{key} shape: {seed}")
            if [str(value) for value in cache["evidence_ids"]] != evidence_ids:
                failures.append(f"evidence ID order: {seed}")
            cache_checkpoint_hash = str(cache["entity_checkpoint_sha256"].item())
            cache_training_protocol = json.loads(
                str(cache["entity_training_protocol_json"].item())
            )
            if bool(
                cache["entity_held_out_occurrences_used_for_optimization"].item()
            ):
                failures.append(f"feature-cache held-out leakage flag: {seed}")
        predictions = pd.read_csv(
            root / f"results/three_module_seeds/seed_{seed}/per_question_metrics.csv"
        )
        if len(predictions) != 2034 * 4 or predictions["question_id"].nunique() != 2034:
            failures.append(f"three-module predictions: {seed}")
        report = json.loads(
            (root / f"data/processed/entity_vectors/seed_{seed}/report.json").read_text(
                encoding="utf-8"
            )
        )
        if report.get("schema") != "geolexvec.entity-adapter.report.v1":
            failures.append(f"entity report schema: {seed}")
        report_protocol = report.get("protocol", {})
        if report_protocol.get("schema") != "geolexvec.entity-adapter.training.v1":
            failures.append(f"entity training protocol schema: {seed}")
        if report_protocol.get("anchor_split") != "train_only":
            failures.append(f"entity report anchor split: {seed}")
        if report_protocol.get("held_out_occurrences_used_for_optimization") is not False:
            failures.append(f"held-out occurrence leakage: {seed}")
        if report_protocol.get("training_anchor_count") != 6605:
            failures.append(f"entity training anchor count: {seed}")
        if len(str(report_protocol.get("training_anchor_sha256", ""))) != 64:
            failures.append(f"entity training anchor fingerprint: {seed}")
        training_inputs = {
            "aliases": root / "data/raw/entity_training/en_zh_equivalence_audit.json",
            "occurrences": root / "data/raw/entity_training/context_occurrences.npy",
            "metadata": root / "data/raw/entity_training/context_occurrences_metadata.json",
            "vectors": root / "data/raw/entity_training/attn_vectors.npy",
            "index": root / "data/raw/entity_training/indexed_docs.pkl",
        }
        input_hashes = report_protocol.get("input_sha256", {})
        for name, input_path in training_inputs.items():
            if input_hashes.get(name) != raw_sha256(input_path):
                failures.append(f"entity training input provenance: {seed}:{name}")
        checkpoint_path = (
            root
            / f"data/processed/entity_vectors/seed_{seed}/canonical_alias_encoder.pt"
        )
        checkpoint_hash = raw_sha256(checkpoint_path)
        if report.get("artifact_sha256", {}).get(checkpoint_path.name) != checkpoint_hash:
            failures.append(f"entity checkpoint report hash: {seed}")
        if protocol.get("checkpoint_sha256") != checkpoint_hash:
            failures.append(f"feature-cache checkpoint provenance: {seed}")
        if protocol.get("training_protocol") != report_protocol:
            failures.append(f"feature-cache training protocol: {seed}")
        if cache_checkpoint_hash != checkpoint_hash:
            failures.append(f"embedded feature-cache checkpoint provenance: {seed}")
        if cache_training_protocol != report_protocol:
            failures.append(f"embedded feature-cache training protocol: {seed}")

    for variant in ("residual_only", "prototype_only"):
        for seed in SEEDS:
            directory = (
                root
                / f"data/processed/entity_vectors/internal_ablation/{variant}/seed_{seed}"
            )
            checkpoint_path = directory / "checkpoint.pt"
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") != "geolexvec.entity-adapter.internal-report.v1":
                failures.append(f"internal report schema: {variant}:{seed}")
            internal_protocol = report.get("protocol", {})
            if internal_protocol.get("variant") != variant:
                failures.append(f"internal report variant: {variant}:{seed}")
            if internal_protocol.get("anchor_split") != "train_only":
                failures.append(f"internal anchor split: {variant}:{seed}")
            if internal_protocol.get("held_out_occurrences_used_for_optimization") is not False:
                failures.append(f"internal held-out leakage: {variant}:{seed}")
            if report.get("checkpoint_sha256") != raw_sha256(checkpoint_path):
                failures.append(f"internal checkpoint hash: {variant}:{seed}")

    internal = pd.read_csv(root / "results/entity_internal_ablation/per_question_metrics.csv")
    if len(internal) != 2034 * 4 or internal["question_id"].nunique() != 2034:
        failures.append("entity-internal predictions")

    required_code = (
        "experiments/entity_alias_metric_learning/entity_training_utils.py",
        "experiments/entity_alias_metric_learning/train_canonical_alias_encoder.py",
        "experiments/entity_alias_metric_learning/transform_canonical_context_vectors.py",
        "experiments/final_alias_aware_study/build_context_entity_cache.py",
        "experiments/final_alias_aware_study/run_three_module_ablation.py",
        "experiments/final_alias_aware_study/run_entity_internal_ablation.py",
        "geolexvec/nested_search.py",
        "geolexvec/entity_aliases.py",
        "geolexvec/entity_provenance.py",
        "geolexvec/validation.py",
        "scripts/reproduce_main.py",
        "scripts/reproduce_tables.py",
        "scripts/repair_entity_index_ids.py",
        "scripts/assemble_entity_internal_results.py",
    )
    for relative in required_code:
        if not (root / relative).is_file():
            failures.append(f"missing code: {relative}")

    forbidden_legacy = (
        "experiments/baselines",
        "experiments/joint_entity_model",
        "experiments/entity_alias_metric_learning/train_alias_projection.py",
        "experiments/entity_alias_metric_learning/train_alias_occurrence_projection.py",
    )
    for relative in forbidden_legacy:
        if (root / relative).exists():
            failures.append(f"legacy code retained: {relative}")

    patterns = (
        re.compile(r"connect\.(?:westd|bjb2)\.seetacloud\.com", re.I),
        re.compile(r"ssh\s+-p\s+\d+\s+root@", re.I),
        re.compile(r"/root/autodl|C:\\Users\\Administrator", re.I),
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        if EXCLUDED_DIRS.intersection(path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in {
            ".py", ".md", ".json", ".jsonl", ".txt", ".toml", ".yml", ".yaml",
            ".sh", ".ps1", ".bat", ".cmd", ".env",
        }:
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        if any(pattern.search(content) for pattern in patterns):
            failures.append(f"sensitive path: {path.relative_to(root)}")

    if failures:
        print(json.dumps({"status": "failed", "failures": sorted(set(failures))}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
