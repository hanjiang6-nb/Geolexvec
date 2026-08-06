from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .metrics import dedupe_doc_ids

DOC_ID_KEYS = ("doc_id", "source_evidence_id", "sentence_id", "source_sentence_id")
SKIP_KEYS = {
    "question_id",
    "query_id",
    "question",
    "question_type",
    "geological_topic",
    "difficulty",
    "answerable",
    "reference_answer",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field} must be a JSON boolean or the string 'true'/'false'")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


_SENT_ID_RE = re.compile(r"^(?P<doc>.+)_sent_(?P<num>\d+)$")
_EVIDENCE_ID_RE = re.compile(r"^(?P<doc>.+)__e(?P<num>\d+)$")


def normalize_question_id(qid: Any) -> str:
    value = str(qid).strip()
    match = re.fullmatch(r"q0*(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"q{int(match.group(1)):04d}"
    return value


def normalize_evidence_id(doc_id: Any) -> str:
    value = str(doc_id).strip()
    if not value:
        return ""
    match = _SENT_ID_RE.fullmatch(value)
    if match:
        return f"{match.group('doc')}__e{int(match.group('num')):06d}"
    match = _EVIDENCE_ID_RE.fullmatch(value)
    if match:
        return f"{match.group('doc')}__e{int(match.group('num')):06d}"
    return value


def load_gold(path: str | Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row.get("question_id") or row.get("query_id")
        if not qid:
            raise ValueError("gold row is missing question_id/query_id")
        qid = normalize_question_id(qid)
        if qid in gold:
            raise ValueError(f"duplicate question_id in gold: {qid}")
        strict = [normalize_evidence_id(x) for x in as_list(row.get("gold_evidence_ids_strict"))]
        context = [normalize_evidence_id(x) for x in as_list(row.get("gold_evidence_ids_context"))] or strict
        judgments = row.get("relevance_judgments") or {}
        if not isinstance(judgments, dict):
            raise ValueError(f"{qid} relevance_judgments must be an object")
        answerable = as_bool(row.get("answerable", True), field=f"{qid}.answerable")
        if answerable and not strict:
            raise ValueError(f"{qid} answerable=true but strict gold is empty")
        row = dict(row)
        row["answerable"] = answerable
        row["_strict_gold"] = dedupe_doc_ids(strict)
        row["_context_gold"] = dedupe_doc_ids(context)
        row["_relevance_judgments"] = {
            normalize_evidence_id(k): float(v) for k, v in judgments.items() if str(k).strip()
        }
        gold[qid] = row
    return gold


def extract_doc_id(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return normalize_evidence_id(value) if value else None
    if not isinstance(item, dict):
        return None
    for key in DOC_ID_KEYS:
        value = item.get(key)
        if value is not None and str(value).strip():
            return normalize_evidence_id(value)
    return None


def extract_retrieved_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("retrieved", "evidence"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for idx, raw in enumerate(value, start=1):
        doc_id = extract_doc_id(raw)
        if not doc_id:
            continue
        score = 0.0
        rank = idx
        if isinstance(raw, dict):
            rank = int(raw.get("rank") or idx)
            try:
                score = float(raw.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
        items.append({"doc_id": doc_id, "rank": rank, "score": score})
    items.sort(key=lambda x: (x["rank"],))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if item["doc_id"] in seen:
            continue
        seen.add(item["doc_id"])
        item["rank"] = len(deduped) + 1
        deduped.append(item)
    return deduped


def model_name_from_path(path: str | Path) -> str:
    return Path(path).stem.replace("_retrieval", "").replace("_", " ").title().replace(" ", "")


def load_runs(
    paths: list[str | Path],
    model_names: list[str] | None = None,
    valid_question_ids: set[str] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for file_idx, path in enumerate(paths):
        fallback = (
            model_names[file_idx]
            if model_names and file_idx < len(model_names)
            else model_name_from_path(path)
        )
        for row in read_jsonl(path):
            qid = row.get("question_id") or row.get("query_id")
            if not qid:
                raise ValueError(f"{path}: run row is missing question_id/query_id")
            qid = normalize_question_id(qid)
            if valid_question_ids is not None and qid not in valid_question_ids:
                raise ValueError(f"{path}: question_id is not in gold: {qid}")
            if isinstance(row.get("retrieved"), list):
                model = str(fallback if model_names else row.get("model") or fallback)
                if qid in runs.setdefault(model, {}):
                    raise ValueError(f"{path}: duplicate run row for {model}/{qid}")
                runs.setdefault(model, {})[qid] = extract_retrieved_items(row["retrieved"])
                continue
            found = False
            for key, value in row.items():
                if key in SKIP_KEYS or not isinstance(value, dict):
                    continue
                if isinstance(value.get("evidence"), list) or isinstance(value.get("retrieved"), list):
                    if model_names or len(paths) > 1:
                        model = f"{fallback}_{key}"
                    else:
                        model = key
                    if qid in runs.setdefault(model, {}):
                        raise ValueError(f"{path}: duplicate run row for {model}/{qid}")
                    runs[model][qid] = extract_retrieved_items(value)
                    found = True
            if not found and isinstance(row.get("evidence"), list):
                if qid in runs.setdefault(fallback, {}):
                    raise ValueError(f"{path}: duplicate run row for {fallback}/{qid}")
                runs.setdefault(fallback, {})[qid] = extract_retrieved_items(row["evidence"])
                found = True
            if not found:
                raise ValueError(f"{path}: no retrieval list found for question {qid}")
    if valid_question_ids is not None:
        for model, by_qid in runs.items():
            missing = valid_question_ids - set(by_qid)
            if missing:
                sample = ", ".join(sorted(missing)[:5])
                raise ValueError(f"{model} is missing {len(missing)} gold questions: {sample}")
    return runs
