from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_every_evidence_record_links_to_a_released_article() -> None:
    evidence = load_jsonl("data/raw/benchmark/evidence_1413.jsonl")
    articles = load_jsonl("data/raw/benchmark/articles_356.jsonl")
    article_ids = {row["article_id"] for row in articles}
    assert len(evidence) == 1413
    assert len(articles) == 356
    assert all(row["article_id"] in article_ids for row in evidence)


def test_evidence_schema_and_ids_are_clean() -> None:
    evidence = load_jsonl("data/raw/benchmark/evidence_1413.jsonl")
    assert all(set(row) == {"id", "source_id", "article_id", "text"} for row in evidence)
    assert len({row["id"] for row in evidence}) == 1413
    placeholder = "unknown" + "_doc_"
    assert not any(placeholder in json.dumps(row, ensure_ascii=False) for row in evidence)
