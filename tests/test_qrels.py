from __future__ import annotations

from rag_eval.qrels import write_qrels
from rag_eval.evaluator import evaluate
from rag_eval.io_utils import as_bool, extract_doc_id, read_jsonl


def test_write_qrels(tmp_path) -> None:
    gold = {
        "q1": {
            "_strict_gold": ["d1"],
            "_relevance_judgments": {"d1": 3, "d2": 2},
        }
    }
    strict, graded = write_qrels(gold, tmp_path)
    assert strict.read_text(encoding="utf-8").strip() == "q1 0 d1 3"
    assert "q1 0 d2 2" in graded.read_text(encoding="utf-8")


def test_string_false_is_not_truthy() -> None:
    assert as_bool("false", field="answerable") is False
    assert as_bool("true", field="answerable") is True


def test_malformed_jsonl_fails_fast(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"question_id":"q1"}\nnot-json\n', encoding="utf-8")
    try:
        read_jsonl(path)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("malformed JSONL must not be silently skipped")


def test_evaluator_rejects_incomplete_runs(tmp_path) -> None:
    gold = {
        "q1": {"_strict_gold": ["d1"], "_context_gold": ["d1"], "_relevance_judgments": {"d1": 3}},
        "q2": {"_strict_gold": ["d2"], "_context_gold": ["d2"], "_relevance_judgments": {"d2": 3}},
    }
    runs = {"model": {"q1": [{"doc_id": "d1", "rank": 1, "score": 1.0}]}}
    try:
        evaluate(gold, runs, [1], tmp_path)
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("incomplete runs must not be evaluated")


def test_string_run_item_uses_the_same_id_normalization_as_dict_item() -> None:
    expected = "doc_1__e000001"
    assert extract_doc_id("doc_1_sent_1") == expected
    assert extract_doc_id({"doc_id": "doc_1_sent_1"}) == expected
