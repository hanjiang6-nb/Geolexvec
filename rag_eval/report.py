from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


QUESTION_TYPE_DISPLAY = {
    "事实型问题": "事实检索类",
    "机制型问题": "机制因果检索类",
    "对比型问题": "对比差异检索类",
    "证据归纳型问题": "证据归纳类",
    "条件限定型问题": "条件限定检索类",
    "多跳推理型问题": "多证据组合类",
    "反事实/不可回答问题": "证据充分性判断类",
}


def _display_question_type(value: Any) -> str:
    text = str(value or "")
    return QUESTION_TYPE_DISPLAY.get(text, text)


def write_summary(
    path: str | Path,
    gold_rows: dict[str, dict[str, Any]],
    results: dict[str, Any],
    k_list: list[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answerable = Counter(bool(r.get("answerable", True)) for r in gold_rows.values())
    qtype = Counter(_display_question_type(r.get("question_type", "")) for r in gold_rows.values())
    overall = results.get("overall", [])
    by_type = results.get("by_question_type", [])

    lines = [
        "# 煤地质知识挖掘检索评估汇总",
        "",
        f"- 问题总数：{len(gold_rows)}",
        f"- 可回答问题：{answerable.get(True, 0)}",
        f"- 不可回答问题：{answerable.get(False, 0)}",
        "",
        "## 问题类型分布",
        "",
        _md_table([{"问题类型": k, "数量": v} for k, v in qtype.items()], ["问题类型", "数量"]),
        "",
        "## 主要指标",
        "",
    ]

    main_cols = ["model", "n_questions"]
    for k in [1, 3, 5, 8, 10]:
        if k in k_list:
            main_cols.extend([f"Strict-Hit@{k}", f"Context-Recall@{k}", f"nDCG@{k}"])
    lines.append(_md_table(overall, [c for c in main_cols if any(c in r for r in overall)]))

    lines.extend(["", "## Strict-Hit 对比", ""])
    strict_cols = ["model"] + [f"Strict-Hit@{k}" for k in [1, 3, 5, 8] if k in k_list]
    lines.append(_md_table(overall, strict_cols))

    lines.extend(["", "## Context-Recall 对比", ""])
    context_cols = ["model"] + [f"Context-Recall@{k}" for k in [1, 3, 5, 8] if k in k_list]
    lines.append(_md_table(overall, context_cols))

    if 10 in k_list:
        lines.extend(["", "## nDCG@10 对比", ""])
        lines.append(_md_table(overall, ["model", "nDCG@10"]))

    lines.extend(["", "## 按问题类型的最佳模型", ""])
    best_rows = []
    for question_type in sorted({r.get("question_type", "") for r in by_type}):
        candidates = [r for r in by_type if r.get("question_type") == question_type]
        if not candidates:
            continue
        metric = "Context-Recall@8" if "Context-Recall@8" in candidates[0] else "nDCG@10"
        best = max(candidates, key=lambda r: float(r.get(metric, 0.0) or 0.0))
        best_rows.append(
            {
                "问题类型": _display_question_type(question_type),
                "最佳模型": best.get("model", ""),
                "指标": metric,
                "分数": best.get(metric, 0.0),
            }
        )
    lines.append(_md_table(best_rows, ["问题类型", "最佳模型", "指标", "分数"]))

    lines.extend(
        [
            "",
            "## 简要结论",
            "",
            "Strict 指标衡量模型是否检索到直接支持地质结论的 silver-gold 证据。Context 指标允许相邻或补充证据进入评价，更接近知识挖掘场景中对证据背景的需求。nDCG 使用分级相关性，并奖励模型把强相关煤地质证据排在更靠前的位置。该评估不依赖 LLM 回答是否拒答，因此能够更稳定地衡量检索系统定位支持性证据的能力。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gold_inspection(path: str | Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Gold 问题文件检查",
        "",
        f"- 问题总数：{stats['total_questions']}",
        f"- strict gold 为空数量：{stats['empty_strict_gold']}",
        f"- context gold 为空数量：{stats['empty_context_gold']}",
        f"- relevance_judgments 为空数量：{stats['empty_relevance_judgments']}",
        f"- 重复 question_id 数量：{len(stats['duplicate_question_ids'])}",
        f"- 重复 source_sentence_id 数量：{len(stats['duplicate_source_sentence_ids'])}",
        "",
        "## 可回答性分布",
        "",
        _md_table([{"answerable": k, "count": v} for k, v in stats["answerable"].items()], ["answerable", "count"]),
        "",
        "## 问题类型分布",
        "",
        _md_table(
            [{"问题类型": _display_question_type(k), "数量": v} for k, v in stats["question_type"].items()],
            ["问题类型", "数量"],
        ),
        "",
        "## 难度分布",
        "",
        _md_table([{"difficulty": k, "count": v} for k, v in stats["difficulty"].items()], ["difficulty", "count"]),
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
