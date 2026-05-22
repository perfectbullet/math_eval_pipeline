#!/usr/bin/env python3
"""
综合归因报告脚本：合并 Math-Verify 和 PRM 结果，生成错误归因报告。

归因标签:
- pass: Math-Verify 正确
- logic_error: Math-Verify 错 + PRM 低分
- answer_or_render_suspect: Math-Verify 错 + PRM 高分
- answer_extract_error: Math-Verify 错 + 无法提取答案
- uncertain: 无法确定

用法:
    python build_error_report.py \
        --verify ../results/verify/verify_results.jsonl \
        --prm ../results/prm/prm_step_scores.jsonl \
        --output_md ../results/reports/error_attribution_report.md \
        --output_jsonl ../results/reports/error_cases.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL 文件。"""
    records = []
    if not path.exists():
        print(f"[WARN] 文件不存在: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_error(verify_record: dict, prm_map: dict) -> str:
    """根据 verify 和 PRM 结果归因。"""
    if verify_record.get("verify_correct", False):
        return "pass"

    qid = verify_record.get("id", "")

    # 无法提取答案
    if verify_record.get("extracted_prediction", "") == "":
        return "answer_extract_error"

    # 有 PRM 结果
    prm = prm_map.get(qid)
    if prm:
        category = prm.get("error_category", "")
        if category in ("logic_error", "answer_or_render_suspect", "uncertain_logic"):
            return category
        # 根据 PRM 逻辑正确性判断
        if prm.get("prm_logic_correct", False):
            return "answer_or_render_suspect"
        else:
            return "logic_error"

    # 没有 PRM 结果
    if verify_record.get("need_prm", False):
        return "uncertain"

    return "uncertain"


def main():
    parser = argparse.ArgumentParser(description="综合归因报告")
    parser.add_argument("--verify", required=True, help="Math-Verify 结果 JSONL")
    parser.add_argument("--prm", required=True, help="PRM 结果 JSONL")
    parser.add_argument("--output_md", required=True, help="输出 Markdown 报告")
    parser.add_argument("--output_jsonl", required=True, help="输出 JSONL 错误案例")
    args = parser.parse_args()

    verify_path = Path(args.verify).expanduser()
    prm_path = Path(args.prm).expanduser()
    output_md = Path(args.output_md).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 加载数据
    verify_records = load_jsonl(verify_path)
    prm_records = load_jsonl(prm_path)

    print(f"Math-Verify 记录: {len(verify_records)}")
    print(f"PRM 记录: {len(prm_records)}")

    # PRM 按 ID 索引
    prm_map = {r["id"]: r for r in prm_records}

    # 归因
    categories = defaultdict(list)
    error_cases = []

    for vr in verify_records:
        category = classify_error(vr, prm_map)
        categories[category].append(vr)

        if category != "pass":
            case = {
                "id": vr.get("id", ""),
                "source": vr.get("source", ""),
                "model_name": vr.get("model_name", ""),
                "question": vr.get("question", "")[:200],
                "reference_answer": vr.get("reference_answer", ""),
                "extracted_prediction": vr.get("extracted_prediction", ""),
                "verify_correct": vr.get("verify_correct", False),
                "verify_error": vr.get("verify_error", ""),
                "answer_type": vr.get("answer_type", ""),
                "category": category,
            }

            # 附加 PRM 信息
            prm = prm_map.get(vr.get("id", ""))
            if prm:
                case["prm_min_score"] = prm.get("min_step_score", 0)
                case["prm_avg_score"] = prm.get("avg_step_score", 0)
                case["first_wrong_step"] = prm.get("first_wrong_step_index", -1)

            error_cases.append(case)

    # 写错误案例 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for case in error_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # 生成报告
    total = len(verify_records)
    correct = len(categories.get("pass", []))
    accuracy = correct / total * 100 if total > 0 else 0

    lines = [
        "# 错误归因报告\n",
        "## 总体概览",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总样本数 | {total} |",
        f"| Math-Verify 正确 | {correct} |",
        f"| Math-Verify 准确率 | {accuracy:.2f}% |",
        f"| 错误样本总数 | {total - correct} |",
        "",
        "## 归因分布",
        "",
        "| 归因类别 | 数量 | 占比 | 说明 |",
        "|----------|------|------|------|",
    ]

    category_desc = {
        "pass": "验证通过",
        "logic_error": "解题逻辑错误（PRM 低分）",
        "answer_or_render_suspect": "疑似答案抽取/渲染/最终汇总问题（PRM 高分但答案错）",
        "answer_extract_error": "答案抽取失败",
        "uncertain": "不确定，需进一步分析",
    }

    for cat in ["pass", "logic_error", "answer_or_render_suspect", "answer_extract_error", "uncertain"]:
        count = len(categories.get(cat, []))
        pct = count / total * 100 if total > 0 else 0
        desc = category_desc.get(cat, "")
        lines.append(f"| {cat} | {count} | {pct:.1f}% | {desc} |")

    # 按来源分组
    by_source = defaultdict(lambda: defaultdict(int))
    for cat, items in categories.items():
        for item in items:
            by_source[item.get("source", "unknown")][cat] += 1

    lines.extend(["", "## 按来源分组", "", "| 来源 | pass | logic_error | render_suspect | extract_error | uncertain | 合计 |", "|------|-------|-------------|----------------|--------------|-----------|------|"])
    for source in sorted(by_source.keys()):
        cats = by_source[source]
        total_s = sum(cats.values())
        lines.append(
            f"| {source} | {cats.get('pass', 0)} | {cats.get('logic_error', 0)} | "
            f"{cats.get('answer_or_render_suspect', 0)} | {cats.get('answer_extract_error', 0)} | "
            f"{cats.get('uncertain', 0)} | {total_s} |"
        )

    # PRM 低分样本
    logic_errors = categories.get("logic_error", [])
    if logic_errors:
        lines.extend(["", "## 解题逻辑错误样本 (Top 20)", ""])
        for item in logic_errors[:20]:
            qid = item.get("id", "")
            ref = item.get("reference_answer", "")
            pred = item.get("extracted_prediction", "")
            lines.append(f"- **{qid}**: ref=`{ref}`, pred=`{pred}`")

    # 疑似渲染问题
    render_suspects = categories.get("answer_or_render_suspect", [])
    if render_suspects:
        lines.extend(["", "## 疑似渲染/答案抽取问题样本 (Top 20)", ""])
        for item in render_suspects[:20]:
            qid = item.get("id", "")
            ref = item.get("reference_answer", "")
            pred = item.get("extracted_prediction", "")
            lines.append(f"- **{qid}**: ref=`{ref}`, pred=`{pred}`")

    # 后续建议
    lines.extend([
        "",
        "## 后续建议",
        "",
    ])

    if render_suspects:
        lines.append(f"1. **{len(render_suspects)}** 条疑似渲染/答案抽取问题样本，建议用 Playwright 做视觉检查")
    if logic_errors:
        lines.append(f"2. **{len(logic_errors)}** 条解题逻辑错误样本，建议换模型重新评测")
    uncertain_count = len(categories.get("uncertain", []))
    if uncertain_count:
        lines.append(f"3. **{uncertain_count}** 条不确定样本，建议人工抽样分析")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n错误案例: {output_jsonl} ({len(error_cases)} 条)")
    print(f"归因报告: {output_md}")


if __name__ == "__main__":
    main()
