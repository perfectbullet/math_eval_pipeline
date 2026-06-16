#!/usr/bin/env python3
"""
导出 Math-Verify + PRM 汇总指标，输出 CSV 和 JSON。

用法:
    # 单组
    python export_verify_prm_metrics.py \
        --verify ../results/verify/verify_results.jsonl \
        --prm ../results/prm/prm_step_scores.jsonl \
        --name tir_en \
        --output_csv reports/verify_prm_metrics.csv \
        --output_json reports/verify_prm_metrics.json

    # 多组
    python export_verify_prm_metrics.py \
        --case tir_en ../results/verify/verify_tir_en.jsonl ../results/prm/prm_tir_en.jsonl \
        --case cot_en ../results/verify/verify_cot_en.jsonl ../results/prm/prm_cot_en.jsonl \
        --output_csv reports/verify_prm_metrics.csv \
        --output_json reports/verify_prm_metrics.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WRONG_STEP_THRESHOLD = 0.45
LOGIC_CORRECT_THRESHOLD = 0.70

FIELDS = [
    "实验名称",
    "总样本数",
    "Verify正确数",
    "Verify错误数",
    "Verify准确率",
    "choice(选择题，answer_type )正确数",
    "choice(选择题，answer_type )错误数",
    "choice(选择题，answer_type )准确率",
    "PRM样本总数",
    "疑似渲染/答案抽取问题",
    "不确定",
    "PRM 高分但答案错数量",
    "修正后的准去率（把PRM高分但答案错的归为正确的）",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """加载 JSONL 文件。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON: {e}") from e
    return records


def as_bool(value: Any) -> bool:
    """兼容 bool / 字符串形式的正确性字段。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def percent(part: int, total: int) -> float:
    """返回百分比数值，保留两位小数。"""
    if total <= 0:
        return 0.0
    return round(part / total * 100, 2)


def get_answer_type(record: dict[str, Any]) -> str:
    """从 verify 结果中读取 answer_type，兼容原始 metadata。"""
    answer_type = record.get("answer_type")
    if answer_type:
        return str(answer_type)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("answer_type", ""))
    return ""


def is_uncertain_prm(record: dict[str, Any]) -> bool:
    """判断 PRM 结果是否属于不确定。"""
    category = str(record.get("error_category", ""))
    return category in {"uncertain", "uncertain_logic"}


def is_render_or_extract_suspect(record: dict[str, Any]) -> bool:
    """判断 PRM 结果是否属于疑似渲染/答案抽取问题。"""
    return str(record.get("error_category", "")) == "answer_or_render_suspect"


def is_prm_high_score_wrong(record: dict[str, Any]) -> bool:
    """判断是否为 PRM 高分但 Verify 答案错。

    优先使用 run_qwen_prm.py 写出的 error_category；如果缺少该字段，
    再用 avg/min 分数按同一套阈值做兜底判断。
    """
    category = str(record.get("error_category", ""))
    if category:
        return category == "answer_or_render_suspect"

    if record.get("prm_logic_correct") is not None:
        return as_bool(record.get("prm_logic_correct"))

    try:
        avg_score = float(record.get("avg_step_score", 0.0))
        min_score = float(record.get("min_step_score", 1.0))
    except (TypeError, ValueError):
        return False
    return min_score >= WRONG_STEP_THRESHOLD and avg_score >= LOGIC_CORRECT_THRESHOLD


def build_metrics(
    name: str,
    verify_records: list[dict[str, Any]],
    prm_records: list[dict[str, Any]],
) -> dict[str, int | float]:
    """生成汇总指标。"""
    total = len(verify_records)
    verify_correct = sum(1 for r in verify_records if as_bool(r.get("verify_correct", False)))
    verify_wrong = total - verify_correct

    choice_records = [r for r in verify_records if get_answer_type(r) == "choice"]
    choice_total = len(choice_records)
    choice_correct = sum(1 for r in choice_records if as_bool(r.get("verify_correct", False)))
    choice_wrong = choice_total - choice_correct

    suspect_count = sum(1 for r in prm_records if is_render_or_extract_suspect(r))
    uncertain_count = sum(1 for r in prm_records if is_uncertain_prm(r))
    high_score_wrong = sum(1 for r in prm_records if is_prm_high_score_wrong(r))
    adjusted_correct = verify_correct + high_score_wrong

    return {
        "实验名称": name,
        "总样本数": total,
        "Verify正确数": verify_correct,
        "Verify错误数": verify_wrong,
        "Verify准确率": percent(verify_correct, total),
        "choice(选择题，answer_type )正确数": choice_correct,
        "choice(选择题，answer_type )错误数": choice_wrong,
        "choice(选择题，answer_type )准确率": percent(choice_correct, choice_total),
        "PRM样本总数": len(prm_records),
        "疑似渲染/答案抽取问题": suspect_count,
        "不确定": uncertain_count,
        "PRM 高分但答案错数量": high_score_wrong,
        "修正后的准去率（把PRM高分但答案错的归为正确的）": percent(adjusted_correct, total),
    }


def infer_name(verify_path: Path) -> str:
    """未显式传名称时，从 verify 文件名生成实验名。"""
    name = verify_path.stem
    if name.startswith("verify-"):
        name = name[len("verify-"):]
    elif name.startswith("verify_"):
        name = name[len("verify_"):]
    return name


def parse_cases(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    """解析单组或多组输入参数。"""
    if args.case:
        if args.verify or args.prm or args.name:
            raise ValueError("--case 多组模式下不要同时传 --verify/--prm/--name")
        return [
            (name, Path(verify).expanduser(), Path(prm).expanduser())
            for name, verify, prm in args.case
        ]

    if not args.verify or not args.prm:
        raise ValueError("单组模式必须同时传 --verify 和 --prm，或使用 --case 多组模式")

    verify_path = Path(args.verify).expanduser()
    prm_path = Path(args.prm).expanduser()
    return [(args.name or infer_name(verify_path), verify_path, prm_path)]


def main():
    parser = argparse.ArgumentParser(description="导出 Math-Verify + PRM 汇总指标")
    parser.add_argument("--verify", help="Math-Verify 结果 JSONL（单组模式）")
    parser.add_argument("--prm", help="PRM 结果 JSONL（单组模式）")
    parser.add_argument("--name", help="实验名称（单组模式，可选）")
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("NAME", "VERIFY_JSONL", "PRM_JSONL"),
        help="一组实验：名称 verify路径 prm路径；可重复传入",
    )
    parser.add_argument("--output_csv", required=True, help="输出 CSV 文件")
    parser.add_argument("--output_json", required=True, help="输出 JSON 文件")
    args = parser.parse_args()

    output_csv = Path(args.output_csv).expanduser()
    output_json = Path(args.output_json).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        cases = parse_cases(args)
    except ValueError as e:
        parser.error(str(e))

    rows = []
    for name, verify_path, prm_path in cases:
        verify_records = load_jsonl(verify_path)
        prm_records = load_jsonl(prm_path)
        rows.append(build_metrics(name, verify_records, prm_records))

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"CSV 汇总已写入: {output_csv} ({len(rows)} 组)")
    print(f"JSON 汇总已写入: {output_json} ({len(rows)} 组)")


if __name__ == "__main__":
    main()
