#!/usr/bin/env python3
"""
数据标准化脚本：将原始数据集统一转换为 JSONL 格式。

支持的数据集：
- Math23K: math23k_train.json, math23k_test.json
- GAOKAO-Bench: 数学相关 JSON 文件
- LinkWiseCoTDataset: training_final.json, test_final.json

用法:
    python standardize_datasets.py \
        --input_dir ../data/raw \
        --output_dir ../data/standardized \
        --limit 1000
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


def _parse_json_variants(raw: str) -> list[dict]:
    """
    解析多种 JSON 格式：
    1. JSON 数组 [...]
    2. JSONL（每行一个完整 JSON）
    3. 多行 JSON 拼接（} { 之间无逗号）
    """
    raw = raw.strip()

    # 1. 标准 JSON 数组
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    # 2. JSONL（每行一个 JSON）
    records = []
    lines = raw.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if records:
        return records

    # 3. 多行 JSON 拼接：用 }{ 或 }\n{ 分隔
    # 找到所有顶层 JSON 对象
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        # 跳过空白
        while pos < len(raw) and raw[pos] in " \t\n\r":
            pos += 1
        if pos >= len(raw):
            break
        try:
            obj, end_pos = decoder.raw_decode(raw, pos)
            records.append(obj)
            pos = end_pos
        except json.JSONDecodeError:
            pos += 1

    return records


def clean_text(text: str) -> str:
    """清理文本：去首尾空白，替换多余空白。"""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_record(
    record_id: str,
    source: str,
    question: str,
    reference_answer: str,
    subset: str = "",
    subject: str = "math",
    grade: str = "",
    difficulty: str = "",
    reference_solution: str = "",
    answer_type: str = "expr",
    has_image: bool = False,
    image_paths: list | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """构造标准化的记录。"""
    return {
        "id": record_id,
        "source": source,
        "subset": subset,
        "subject": subject,
        "grade": grade,
        "difficulty": difficulty,
        "question": clean_text(question),
        "reference_answer": clean_text(reference_answer),
        "reference_solution": clean_text(reference_solution),
        "answer_type": answer_type,
        "has_image": has_image,
        "image_paths": image_paths or [],
        "metadata": metadata or {},
    }


# ── Math23K ──────────────────────────────────────────────────────────────


def process_math23k(input_dir: Path, output_dir: Path, limit: int | None) -> tuple[int, int]:
    """处理 Math23K 数据集。返回 (成功数, 跳过数)。"""
    source_dir = input_dir / "Math23k"
    if not source_dir.exists():
        print(f"[WARN] Math23k 目录不存在: {source_dir}")
        return 0, 0

    success, skipped = 0, 0
    counter = 0
    output_path = output_dir / "math23k.jsonl"
    collected = []

    json_files = sorted(source_dir.glob("*.json"))
    # 排除非数据文件
    data_files = [f for f in json_files if f.name not in ("package.json",)]

    for jf in data_files:
        print(f"  处理: {jf.name}")
        with open(jf, "r", encoding="utf-8") as fin:
            raw = fin.read()

        # Math23K 可能是 JSONL、JSON 数组、或多行 JSON 拼接
        records = _parse_json_variants(raw)

        for item in records:
            if limit is not None and counter >= limit:
                break

            counter += 1
            question = item.get("original_text", "") or item.get("segmented_text", "")
            answer = item.get("ans", "")
            equation = item.get("equation", "")
            item_id = item.get("id", str(counter))

            if not question:
                skipped += 1
                continue

            collected.append(make_record(
                record_id=f"math23k_{int(item_id):06d}",
                source="Math23K",
                subset=jf.stem,
                question=question,
                reference_answer=answer,
                reference_solution=equation,
                answer_type="expr",
                metadata={"original_id": item_id},
            ))
            success += 1

        if limit is not None and counter >= limit:
            break

    random.shuffle(collected)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in collected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Math23K: 成功 {success}, 跳过 {skipped}")
    return success, skipped


# ── GAOKAO-Bench ─────────────────────────────────────────────────────────


# 数学相关文件名关键字
MATH_KEYWORDS = ["Math_I", "Math_II"]

GAOKAO_ANSWER_TYPE_MAP = {
    "MCQs": "choice",
    "Fill-in-the-Blank": "expr",
    "Open-ended": "expr",
}


def _guess_answer_type(filename: str) -> str:
    """根据文件名猜测答案类型。"""
    for key, atype in GAOKAO_ANSWER_TYPE_MAP.items():
        if key in filename:
            return atype
    return "expr"


def _normalize_answer(answer) -> str:
    """将答案统一为字符串。"""
    if isinstance(answer, list):
        return ",".join(str(a) for a in answer)
    return str(answer)


def process_gaokao_bench(input_dir: Path, output_dir: Path, limit: int | None) -> tuple[int, int]:
    """处理 GAOKAO-Bench 数据集（仅数学）。返回 (成功数, 跳过数)。"""
    source_dir = input_dir / "GAOKAO-Bench"
    if not source_dir.exists():
        print(f"[WARN] GAOKAO-Bench 目录不存在: {source_dir}")
        return 0, 0

    success, skipped = 0, 0
    counter = 0
    output_path = output_dir / "gaokao_bench.jsonl"
    collected = []

    # 收集所有数学相关文件
    data_dir = source_dir / "Data"
    math_files = []
    if data_dir.exists():
        for sub in ["Objective_Questions", "Subjective_Questions"]:
            sub_dir = data_dir / sub
            if sub_dir.exists():
                for f in sorted(sub_dir.glob("*.json")):
                    if any(kw in f.name for kw in MATH_KEYWORDS):
                        math_files.append(f)

    print(f"  找到 {len(math_files)} 个数学相关文件")

    for mf in math_files:
        print(f"  处理: {mf.name}")
        with open(mf, "r", encoding="utf-8") as fin:
            data = json.load(fin)

        examples = data.get("example", [])
        subset_name = data.get("keywords", mf.stem)
        answer_type = _guess_answer_type(mf.name)

        for item in examples:
            if limit is not None and counter >= limit:
                break

            counter += 1
            question = item.get("question", "")
            answer = item.get("answer", "")
            analysis = item.get("analysis", "")
            year = item.get("year", "")
            score = item.get("score", "")
            idx = item.get("index", counter)

            if not question:
                skipped += 1
                continue

            ref_answer = _normalize_answer(answer)

            collected.append(make_record(
                record_id=f"gaokao_{counter:06d}",
                source="GAOKAO-Bench",
                subset=subset_name,
                question=question,
                reference_answer=ref_answer,
                reference_solution=analysis,
                answer_type=answer_type,
                metadata={
                    "year": year,
                    "category": item.get("category", ""),
                    "score": score,
                    "index": idx,
                },
            ))
            success += 1

        if limit is not None and counter >= limit:
            break

    random.shuffle(collected)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in collected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  GAOKAO-Bench: 成功 {success}, 跳过 {skipped}")
    return success, skipped


# ── LinkWiseCoTDataset ───────────────────────────────────────────────────


def process_linkwise_cot(input_dir: Path, output_dir: Path, limit: int | None) -> tuple[int, int]:
    """处理 LinkWiseCoTDataset。返回 (成功数, 跳过数)。"""
    source_dir = input_dir / "LinkWiseCoTDataset"
    if not source_dir.exists():
        print(f"[WARN] LinkWiseCoTDataset 目录不存在: {source_dir}")
        return 0, 0

    success, skipped = 0, 0
    counter = 0
    output_path = output_dir / "linkwise_cot.jsonl"
    collected = []

    json_files = sorted(source_dir.glob("*.json"))
    # 排除非数据文件
    data_files = [f for f in json_files if "介绍" not in f.name and f.stat().st_size > 1000]

    for jf in data_files:
        print(f"  处理: {jf.name}")
        with open(jf, "r", encoding="utf-8") as fin:
            data = json.load(fin)

        if not isinstance(data, list):
            data = [data]

        for item in data:
            if limit is not None and counter >= limit:
                break

            counter += 1
            question = item.get("question_stem", "")
            answer = item.get("answer", "")
            solution = item.get("solution", "")
            reasoning = item.get("reasoning", "")
            job_id = item.get("job_id", counter)
            q_type = item.get("question_type", "")
            difficulty = item.get("difficulty", "")

            if not question:
                skipped += 1
                continue

            # 判断答案类型
            if "选择" in q_type:
                answer_type = "choice"
            else:
                answer_type = "expr"

            # reference_solution 优先用 reasoning（更完整）
            ref_solution = reasoning or solution

            collected.append(make_record(
                record_id=f"linkwise_{counter:06d}",
                source="LinkWiseCoTDataset",
                subset=jf.stem,
                question=question,
                reference_answer=answer,
                reference_solution=ref_solution,
                answer_type=answer_type,
                difficulty=difficulty,
                metadata={
                    "job_id": job_id,
                    "question_type": q_type,
                    "knowledge_points": item.get("knowledge_points", ""),
                    "volume_name": item.get("volume_name", ""),
                },
            ))
            success += 1

        if limit is not None and counter >= limit:
            break

    random.shuffle(collected)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in collected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  LinkWiseCoTDataset: 成功 {success}, 跳过 {skipped}")
    return success, skipped


# ── 合并与统计 ────────────────────────────────────────────────────────────


def merge_all(output_dir: Path) -> int:
    """合并所有 JSONL 到 train_or_eval_all.jsonl，打乱顺序。返回总行数。"""
    all_path = output_dir / "train_or_eval_all.jsonl"
    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    # 排除合并文件本身
    jsonl_files = [f for f in jsonl_files if f.name != "train_or_eval_all.jsonl"]

    lines = []
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    lines.append(line)

    random.shuffle(lines)

    with open(all_path, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")

    print(f"\n合并完成（已打乱）: train_or_eval_all.jsonl 共 {len(lines)} 条")
    return len(lines)


def write_summary(output_dir: Path, stats: dict[str, tuple[int, int]]):
    """写统计报告，包含题型分布、来源分布、年份分布。"""
    summary_path = output_dir / "dataset_summary.md"
    from collections import Counter

    total_success = sum(s for s, _ in stats.values())
    total_skipped = sum(sk for _, sk in stats.values())

    lines = [
        "# 数据集标准化统计报告\n",
        "## 总览\n",
        "| 数据集 | 题目数 | 跳过数 |",
        "|--------|--------|--------|",
    ]
    for name, (s, sk) in stats.items():
        lines.append(f"| {name} | {s} | {sk} |")
    lines.append(f"| **合计** | **{total_success}** | **{total_skipped}** |")
    lines.append("")

    # 每个数据集的详细统计
    dataset_files = {
        "Math23K": "math23k.jsonl",
        "GAOKAO-Bench": "gaokao_bench.jsonl",
        "LinkWiseCoTDataset": "linkwise_cot.jsonl",
    }

    for ds_name, fname in dataset_files.items():
        ds_path = output_dir / fname
        if not ds_path.exists():
            continue

        records = []
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        n = len(records)
        lines.append("---\n")
        lines.append(f"## {ds_name}（{n} 题）\n")

        # 题型分布
        type_counts = Counter(r.get("answer_type", "?") for r in records)
        type_label = {"choice": "选择题", "expr": "填空/解答题"}
        lines.append("### 题型分布\n")
        lines.append("| 题型 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for t, c in type_counts.most_common():
            label = type_label.get(t, t)
            lines.append(f"| {label} ({t}) | {c} | {c/n*100:.1f}% |")
        lines.append("")

        # 来源/子集分布
        subset_counts = Counter(r.get("subset", "") for r in records)
        if subset_counts:
            lines.append("### 来源分布\n")
            lines.append("| 子集 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            for s, c in subset_counts.most_common():
                lines.append(f"| {s} | {c} | {c/n*100:.1f}% |")
            lines.append("")

        # 年份分布（仅 GAOKAO-Bench 有年份）
        year_counts = Counter(
            r.get("metadata", {}).get("year", "")
            for r in records
            if r.get("metadata", {}).get("year", "")
        )
        if year_counts:
            lines.append("### 年份分布\n")
            lines.append("| 年份 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            for y, c in sorted(year_counts.items()):
                lines.append(f"| {y} | {c} | {c/n*100:.1f}% |")
            lines.append("")

        # 难度分布（仅 LinkWise 有难度）
        diff_counts = Counter(r.get("difficulty", "") for r in records)
        diff_counts = {k: v for k, v in diff_counts.items() if k}
        if diff_counts:
            lines.append("### 难度分布\n")
            lines.append("| 难度 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            for d, c in sorted(diff_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {d} | {c} | {c/n*100:.1f}% |")
            lines.append("")

        # 含图片
        img_count = sum(1 for r in records if r.get("has_image"))
        lines.append(f"- **含图片**: {img_count}")
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"统计报告已写入: {summary_path}")


# ── 日志：坏样本 ─────────────────────────────────────────────────────────


def check_quality(output_dir: Path, log_dir: Path):
    """检查标准化后的数据质量，记录空题目/空答案的坏样本。"""
    bad_path = log_dir / "standardize_bad_samples.jsonl"
    bad_count = 0

    with open(bad_path, "w", encoding="utf-8") as fout:
        for jf in sorted(output_dir.glob("*.jsonl")):
            if jf.name == "train_or_eval_all.jsonl":
                continue
            with open(jf, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    issues = []
                    if not record.get("question"):
                        issues.append("empty_question")
                    if not record.get("reference_answer"):
                        issues.append("empty_answer")
                    if not record.get("id"):
                        issues.append("empty_id")

                    if issues:
                        bad_count += 1
                        fout.write(json.dumps({
                            "id": record.get("id", ""),
                            "source": record.get("source", ""),
                            "issues": issues,
                        }, ensure_ascii=False) + "\n")

    if bad_count > 0:
        print(f"[WARN] 发现 {bad_count} 条坏样本，已记录到 {bad_path}")
    else:
        print("数据质量检查通过，无坏样本")


# ── 主函数 ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="数据标准化：将原始数据集转为统一 JSONL")
    parser.add_argument("--input_dir", required=True, help="原始数据集目录")
    parser.add_argument("--output_dir", required=True, help="标准化输出目录")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集最多处理 N 条")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("数据标准化开始")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    if args.limit:
        print(f"限制: 每个数据集最多 {args.limit} 条")
    print("=" * 60)

    stats = {}

    # 1. Math23K
    print("\n[1/3] 处理 Math23K ...")
    stats["Math23K"] = process_math23k(input_dir, output_dir, args.limit)

    # 2. GAOKAO-Bench
    print("\n[2/3] 处理 GAOKAO-Bench (数学) ...")
    stats["GAOKAO-Bench"] = process_gaokao_bench(input_dir, output_dir, args.limit)

    # 3. LinkWiseCoTDataset
    print("\n[3/3] 处理 LinkWiseCoTDataset ...")
    stats["LinkWiseCoTDataset"] = process_linkwise_cot(input_dir, output_dir, args.limit)

    # 4. 合并
    print("\n[合并] 合并所有数据集 ...")
    merge_all(output_dir)

    # 5. 统计
    print("\n[统计] 生成统计报告 ...")
    write_summary(output_dir, stats)

    # 6. 质量检查
    print("\n[质检] 检查数据质量 ...")
    check_quality(output_dir, log_dir)

    print("\n" + "=" * 60)
    print("数据标准化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
