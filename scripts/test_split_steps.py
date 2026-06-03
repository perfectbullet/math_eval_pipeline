#!/usr/bin/env python3
"""
测试 split_steps 切分效果：对验证结果中的模型输出做步骤切分，
输出新旧对比报告，方便人工审阅切分质量。

用法:
    python scripts/test_split_steps.py \
        --input results/verify/verify_results_gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl \
        --output results/test_split_steps_report.md \
        --limit 50
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── 新版 split_steps（直接内联，避免导入 torch） ───────────────────────────

STEP_MARKERS_NEW = [
    re.compile(r"(?:^|\n)\s*(?:步骤|step)\s*(\d+)\s*[.。：:]\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*（(\d+)）\s*"),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.。]\s+"),
]
SENTENCE_SPLIT_NEW = re.compile(r"[。；\n]")

_DECORATIVE_RE = re.compile(
    r"^(?:-{3,}|={3,}|#{1,4}\s|"
    r"第\s*[（(]\s*[IVXivx]+\s*[）)]\s*[部分问]|"
    r"第\s*[（(]\s*\d+\s*[）)]\s*[部分问]|"
    r"\*{3,})"
)


def _merge_paragraphs(paragraphs: list[str], min_length: int = 20) -> list[str]:
    filtered = [p for p in paragraphs if not _DECORATIVE_RE.match(p)]
    if not filtered:
        return filtered
    merged = []
    i = 0
    while i < len(filtered):
        p = filtered[i]
        if len(p) < min_length and i + 1 < len(filtered):
            merged.append(p + "\n\n" + filtered[i + 1])
            i += 2
        else:
            merged.append(p)
            i += 1
    return merged


def split_steps(text: str, min_length: int = 5) -> list[str]:
    """新版：按 \\n\\n 双换行切分 + 后处理（过滤装饰行 + 合并短段落）。"""
    if not text:
        return []
    for pat in STEP_MARKERS_NEW:
        parts = pat.split(text)
        if len(parts) >= 3:
            steps = []
            for i in range(1, len(parts), 2):
                step_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if step_text and len(step_text) >= min_length:
                    steps.append(step_text)
            if len(steps) >= 2:
                return steps
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        paragraphs = _merge_paragraphs(paragraphs)
        if paragraphs:
            return paragraphs
    sentences = SENTENCE_SPLIT_NEW.split(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
    if len(sentences) >= 2:
        return sentences
    return [text.strip()] if text.strip() else []


# ── 旧版 split_steps（按单换行切分）用于对比 ─────────────────────────────

_STEP_MARKERS = [
    re.compile(r"(?:^|\n)\s*(?:步骤|step)\s*(\d+)\s*[.。：:]\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*（(\d+)）\s*"),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.。]\s+"),
]
_SENTENCE_SPLIT = re.compile(r"[。；\n]")


def old_split_steps(text: str, min_length: int = 5) -> list[str]:
    """旧版：按单换行切分 + 合并短行（修改前的逻辑）。"""
    if not text:
        return []

    # 1. 按明确步骤标记切分
    for pat in _STEP_MARKERS:
        parts = pat.split(text)
        if len(parts) >= 3:
            steps = []
            for i in range(1, len(parts), 2):
                step_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if step_text and len(step_text) >= min_length:
                    steps.append(step_text)
            if len(steps) >= 2:
                return steps

    # 2. 按单换行切分，合并过短相邻行
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 3:
        merged = []
        buffer = ""
        for line in lines:
            if len(buffer) + len(line) < min_length:
                buffer = (buffer + " " + line).strip()
            else:
                if buffer:
                    merged.append(buffer)
                buffer = line
        if buffer:
            merged.append(buffer)
        if len(merged) >= 2:
            return [m for m in merged if len(m) >= min_length]

    # 3. 按句号切分
    sentences = _SENTENCE_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
    if len(sentences) >= 2:
        return sentences

    return [text.strip()] if text.strip() else []


def preview(text: str, max_len: int = 120) -> str:
    """截取预览，替换换行为 ↵ 方便阅读。"""
    t = text.replace("\n", "↵")
    if len(t) > max_len:
        return t[:max_len] + "…"
    return t


def main():
    parser = argparse.ArgumentParser(description="测试 split_steps 切分效果")
    parser.add_argument("--input", required=True, help="验证结果 JSONL")
    parser.add_argument("--output", required=True, help="对比报告输出路径")
    parser.add_argument("--limit", type=int, default=50, help="最多处理 N 条错误样本")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 只取错误样本（有模型输出的）
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("verify_correct") and r.get("model_output", "").strip():
                samples.append(r)

    samples = samples[: args.limit]
    print(f"共 {len(samples)} 条错误样本待分析")

    # 统计
    new_counts = []
    old_counts = []
    lines = ["# split_steps 切分效果对比报告\n"]

    for idx, r in enumerate(samples):
        rid = r.get("id", f"unknown_{idx}")
        output = r["model_output"]

        new_steps = split_steps(output)
        old_steps = old_split_steps(output)
        new_counts.append(len(new_steps))
        old_counts.append(len(old_steps))

        lines.append(f"---\n\n## {rid}\n")
        lines.append(f"- 来源: {r.get('source', '')}")
        lines.append(f"- 题型: {r.get('answer_type', '')}")
        lines.append(f"- 模型输出长度: {len(output)} 字符")
        lines.append(f"| | 旧版（单换行） | 新版（双换行） |")
        lines.append(f"|---|---|---|")
        lines.append(f"| 步骤数 | {len(old_steps)} | {len(new_steps)} |")
        lines.append("")

        # 新版步骤详情
        lines.append("### 新版步骤（\\n\\n 切分）\n")
        for i, s in enumerate(new_steps):
            lines.append(f"**步骤 {i+1}** ({len(s)}字):")
            lines.append(f"> {preview(s, 200)}")
            lines.append("")

        # 如果差异大，也列出旧版
        if abs(len(new_steps) - len(old_steps)) >= 5:
            lines.append("<details><summary>旧版步骤详情（单换行切分）</summary>\n")
            for i, s in enumerate(old_steps):
                lines.append(f"{i+1}. ({len(s)}字) {preview(s, 150)}")
            lines.append("\n</details>\n")

    # 汇总统计
    lines.insert(1, "## 总体统计\n")
    lines.insert(2, f"- 样本数: {len(samples)}")
    if new_counts:
        lines.insert(3, f"| 指标 | 旧版（单换行） | 新版（双换行） |")
        lines.insert(4, f"|------|------|------|")
        avg_new = sum(new_counts) / len(new_counts)
        avg_old = sum(old_counts) / len(old_counts)
        max_new = max(new_counts)
        max_old = max(old_counts)
        min_new = min(new_counts)
        min_old = min(old_counts)
        lines.insert(5, f"| 平均步骤数 | {avg_old:.1f} | {avg_new:.1f} |")
        lines.insert(6, f"| 最大步骤数 | {max_old} | {max_new} |")
        lines.insert(7, f"| 最小步骤数 | {min_old} | {min_new} |")

        # 步骤数分布
        from collections import Counter
        new_dist = Counter(new_counts)
        old_dist = Counter(old_counts)
        lines.insert(8, "")
        lines.insert(9, "### 步骤数分布\n")
        lines.insert(10, "| 步骤数 | 旧版样本数 | 新版样本数 |")
        lines.insert(11, "|--------|-----------|-----------|")
        all_counts = sorted(set(list(new_dist.keys()) + list(old_dist.keys())))
        for c in all_counts:
            lines.insert(12, f"| {c} | {old_dist.get(c, 0)} | {new_dist.get(c, 0)} |")

    lines.insert(13, "\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"对比报告已写入: {output_path}")

    # 快速统计摘要
    if new_counts:
        print(f"\n步骤数变化:")
        print(f"  旧版: avg={sum(old_counts)/len(old_counts):.1f}, max={max(old_counts)}, min={min(old_counts)}")
        print(f"  新版: avg={sum(new_counts)/len(new_counts):.1f}, max={max(new_counts)}, min={min(new_counts)}")
        # 步骤数大幅减少的样本
        big_change = [(samples[i]['id'], old_counts[i], new_counts[i])
                      for i in range(len(samples)) if old_counts[i] - new_counts[i] >= 10]
        if big_change:
            print(f"\n步骤数减少 >= 10 的样本 ({len(big_change)} 条):")
            for rid, oc, nc in big_change[:10]:
                print(f"  {rid}: {oc} → {nc}")


if __name__ == "__main__":
    main()
