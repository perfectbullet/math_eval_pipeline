#!/usr/bin/env python3
"""
Math-Verify 验证脚本：验证模型输出与标准答案是否等价。

核心策略：
1. 优先用 math-verify 库做端到端验证（抽取 + 比较）
2. math-verify 失败时降级为手工正则抽取 + 数值/字符串比较

用法:
    python run_math_verify.py \
        --input data/model_outputs/model_outputs.jsonl \
        --output results/verify/verify_results.jsonl \
        --summary results/verify/verify_summary.md
"""

import argparse
import html as html_module
import json
import re
from collections import defaultdict
from pathlib import Path

from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
)

CHOICE_OPTIONS = ("A", "B", "C", "D", "E", "F")


# ── math-verify 端到端验证 ───────────────────────────────────────────────


def _mv_extract(text: str, answer_type: str) -> list:
    """用 math-verify 的 parse() 从文本中抽取答案。"""
    if answer_type == "choice":
        configs = [StringExtractionConfig(strings=CHOICE_OPTIONS), LatexExtractionConfig()]
    else:
        configs = [LatexExtractionConfig(), ExprExtractionConfig()]
    try:
        return parse(text, extraction_config=configs)
    except Exception:
        return []


def _mv_verify_with_extract(
    reference_answer: str, model_output: str, answer_type: str
) -> tuple[bool, str, str]:
    """
    用 math-verify 同时做答案抽取和比较。
    返回: (是否正确, 抽取到的预测文本, 错误信息)
    """
    gold_parsed = _mv_extract(reference_answer, answer_type)
    pred_parsed = _mv_extract(model_output, answer_type)

    if not gold_parsed:
        return False, "", "gold_parse_empty"
    if not pred_parsed:
        return False, "", "pred_parse_empty"

    extracted = str(pred_parsed[0])

    try:
        is_correct = verify(gold_parsed, pred_parsed)
        return bool(is_correct), extracted, ""
    except Exception as e:
        return False, extracted, f"verify_error: {e}"


# ── 手工答案抽取（降级用） ────────────────────────────────────────────────

EXTRACT_PATTERNS = [
    re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL),
    re.compile(r"最终答案[：:]\s*(.+?)$", re.MULTILINE),
    re.compile(r"答案[：:是为]\s*(.+?)$", re.MULTILINE),
]


def _manual_extract(model_output: str, answer_type: str = "expr") -> str:
    """手工正则抽取答案（math-verify 抽取失败时的降级）。"""
    if not model_output:
        return ""

    if answer_type == "choice":
        boxed_matches = EXTRACT_PATTERNS[0].findall(model_output)
        for m in reversed(boxed_matches):
            if re.fullmatch(r"\s*[A-F]\s*", m.strip()):
                return m.strip()
        for pat in [
            re.compile(r"最终答案[：:]\s*([A-F])"),
            re.compile(r"答案[：:是为]\s*([A-F])"),
            re.compile(r"选\s*([A-F])"),
        ]:
            m = pat.search(model_output)
            if m:
                return m.group(1)
        choices = re.findall(r"\b([A-F])\b", model_output)
        if choices:
            return choices[-1]
        return ""

    for pat in EXTRACT_PATTERNS:
        m = pat.search(model_output)
        if m:
            return m.group(1).strip()

    lines = [l.strip() for l in model_output.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["最终答案：", "最终答案:", "答案：", "答案:"]:
            if last.startswith(prefix):
                return last[len(prefix):].strip()
        return last

    return ""


# ── 降级验证 ─────────────────────────────────────────────────────────────


def _string_verify(gold_str: str, pred_str: str) -> tuple[bool, str]:
    """字符串级别验证（选择题兜底）。"""
    gold = gold_str.strip().upper()
    pred = pred_str.strip().upper()
    if gold == pred:
        return True, ""
    gold_clean = re.sub(r"[,，;；\s]", "", gold)
    pred_clean = re.sub(r"[,，;；\s]", "", pred)
    if gold_clean == pred_clean:
        return True, ""
    return False, "string_mismatch"


def _numeric_verify(gold_str: str, pred_str: str, tolerance: float = 1e-3) -> tuple[bool, str]:
    """数值验证（兜底）。"""
    try:
        gold_val = float(gold_str.replace(",", "").replace("，", ""))
        pred_val = float(pred_str.replace(",", "").replace("，", ""))
        if abs(gold_val - pred_val) < tolerance:
            return True, ""
        return False, f"numeric_mismatch: {gold_val} vs {pred_val}"
    except (ValueError, TypeError):
        return False, "not_numeric"


def _fallback_verify(
    reference_answer: str, model_output: str, answer_type: str
) -> tuple[bool, str, str]:
    """降级验证：手工抽取 + 简单比较。"""
    extracted = _manual_extract(model_output, answer_type)

    if not extracted:
        return False, "", "extraction_failed"
    if not reference_answer:
        return False, extracted, "empty_reference"

    if answer_type == "choice":
        ok, err = _string_verify(reference_answer, extracted)
        return ok, extracted, err

    ok, err = _numeric_verify(reference_answer, extracted)
    if err == "":
        return ok, extracted, err
    gold_clean = reference_answer.strip().replace(" ", "").replace("$", "")
    pred_clean = extracted.strip().replace(" ", "").replace("$", "")
    if gold_clean == pred_clean:
        return True, extracted, ""
    return False, extracted, "string_mismatch"


# ── 统一入口 ─────────────────────────────────────────────────────────────


def verify_answer(
    reference_answer: str,
    model_output: str,
    answer_type: str = "expr",
) -> tuple[bool, str, str]:
    """
    验证答案：优先 math-verify 端到端，失败则降级手工验证。
    返回: (是否正确, 提取的预测文本, 错误信息)
    """
    ok, extracted, err = _mv_verify_with_extract(reference_answer, model_output, answer_type)
    if err == "":
        return ok, extracted, ""
    # math-verify 抽取或验证失败，降级
    return _fallback_verify(reference_answer, model_output, answer_type)


# ── 主流程 ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Math-Verify 验证模型输出")
    parser.add_argument("--input", required=True, help="模型输出 JSONL")
    parser.add_argument("--output", required=True, help="验证结果 JSONL")
    parser.add_argument("--summary", required=True, help="汇总报告 Markdown")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 条")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    count = 0

    print("=" * 60)
    print("Math-Verify 验证开始")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and count >= args.limit:
                break

            record = json.loads(line)
            count += 1

            ref_answer = record.get("reference_answer", "")
            model_output = record.get("model_output", "")
            # 去掉 <think...>...</think()> 推理过程，只保留正式回复
            model_output = re.sub(r"<think[^>]*>.*?</think\s*>", "", model_output, flags=re.DOTALL).strip()
            answer_type = record.get("metadata", {}).get("answer_type", "expr")

            # 模型输出为空，不计入准确率
            if not model_output:
                is_correct, extracted, error = False, "", "empty_output"
            else:
                is_correct, extracted, error = verify_answer(
                    ref_answer, model_output, answer_type
                )

            result = {
                "id": record.get("id", ""),
                "source": record.get("source", ""),
                "difficulty": record.get("difficulty", ""),
                "model_name": record.get("model_name", ""),
                "question": record.get("question", ""),
                "reference_answer": ref_answer,
                "model_output": model_output,
                "extracted_prediction": extracted,
                "verify_correct": is_correct,
                "verify_error": error,
                "answer_type": answer_type,
                "need_prm": not is_correct,
                "need_render_check": False,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

            if count % 100 == 0:
                print(f"  已处理 {count} 条...")

    print(f"\n共处理 {count} 条")

    _write_summary(results, summary_path)
    print(f"汇总报告已写入: {summary_path}")

    # 生成错误样本 HTML
    _write_error_html(results, output_path.parent)
    print(f"错误样本 HTML 已写入: {output_path.parent / 'errors/'}")


def _write_summary(results: list[dict], path: Path):
    """生成验证汇总报告。"""
    # 排除空输出的记录
    valid = [r for r in results if r.get("verify_error") != "empty_output"]
    skipped = len(results) - len(valid)

    total = len(valid)
    correct = sum(1 for r in valid if r["verify_correct"])
    wrong = total - correct
    parse_fail = sum(1 for r in valid if r["verify_error"] == "extraction_failed")
    accuracy = correct / total * 100 if total > 0 else 0

    by_source = defaultdict(list)
    for r in valid:
        by_source[r.get("source", "unknown")].append(r)

    lines = [
        "# Math-Verify 验证汇总报告\n",
        f"## 总体统计",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总样本数 | {total} |",
        f"| 正确数 | {correct} |",
        f"| 错误数 | {wrong} |",
        f"| 准确率 | {accuracy:.2f}% |",
        f"| 解析失败数 | {parse_fail} |",
        f"| 空输出（排除） | {skipped} |",
        f"",
        f"## 按 source 分组",
        f"",
        f"| 来源 | 总数 | 正确 | 错误 | 准确率 |",
        f"|------|------|------|------|--------|",
    ]

    for source, items in sorted(by_source.items()):
        s_total = len(items)
        s_correct = sum(1 for r in items if r["verify_correct"])
        s_acc = s_correct / s_total * 100 if s_total > 0 else 0
        lines.append(f"| {source} | {s_total} | {s_correct} | {s_total - s_correct} | {s_acc:.2f}% |")

    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("answer_type", "unknown")].append(r)

    lines.extend([
        "",
        "## 按 answer_type 分组",
        "",
        "| 类型 | 总数 | 正确 | 错误 | 准确率 |",
        "|------|------|------|------|--------|",
    ])
    for atype, items in sorted(by_type.items()):
        s_total = len(items)
        s_correct = sum(1 for r in items if r["verify_correct"])
        s_acc = s_correct / s_total * 100 if s_total > 0 else 0
        lines.append(f"| {atype} | {s_total} | {s_correct} | {s_total - s_correct} | {s_acc:.2f}% |")

    # 按 difficulty 分组
    by_diff = defaultdict(list)
    for r in valid:
        by_diff[r.get("difficulty", "") or "unknown"].append(r)

    lines.extend([
        "",
        "## 按 difficulty 分组",
        "",
        "| 难度 | 总数 | 正确 | 错误 | 准确率 |",
        "|------|------|------|------|--------|",
    ])
    for diff, items in sorted(by_diff.items()):
        s_total = len(items)
        s_correct = sum(1 for r in items if r["verify_correct"])
        s_acc = s_correct / s_total * 100 if s_total > 0 else 0
        lines.append(f"| {diff} | {s_total} | {s_correct} | {s_total - s_correct} | {s_acc:.2f}% |")

    wrong_results = [r for r in valid if not r["verify_correct"]]
    lines.extend(["", "## Top 20 错误样本", ""])
    for r in wrong_results[:20]:
        lines.append(f"- **{r['id']}** ({r.get('source', '')}): ref=`{r['reference_answer']}`, pred=`{r['extracted_prediction']}`, error={r['verify_error']}")

    parse_failed = [r for r in valid if r["verify_error"] == "extraction_failed"]
    if parse_failed:
        lines.extend(["", "## Top 20 解析失败样本", ""])
        for r in parse_failed[:20]:
            lines.append(f"- **{r['id']}** ({r.get('source', '')}): model_output 前100字=`{r['model_output'][:100]}`")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── 错误样本 HTML 报告 ────────────────────────────────────────────────────

FIELD_LABELS = {
    "id": "编号",
    "source": "来源",
    "difficulty": "难度",
    "model_name": "模型",
    "question": "题目",
    "model_output": "模型输出",
    "reference_answer": "标准答案",
    "extracted_prediction": "提取预测",
    "verify_correct": "验证正确",
    "verify_error": "验证错误",
    "answer_type": "题型",
    "need_prm": "需PRM评分",
    "need_render_check": "需渲染检查",
}

DISPLAY_ORDER = [
    "id", "source", "difficulty", "model_name", "answer_type",
    "verify_correct", "verify_error", "need_prm", "need_render_check",
    "question", "model_output", "reference_answer", "extracted_prediction",
]

KATEX_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}},
    ]}});"></script>
<style>
body {{ font-family: "Noto Sans SC", sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; padding-bottom: 8px; }}
.field {{ margin-bottom: 16px; background: #fff; border-radius: 6px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
.field-label {{ font-weight: bold; color: #2c3e50; font-size: 14px; margin-bottom: 4px; }}
.field-value {{ white-space: pre-wrap; word-break: break-word; font-size: 15px; line-height: 1.6; color: #333; }}
.field-value.long {{ max-height: 400px; overflow-y: auto; }}
</style>
</head>
<body>
<h1>{title}</h1>
"""


def _write_error_html(results: list[dict], output_dir: Path):
    """为错误样本生成独立 HTML 文件。"""
    errors = [r for r in results if not r.get("verify_correct") and r.get("verify_error") != "empty_output"]
    if not errors:
        return

    html_dir = output_dir / "errors"
    html_dir.mkdir(parents=True, exist_ok=True)

    for r in errors:
        rid = r.get("id", "unknown")
        title = f"错误样本 - {rid}"
        html = KATEX_HTML_HEAD.format(title=title)

        for key in DISPLAY_ORDER:
            label = FIELD_LABELS.get(key, key)
            value = r.get(key, "")
            if isinstance(value, bool):
                value = "✅ 是" if value else "❌ 否"
            value = str(value) if value else "（空）"
            value = html_module.escape(value)
            # 把 $$...$$ 多行块用 <div> 包裹，让 KaTeX 能正确识别 display math
            value = re.sub(
                r"(\$\$.*?\$\$)",
                r'<div style="white-space:normal;text-align:center">\1</div>',
                value,
                flags=re.DOTALL,
            )
            long_cls = " long" if key in ("question", "model_output", "reference_answer") else ""
            html += f'<div class="field"><div class="field-label">{label}</div><div class="field-value{long_cls}">{value}</div></div>\n'

        html += "</body></html>"

        (html_dir / f"{rid}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
