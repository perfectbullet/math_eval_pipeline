#!/usr/bin/env python3
"""
DeepSeek-Math PRM 过程评分脚本：对 Math-Verify 错误样本做步骤级评分。

默认模型: mukaj/deepseek-math-7b-rl-prm-v0.1

用法:
    python run_deepseek_prm.py \
        --input results/verify/verify_results.jsonl \
        --output results/prm/prm_step_scores.jsonl \
        --summary results/prm/prm_summary.md \
        --model mukaj/deepseek-math-7b-rl-prm-v0.1 \
        --limit 50
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 步骤切分 ──────────────────────────────────────────────────────────────

STEP_MARKERS = [
    re.compile(r"(?:^|\n)\s*(?:步骤|step)\s*(\d+)\s*[.。：:]\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*（(\d+)）\s*"),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.。]\s+"),
]

SENTENCE_SPLIT = re.compile(r"[。；\n]")

# 归因阈值
WRONG_STEP_THRESHOLD = 0.45
LOGIC_CORRECT_THRESHOLD = 0.70


def split_steps(text: str, min_length: int = 5) -> list[str]:
    """将模型输出切分为步骤。三级降级：步骤标记 → 换行 → 句号。"""
    if not text:
        return []

    # 1. 按明确步骤标记切分
    for pat in STEP_MARKERS:
        parts = pat.split(text)
        if len(parts) >= 3:
            steps = []
            for i in range(1, len(parts), 2):
                step_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if step_text and len(step_text) >= min_length:
                    steps.append(step_text)
            if len(steps) >= 2:
                return steps

    # 2. 按换行切分，合并过短相邻行
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
    sentences = SENTENCE_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
    if len(sentences) >= 2:
        return sentences

    # 兜底：整个文本作为一个步骤
    return [text.strip()] if text.strip() else []


# ── PRM 模型加载 ─────────────────────────────────────────────────────────


class PRMScorer:
    """DeepSeek-Math PRM 评分器。"""

    def __init__(self, model_name: str, device: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

        # 检测设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        """加载模型。"""
        print(f"加载 PRM 模型: {self.model_name} ...")

        kwargs = {
            "device_map": self.device,
            "trust_remote_code": True,
        }
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

        print(f"PRM 模型加载完成，设备: {self.device}")

    def score_step(self, question: str, previous_steps: str, current_step: str) -> float:
        """对单步评分（增量方式）。返回 0~1 之间的分数。"""
        prompt = (
            f"Problem:\n{question}\n\n"
            f"Previous steps:\n{previous_steps}\n\n"
            f"Current step:\n{current_step}\n\n"
            f"Is the current reasoning step correct?"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]

        yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("no", add_special_tokens=False)

        yes_score = sum(logits[t].item() for t in yes_tokens) / len(yes_tokens) if yes_tokens else 0
        no_score = sum(logits[t].item() for t in no_tokens) / len(no_tokens) if no_tokens else 0

        # softmax 归一化
        exp_yes = math.exp(yes_score)
        exp_no = math.exp(no_score)
        total = exp_yes + exp_no

        return exp_yes / total if total > 0 else 0.5

    def score_sample(self, question: str, model_output: str) -> dict[str, Any]:
        """对一个样本的所有步骤评分。"""
        steps = split_steps(model_output)
        if not steps:
            return {
                "steps": [],
                "min_step_score": 0.0,
                "avg_step_score": 0.0,
                "first_wrong_step_index": -1,
            }

        step_scores = []
        previous = ""

        for i, step in enumerate(steps):
            score = self.score_step(question, previous, step)
            step_scores.append({
                "step_index": i + 1,
                "text": step,
                "score": round(score, 4),
                "label": "correct" if score >= WRONG_STEP_THRESHOLD else "wrong",
            })
            previous += step + "\n"

        scores = [s["score"] for s in step_scores]
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        first_wrong = -1
        for i, s in enumerate(step_scores):
            if s["score"] < WRONG_STEP_THRESHOLD:
                first_wrong = i + 1
                break

        return {
            "steps": step_scores,
            "min_step_score": round(min_score, 4),
            "avg_step_score": round(avg_score, 4),
            "first_wrong_step_index": first_wrong,
        }


# ── 归因 ─────────────────────────────────────────────────────────────────


def classify_error(prm_result: dict) -> tuple[str, bool]:
    """根据 PRM 评分结果归因。返回 (error_category, need_render_check)。"""
    min_score = prm_result["min_step_score"]
    avg_score = prm_result["avg_step_score"]

    if min_score < WRONG_STEP_THRESHOLD:
        return "logic_error", False
    elif avg_score >= LOGIC_CORRECT_THRESHOLD:
        return "answer_or_render_suspect", True
    else:
        return "uncertain_logic", True


# ── 主流程 ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-Math PRM 过程评分")
    parser.add_argument("--input", required=True, help="Math-Verify 结果 JSONL")
    parser.add_argument("--output", required=True, help="PRM 评分结果 JSONL")
    parser.add_argument("--summary", required=True, help="汇总报告 Markdown")
    parser.add_argument("--model", default="mukaj/deepseek-math-7b-rl-prm-v0.1", help="PRM 模型名")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 条错误样本")
    parser.add_argument("--max_samples", type=int, default=None, help="同 --limit（兼容）")
    parser.add_argument("--device", default="auto", help="设备: auto/cuda/cpu")
    parser.add_argument("--load_in_8bit", action="store_true", help="8bit 量化加载")
    parser.add_argument("--load_in_4bit", action="store_true", help="4bit 量化加载")
    args = parser.parse_args()

    limit = args.limit or args.max_samples

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepSeek-Math PRM 过程评分")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print("=" * 60)

    # 读取错误样本
    wrong_samples = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("verify_correct", False):
                wrong_samples.append(record)

    print(f"从 {input_path} 中找到 {len(wrong_samples)} 条错误样本")

    if limit is not None:
        wrong_samples = wrong_samples[:limit]
        print(f"限制处理 {limit} 条")

    if not wrong_samples:
        print("无错误样本需要处理，退出")
        output_path.write_text("", encoding="utf-8")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# PRM 评分汇总\n\n无错误样本\n", encoding="utf-8")
        return

    # 加载模型
    scorer = PRMScorer(args.model, device=args.device, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)

    # 评分
    results = []
    for i, sample in enumerate(wrong_samples):
        qid = sample.get("id", f"unknown_{i}")
        question = sample.get("question", "")
        model_output = sample.get("model_output", "")

        print(f"  [{i + 1}/{len(wrong_samples)}] 评分: {qid}")

        try:
            prm_result = scorer.score_sample(question, model_output)
        except Exception as e:
            print(f"    [ERROR] {e}")
            prm_result = {
                "steps": [],
                "min_step_score": 0.0,
                "avg_step_score": 0.0,
                "first_wrong_step_index": -1,
            }

        error_category, need_render = classify_error(prm_result)
        logic_correct = prm_result["avg_step_score"] >= LOGIC_CORRECT_THRESHOLD

        result = {
            "id": qid,
            "source": sample.get("source", ""),
            "model_name": sample.get("model_name", ""),
            "verify_correct": False,
            "question": question,
            "reference_answer": sample.get("reference_answer", ""),
            "model_output": model_output,
            **prm_result,
            "prm_logic_correct": logic_correct,
            "need_render_check": need_render,
            "error_category": error_category,
        }

        results.append(result)

    # 写入结果
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nPRM 评分完成，共 {len(results)} 条 -> {output_path}")

    # 生成汇总
    _write_summary(results, summary_path)
    print(f"汇总报告: {summary_path}")


def _write_summary(results: list[dict], path: Path):
    """生成 PRM 汇总报告。"""
    total = len(results)
    if total == 0:
        path.write_text("# PRM 评分汇总\n\n无样本\n", encoding="utf-8")
        return

    logic_errors = sum(1 for r in results if r.get("error_category") == "logic_error")
    render_suspects = sum(1 for r in results if r.get("error_category") == "answer_or_render_suspect")
    uncertain = sum(1 for r in results if r.get("error_category") == "uncertain_logic")

    avg_scores = [r["avg_step_score"] for r in results]
    min_scores = [r["min_step_score"] for r in results]

    # 按 source 分组
    by_source = defaultdict(list)
    for r in results:
        by_source[r.get("source", "unknown")].append(r)

    lines = [
        "# PRM 过程评分汇总报告\n",
        "## 总体统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 错误样本总数 | {total} |",
        f"| 解题逻辑错误 | {logic_errors} |",
        f"| 疑似渲染/答案抽取问题 | {render_suspects} |",
        f"| 不确定 | {uncertain} |",
        f"| 平均步骤分 (avg) | {sum(avg_scores)/len(avg_scores):.4f} |",
        f"| 最低步骤分 (min) | {sum(min_scores)/len(min_scores):.4f} |",
        "",
        "## 归因分布",
        "",
        "| 类别 | 数量 | 占比 |",
        "|------|------|------|",
        f"| logic_error | {logic_errors} | {logic_errors/total*100:.1f}% |",
        f"| answer_or_render_suspect | {render_suspects} | {render_suspects/total*100:.1f}% |",
        f"| uncertain_logic | {uncertain} | {uncertain/total*100:.1f}% |",
    ]

    # 按来源分组
    lines.extend([
        "",
        "## 按来源分组",
        "",
        "| 来源 | 总数 | logic_error | render_suspect | uncertain |",
        "|------|------|-------------|----------------|-----------|",
    ])
    for source in sorted(by_source.keys()):
        items = by_source[source]
        le = sum(1 for r in items if r.get("error_category") == "logic_error")
        rs = sum(1 for r in items if r.get("error_category") == "answer_or_render_suspect")
        uc = sum(1 for r in items if r.get("error_category") == "uncertain_logic")
        lines.append(f"| {source} | {len(items)} | {le} | {rs} | {uc} |")

    # 低分步骤最多的样本 (Top 10)
    lines.extend(["", "## 低分步骤最多的样本 (Top 10)", ""])
    sorted_results = sorted(results, key=lambda r: r["min_step_score"])
    for r in sorted_results[:10]:
        lines.append(
            f"- **{r['id']}** ({r.get('source', '')}): "
            f"min_score={r['min_step_score']:.4f}, "
            f"avg_score={r['avg_step_score']:.4f}, "
            f"first_wrong_step={r['first_wrong_step_index']}, "
            f"category={r['error_category']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
