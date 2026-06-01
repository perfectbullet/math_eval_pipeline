#!/usr/bin/env python3
"""
Qwen2.5-Math PRM 过程评分脚本：对 Math-Verify 错误样本做步骤级评分。

基于 Qwen2.5-Math-PRM-7B 模型，使用 <extra_0> token 分隔步骤，
单次前向传播获取所有步骤分数（比 DeepSeek PRM 增量评分更高效）。

用法:
    # 使用 HuggingFace 模型 ID
    python run_qwen_prm.py \\
        --input results/verify/verify_results.jsonl \\
        --output results/prm/prm_step_scores.jsonl \\
        --summary results/prm/prm_summary.md \\
        --limit 50

    # 指定本地模型路径
    python run_qwen_prm.py \\
        --input results/verify/verify_results.jsonl \\
        --output results/prm/prm_step_scores.jsonl \\
        --summary results/prm/prm_summary.md \\
        --model models/Qwen2.5-Math-PRM-7B

    # 8bit 量化（节省显存）
    python run_qwen_prm.py \\
        --input results/verify/verify_results.jsonl \\
        --output results/prm/prm_step_scores.jsonl \\
        --summary results/prm/prm_summary.md \\
        --load_in_8bit --limit 50
"""

import argparse
import json
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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

# Qwen PRM 推荐的系统提示（与模型推理时一致）
DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


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


# ── PRM 模型加载与评分 ────────────────────────────────────────────────────


class QwenPRMScorer:
    """Qwen2.5-Math PRM 评分器。

    使用 <extra_0> token 分隔步骤，单次前向传播获取所有步骤分数。
    参考: https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 4096,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.step_sep_id = None

        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # dtype：auto 根据设备和 GPU 能力自动选择
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        if torch_dtype == "auto":
            if self.device == "cpu":
                resolved_dtype = torch.float32
            elif torch.cuda.is_bf16_supported():
                resolved_dtype = torch.bfloat16
            else:
                resolved_dtype = torch.float16  # V100 等不支持 bf16 的 GPU
        else:
            resolved_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.torch_dtype = resolved_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self._load_model()

    def _load_model(self):
        """加载 Qwen PRM 模型。"""
        print(f"加载 Qwen PRM 模型: {self.model_name} ...")

        kwargs = {
            "device_map": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }

        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype", None)
        elif self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs.pop("torch_dtype", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

        # 获取 <extra_0> token id（Qwen PRM 用于标记步骤边界的特殊 token）
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"Qwen PRM 模型加载完成")
        print(f"  设备: {self.device}")
        print(f"  精度: {self.torch_dtype}")
        print(f"  参数量: {param_count:.1f}B")
        print(f"  <extra_0> token id: {self.step_sep_id}")

    @staticmethod
    def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> list[list[float]]:
        """从 PRM 输出 logits 中提取每个步骤的正类概率。

        Qwen PRM 的输出头是二分类（positive/negative），在每个 <extra_0> 位置
        输出该步骤正确的概率。

        Args:
            logits: 模型输出, shape (batch, seq_len, num_labels=2)
            token_masks: <extra_0> 位置掩码, shape (batch, seq_len)

        Returns:
            每个 batch 中每个步骤的正类概率列表
        """
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # (bs, seq_len, 2)

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # (seq_len, 2)
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # 取正类概率
            all_scores_res.append(positive_probs.cpu().tolist())
        return all_scores_res

    def score_sample(
        self,
        question: str,
        model_output: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        """对一个样本的所有步骤评分。

        核心流程:
        1. 将模型输出切分为步骤
        2. 用 <extra_0> 连接步骤，构造 chat template 输入
        3. 单次前向传播
        4. 在 <extra_0> token 位置提取步骤分数

        Args:
            question: 题目文本
            model_output: 模型完整输出
            system_prompt: 推理时使用的系统提示

        Returns:
            包含步骤分数和统计信息的字典
        """
        steps = split_steps(model_output)
        if not steps:
            return {
                "steps": [],
                "min_step_score": 0.0,
                "avg_step_score": 0.0,
                "first_wrong_step_index": -1,
            }

        # 构造 assistant 消息：步骤用 <extra_0> 分隔，末尾也加 <extra_0>
        assistant_content = "<extra_0>".join(steps) + "<extra_0>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]

        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        # 构建 token mask：标记所有 <extra_0> 的位置
        token_masks = (input_ids == self.step_sep_id)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        # 提取步骤分数
        step_scores_raw = self.make_step_rewards(outputs[0], token_masks)

        if not step_scores_raw or not step_scores_raw[0]:
            return {
                "steps": [],
                "min_step_score": 0.0,
                "avg_step_score": 0.0,
                "first_wrong_step_index": -1,
            }

        scores = step_scores_raw[0]

        # 构建步骤结果（处理截断导致步骤数与分数数不一致的情况）
        n_valid = min(len(steps), len(scores))
        if n_valid < len(steps):
            print(f"    [WARN] 步骤数({len(steps)}) > 分数数({len(scores)})，"
                  f"可能因 max_length={self.max_length} 截断导致，仅保留前 {n_valid} 步")

        step_results = []
        for i in range(n_valid):
            step_results.append({
                "step_index": i + 1,
                "text": steps[i],
                "score": round(scores[i], 4),
                "label": "correct" if scores[i] >= WRONG_STEP_THRESHOLD else "wrong",
            })

        if not step_results:
            return {
                "steps": [],
                "min_step_score": 0.0,
                "avg_step_score": 0.0,
                "first_wrong_step_index": -1,
            }

        valid_scores = [s["score"] for s in step_results]
        min_score = min(valid_scores)
        avg_score = sum(valid_scores) / len(valid_scores)

        first_wrong = -1
        for i, s in enumerate(step_results):
            if s["score"] < WRONG_STEP_THRESHOLD:
                first_wrong = i + 1
                break

        return {
            "steps": step_results,
            "min_step_score": round(min_score, 4),
            "avg_step_score": round(avg_score, 4),
            "first_wrong_step_index": first_wrong,
        }


# ── 归因 ─────────────────────────────────────────────────────────────────


def classify_error(prm_result: dict) -> tuple[str, bool]:
    """根据 PRM 评分结果归因。返回 (error_category, need_render_check)。

    归因规则:
    - min_step_score < 0.45 → 解题逻辑错误
    - avg_step_score >= 0.70 但答案错 → 疑似渲染/答案抽取问题
    - 其他 → 不确定
    """
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
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Math PRM 过程评分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Math-Verify 结果 JSONL")
    parser.add_argument("--output", required=True, help="PRM 评分结果 JSONL")
    parser.add_argument("--summary", required=True, help="汇总报告 Markdown")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="PRM 模型 HuggingFace ID 或本地路径 (默认: Qwen/Qwen2.5-Math-PRM-7B)",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 条错误样本")
    parser.add_argument("--max_samples", type=int, default=None, help="同 --limit（兼容旧参数）")
    parser.add_argument("--max_length", type=int, default=4096, help="最大输入 token 长度 (默认: 4096)")
    parser.add_argument("--device", default="auto", help="设备: auto/cuda/cpu (默认: auto)")
    parser.add_argument("--load_in_8bit", action="store_true", help="8bit 量化加载（约 8-10GB 显存）")
    parser.add_argument("--load_in_4bit", action="store_true", help="4bit 量化加载（约 5-7GB 显存）")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="模型精度 (默认: auto，根据 GPU 自动选择)",
    )
    parser.add_argument("--system_prompt", default=None, help="推理时使用的系统提示（默认使用 Qwen PRM 推荐提示）")
    args = parser.parse_args()

    limit = args.limit or args.max_samples

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen2.5-Math PRM 过程评分")
    print(f"  输入:   {input_path}")
    print(f"  输出:   {output_path}")
    print(f"  模型:   {args.model}")
    print(f"  设备:   {args.device}")
    print(f"  精度:   {args.torch_dtype}")
    print(f"  长度:   {args.max_length}")
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
        summary_path.write_text("# Qwen2.5-Math PRM 评分汇总\n\n无错误样本\n", encoding="utf-8")
        return

    # 加载模型
    scorer = QwenPRMScorer(
        args.model,
        device=args.device,
        max_length=args.max_length,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    # 评分
    results = []
    for i, sample in enumerate(wrong_samples):
        qid = sample.get("id", f"unknown_{i}")
        question = sample.get("question", "")
        model_output = sample.get("model_output", "")

        print(f"  [{i + 1}/{len(wrong_samples)}] 评分: {qid}")

        try:
            prm_result = scorer.score_sample(question, model_output, system_prompt=system_prompt)
        except Exception as e:
            print(f"    [ERROR] {e}")
            traceback.print_exc()
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
        path.write_text("# Qwen2.5-Math PRM 评分汇总\n\n无样本\n", encoding="utf-8")
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
        "# Qwen2.5-Math PRM 过程评分汇总报告\n",
        "## 总体统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 错误样本总数 | {total} |",
        f"| 解题逻辑错误 | {logic_errors} |",
        f"| 疑似渲染/答案抽取问题 | {render_suspects} |",
        f"| 不确定 | {uncertain} |",
        f"| 平均步骤分 (avg) | {sum(avg_scores) / len(avg_scores):.4f} |",
        f"| 最低步骤分 (min) | {sum(min_scores) / len(min_scores):.4f} |",
        "",
        "## 归因分布",
        "",
        "| 类别 | 数量 | 占比 |",
        "|------|------|------|",
        f"| logic_error | {logic_errors} | {logic_errors / total * 100:.1f}% |",
        f"| answer_or_render_suspect | {render_suspects} | {render_suspects / total * 100:.1f}% |",
        f"| uncertain_logic | {uncertain} | {uncertain / total * 100:.1f}% |",
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

    # PRM 高分但答案错误的样本（最值得分析）
    high_prm_wrong = [r for r in results if r["avg_step_score"] >= LOGIC_CORRECT_THRESHOLD]
    if high_prm_wrong:
        lines.extend(["", "## PRM 高分但答案错误（最值得分析）", ""])
        lines.append(f"共 {len(high_prm_wrong)} 条，按 avg_score 降序排列：\n")
        for r in sorted(high_prm_wrong, key=lambda x: x["avg_step_score"], reverse=True)[:10]:
            lines.append(
                f"- **{r['id']}** ({r.get('source', '')}): "
                f"avg_score={r['avg_step_score']:.4f}, "
                f"min_score={r['min_step_score']:.4f}, "
                f"steps={len(r['steps'])}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
