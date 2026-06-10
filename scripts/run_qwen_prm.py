#!/usr/bin/env python3
"""
Qwen2.5-Math PRM 过程评分脚本：对 Math-Verify 错误样本做步骤级评分。

基于 Qwen2.5-Math-PRM-7B 模型，使用 <extra_0> token 分隔步骤，
单次前向传播获取所有步骤分数（比 DeepSeek PRM 增量评分更高效）。

用法:
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
import html
import json
import markdown
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 兼容旧版 transformers：DynamicCache 缺少 get_usable_length 方法
from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        past_length = self._seen_tokens if hasattr(self, "_seen_tokens") else 0
        return past_length
    DynamicCache.get_usable_length = _get_usable_length

# ── 步骤切分 ──────────────────────────────────────────────────────────────

STEP_MARKERS = [
    re.compile(r"(?:^|\n)\s*(?:步骤|step)\s*(\d+)\s*[.。：:]\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*（(\d+)）\s*"),
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.。]\s+"),
]

SENTENCE_SPLIT = re.compile(r"[。；\n]")

# 装饰性行：水平分隔线、Markdown 标题、子题标题
_DECORATIVE_RE = re.compile(
    r"^(?:-{3,}|={3,}|#{1,4}\s|"
    r"第\s*[（(]\s*[IVXivx]+\s*[）)]\s*[部分问]|"
    r"第\s*[（(]\s*\d+\s*[）)]\s*[部分问]|"
    r"\*{3,})"
)

# 归因阈值
WRONG_STEP_THRESHOLD = 0.45
LOGIC_CORRECT_THRESHOLD = 0.70

# Qwen PRM 推荐的系统提示（与模型推理时一致）
DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _merge_paragraphs(paragraphs: list[str], min_length: int = 20) -> list[str]:
    """后处理：过滤装饰行，合并过短段落到相邻步骤。

    策略：
    1. 跳过纯装饰性行（---, ### 标题等）
    2. 过短的段落（< min_length）向后合并到下一个段落
       （通常是引导语如 "代入公式：" 后面紧跟公式块）
    """
    # 先过滤装饰行
    filtered = [p for p in paragraphs if not _DECORATIVE_RE.match(p)]

    if not filtered:
        return filtered

    # 合并过短段落
    merged = []
    i = 0
    while i < len(filtered):
        p = filtered[i]
        # 当前段落太短，且有下一段 → 合并
        if len(p) < min_length and i + 1 < len(filtered):
            merged.append(p + "\n\n" + filtered[i + 1])
            i += 2
        else:
            merged.append(p)
            i += 1

    return merged


def split_steps(text: str, min_length: int = 5) -> list[str]:
    """将模型输出切分为步骤。

    遵循 Qwen PRM 官方推荐：使用双换行 (\\n\\n) 分隔步骤。
    三级降级：明确步骤标记 → 双换行（含后处理） → 句号。

    后处理：过滤装饰行（---, ### 标题）并合并过短引导段落。

    参考: https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
    > We recommend using double line breaks ("\\n\\n") to separate individual steps.
    """
    if not text:
        return []

    # 1. 按明确步骤标记切分（步骤一/Step 1/（1）/1. 等格式）
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

    # 2. 按双换行切分（Qwen PRM 官方推荐方式）
    #    这样 $$...$$ 数学块会保留在步骤内部，不会被拆碎
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        # 后处理：过滤装饰行 + 合并过短段落
        paragraphs = _merge_paragraphs(paragraphs)
        if paragraphs:
            return paragraphs

    # 3. 按句号切分（兜底：没有双换行分隔的长文本）
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
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.step_sep_id = None
        self.cache_dir = cache_dir

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
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype", None)
        elif self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs.pop("torch_dtype", None)

        tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.cache_dir:
            tok_kwargs["cache_dir"] = self.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tok_kwargs
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
    parser.add_argument("--cache_dir", default=None, help="HuggingFace 缓存目录（解决权限问题时指定）")
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
    skipped_empty = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("verify_correct", False):
                # 跳过模型输出为空的样本
                output = record.get("model_output", "").strip()
                if not output:
                    skipped_empty += 1
                    continue
                wrong_samples.append(record)

    print(f"从 {input_path} 中找到 {len(wrong_samples)} 条错误样本")
    if skipped_empty:
        print(f"  跳过 {skipped_empty} 条模型输出为空的样本")

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
        cache_dir=args.cache_dir,
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
    _write_summary(results, summary_path, skipped_empty)
    print(f"汇总报告: {summary_path}")

    # 生成错误样本 HTML
    _write_error_html(results, output_path.parent)
    print(f"错误样本 HTML: {output_path.parent / 'errors_prm/'}")


def _write_summary(results: list[dict], path: Path, skipped_empty: int = 0):
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
        f"| 模型输出为空（跳过） | {skipped_empty} |",
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


# ── 错误样本 HTML 报告 ────────────────────────────────────────────────────

# 数学块保护：markdown.markdown() 会破坏 LaTeX（* 变 <em>，{} 被吃等）
# 先把 $...$ 和 $$...$$ 替换为占位符，markdown 处理后再恢复
_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+?\$)", re.DOTALL)


def _safe_markdown(text: str) -> str:
    """安全地将含 LaTeX 的文本转为 HTML：先保护数学块，再 markdown，再恢复。"""
    if not text:
        return ""

    # 提取所有数学块并替换为占位符
    math_blocks = []

    def _replace(m):
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"%%MATH{idx}%%"

    protected = _MATH_BLOCK_RE.sub(_replace, text)

    # 防止行首的 "数字.空格" 被 Markdown 解析为有序列表
    # 例如 "0. 98." → "0\. 98."，渲染结果不变但不会被吞掉
    protected = re.sub(r"(^|\n)(\d+)\. ", r"\1\2\\. ", protected)

    # markdown 处理（此时文本中没有 $...$ 数学块了）
    result = markdown.markdown(protected, extensions=["fenced_code", "tables"])

    # 恢复数学块，HTML 转义防止 < > & 被浏览器解析为 HTML 标签
    for idx, block in enumerate(math_blocks):
        placeholder = f"%%MATH{idx}%%"
        escaped_block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        result = result.replace(placeholder, escaped_block)

    return result


_KATEX_HEAD = """\
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
body {{ font-family: "Noto Sans SC", sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; padding-bottom: 8px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.tag {{ padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600; color: #fff; }}
.tag-id {{ background: #2c3e50; }}
.tag-source {{ background: #2980b9; }}
.tag-model {{ background: #8e44ad; }}
.tag-category {{ background: #c0392b; }}
.tag-correct {{ background: #27ae60; }}
.section {{ margin-bottom: 20px; background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
.section-title {{ font-weight: bold; font-size: 15px; color: #2c3e50; margin-bottom: 8px; border-left: 4px solid #3498db; padding-left: 8px; }}
.section-content {{ white-space: pre-wrap; word-break: break-word; line-height: 1.7; font-size: 14px; }}
.section-content.scroll {{ max-height: 400px; overflow-y: auto; }}
.score-bar {{ display: inline-block; height: 14px; border-radius: 7px; }}
.score-bar.green {{ background: #27ae60; }}
.score-bar.red {{ background: #e74c3c; }}
.step-card {{ margin-bottom: 12px; border-radius: 6px; padding: 12px; border-left: 4px solid #bdc3c7; }}
.step-card.wrong {{ border-left-color: #e74c3c; background: #fdf0f0; }}
.step-card.correct {{ border-left-color: #27ae60; background: #f0fdf4; }}
.step-card.first-wrong {{ border-left-color: #e74c3c; background: #fde8e8; box-shadow: 0 0 0 2px #e74c3c; }}
.step-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.step-num {{ font-weight: bold; font-size: 14px; color: #2c3e50; }}
.step-score {{ font-size: 13px; color: #7f8c8d; }}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.summary-item {{ padding: 8px 12px; background: #f8f9fa; border-radius: 4px; }}
.summary-label {{ font-size: 12px; color: #7f8c8d; }}
.summary-value {{ font-size: 16px; font-weight: bold; color: #2c3e50; }}
</style>
</head>
<body>
<h1>{title}</h1>
"""


def _write_error_html(results: list[dict], output_dir: Path):
    """为 PRM 评分的错误样本生成独立 HTML 文件。"""
    html_dir = output_dir / "errors_prm"
    html_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        rid = r.get("id", "unknown")
        title = f"PRM 错误样本 - {rid}"
        buf = _KATEX_HEAD.format(title=title)

        # 头部标签
        category = r.get("error_category", "")
        category_map = {
            "logic_error": "解题逻辑错误",
            "answer_or_render_suspect": "疑似渲染问题",
            "uncertain_logic": "不确定",
        }
        buf += '<div class="meta">\n'
        buf += f'  <span class="tag tag-id">{html.escape(rid)}</span>\n'
        buf += f'  <span class="tag tag-source">{html.escape(r.get("source", ""))}</span>\n'
        buf += f'  <span class="tag tag-model">{html.escape(r.get("model_name", ""))}</span>\n'
        buf += f'  <span class="tag tag-category">{html.escape(category_map.get(category, "") or category)}</span>\n'
        logic_ok = r.get("prm_logic_correct", False)
        buf += f'  <span class="tag tag-correct">逻辑{"✓" if logic_ok else "✗"}</span>\n'
        buf += '</div>\n'

        # 评分汇总
        buf += '<div class="section">\n'
        buf += '  <div class="section-title">评分汇总</div>\n'
        buf += '  <div class="summary-grid">\n'
        for lbl, key in [("最低步骤分", "min_step_score"), ("平均步骤分", "avg_step_score"),
                          ("首个错误步骤", "first_wrong_step_index"), ("步骤数", None)]:
            if key:
                val = r.get(key, "")
            else:
                val = len(r.get("steps", []))
            buf += f'    <div class="summary-item"><div class="summary-label">{lbl}</div><div class="summary-value">{val}</div></div>\n'
        buf += '  </div>\n</div>\n'

        # 题目
        q_html = _safe_markdown(r.get("question", ""))
        buf += '<div class="section">\n'
        buf += f'  <div class="section-title">题目</div>\n'
        buf += f'  <div class="section-content">{q_html}</div>\n</div>\n'

        # 标准答案（纯文本，KaTeX 浏览器端渲染 $...$）
        a_html = html.escape(r.get("reference_answer", ""))
        buf += '<div class="section">\n'
        buf += '  <div class="section-title">标准答案</div>\n'
        buf += f'  <div class="section-content">{a_html}</div>\n</div>\n'

        # 步骤评分
        steps = r.get("steps", [])
        if steps:
            buf += '<div class="section">\n  <div class="section-title">步骤评分</div>\n'
            first_wrong = r.get("first_wrong_step_index", -1)
            for s in steps:
                idx = s.get("step_index", 0)
                score = s.get("score", 0)
                slabel = s.get("label", "")
                is_first_wrong = (idx == first_wrong)
                css_class = "first-wrong" if is_first_wrong else slabel
                bar_width = int(score * 100)
                bar_color = "green" if score >= 0.45 else "red"
                stext = _safe_markdown(s.get("text", ""))
                buf += f'  <div class="step-card {css_class}">\n'
                buf += f'    <div class="step-header">\n'
                buf += f'      <span class="step-num">步骤 {idx}</span>\n'
                buf += f'      <span class="score-bar {bar_color}" style="width:{bar_width}px"></span>\n'
                buf += f'      <span class="step-score">{score:.4f}</span>\n'
                if is_first_wrong:
                    buf += '      <span style="color:#c0392b;font-weight:bold">⚠ 首个错误</span>\n'
                buf += f'    </div>\n'
                buf += f'    <div class="section-content">{stext}</div>\n'
                buf += f'  </div>\n'
            buf += '</div>\n'

        # 模型输出
        out_html = _safe_markdown(r.get("model_output", ""))
        buf += '<div class="section">\n'
        buf += '  <div class="section-title">模型输出</div>\n'
        buf += f'  <div class="section-content">{out_html}</div>\n</div>\n'

        buf += "</body></html>"
        (html_dir / f"{rid}.html").write_text(buf, encoding="utf-8")


if __name__ == "__main__":
    main()
