#!/usr/bin/env python3
"""
构建 dummy 模型输出：将标准答案包进 \\boxed{} 用于验证流水线。

用法:
    python build_dummy_model_outputs.py \
        --dataset ../data/standardized/train_or_eval_all.jsonl \
        --output ../data/model_outputs/model_outputs_dummy.jsonl \
        --limit 100
"""

import argparse
import json
import re
from pathlib import Path


def build_dummy_output(record: dict, model_name: str = "dummy") -> dict:
    """将标准答案包装成 dummy 模型输出。"""
    ref_answer = record.get("reference_answer", "")
    answer_type = record.get("answer_type", "expr")

    # 对于选择题直接输出答案
    if answer_type == "choice":
        model_output = f"解题过程：\n经过分析，本题答案为 {ref_answer}\n最终答案：{ref_answer}"
        final_answer_raw = ref_answer
    else:
        # 对于表达式/填空题，把答案放进 \\boxed{}
        # 如果答案已经是 LaTeX 格式，直接用
        if ref_answer.startswith("$") and ref_answer.endswith("$"):
            boxed = f"\\boxed{{{ref_answer[1:-1]}}}"
        else:
            boxed = f"\\boxed{{{ref_answer}}}"

        model_output = f"解题过程：\n根据题意计算得：{boxed}\n最终答案：{boxed}"
        final_answer_raw = boxed

    return {
        "id": record["id"],
        "source": record.get("source", ""),
        "question": record.get("question", ""),
        "reference_answer": ref_answer,
        "model_name": model_name,
        "model_output": model_output,
        "final_answer_raw": final_answer_raw,
        "metadata": {
            "answer_type": answer_type,
            "is_dummy": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="构建 dummy 模型输出")
    parser.add_argument("--dataset", required=True, help="标准化数据集 JSONL")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 条")
    parser.add_argument("--model_name", default="dummy", help="模型名称")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(dataset_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and count >= args.limit:
                break

            record = json.loads(line)
            dummy = build_dummy_output(record, args.model_name)
            fout.write(json.dumps(dummy, ensure_ascii=False) + "\n")
            count += 1

    print(f"生成 {count} 条 dummy 模型输出 -> {output_path}")


if __name__ == "__main__":
    main()
