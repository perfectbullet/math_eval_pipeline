#!/usr/bin/env python3
"""
TIR（工具集成推理）批量推理脚本：用 qwen_agent 的 TIRMathAgent 处理数学题。

用法:
    cd tir_math/
    python run_model_tir_inference.py \
        --input ../data/math_questions_31.jsonl \
        --output ../data/model_outputs/model_outputs-math_questions_31-Qwen2.5-Math-72B-Instruct.jsonl \
        --api_base http://192.168.100.203:8200/v1 \
        --model Qwen2.5-Math-72B-Instruct \
        --limit 10
"""

import argparse
import json
import time
from pathlib import Path

from qwen_agent.agents import TIRMathAgent
from tqdm import tqdm
from loguru import logger

# TIR_SYSTEM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
# COT_SYSTEM = (
#     """Please reason step by step, and put your final answer within \\boxed{}."""
# )
TIR_SYSTEM_ZH = "请结合自然语言推理和程序来解决上述问题，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"
COT_SYSTEM_ZH = "请逐步推理，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"


def run_tir_query(bot: TIRMathAgent, question: str) -> tuple[str, dict]:
    """调用 TIR agent 推理单条题目。

    Returns:
        (model_output, metadata) — metadata 包含 latency_seconds
    """
    meta = {}
    start_time = time.time()

    try:
        messages = [{"role": "user", "content": question}]
        last_response = None
        for response in bot.run(messages):
            last_response = response

        # 从最后的响应中提取 assistant 内容
        model_output = ""
        if last_response:
            if isinstance(last_response, list):
                # 取最后一条 assistant 消息
                for msg in reversed(last_response):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = "\n".join(
                                p.get("text", "")
                                for p in content
                                if isinstance(p, dict) and "text" in p
                            )
                        model_output = content
                        break
            elif isinstance(last_response, dict):
                model_output = last_response.get("content", "")

        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return model_output, meta

    except Exception as e:
        meta["error"] = str(e)
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        logger.exception(e)
        return "", meta


def load_existing_ids(output_path: Path) -> set[str]:
    """读取已有输出文件，收集成功推理的 ID 集合。"""
    existing_ids: set[str] = set()
    if not output_path.exists():
        return existing_ids

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model_output", ""):
                existing_ids.add(rec["id"])

    return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="TIR 批量推理：用 TIRMathAgent 处理数学题"
    )
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--api_base",
        required=True,
        help="模型 API 地址（如 http://192.168.100.203:8200/v1）",
    )
    parser.add_argument(
        "--model", required=True, help="模型名称（如 Qwen2.5-Math-72B-Instruct）"
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有结果（默认增量）
    existing_ids = load_existing_ids(output_path)

    # 读取输入
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    total = len(records)
    pending = [r for r in records if r["id"] not in existing_ids]
    skipped = total - len(pending)

    print("=" * 60)
    print("TIR 批量推理")
    print(f"API:    {args.api_base}")
    print(f"模型:   {args.model}")
    print(f"输入:   {input_path} ({total} 条)")
    print(f"输出:   {output_path}")
    print(f"已完成: {skipped} 条，待处理: {len(pending)} 条")
    print("=" * 60)

    # 初始化 TIR agent
    llm_cfg = {
        "model": args.model,
        "model_server": args.api_base,
        "api_key": "no-key",
        "generate_cfg": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 8012,
        },
    }
    bot = TIRMathAgent(llm=llm_cfg, name=args.model, system_message=TIR_SYSTEM_ZH)

    success, failed = 0, 0

    with open(output_path, "a", encoding="utf-8") as fout:
        for record in tqdm(pending, desc="TIR 推理中"):
            qid = record["id"]
            question = record["question"]

            model_output, meta = run_tir_query(bot, question)

            if meta.get("error"):
                failed += 1
                print(f"  [FAIL] {qid}: {meta['error']}")
            else:
                success += 1

            result = {
                "id": qid,
                "source": record.get("source", ""),
                "question": question,
                "reference_answer": record.get("reference_answer", ""),
                "model_name": args.model,
                "model_output": model_output,
                "final_answer_raw": "",
                "metadata": {
                    "answer_type": record.get("answer_type", "expr"),
                    **meta,
                },
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print("\n" + "=" * 60)
    print("推理完成")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
