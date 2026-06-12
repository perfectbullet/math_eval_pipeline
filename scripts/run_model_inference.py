#!/usr/bin/env python3
"""
模型推理脚本：将标准化数据集发送给 vLLM 数学模型，收集推理结果。

用法:
    python run_model_inference.py \
        --input ../data/standardized/train_or_eval_all.jsonl \
        --output ../data/model_outputs/model_outputs.jsonl \
        --api_base http://192.168.8.231:8000/v1 \
        --model Qwen3-32B-GPTQ-Int8 \
        --limit 100
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI, APIConnectionError, APITimeoutError
from tqdm import tqdm


# 数学解题系统提示
# 这个是官方推荐的提示词
MATH_SYSTEM_PROMPT = """请逐步推理，并将您的最终答案放在\\boxed{}内。"""


def call_vllm(
    question: str,
    client: OpenAI,
    model: str,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> tuple[str, dict]:
    """
    调用 vLLM OpenAI 兼容 API。
    返回: (模型输出文本, 元数据)
    """
    meta = {}
    start_time = time.time()

    extra_body = {
        # enable thinking, set to False to disable test
        # "enable_thinking": True,
        # use thinking_budget to contorl num of tokens used for thinking
        # "thinking_budget": 4096
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            # temperature=temperature,
            timeout=timeout,
            extra_body=extra_body
        )

        content = resp.choices[0].message.content or ""
        usage = resp.usage

        meta["latency_seconds"] = round(time.time() - start_time, 2)
        meta["prompt_tokens"] = usage.prompt_tokens if usage else 0
        meta["completion_tokens"] = usage.completion_tokens if usage else 0
        meta["total_tokens"] = usage.total_tokens if usage else 0
        meta["finish_reason"] = resp.choices[0].finish_reason or ""

        return content, meta

    except APITimeoutError:
        meta["error"] = "timeout"
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta
    except APIConnectionError:
        meta["error"] = "connection_error"
        return "", meta
    except Exception as e:
        meta["error"] = str(e)
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta


def main():
    parser = argparse.ArgumentParser(description="调用 vLLM 模型做数学推理")
    parser.add_argument("--input", required=True, help="标准化数据集 JSONL")
    parser.add_argument("--output", required=True, help="模型输出 JSONL")
    parser.add_argument(
        "--api_base", default="http://192.168.8.231:8000/v1", help="vLLM API 地址"
    )
    parser.add_argument(
        "--api_key", default="EMPTY", help="API Key（vLLM 本地部署可忽略）"
    )
    parser.add_argument("--model", default="Qwen3-32B-GPTQ-Int8", help="模型名称")
    parser.add_argument("--limit", type=int, default=None, help="最多推理 N 条")
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="最大生成 token 数"
    )
    # parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--timeout", type=int, default=120, help="单条超时（秒）")
    parser.add_argument(
        "--skip_existing", action="store_true", help="跳过已有结果（断点续跑）"
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有结果（断点续跑，只跳过成功的，失败记录重试）
    existing_ids = set()
    if args.skip_existing and output_path.exists():
        # 按ID去重：优先保留成功记录，多条失败时保留最后一条
        best_records = {}  # id -> json line
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    rec = json.loads(line_stripped)
                except (json.JSONDecodeError, KeyError):
                    continue
                qid = rec["id"]
                # 成功记录直接覆盖（优先级最高）
                if rec.get("model_output", ""):
                    best_records[qid] = line_stripped
                    existing_ids.add(qid)
                elif qid not in best_records:
                    # 仅在没有成功记录时保留失败记录
                    best_records[qid] = line_stripped

        # 去重后写回文件
        with open(output_path, "w", encoding="utf-8") as f:
            for line in best_records.values():
                f.write(line + "\n")

        print(f"已有 {len(existing_ids)} 条成功结果，将跳过")

    # 读取输入
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if args.limit:
        records = records[: args.limit]
        total = len(records)
        print(f"限制处理 {total} 条")

    print("=" * 60)
    print("模型推理开始")
    print(f"API: {args.api_base}")
    print(f"模型: {args.model}")
    print(f"输入: {input_path} ({total} 条)")
    print(f"输出: {output_path}")
    print("=" * 60)

    # 创建 OpenAI 客户端
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # 测试连接
    try:
        client.models.list()
        print("API 连接正常")
    except Exception as e:
        print(f"[ERROR] 无法连接到 {args.api_base}: {e}")
        return

    success, failed, skipped = 0, 0, 0
    total_tokens = 0
    total_latency = 0.0

    # 追加模式写入
    mode = "a" if args.skip_existing else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for record in tqdm(records, desc="推理中"):
            qid = record["id"]

            if qid in existing_ids:
                skipped += 1
                continue

            question = record["question"]
            answer_type = record.get("answer_type", "expr")

            model_output, meta = call_vllm(
                question=question,
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                # temperature=args.temperature,
                timeout=args.timeout,
            )

            if meta.get("error"):
                failed += 1
                print(f"  [FAIL] {qid}: {meta['error']}")
            else:
                success += 1
                total_tokens += meta.get("total_tokens", 0)
                total_latency += meta.get("latency_seconds", 0)

            result = {
                "id": qid,
                "source": record.get("source", ""),
                "question": question,
                "reference_answer": record.get("reference_answer", ""),
                "model_name": args.model,
                "model_output": model_output,
                "final_answer_raw": "",
                "metadata": {
                    "answer_type": answer_type,
                    **meta,
                },
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    # 统计
    print("\n" + "=" * 60)
    print("推理完成")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    if success > 0:
        print(f"平均延迟: {total_latency / success:.2f}s")
        print(f"总 token: {total_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    main()
