#!/usr/bin/env python3
"""
模型推理脚本：将标准化数据集发送给 Ollama 数学模型，收集推理结果。

用法:
    python run_model_inference.py \
        --input ../data/standardized/train_or_eval_all.jsonl \
        --output ../data/model_outputs/model_outputs.jsonl \
        --api_base http://192.168.8.231:11434/v1 \
        --model qwen2.5-math-Q6_K_L:7b \
        --limit 100
"""

import argparse
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm


# 数学解题系统提示
# 这个是官方推荐的提示词
MATH_SYSTEM_PROMPT = """请逐步推理，并将您的最终答案放在\boxed{}内。"""


def call_ollama(
    question: str,
    api_base: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout: int = 120,
) -> tuple[str, dict]:
    """
    调用 Ollama OpenAI 兼容 API。
    返回: (模型输出文本, 元数据)
    """
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    meta = {}
    start_time = time.time()

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        meta["latency_seconds"] = round(time.time() - start_time, 2)
        meta["prompt_tokens"] = usage.get("prompt_tokens", 0)
        meta["completion_tokens"] = usage.get("completion_tokens", 0)
        meta["total_tokens"] = usage.get("total_tokens", 0)
        meta["finish_reason"] = data["choices"][0].get("finish_reason", "")

        return content, meta

    except requests.exceptions.Timeout:
        meta["error"] = "timeout"
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta
    except requests.exceptions.ConnectionError:
        meta["error"] = "connection_error"
        return "", meta
    except Exception as e:
        meta["error"] = str(e)
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta


def main():
    parser = argparse.ArgumentParser(description="调用 Ollama 模型做数学推理")
    parser.add_argument("--input", required=True, help="标准化数据集 JSONL")
    parser.add_argument("--output", required=True, help="模型输出 JSONL")
    parser.add_argument("--api_base", default="http://192.168.8.231:11434/v1", help="Ollama API 地址")
    parser.add_argument("--model", default="qwen2.5-math-Q6_K_L:7b", help="模型名称")
    parser.add_argument("--limit", type=int, default=None, help="最多推理 N 条")
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--timeout", type=int, default=120, help="单条超时（秒）")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已有结果（断点续跑）")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有结果（断点续跑，只跳过成功的，失败记录重试）
    existing_ids = set()
    if args.skip_existing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("model_output", ""):
                            existing_ids.add(rec["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
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
        records = records[:args.limit]
        total = len(records)
        print(f"限制处理 {total} 条")

    print("=" * 60)
    print("模型推理开始")
    print(f"API: {args.api_base}")
    print(f"模型: {args.model}")
    print(f"输入: {input_path} ({total} 条)")
    print(f"输出: {output_path}")
    print("=" * 60)

    # 先测试连接
    try:
        resp = requests.get(f"{args.api_base}/models", timeout=10)
        resp.raise_for_status()
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

            model_output, meta = call_ollama(
                question=question,
                api_base=args.api_base,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
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
