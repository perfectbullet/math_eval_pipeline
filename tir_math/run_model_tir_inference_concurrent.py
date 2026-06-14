#!/usr/bin/env python3
"""
TIR/COT 并发批量推理脚本。

- `tir` 模式：使用 qwen_agent 的 `TIRMathAgent`
- `cot` 模式：使用 qwen_agent 的 `Assistant(function_list=[])`

用法:
    cd tir_math/
    python run_model_tir_inference_concurrent.py \
        --input ../data/math_questions_31.jsonl \
        --output ../data/model_outputs/model_outputs-concurrent-test.jsonl \
        --api_base http://192.168.100.203:8200/v1 \
        --model Qwen2.5-Math-72B-Instruct \
        --workers 4 \
        --limit 10
"""

import argparse
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from qwen_agent.agents import Assistant, TIRMathAgent
from tqdm import tqdm

TIR_SYSTEM_EN = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
COT_SYSTEM_EN = """Please reason step by step, and put your final answer within \\boxed{}."""

TIR_SYSTEM_ZH = "请结合自然语言推理和程序来解决上述问题，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"
COT_SYSTEM_ZH = "请逐步推理，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"

# (mode, lang) → system prompt
SYSTEM_PROMPTS = {
    ("tir", "zh"): TIR_SYSTEM_ZH,
    ("tir", "en"): TIR_SYSTEM_EN,
    ("cot", "zh"): COT_SYSTEM_ZH,
    ("cot", "en"): COT_SYSTEM_EN,
}


def run_agent_query(bot, question: str) -> tuple[str, dict]:
    """调用 agent 推理单条题目。

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
        if not model_output.strip():
            meta["error"] = "empty_model_output"
        return model_output, meta

    except Exception as e:
        meta["error"] = str(e)
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta


def _read_existing_best_records(output_path: Path) -> dict[str, dict]:
    """读取已有输出并按 ID 去重。

    规则：
    1. 优先保留有 `model_output` 的成功记录；
    2. 若均失败，则保留最后一条失败记录。
    """
    best_records: dict[str, dict] = {}
    if not output_path.exists():
        return best_records

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("id")
            if not qid:
                continue
            if rec.get("model_output", ""):
                best_records[qid] = rec
            elif qid not in best_records or not best_records[qid].get("model_output", ""):
                best_records[qid] = rec

    return best_records


def prepare_output_file(output_path: Path) -> tuple[set[str], dict[str, dict]]:
    """规范化已有输出文件并返回成功 ID 集合。"""
    best_records = _read_existing_best_records(output_path)
    existing_ids = {
        qid for qid, rec in best_records.items()
        if rec.get("model_output", "")
    }

    if output_path.exists():
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in best_records.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return existing_ids, best_records


def normalize_input_stem(input_path: Path) -> str:
    """把 `xxx.mineru.json` 规整成 `xxx`。"""
    stem = input_path.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    elif stem.endswith(".json"):
        stem = stem[:-5]
    if stem.endswith(".mineru"):
        stem = stem[:-7]
    return stem


def process_one(
    record: dict,
    llm_cfg: dict,
    system_msg: str,
    model_name: str,
    input_stem: str,
    fout_lock: threading.Lock,
    fout,
    mode,
    lang,
) -> dict:
    """单个 worker 处理一条题目：独立创建 bot → 推理 → 写入结果。"""
    qid = record["id"]
    question = record["question"]
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"  [START] {qid} mode={mode} lang={lang}")

    # 每个 worker 独立创建 bot，避免共享 agent 内部状态。
    if mode == "tir":
        agent_type = "TIRMathAgent"
        bot = TIRMathAgent(llm=llm_cfg, name=model_name, system_message=system_msg)
    else:
        agent_type = "Assistant"
        bot = Assistant(
            llm=llm_cfg,
            name=model_name,
            system_message=system_msg,
            function_list=[],
        )
    model_output, meta = run_agent_query(bot, question)
    finished_at = datetime.now().isoformat(timespec="seconds")

    result = {
        "id": qid,
        "source": f"{input_stem}-{record.get('source', '')}",
        "question": question,
        "reference_answer": record.get("reference_answer", ""),
        "model_name": f"{model_name}_{mode}",
        "model_output": model_output,
        "final_answer_raw": "",
        "metadata": {
            "answer_type": record.get("answer_type", "expr"),
            "mode": mode,
            "lang": lang,
            "agent_type": agent_type,
            "started_at": started_at,
            "finished_at": finished_at,
            "retry_count": 0,
            **meta,
        },
    }

    # 线程安全写入
    with fout_lock:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="TIR/COT 并发批量推理"
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
    parser.add_argument(
        "--api_key", default="no-key", help="API Key（本地部署可忽略，默认 no-key）"
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument(
        "--workers", type=int, default=8, help="并发 worker 数量（默认 8）"
    )
    parser.add_argument(
        "--cot_workers",
        type=int,
        default=None,
        help="COT 模式并发数，未指定时回退到 --workers",
    )
    parser.add_argument(
        "--tir_workers",
        type=int,
        default=None,
        help="TIR 模式并发数，未指定时回退到 --workers",
    )
    parser.add_argument(
        "--warn_after",
        type=int,
        default=600,
        help="单条任务运行超过多少秒后打印卡顿告警，默认 600 秒",
    )
    parser.add_argument(
        "--status_interval",
        type=int,
        default=60,
        help="运行中状态汇总打印间隔秒数，默认 60 秒",
    )
    parser.add_argument(
        "--mode",
        choices=["tir", "cot"],
        default="tir",
        help="推理模式：tir=TIRMathAgent，cot=Assistant(function_list=[])",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="系统提示词语言：zh=中文（默认），en=英文",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有结果（默认增量），并先做一轮去重规整
    existing_ids, existing_records = prepare_output_file(output_path)

    # 读取输入（兼容 JSON 数组和 JSONL 两种格式）
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            pass
        elif content[0] == "[":
            # JSON 数组格式（如 .mineru.json）
            records = json.loads(content)
        else:
            # JSONL 格式（每行一个 JSON 对象）
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    total = len(records)
    # 过滤掉 question 为空的记录
    pending = [r for r in records if r["id"] not in existing_ids and r.get("question")]
    skipped = total - len(pending)
    effective_workers = (
        args.tir_workers if args.mode == "tir" and args.tir_workers is not None
        else args.cot_workers if args.mode == "cot" and args.cot_workers is not None
        else args.workers
    )
    input_stem = normalize_input_stem(input_path)

    print("=" * 60)
    print("TIR 并发批量推理")
    print(f"API:    {args.api_base}")
    print(f"模型:   {args.model}")
    print(f"模式:   {args.mode.upper()} ({args.lang})")
    print(f"并发:   {effective_workers} workers")
    print(f"输入:   {input_path} ({total} 条)")
    print(f"输出:   {output_path}")
    print(f"已规整: {len(existing_records)} 条历史记录")
    print(f"已完成: {skipped} 条，待处理: {len(pending)} 条")
    print("=" * 60)

    llm_cfg = {
        "model": args.model,
        "model_server": args.api_base,
        "api_key": args.api_key,
        "generate_cfg": {
            # "temperature": 0.1,
            # "top_p": 0.95,
            "max_tokens": 30720,
        },
    }

    success, failed = 0, 0
    fout_lock = threading.Lock()
    system_msg = SYSTEM_PROMPTS[(args.mode, args.lang)]
    last_status_log_at = time.time()

    if not pending:
        print("\n" + "=" * 60)
        print("没有待处理任务，直接结束")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        print(f"跳过: {skipped}")
        print("=" * 60)
        return

    with open(output_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_info = {
                pool.submit(
                    process_one,
                    record,
                    llm_cfg,
                    system_msg,
                    args.model,
                    input_stem,
                    fout_lock,
                    fout,
                    args.mode,
                    args.lang,
                ): {
                    "record": record,
                    "submitted_at": time.time(),
                    "warned": False,
                }
                for record in pending
            }
            pending_futures = set(future_info.keys())

            with tqdm(total=len(pending_futures), desc="TIR 并发推理中") as pbar:
                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures,
                        timeout=5,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done:
                        now = time.time()
                        if now - last_status_log_at >= args.status_interval:
                            active_infos = [
                                (info["record"]["id"], now - info["submitted_at"])
                                for future, info in future_info.items()
                                if future in pending_futures
                            ]
                            active_infos.sort(key=lambda item: item[1], reverse=True)
                            top_active = ", ".join(
                                f"{qid}:{elapsed:.1f}s"
                                for qid, elapsed in active_infos[: min(3, len(active_infos))]
                            )
                            print(
                                f"  [STATUS] mode={args.mode} lang={args.lang} "
                                f"active={len(pending_futures)} success={success} failed={failed} "
                                f"top_running=[{top_active}]"
                            )
                            last_status_log_at = now
                        for future in pending_futures:
                            info = future_info[future]
                            elapsed = now - info["submitted_at"]
                            if elapsed >= args.warn_after and not info["warned"]:
                                qid = info["record"]["id"]
                                print(
                                    f"  [WARN] {qid} 运行 {elapsed:.1f}s 仍未完成，"
                                    f"mode={args.mode}, lang={args.lang}"
                                )
                                info["warned"] = True
                        continue

                    for future in done:
                        record = future_info[future]["record"]
                        try:
                            result = future.result()
                            if result.get("metadata", {}).get("error"):
                                failed += 1
                                print(f"  [FAIL] {result['id']}: {result['metadata']['error']}")
                            else:
                                success += 1
                                print(
                                    f"  [OK] {result['id']} "
                                    f"({result['metadata'].get('latency_seconds', '?')}s)"
                                )
                        except Exception as e:
                            failed += 1
                            print(f"  [FAIL] {record['id']}: {e}")
                        finally:
                            pbar.update(1)

    print("\n" + "=" * 60)
    print("推理完成")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
