"""
流水线离线测试脚本。

用法:
    # 处理指定 id
    python test_pipeline.py --id gaokao_000829

    # 处理多个 id
    python test_pipeline.py --id gaokao_000811 gaokao_000814

    # 处理全部样本
    python test_pipeline.py

    # 指定数据文件
    python test_pipeline.py --file ../results/prm/other.jsonl --id gaokao_000123
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 抑制 SentenceBuffer 的 debug/warning 日志噪音
logging.getLogger("app.sentence_buffer").setLevel(logging.ERROR)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="ERROR")
except ImportError:
    pass

from app.math_render_service import load_jsonl, _process_model_output


def main():
    parser = argparse.ArgumentParser(description="流水线离线测试")
    parser.add_argument(
        "--file",
        type=str,
        default="../results/prm/prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl",
        help="JSONL 数据文件路径",
    )
    parser.add_argument("--id", type=str, nargs="+", default=None, help="指定记录 id（支持多个，不指定则处理全部）")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"文件不存在: {file_path}", file=sys.stderr)
        sys.exit(1)

    data = load_jsonl(file_path)
    print(f"共加载 {len(data)} 条记录")

    # 筛选目标记录
    if args.id:
        missing = [rid for rid in args.id if rid not in data]
        if missing:
            print(f"id 未找到: {missing}，可用 id 前几个: {list(data.keys())[:10]}", file=sys.stderr)
            sys.exit(1)
        targets = {rid: data[rid] for rid in args.id}
    else:
        targets = data

    # 逐条跑管线
    all_sentences = []
    all_latex_fixed = []

    for idx, (rid, record) in enumerate(targets.items(), 1):
        model_output = record.get("model_output", "")
        if not model_output:
            print(f"[{idx}/{len(targets)}] {rid}: model_output 为空，跳过")
            continue

        print(f"[{idx}/{len(targets)}] 处理 {rid} (原始长度 {len(model_output)})")

        result = asyncio.run(_process_model_output(rid, model_output))

        # 保留原始 model_output 方便对照
        result["model_output_original"] = model_output
        all_sentences.append({"id": rid, "segments": result["segments"]})
        all_latex_fixed.append({
            "id": rid,
            "fixed_segments": result["fixed_segments"],
            "model_output_new": result["model_output_new"],
        })

        print(f"  → 断句 {len(result['segments'])} 段, 修复后 {len(result['fixed_segments'])} 段")

    # 写中间结果
    stem = file_path.stem
    out_dir = file_path.parent
    sentences_file = out_dir / f"{stem}_sentences.json"
    latex_fixed_file = out_dir / f"{stem}_latex_fixed.json"

    sentences_file.write_text(json.dumps(all_sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    latex_fixed_file.write_text(json.dumps(all_latex_fixed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成。共处理 {len(all_sentences)} 条")
    print(f"断句结果: {sentences_file}")
    print(f"修复结果: {latex_fixed_file}")


if __name__ == "__main__":
    main()
