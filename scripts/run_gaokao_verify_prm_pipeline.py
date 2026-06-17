#!/usr/bin/env python3
"""
串行执行 gaokao_bench 的 Math-Verify、Qwen PRM 和汇总导出流程。

默认处理 4 组 case：tir_en、cot_en、cot_zh、tir_zh。
脚本建议放在 scripts/ 目录下运行，也会强制把子脚本执行目录设为 scripts/。

示例：
    cd scripts
    python run_gaokao_verify_prm_pipeline.py

只打印命令不执行：
    python run_gaokao_verify_prm_pipeline.py --dry-run

指定其他模型名：
    python run_gaokao_verify_prm_pipeline.py \
        --model-name qwen2.5-math-72b
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_CASES = ("tir_en", "cot_en", "cot_zh", "tir_zh")
DEFAULT_MODEL_NAME = "deepseek-r1-distill-llama-70b"
DEFAULT_PRM_MODEL = "/nfs-data/math_model_deployment/math_eval_pipeline/models/Qwen2.5-Math-PRM-7B"
DEFAULT_CACHE_DIR = "/data/metahuman_work/.hf_cache"


@dataclass(frozen=True)
class CaseInfo:
    case_name: str
    input_path: Path
    exp_key: str
    report_key: str
    verify_path: Path
    verify_summary_path: Path
    prm_path: Path
    prm_summary_path: Path


@dataclass(frozen=True)
class ParsedStem:
    exp_key: str
    case_name: str
    model_name: str
    report_key: str


def normalize_input_stem(path: Path) -> str:
    """从 input 文件名中去掉 model_outputs 前缀，得到实验 key。"""
    stem = path.stem
    return re.sub(r"^model_outputs[-_]", "", stem)


def parse_input_stem(path: Path) -> ParsedStem:
    """解析 input 文件名，识别 case、模型名和报告 key。

    支持两类文件名：
    1. model_outputs-gaokao_bench_20260616_cot_en_xxx.jsonl
    2. cot_en_xxx.jsonl

    对第 1 类：
      exp_key    = gaokao_bench_20260616_cot_en_xxx
      case_name  = cot_en
      model_name = xxx
      report_key = gaokao_bench_20260616_xxx

    对第 2 类：
      exp_key    = cot_en_xxx
      case_name  = cot_en
      model_name = xxx
      report_key = xxx
    """
    exp_key = normalize_input_stem(path)
    match = re.search(r"(?:^|_)(cot|tir)_(en|zh)_(.+)$", exp_key)
    if not match:
        raise ValueError(
            f"无法从文件名识别 case：{path.name}。"
            "期望包含 cot_en/cot_zh/tir_en/tir_zh，例如 "
            "model_outputs-gaokao_bench_20260616_cot_en_deepseek-r1-distill-llama-70b.jsonl"
        )

    case_name = f"{match.group(1)}_{match.group(2)}"
    model_name = match.group(3)
    prefix = exp_key[: match.start()].strip("_")
    report_key = f"{prefix}_{model_name}" if prefix else model_name
    return ParsedStem(
        exp_key=exp_key,
        case_name=case_name,
        model_name=model_name,
        report_key=report_key,
    )


def resolve_from_scripts(scripts_dir: Path, path_text: str) -> Path:
    """把相对路径按 scripts/ 目录解释。"""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (scripts_dir / path).resolve()


def parse_cases(cases_text: str) -> list[str]:
    cases = [item.strip() for item in cases_text.split(",") if item.strip()]
    valid = set(DEFAULT_CASES)
    invalid = [case for case in cases if case not in valid]
    if invalid:
        raise ValueError(f"非法 case：{invalid}，可选值：{sorted(valid)}")
    return cases


def find_input_for_case(
    input_dir: Path,
    case_name: str,
    model_name: str,
    input_tag: str | None = None,
) -> tuple[Path, ParsedStem]:
    """在 input_dir 中按 case 和 model_name 查找 input JSONL。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"input 目录不存在：{input_dir}")

    candidates: list[tuple[Path, ParsedStem]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        try:
            parsed = parse_input_stem(path)
        except ValueError:
            continue

        if parsed.case_name != case_name:
            continue
        if parsed.model_name != model_name:
            continue
        if input_tag and input_tag not in parsed.exp_key:
            continue
        candidates.append((path, parsed))

    if not candidates:
        hint = f"，并包含 input_tag={input_tag}" if input_tag else ""
        raise FileNotFoundError(
            f"未找到 case={case_name}, model={model_name}{hint} 的 input 文件。"
            f"查找目录：{input_dir}"
        )

    if len(candidates) > 1:
        # 有多个日期/批次时，优先选文件名排序最后的那个；通常日期越新越靠后。
        print(f"[WARN] case={case_name} 找到多个 input，默认使用排序最后一个：", file=sys.stderr)
        for path, _ in candidates:
            print(f"       - {path}", file=sys.stderr)

    return candidates[-1]


def is_non_empty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def format_command(command: Sequence[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(text: str) -> str:
    if not text:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:=,+@%-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> None:
    print("\n$ " + format_command(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def build_case_infos(args: argparse.Namespace, scripts_dir: Path) -> list[CaseInfo]:
    input_dir = resolve_from_scripts(scripts_dir, args.input_dir)
    verify_dir = resolve_from_scripts(scripts_dir, args.verify_dir)
    prm_dir = resolve_from_scripts(scripts_dir, args.prm_dir)

    verify_dir.mkdir(parents=True, exist_ok=True)
    prm_dir.mkdir(parents=True, exist_ok=True)

    case_infos: list[CaseInfo] = []
    for case_name in parse_cases(args.cases):
        input_path, parsed = find_input_for_case(
            input_dir=input_dir,
            case_name=case_name,
            model_name=args.model_name,
            input_tag=args.input_tag,
        )

        verify_path = verify_dir / f"verify_{parsed.exp_key}.jsonl"
        verify_summary_path = verify_dir / f"verify_summary_{parsed.exp_key}.md"
        prm_path = prm_dir / f"prm_step_{parsed.exp_key}.jsonl"
        prm_summary_path = prm_dir / f"prm_summary-{parsed.exp_key}.md"

        case_infos.append(
            CaseInfo(
                case_name=case_name,
                input_path=input_path,
                exp_key=parsed.exp_key,
                report_key=parsed.report_key,
                verify_path=verify_path,
                verify_summary_path=verify_summary_path,
                prm_path=prm_path,
                prm_summary_path=prm_summary_path,
            )
        )

    return case_infos


def build_report_key(case_infos: list[CaseInfo], override: str | None) -> str:
    if override:
        return override
    if not case_infos:
        raise ValueError("case_infos 为空，无法生成 report_key")

    report_keys = {item.report_key for item in case_infos}
    if len(report_keys) == 1:
        return next(iter(report_keys))

    # 正常同一批 case 的 report_key 应该一致；不一致时给出可用的保守命名。
    print("[WARN] 多组 case 的 report_key 不一致：", sorted(report_keys), file=sys.stderr)
    return "mixed_" + case_infos[0].report_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="串行执行 gaokao_bench 的 Math-Verify、Qwen PRM 和指标汇总",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="逗号分隔的 case 列表")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="待处理模型名，用于匹配 input 文件")
    parser.add_argument("--input-tag", default=None, help="input 文件名额外过滤条件，例如 gaokao_bench_20260616")
    parser.add_argument("--input-dir", default="../data/model_outputs/gaokao_bench", help="模型输出 JSONL 目录")
    parser.add_argument("--verify-dir", default="../results/verify/gaokao_bench", help="Math-Verify 输出目录")
    parser.add_argument("--prm-dir", default="../results/prm/gaokao_bench", help="PRM 输出目录")
    parser.add_argument("--reports-dir", default="../reports/gaokao_bench", help="汇总报告输出目录")
    parser.add_argument("--report-key", default=None, help="汇总文件名 key；默认从 input 文件名推导")

    parser.add_argument("--limit", type=int, default=1000, help="传给 run_math_verify.py 的 --limit")
    parser.add_argument("--prm-limit", type=int, default=None, help="传给 run_qwen_prm.py 的 --limit；默认不限制")
    parser.add_argument("--prm-model", default=DEFAULT_PRM_MODEL, help="Qwen PRM 模型路径")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="HuggingFace cache_dir")
    parser.add_argument("--max-length", type=int, default=None, help="传给 run_qwen_prm.py 的 --max_length")
    parser.add_argument("--device", default=None, help="传给 run_qwen_prm.py 的 --device")
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=["auto", "bfloat16", "float16", "float32"],
        help="传给 run_qwen_prm.py 的 --torch_dtype",
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="传给 run_qwen_prm.py 的 --load_in_8bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="传给 run_qwen_prm.py 的 --load_in_4bit")

    parser.add_argument("--python", default=sys.executable, help="执行子脚本使用的 Python 解释器")
    parser.add_argument("--skip-existing", action="store_true", help="如果输出文件已存在且非空，则跳过对应步骤")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--no-export", action="store_true", help="不执行第三步汇总")
    parser.add_argument("--keep-going", action="store_true", help="某个 case 失败后继续跑后续 case，最后仍会尝试汇总已有结果")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    case_infos = build_case_infos(args, scripts_dir)

    print("=" * 80)
    print("Verify + PRM pipeline")
    print(f"scripts_dir: {scripts_dir}")
    print(f"cases:       {', '.join(item.case_name for item in case_infos)}")
    print(f"model_name:  {args.model_name}")
    print("=" * 80)

    finished_cases: list[CaseInfo] = []

    for item in case_infos:
        print("\n" + "-" * 80)
        print(f"case:    {item.case_name}")
        print(f"input:   {item.input_path}")
        print(f"exp_key: {item.exp_key}")
        print("-" * 80)

        try:
            verify_cmd = [
                args.python,
                "run_math_verify.py",
                "--input",
                str(item.input_path),
                "--output",
                str(item.verify_path),
                "--summary",
                str(item.verify_summary_path),
                "--limit",
                str(args.limit),
            ]
            if args.skip_existing and is_non_empty_file(item.verify_path):
                print(f"[SKIP] Math-Verify 已存在：{item.verify_path}")
            else:
                run_command(verify_cmd, cwd=scripts_dir, dry_run=args.dry_run)

            prm_cmd = [
                args.python,
                "run_qwen_prm.py",
                "--input",
                str(item.verify_path),
                "--output",
                str(item.prm_path),
                "--summary",
                str(item.prm_summary_path),
                "--model",
                args.prm_model,
                "--cache_dir",
                args.cache_dir,
            ]
            if args.prm_limit is not None:
                prm_cmd.extend(["--limit", str(args.prm_limit)])
            if args.max_length is not None:
                prm_cmd.extend(["--max_length", str(args.max_length)])
            if args.device is not None:
                prm_cmd.extend(["--device", args.device])
            if args.torch_dtype is not None:
                prm_cmd.extend(["--torch_dtype", args.torch_dtype])
            if args.load_in_8bit:
                prm_cmd.append("--load_in_8bit")
            if args.load_in_4bit:
                prm_cmd.append("--load_in_4bit")

            if args.skip_existing and is_non_empty_file(item.prm_path):
                print(f"[SKIP] PRM 已存在：{item.prm_path}")
            else:
                run_command(prm_cmd, cwd=scripts_dir, dry_run=args.dry_run)

            finished_cases.append(item)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] case={item.case_name} 执行失败，returncode={e.returncode}", file=sys.stderr)
            if not args.keep_going:
                raise
        except Exception as e:
            print(f"[ERROR] case={item.case_name} 执行失败：{e}", file=sys.stderr)
            if not args.keep_going:
                raise

    if args.no_export:
        print("\n[OK] 已跳过第三步汇总。")
        return

    export_cases = finished_cases if args.keep_going else case_infos
    if not export_cases:
        raise RuntimeError("没有成功完成的 case，无法执行汇总")

    reports_dir = resolve_from_scripts(scripts_dir, args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_key = build_report_key(export_cases, args.report_key)

    output_csv = reports_dir / f"verify_prm_metrics-{report_key}.csv"
    output_json = reports_dir / f"verify_prm_metrics-{report_key}.json"

    export_cmd = [args.python, "export_verify_prm_metrics.py"]
    for item in export_cases:
        export_cmd.extend([
            "--case",
            item.case_name,
            str(item.verify_path),
            str(item.prm_path),
        ])
    export_cmd.extend(["--output_csv", str(output_csv), "--output_json", str(output_json)])

    run_command(export_cmd, cwd=scripts_dir, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("Pipeline 完成")
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
