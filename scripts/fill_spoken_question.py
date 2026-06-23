#!/usr/bin/env python3
"""Convert JSONL question fields into spoken Chinese math questions.

This script reads a MinerU JSONL file, sends each record's `question` value to an
OpenAI-compatible LLM service, writes the oralized text into a target field, and
saves the JSONL back atomically.

Default target file:
    data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl

Recommended first run:
    python scripts/fill_spoken_question.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, OpenAI

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = "data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl"

SYSTEM_PROMPT = """你是一个数学题目口语化改写助手。

你的任务是把书面数学题改写成适合直接朗读、TTS 播放、语音识别回放的中文口语文本。

必须遵守：
1. 只输出改写后的口语题目，不要解释，不要解题，不要给答案。
2. 完全保留原题的数学含义、条件、问法、选项和符号关系。
3. 将 LaTeX / 数学符号自然读出来，不要保留美元符号、反斜杠命令或 Markdown。
4. 英文字母变量按字母读，例如 M、F、N、O 可以保留为大写字母，适合朗读即可。
5. 分式、指数、根号、向量、绝对值、集合、区间等要读成清楚的中文表达。
6. 题号可以保留并口语化，例如“第 5 题”。
7. 括号里的空，例如“(  )”，读成“空”或“应填什么”，不要自作主张补答案。
8. 原题如果有 A、B、C、D 选项，要逐项读出。
"""

USER_PROMPT_TEMPLATE = """请把下面的数学题转成适合口语念出来的中文内容。

参考改写方式：
- $\\frac{x^2}{25}$ 可以读成：x 的平方除以二十五。
- $|\\overrightarrow{ON}| = 3$ 可以读成：向量 O N 的长度等于三。
- $x \\geq 0$ 可以读成：x 大于等于零。
- $a_n$ 可以读成：a 下标 n。
- “(  )” 可以读成：空。

原题：
{question}
"""


class SpokenQuestionError(RuntimeError):
    """Raised when conversion cannot proceed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read JSONL records, convert `question` into a spoken Chinese math question "
            "with an OpenAI-compatible LLM, and write the result back."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_JSONL,
        help=f"input JSONL file. Default: {DEFAULT_JSONL}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output JSONL file. If omitted and --in-place is not set, writes to "
            "<input-stem>.spoken_question.jsonl."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="write changes back to --input-jsonl atomically.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="when --in-place is set, do not create a timestamped .bak file.",
    )
    parser.add_argument(
        "--source-field",
        default="question",
        help="field to read from each JSON object. Default: question.",
    )
    parser.add_argument(
        "--target-field",
        default="spoken_question",
        help="field to write converted text. Default: spoken_question.",
    )
    parser.add_argument(
        "--replace-question",
        action="store_true",
        help=(
            "write the oralized text back into the source field itself. "
            "This is destructive unless you keep the backup."
        ),
    )
    parser.add_argument(
        "--overwrite-target",
        action="store_true",
        help="re-convert records that already have a non-empty target field.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of records to convert in this run.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated record ids to process, for example: MATH-238,MATH-239.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be processed without calling the model or writing files.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="number of candidate ids to show in dry-run/summary. Default: 20.",
    )
    parser.add_argument(
        "--api-base",
        default="http://192.168.8.233:8200/v1",
        help="OpenAI-compatible API base URL. Default: http://192.168.8.233:8200/v1",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key. Local vLLM normally uses EMPTY.",
    )
    parser.add_argument(
        "--model",
        default="Qwen3-14B-AWQ",
        help="served model name. Default: Qwen3-14B-AWQ.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="maximum completion tokens for one converted question. Default: 512.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature. Default: 0.0 for deterministic conversion.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top_p sampling parameter. Default: 1.0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="optional top_k value passed through extra_body.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Qwen thinking mode. Default is disabled for direct conversion.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-request timeout in seconds. Default: 120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retry count for one failed model call. Default: 2.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="base sleep seconds between retries. Default: 2.0.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="write checkpoint every N successful conversions. Default: 5.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record errors into <target_field>_error and continue.",
    )
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="write request metadata into <target_field>_meta.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [
        Path.cwd() / raw,
        REPO_ROOT / raw,
        REPO_ROOT.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()

    return candidates[1].resolve()


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.spoken_question{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}.spoken_question.jsonl")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SpokenQuestionError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise SpokenQuestionError(f"{label} is not a file: {path}")


def read_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SpokenQuestionError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise SpokenQuestionError(f"JSONL line must be an object at {path}:{line_no}")
            records.append(item)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

        # Validate before replacing the real file.
        with tmp_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    json.loads(text)

        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def create_backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def parse_id_filter(ids_text: Optional[str]) -> Optional[set[str]]:
    if not ids_text:
        return None
    ids = {item.strip() for item in ids_text.split(",") if item.strip()}
    return ids or None


def build_candidates(
    records: List[Dict[str, Any]],
    source_field: str,
    target_field: str,
    overwrite_target: bool,
    id_filter: Optional[set[str]],
    limit: Optional[int],
) -> List[int]:
    candidates: List[int] = []

    for idx, record in enumerate(records):
        item_id = record.get("id", f"index-{idx}")
        if id_filter is not None and str(item_id) not in id_filter:
            continue

        source_value = record.get(source_field)
        if not is_non_empty_string(source_value):
            continue

        target_value = record.get(target_field)
        if target_field != source_field and is_non_empty_string(target_value) and not overwrite_target:
            continue

        candidates.append(idx)
        if limit is not None and len(candidates) >= limit:
            break

    return candidates


def build_user_prompt(question: str) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question.strip())


def clean_model_output(text: str) -> str:
    text = text.strip()

    # Drop accidental thinking blocks if a server/template returns them in content.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Drop fenced code wrappers.
    text = re.sub(r"^```(?:text|markdown|md)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # Drop common assistant prefixes.
    text = re.sub(r"^(口语化(?:题目|文本|结果)?|改写后|输出|答案)\s*[：:]\s*", "", text).strip()

    # Remove one pair of surrounding quotes if the whole output is quoted.
    quote_pairs = [("“", "”"), ("\"", "\""), ("'", "'"), ("「", "」")]
    for left, right in quote_pairs:
        if text.startswith(left) and text.endswith(right) and len(text) >= 2:
            text = text[1:-1].strip()
            break

    return text


def call_llm_once(
    client: OpenAI,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    enable_thinking: bool,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    start_time = time.time()

    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if top_k is not None:
        extra_body["top_k"] = top_k

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question)},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            timeout=timeout,
            extra_body=extra_body,
        )

        content = clean_model_output(resp.choices[0].message.content or "")
        usage = resp.usage

        meta["latency_seconds"] = round(time.time() - start_time, 2)
        meta["finish_reason"] = resp.choices[0].finish_reason or ""
        meta["prompt_tokens"] = usage.prompt_tokens if usage else 0
        meta["completion_tokens"] = usage.completion_tokens if usage else 0
        meta["total_tokens"] = usage.total_tokens if usage else 0

        if not content:
            meta["error"] = "empty_model_output"
            return "", meta

        return content, meta

    except APITimeoutError:
        meta["error"] = "timeout"
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta
    except APIConnectionError:
        meta["error"] = "connection_error"
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta
    except Exception as exc:
        meta["error"] = str(exc)
        meta["latency_seconds"] = round(time.time() - start_time, 2)
        return "", meta


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    enable_thinking: bool,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> Tuple[str, Dict[str, Any]]:
    last_meta: Dict[str, Any] = {}

    for attempt in range(retries + 1):
        content, meta = call_llm_once(
            client=client,
            model=model,
            question=question,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        meta["attempt"] = attempt + 1

        if not meta.get("error"):
            return content, meta

        last_meta = meta
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))

    return "", last_meta


def print_plan(
    input_path: Path,
    output_path: Path,
    records: List[Dict[str, Any]],
    candidates: List[int],
    source_field: str,
    target_field: str,
    dry_run: bool,
    in_place: bool,
    preview_limit: int,
) -> None:
    print("=" * 80)
    print("Convert question to spoken question")
    print("=" * 80)
    print(f"input_jsonl:       {input_path}")
    print(f"output_jsonl:      {output_path}")
    print(f"in_place:          {in_place}")
    print(f"dry_run:           {dry_run}")
    print(f"source_field:      {source_field}")
    print(f"target_field:      {target_field}")
    print(f"total_records:     {len(records)}")
    print(f"to_process:        {len(candidates)}")

    if candidates:
        print("candidate ids preview:")
        for idx in candidates[:preview_limit]:
            record = records[idx]
            print(f"  - {record.get('id', f'index-{idx}')}")
        if len(candidates) > preview_limit:
            print(f"  ... (+{len(candidates) - preview_limit} more)")
    print("=" * 80)


def main() -> int:
    args = parse_args()

    try:
        input_path = resolve_path(args.input_jsonl)
        require_file(input_path, "input JSONL")

        source_field = args.source_field
        target_field = source_field if args.replace_question else args.target_field

        if args.output:
            output_path = resolve_path(args.output)
        elif args.in_place:
            output_path = input_path
        else:
            output_path = default_output_path(input_path).resolve()

        if output_path == input_path and not args.in_place:
            raise SpokenQuestionError("output path equals input path; use --in-place explicitly.")

        if args.save_every <= 0:
            raise SpokenQuestionError("--save-every must be greater than 0.")

        id_filter = parse_id_filter(args.ids)
        records = read_jsonl_objects(input_path)
        candidates = build_candidates(
            records=records,
            source_field=source_field,
            target_field=target_field,
            overwrite_target=args.overwrite_target or args.replace_question,
            id_filter=id_filter,
            limit=args.limit,
        )

        print_plan(
            input_path=input_path,
            output_path=output_path,
            records=records,
            candidates=candidates,
            source_field=source_field,
            target_field=target_field,
            dry_run=args.dry_run,
            in_place=args.in_place,
            preview_limit=args.preview_limit,
        )

        if args.dry_run:
            print("DRY-RUN: no model call and no file written.")
            return 0

        if not candidates:
            print("No records need conversion.")
            return 0

        backup_path: Optional[Path] = None
        if args.in_place and not args.no_backup:
            backup_path = create_backup(input_path)
            print(f"backup:            {backup_path}")

        client = OpenAI(api_key=args.api_key, base_url=args.api_base)

        success = 0
        failed = 0
        total_tokens = 0
        total_latency = 0.0
        last_save_success = 0

        for idx in tqdm(candidates, desc="口语化转换中"):
            record = records[idx]
            item_id = record.get("id", f"index-{idx}")
            question = str(record[source_field]).strip()

            spoken_text, meta = call_llm_with_retry(
                client=client,
                model=args.model,
                question=question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                enable_thinking=args.enable_thinking,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )

            if meta.get("error"):
                failed += 1
                error_text = str(meta["error"])
                print(f"[FAIL] {item_id}: {error_text}", file=sys.stderr)

                if args.continue_on_error:
                    record[f"{target_field}_error"] = {
                        "error": error_text,
                        "model": args.model,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    write_jsonl_atomic(output_path, records)
                    continue

                raise SpokenQuestionError(f"model call failed for {item_id}: {error_text}")

            record[target_field] = spoken_text
            record.pop(f"{target_field}_error", None)

            if args.write_meta:
                record[f"{target_field}_meta"] = {
                    "model": args.model,
                    "api_base": args.api_base,
                    "source_field": source_field,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    **meta,
                }

            success += 1
            total_tokens += int(meta.get("total_tokens", 0) or 0)
            total_latency += float(meta.get("latency_seconds", 0) or 0.0)

            if success - last_save_success >= args.save_every:
                write_jsonl_atomic(output_path, records)
                last_save_success = success

        write_jsonl_atomic(output_path, records)

        print("\n" + "=" * 80)
        print("Done")
        print("=" * 80)
        print(f"output_jsonl:      {output_path}")
        if backup_path:
            print(f"backup:            {backup_path}")
        print(f"success:           {success}")
        print(f"failed:            {failed}")
        if success > 0:
            print(f"avg_latency:       {total_latency / success:.2f}s")
            print(f"total_tokens:      {total_tokens}")
        print("=" * 80)
        return 0

    except SpokenQuestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
