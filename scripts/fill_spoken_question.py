#!/usr/bin/env python3
"""Convert JSONL question fields into spoken Chinese math questions."""
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
QUESTION_PLACEHOLDER = "__QUESTION_PLACEHOLDER__"

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

# 不要对该模板使用 str.format()。模板中有 LaTeX 花括号，如 \frac{x^2}{25}，
# str.format() 会把 {x^2} 当作占位符，触发 KeyError: 'x^2'。
USER_PROMPT_TEMPLATE = """请把下面的数学题转成适合口语念出来的中文内容。

参考改写方式：
- $\\frac{x^2}{25}$ 可以读成：x 的平方除以二十五。
- $|\\overrightarrow{ON}| = 3$ 可以读成：向量 O N 的长度等于三。
- $x \\geq 0$ 可以读成：x 大于等于零。
- $a_n$ 可以读成：a 下标 n。
- “(  )” 可以读成：空。

原题：
__QUESTION_PLACEHOLDER__
"""


class SpokenQuestionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL question field into spoken Chinese math text.")
    parser.add_argument("--input-jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--source-field", default="question")
    parser.add_argument("--target-field", default="spoken_question")
    parser.add_argument("--replace-question", action="store_true")
    parser.add_argument("--overwrite-target", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--api-base", default="http://192.168.8.233:8200/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-14B-AWQ")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-meta", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw
    for candidate in (Path.cwd() / raw, REPO_ROOT / raw, REPO_ROOT.parent / raw):
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw).resolve()


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
        with tmp_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    try:
                        json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise SpokenQuestionError(f"invalid checkpoint JSONL at {tmp_path}:{line_no}: {exc}") from exc
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def create_backup(path: Path) -> Path:
    backup_path = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(path, backup_path)
    return backup_path


def parse_id_filter(ids_text: Optional[str]) -> Optional[set[str]]:
    if not ids_text:
        return None
    ids = {item.strip() for item in ids_text.split(",") if item.strip()}
    return ids or None


def build_candidates(records: List[Dict[str, Any]], source_field: str, target_field: str,
                     overwrite_target: bool, id_filter: Optional[set[str]], limit: Optional[int]) -> List[int]:
    candidates: List[int] = []
    for idx, record in enumerate(records):
        item_id = str(record.get("id", f"index-{idx}"))
        if id_filter is not None and item_id not in id_filter:
            continue
        source_value = record.get(source_field)
        if not isinstance(source_value, str) or not source_value.strip():
            continue
        target_value = record.get(target_field)
        if target_field != source_field and isinstance(target_value, str) and target_value.strip() and not overwrite_target:
            continue
        candidates.append(idx)
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def build_user_prompt(question: str) -> str:
    return USER_PROMPT_TEMPLATE.replace(QUESTION_PLACEHOLDER, question.strip())


def clean_model_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:text|markdown|md)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(口语化(?:题目|文本|结果)?|改写后|输出|答案)\s*[：:]\s*", "", text).strip()
    for left, right in [("“", "”"), ('"', '"'), ("'", "'"), ("「", "」")]:
        if text.startswith(left) and text.endswith(right) and len(text) >= 2:
            return text[1:-1].strip()
    return text


def call_llm_once(client: OpenAI, model: str, question: str, max_tokens: int, temperature: float,
                  top_p: float, top_k: Optional[int], enable_thinking: bool, timeout: int) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    start_time = time.time()
    extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
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
    except APIConnectionError:
        meta["error"] = "connection_error"
    except Exception as exc:
        meta["error"] = str(exc)
    meta["latency_seconds"] = round(time.time() - start_time, 2)
    return "", meta


def call_llm_with_retry(client: OpenAI, model: str, question: str, max_tokens: int, temperature: float,
                        top_p: float, top_k: Optional[int], enable_thinking: bool, timeout: int,
                        retries: int, retry_sleep: float) -> Tuple[str, Dict[str, Any]]:
    last_meta: Dict[str, Any] = {}
    for attempt in range(retries + 1):
        content, meta = call_llm_once(client, model, question, max_tokens, temperature, top_p, top_k, enable_thinking, timeout)
        meta["attempt"] = attempt + 1
        if not meta.get("error"):
            return content, meta
        last_meta = meta
        if attempt < retries:
            time.sleep(retry_sleep * (attempt + 1))
    return "", last_meta


def print_plan(input_path: Path, output_path: Path, records: List[Dict[str, Any]], candidates: List[int],
               source_field: str, target_field: str, dry_run: bool, in_place: bool, preview_limit: int) -> None:
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
            print(f"  - {records[idx].get('id', f'index-{idx}')}")
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
        output_path = resolve_path(args.output) if args.output else input_path if args.in_place else default_output_path(input_path).resolve()

        if output_path == input_path and not args.in_place:
            raise SpokenQuestionError("output path equals input path; use --in-place explicitly.")
        if args.save_every <= 0:
            raise SpokenQuestionError("--save-every must be greater than 0.")

        records = read_jsonl_objects(input_path)
        candidates = build_candidates(records, source_field, target_field, args.overwrite_target or args.replace_question,
                                      parse_id_filter(args.ids), args.limit)
        print_plan(input_path, output_path, records, candidates, source_field, target_field,
                   args.dry_run, args.in_place, args.preview_limit)

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
        success = failed = total_tokens = 0
        total_latency = 0.0
        last_save_success = 0

        for idx in tqdm(candidates, desc="口语化转换中"):
            record = records[idx]
            item_id = record.get("id", f"index-{idx}")
            spoken_text, meta = call_llm_with_retry(
                client, args.model, str(record[source_field]).strip(), args.max_tokens, args.temperature,
                args.top_p, args.top_k, args.enable_thinking, args.timeout, args.retries, args.retry_sleep
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
