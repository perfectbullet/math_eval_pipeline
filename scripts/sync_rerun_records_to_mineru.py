#!/usr/bin/env python3
"""Sync manually corrected rerun records back to a MinerU dataset.

This script reads model output JSONL records saved by the math render service,
selects records with ``needs_model_rerun is True``, and updates the matching
records in a MinerU JSON dataset by id.

Only these fields are overwritten in the MinerU records:
- question
- reference_answer
- needs_model_rerun

The output is written to a new JSON file by default. Use --dry-run first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_OUTPUT = (
    "data/model_outputs/math_qa_275/"
    "model_outputs-math_qa_275_20260612_tir_en_qwen3-32b.jsonl"
)
DEFAULT_MINERU_INPUT = "data/math_qa_275_20260612.mineru.json"

UPDATE_FIELDS = ("question", "reference_answer", "needs_model_rerun")


class SyncError(RuntimeError):
    """Raised when input data is unsafe to sync."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update a MinerU JSON dataset with records whose needs_model_rerun "
            "field is true in a model output JSONL file."
        )
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_OUTPUT,
        help=f"model output JSONL file. Default: {DEFAULT_MODEL_OUTPUT}",
    )
    parser.add_argument(
        "--mineru-input",
        default=DEFAULT_MINERU_INPUT,
        help=f"original MinerU JSON file. Default: {DEFAULT_MINERU_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output MinerU JSON file. Default: replace the date in "
            "--mineru-input with today's YYYYMMDD."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip update ids that are absent from the MinerU input instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be updated without writing the output file.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="maximum number of ids to show in the update preview. Default: 20.",
    )
    return parser.parse_args()


def resolve_existing_path(path_text: str) -> Path:
    """Resolve an input path from cwd first, then repo root."""
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw

    cwd_path = Path.cwd() / raw
    if cwd_path.exists():
        return cwd_path.resolve()

    repo_path = REPO_ROOT / raw
    if repo_path.exists():
        return repo_path.resolve()

    # Return the repo-root form for clearer errors with project-relative paths.
    return repo_path.resolve()


def resolve_output_path(path_text: str | None, mineru_input: Path) -> Path:
    if path_text:
        raw = Path(path_text).expanduser()
        if raw.is_absolute():
            return raw

        cwd_path = Path.cwd() / raw
        repo_path = REPO_ROOT / raw

        # If the parent exists in cwd, respect cwd. Otherwise prefer repo-root
        # relative paths such as data/foo.json even when called from scripts/.
        if cwd_path.parent.exists() and not repo_path.parent.exists():
            return cwd_path.resolve()
        return repo_path.resolve()

    today = datetime.now().strftime("%Y%m%d")
    name = mineru_input.name
    match = re.match(r"^(?P<prefix>.+_)\d{8}(?P<suffix>\.mineru\.json)$", name)
    if match:
        output_name = f"{match.group('prefix')}{today}{match.group('suffix')}"
    else:
        output_name = f"{mineru_input.stem}_{today}{mineru_input.suffix}"
    return mineru_input.with_name(output_name)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SyncError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise SyncError(f"{label} is not a file: {path}")


def read_jsonl_updates(model_output: Path) -> Dict[str, Dict[str, Any]]:
    updates: Dict[str, Dict[str, Any]] = {}
    seen_lines: Dict[str, int] = {}

    with model_output.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SyncError(f"invalid JSONL at {model_output}:{line_no}: {exc}") from exc

            if not isinstance(record, dict):
                raise SyncError(f"JSONL line must be an object at {model_output}:{line_no}")

            if record.get("needs_model_rerun") is not True:
                continue

            item_id = record.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise SyncError(f"missing or invalid id at {model_output}:{line_no}")
            item_id = item_id.strip()

            if item_id in updates:
                prev_line = seen_lines[item_id]
                raise SyncError(
                    f"duplicate update id {item_id!r} in JSONL: "
                    f"line {prev_line} and line {line_no}"
                )

            missing_fields = [field for field in UPDATE_FIELDS if field not in record]
            if missing_fields:
                raise SyncError(
                    f"record {item_id!r} at line {line_no} misses fields: "
                    f"{', '.join(missing_fields)}"
                )

            # question/reference_answer come from the edit API output and may be
            # strings, null, lists, or other JSON values. Keep them exactly as
            # saved instead of forcing them to strings.
            updates[item_id] = {
                "question": record["question"],
                "reference_answer": record["reference_answer"],
                "needs_model_rerun": True,
            }
            seen_lines[item_id] = line_no

    return updates


def read_mineru_records(mineru_input: Path) -> List[Dict[str, Any]]:
    try:
        with mineru_input.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SyncError(f"invalid MinerU JSON file {mineru_input}: {exc}") from exc

    if not isinstance(data, list):
        raise SyncError("MinerU input must be a top-level JSON array")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SyncError(f"MinerU item at index {index} must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise SyncError(f"MinerU item at index {index} misses a valid string id")

    return data


def build_mineru_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    seen_indexes: Dict[str, int] = {}

    for row_index, record in enumerate(records):
        item_id = record["id"].strip()
        if item_id in index:
            raise SyncError(
                f"duplicate id {item_id!r} in MinerU input: "
                f"index {seen_indexes[item_id]} and index {row_index}"
            )
        index[item_id] = record
        seen_indexes[item_id] = row_index

    return index


def apply_updates(
    mineru_records: List[Dict[str, Any]],
    updates: Dict[str, Dict[str, Any]],
    allow_missing: bool,
) -> Tuple[int, List[str]]:
    mineru_index = build_mineru_index(mineru_records)
    missing_ids = sorted(set(updates) - set(mineru_index))
    if missing_ids and not allow_missing:
        preview = ", ".join(missing_ids[:20])
        more = "" if len(missing_ids) <= 20 else f" ... (+{len(missing_ids) - 20} more)"
        raise SyncError(
            f"{len(missing_ids)} update ids are absent from MinerU input: {preview}{more}. "
            "Use --allow-missing only if this is expected."
        )

    updated_count = 0
    for item_id, patch in updates.items():
        target = mineru_index.get(item_id)
        if target is None:
            continue
        for field in UPDATE_FIELDS:
            target[field] = patch[field]
        updated_count += 1

    return updated_count, missing_ids


def write_json_atomic(output_path: Path, records: List[Dict[str, Any]], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise SyncError(f"output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")

        # Re-read the temporary file so a broken JSON file is never promoted.
        with tmp_path.open("r", encoding="utf-8") as f:
            json.load(f)

        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def print_summary(
    *,
    model_output: Path,
    mineru_input: Path,
    output_path: Path,
    total_mineru: int,
    total_updates: int,
    updated_count: int,
    missing_ids: List[str],
    dry_run: bool,
    preview_limit: int,
) -> None:
    print("=" * 80)
    print("Sync rerun records to MinerU")
    print("=" * 80)
    print(f"model_output:       {model_output}")
    print(f"mineru_input:       {mineru_input}")
    print(f"output:             {output_path}")
    print(f"dry_run:            {dry_run}")
    print(f"mineru records:     {total_mineru}")
    print(f"rerun updates:      {total_updates}")
    print(f"matched updates:    {updated_count}")
    print(f"missing update ids: {len(missing_ids)}")

    if missing_ids:
        print("missing ids preview:")
        for item_id in missing_ids[:preview_limit]:
            print(f"  - {item_id}")
        if len(missing_ids) > preview_limit:
            print(f"  ... (+{len(missing_ids) - preview_limit} more)")


def main() -> int:
    args = parse_args()

    try:
        model_output = resolve_existing_path(args.model_output)
        mineru_input = resolve_existing_path(args.mineru_input)
        output_path = resolve_output_path(args.output, mineru_input)

        require_file(model_output, "model output")
        require_file(mineru_input, "MinerU input")

        updates = read_jsonl_updates(model_output)
        mineru_records = read_mineru_records(mineru_input)
        updated_count, missing_ids = apply_updates(
            mineru_records,
            updates,
            allow_missing=args.allow_missing,
        )

        print_summary(
            model_output=model_output,
            mineru_input=mineru_input,
            output_path=output_path,
            total_mineru=len(mineru_records),
            total_updates=len(updates),
            updated_count=updated_count,
            missing_ids=missing_ids,
            dry_run=args.dry_run,
            preview_limit=args.preview_limit,
        )

        if args.dry_run:
            print("\nDRY-RUN: no file written.")
            return 0

        write_json_atomic(output_path, mineru_records, overwrite=args.overwrite)
        print(f"\nDone. Wrote: {output_path}")
        return 0
    except SyncError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
