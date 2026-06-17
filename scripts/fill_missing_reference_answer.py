#!/usr/bin/env python3
"""Fill missing reference_answer values in a MinerU JSONL file.

The script reads a source model-output JSONL file and builds an id ->
reference_answer map. Then it reads a target MinerU JSONL file and fills only
records whose reference_answer is null or an empty string.

Use --dry-run first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = (
    "data/model_outputs/math_qa_275/"
    "model_outputs-math_qa_275_20260612_tir_en_qwen3-32b.jsonl"
)
DEFAULT_TARGET = "data/math_qa_275_20260617.mineru.jsonl"
DEFAULT_OUTPUT = "data/math_qa_275_20260617.mineru.filled_reference_answer.jsonl"


class FillError(RuntimeError):
    """Raised when input data is unsafe to merge."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill null/empty reference_answer fields in a MinerU JSONL file "
            "from a source model-output JSONL file by matching id."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        default=DEFAULT_SOURCE,
        help=f"source model-output JSONL file. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--target-jsonl",
        default=DEFAULT_TARGET,
        help=f"target MinerU JSONL file to patch. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output JSONL file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting the output file if it already exists.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "replace --target-jsonl in place after writing and validating a temporary file. "
            "When set, --output is ignored."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print merge statistics without writing files.",
    )
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help="do not fail when a target record needs filling but has no source id match.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="maximum number of ids to show in previews. Default: 20.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw

    cwd_path = Path.cwd() / raw
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path.resolve()

    return (REPO_ROOT / raw).resolve()


def resolve_input_path(path_text: str) -> Path:
    """Resolve input paths from cwd, repo root, then repo parent.

    The repo-parent fallback makes paths like ../data/... work when the user runs
    the script from the repository root and keeps project-relative data/... paths
    working as well.
    """
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [Path.cwd() / raw, REPO_ROOT / raw, REPO_ROOT.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[1].resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FillError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FillError(f"{label} is not a file: {path}")


def is_missing_reference_answer(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def has_usable_reference_answer(value: Any) -> bool:
    return not is_missing_reference_answer(value)


def read_jsonl_objects(path: Path, label: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FillError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise FillError(f"{label} line must be a JSON object at {path}:{line_no}")
            records.append(item)
    return records


def get_record_id(record: Dict[str, Any], path: Path, line_no: int) -> str:
    item_id = record.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise FillError(f"missing or invalid id at {path}:{line_no}")
    return item_id.strip()


def build_source_reference_map(source_path: Path) -> Dict[str, Any]:
    source_map: Dict[str, Any] = {}
    seen_lines: Dict[str, int] = {}

    with source_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FillError(f"invalid JSONL at {source_path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise FillError(f"source line must be a JSON object at {source_path}:{line_no}")

            item_id = get_record_id(record, source_path, line_no)
            if item_id in source_map:
                prev_line = seen_lines[item_id]
                raise FillError(
                    f"duplicate source id {item_id!r}: line {prev_line} and line {line_no}"
                )

            if "reference_answer" not in record:
                continue
            reference_answer = record["reference_answer"]
            if not has_usable_reference_answer(reference_answer):
                continue

            source_map[item_id] = reference_answer
            seen_lines[item_id] = line_no

    return source_map


def fill_missing_reference_answers(
    target_records: List[Dict[str, Any]],
    source_map: Dict[str, Any],
    allow_missing_source: bool,
) -> Tuple[int, int, List[str], List[str]]:
    """Return total_missing_before, filled_count, missing_source_ids, filled_ids."""
    target_seen: Dict[str, int] = {}
    total_missing_before = 0
    filled_count = 0
    missing_source_ids: List[str] = []
    filled_ids: List[str] = []

    for index, record in enumerate(target_records):
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise FillError(f"target item at index {index} misses a valid string id")
        item_id = item_id.strip()

        if item_id in target_seen:
            raise FillError(
                f"duplicate target id {item_id!r}: index {target_seen[item_id]} and index {index}"
            )
        target_seen[item_id] = index

        if not is_missing_reference_answer(record.get("reference_answer")):
            continue

        total_missing_before += 1
        if item_id not in source_map:
            missing_source_ids.append(item_id)
            continue

        record["reference_answer"] = source_map[item_id]
        filled_count += 1
        filled_ids.append(item_id)

    if missing_source_ids and not allow_missing_source:
        preview = ", ".join(missing_source_ids[:20])
        more = "" if len(missing_source_ids) <= 20 else f" ... (+{len(missing_source_ids) - 20} more)"
        raise FillError(
            f"{len(missing_source_ids)} target records need reference_answer but have no usable source match: "
            f"{preview}{more}. Use --allow-missing-source only if this is expected."
        )

    return total_missing_before, filled_count, missing_source_ids, filled_ids


def write_jsonl_atomic(output_path: Path, records: Iterable[Dict[str, Any]], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FillError(f"output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

        with tmp_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise FillError(f"invalid JSONL written at {tmp_path}:{line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise FillError(f"written JSONL line must be an object at {tmp_path}:{line_no}")

        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def print_summary(
    *,
    source_path: Path,
    target_path: Path,
    output_path: Path,
    total_target: int,
    usable_source_refs: int,
    total_missing_before: int,
    filled_count: int,
    missing_source_ids: List[str],
    filled_ids: List[str],
    dry_run: bool,
    in_place: bool,
    preview_limit: int,
) -> None:
    print("=" * 80)
    print("Fill missing reference_answer")
    print("=" * 80)
    print(f"source_jsonl:          {source_path}")
    print(f"target_jsonl:          {target_path}")
    print(f"output_jsonl:          {output_path}")
    print(f"dry_run:               {dry_run}")
    print(f"in_place:              {in_place}")
    print(f"target records:        {total_target}")
    print(f"usable source refs:    {usable_source_refs}")
    print(f"missing before:        {total_missing_before}")
    print(f"filled:                {filled_count}")
    print(f"missing source ids:    {len(missing_source_ids)}")

    if filled_ids:
        print("filled ids preview:")
        for item_id in filled_ids[:preview_limit]:
            print(f"  - {item_id}")
        if len(filled_ids) > preview_limit:
            print(f"  ... (+{len(filled_ids) - preview_limit} more)")

    if missing_source_ids:
        print("missing source ids preview:")
        for item_id in missing_source_ids[:preview_limit]:
            print(f"  - {item_id}")
        if len(missing_source_ids) > preview_limit:
            print(f"  ... (+{len(missing_source_ids) - preview_limit} more)")


def main() -> int:
    args = parse_args()

    try:
        source_path = resolve_input_path(args.source_jsonl)
        target_path = resolve_input_path(args.target_jsonl)
        output_path = target_path if args.in_place else resolve_path(args.output)

        require_file(source_path, "source JSONL")
        require_file(target_path, "target JSONL")

        source_map = build_source_reference_map(source_path)
        target_records = read_jsonl_objects(target_path, "target")
        total_missing_before, filled_count, missing_source_ids, filled_ids = fill_missing_reference_answers(
            target_records=target_records,
            source_map=source_map,
            allow_missing_source=args.allow_missing_source,
        )

        print_summary(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            total_target=len(target_records),
            usable_source_refs=len(source_map),
            total_missing_before=total_missing_before,
            filled_count=filled_count,
            missing_source_ids=missing_source_ids,
            filled_ids=filled_ids,
            dry_run=args.dry_run,
            in_place=args.in_place,
            preview_limit=args.preview_limit,
        )

        if args.dry_run:
            print("\nDRY-RUN: no file written.")
            return 0

        write_jsonl_atomic(output_path, target_records, overwrite=args.overwrite or args.in_place)
        print(f"\nDone. Wrote: {output_path}")
        return 0
    except FillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
