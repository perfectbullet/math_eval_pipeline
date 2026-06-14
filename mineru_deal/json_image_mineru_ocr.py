#!/usr/bin/env python3
"""读取 excel_to_json.py 导出的 JSON，调用 MinerU 识别题目/答案图片，并回写 JSON。

功能：
1. 根据每条记录的 image_path 读取图片，识别结果写入 question 字段；
2. 根据每条记录的 reference_answer 读取答案图片，识别结果写回 reference_answer；
3. 原 reference_answer 图片路径保存到 reference_answer_image_path；
4. 支持相对路径：默认相对于输入 JSON 所在目录；
5. 支持断点续跑：若 -o 输出文件已存在，优先读取输出文件并跳过已处理记录；
6. 每处理完一个 MinerU 批次就立即保存一次完整 JSON；
7. 支持 dry-run 只检查路径，不调用 MinerU。

准备：
    pip install requests tqdm
    export MINERU_API_TOKEN="你的 MinerU token"

用法：
    # 推荐：输出新文件，避免覆盖原始 JSON
    python3 json_image_mineru_ocr.py.py output_qwen3.json -o output_qwen3.mineru.json

    # 原地覆盖，同时自动生成 .bak 备份
    python3 json_image_mineru_ocr.py.py output_qwen3.json --inplace

    # 只检查图片路径是否存在，不请求 MinerU
    python3 json_image_mineru_ocr.py.py output_qwen3.json --dry-run

    # 只处理前 5 条记录，适合小批量测试
    python3 json_image_mineru_ocr.py.py output_qwen3.json --limit 5 -o output_qwen3.part5.mineru.json

    # 断点续跑：若 output_qwen3.mineru.json 已存在，会读取它并跳过已处理记录
    python3 json_image_mineru_ocr.py.py output_qwen3.json -o output_qwen3.mineru.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://mineru.net"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_BATCH_SIZE = 5
DEFAULT_UPLOAD_TIMEOUT = 30
DEFAULT_RESULT_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"
}

_MD5_CHUNK_SIZE = 8192


def load_env_file(env_path: Path) -> None:
    """加载 .env 文件到环境变量，已有环境变量不覆盖。"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def compute_file_md5(path: Path) -> str:
    """计算文件内容 MD5，用于去重与生成 data_id。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(_MD5_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


class MinerUClient:
    """MinerU 精准解析 API 客户端，接口逻辑参考 extract_mineru_sdk.py。"""

    def __init__(self, token: str, base_url: str = DEFAULT_BASE_URL):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def upload_batch(
        self,
        file_paths: list[Path],
        data_ids: list[str] | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        """批量上传本地文件，返回 batch_id。"""
        url = f"{self.base_url}/api/v4/file-urls/batch"
        files_payload = []
        for i, fp in enumerate(file_paths):
            item = {"name": fp.name}
            item["data_id"] = data_ids[i] if data_ids and i < len(data_ids) else fp.stem
            files_payload.append(item)

        payload = {
            "files": files_payload,
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }

        logger.info("申请 %d 个文件的上传链接...", len(file_paths))
        resp = requests.post(url, headers=self.headers, json=payload, timeout=DEFAULT_UPLOAD_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")

        batch_id = result["data"]["batch_id"]
        upload_urls = result["data"]["file_urls"]

        logger.info("上传 %d 个文件...", len(file_paths))
        for i, (fp, upload_url) in enumerate(zip(file_paths, upload_urls), start=1):
            with fp.open("rb") as f:
                put_resp = requests.put(upload_url, data=f, timeout=120)
                put_resp.raise_for_status()
            logger.debug("[%d/%d] %s 上传成功", i, len(file_paths), fp)

        logger.info("batch_id=%s 上传完成，等待服务端解析", batch_id)
        return batch_id

    def get_batch_result(self, batch_id: str) -> dict[str, Any]:
        """查询批量任务结果。"""
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        resp = requests.get(url, headers=self.headers, timeout=DEFAULT_UPLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def poll_batch_result(
        self,
        batch_id: str,
        timeout: int = DEFAULT_RESULT_TIMEOUT,
        interval: int = DEFAULT_POLL_INTERVAL,
    ) -> list[dict[str, Any]]:
        """轮询批量任务直到全部完成或超时。"""
        start = time.time()
        pbar = tqdm(desc=f"batch {batch_id[:8]}...", unit="poll")
        try:
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise TimeoutError(f"batch {batch_id} 轮询超时 ({timeout}s)")

                result = self.get_batch_result(batch_id)
                if result.get("code") != 0:
                    raise RuntimeError(f"查询结果失败: {result.get('msg')}")

                extract_result = result["data"].get("extract_result", [])
                done_count = sum(1 for r in extract_result if r.get("state") in ("done", "failed"))
                pbar.set_postfix(done=f"{done_count}/{len(extract_result)}")
                pbar.update(1)

                # 有些情况下接口刚开始可能短暂返回空列表；不能把 0/0 当作完成。
                if extract_result and done_count == len(extract_result):
                    return extract_result

                time.sleep(interval)
        finally:
            pbar.close()


def _download_via_curl(zip_url: str, tmp_path: Path) -> bool:
    """用 curl 下载，兼容某些环境下 requests SSL/代理问题。"""
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "-o", str(tmp_path), zip_url],
        capture_output=True,
        timeout=120,
    )
    return r.returncode == 0


def download_and_extract_markdown(zip_url: str, max_retries: int = 3) -> str | None:
    """下载 MinerU 结果 zip，并提取 full.md 内容。"""
    last_err: Exception = RuntimeError("未知下载错误")

    for attempt in range(1, max_retries + 1):
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ok = _download_via_curl(zip_url, tmp_path)
            if not ok:
                try:
                    # verify=False 是为了兼容部分公司内网/代理证书场景。
                    resp = requests.get(
                        zip_url,
                        timeout=120,
                        verify=False,
                        proxies={"http": "", "https": ""},
                    )
                    resp.raise_for_status()
                    tmp_path.write_bytes(resp.content)
                except Exception as e:
                    last_err = e
                    continue

            with zipfile.ZipFile(tmp_path) as zf:
                for name in zf.namelist():
                    if name.endswith("full.md"):
                        return zf.read(name).decode("utf-8").strip()
            return None
        except Exception as e:
            last_err = e
        finally:
            tmp_path.unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(attempt * 3)

    raise last_err


def validate_records(records: Any, source: str) -> list[dict[str, Any]]:
    """校验读取结果必须是 list[dict]。"""
    if not isinstance(records, list):
        raise ValueError(f"{source} 必须是 JSON 数组，实际类型是 {type(records).__name__}")

    for i, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{source} 第 {i} 条记录不是 JSON object，实际类型是 {type(item).__name__}")

    return records


def read_json_records(json_path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组；如果不是标准 JSON，再按 JSONL 逐行兼容读取。"""
    if json_path.stat().st_size == 0:
        return []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_records(data, "输入文件")
    except json.JSONDecodeError as json_error:
        # 兼容 JSONL：一行一个 JSON object。标准 JSON 数组解析失败时才走这里。
        records: list[dict[str, Any]] = []
        try:
            with json_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"JSONL 第 {line_no} 行不是 JSON object，实际类型是 {type(item).__name__}"
                        )
                    records.append(item)
        except json.JSONDecodeError as jsonl_error:
            raise ValueError(
                "输入文件既不是标准 JSON 数组，也不是合法 JSONL。"
                f"JSON 解析错误: line {json_error.lineno}, column {json_error.colno}; "
                f"JSONL 解析错误: line {jsonl_error.lineno}, column {jsonl_error.colno}"
            ) from jsonl_error

        return records


def write_json_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """写入 pretty JSON，保留中文。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def load_records_for_run(json_path: Path, output_path: Path, inplace: bool) -> tuple[list[dict[str, Any]], Path]:
    """读取本次运行要继续处理的记录。

    非 inplace 模式下，如果 -o 输出文件已经存在，就优先读取输出文件，
    这样已经写入 question / reference_answer_image_path 的记录会被自动跳过。
    """
    if not inplace and output_path.exists():
        return read_json_records(output_path), output_path
    return read_json_records(json_path), json_path


def resolve_path(raw_value: Any, base_dir: Path) -> Path | None:
    """把 JSON 字段中的相对路径解析成真实文件路径。"""
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if not raw:
        return None

    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return base_dir / p


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Task:
    record_index: int
    kind: str                 # question 或 answer
    image_path: Path
    raw_json_path: str


@dataclass(frozen=True)
class UniqueImage:
    data_id: str
    image_path: Path
    file_md5: str


def collect_tasks(
    records: list[dict[str, Any]],
    base_dir: Path,
    force: bool = False,
    limit: int | None = None,
) -> tuple[list[Task], list[str]]:
    """收集需要识别的图片任务。

    limit 表示只扫描/处理前 N 条 JSON 记录；未处理记录会原样保留在输出 JSON 中。
    """
    tasks: list[Task] = []
    warnings: list[str] = []

    if limit is not None and limit < 0:
        raise ValueError("limit 不能小于 0")

    target_records = records if limit is None else records[:limit]

    for idx, rec in enumerate(target_records):
        rec_id = rec.get("id", idx + 1)

        # 题目图片：image_path -> question
        image_raw = rec.get("image_path")
        question_path = resolve_path(image_raw, base_dir)
        if question_path is None:
            warnings.append(f"记录 {rec_id}: 缺少 image_path")
        elif not question_path.exists():
            warnings.append(f"记录 {rec_id}: image_path 文件不存在: {question_path}")
        elif not is_supported_image(question_path):
            warnings.append(f"记录 {rec_id}: image_path 不是支持的图片格式: {question_path}")
        elif force or not str(rec.get("question", "")).strip():
            tasks.append(Task(idx, "question", question_path, str(image_raw)))

        # 答案图片：reference_answer -> reference_answer，并备份到 reference_answer_image_path
        # 兼容已跑过一半的情况：若已有 reference_answer_image_path，就优先用它作为答案图片路径。
        answer_raw = rec.get("reference_answer_image_path") or rec.get("reference_answer")
        answer_path = resolve_path(answer_raw, base_dir)
        if answer_path is None:
            warnings.append(f"记录 {rec_id}: 缺少 reference_answer / reference_answer_image_path")
            continue

        if not answer_path.exists():
            # 如果 reference_answer 已经是识别后的文本，这里不要强行报错为致命错误。
            if not rec.get("reference_answer_image_path"):
                warnings.append(f"记录 {rec_id}: reference_answer 图片文件不存在: {answer_path}")
            continue

        if not is_supported_image(answer_path):
            warnings.append(f"记录 {rec_id}: reference_answer 不是支持的图片格式: {answer_path}")
            continue

        answer_already_converted = bool(rec.get("reference_answer_image_path")) and bool(str(rec.get("reference_answer", "")).strip())
        if force or not answer_already_converted:
            tasks.append(Task(idx, "answer", answer_path, str(answer_raw)))

    return tasks, warnings


def build_unique_images(tasks: list[Task]) -> tuple[list[UniqueImage], dict[Path, UniqueImage]]:
    """同一张图只上传识别一次。"""
    unique_images: list[UniqueImage] = []
    path_map: dict[Path, UniqueImage] = {}

    for task in tasks:
        path = task.image_path.resolve()
        if path in path_map:
            continue
        file_md5 = compute_file_md5(path)
        safe_stem = "".join(ch if ch.isalnum() else "_" for ch in path.stem)[:60]
        data_id = f"img_{len(unique_images) + 1:06d}_{safe_stem}_{file_md5[:8]}"
        item = UniqueImage(data_id=data_id, image_path=path, file_md5=file_md5)
        unique_images.append(item)
        path_map[path] = item

    return unique_images, path_map


def recognize_image_batch(
    client: MinerUClient,
    batch: list[UniqueImage],
    model_version: str,
    poll_interval: int,
    result_timeout: int,
    enable_formula: bool,
    enable_table: bool,
) -> dict[str, dict[str, Any]]:
    """调用 MinerU 识别一批图片，返回 data_id -> 识别结果。"""
    if not batch:
        return {}

    file_paths = [item.image_path for item in batch]
    data_ids = [item.data_id for item in batch]

    batch_id = client.upload_batch(
        file_paths=file_paths,
        data_ids=data_ids,
        model_version=model_version,
        enable_formula=enable_formula,
        enable_table=enable_table,
    )
    extract_results = client.poll_batch_result(
        batch_id=batch_id,
        timeout=result_timeout,
        interval=poll_interval,
    )

    batch_results: dict[str, dict[str, Any]] = {}
    for r in extract_results:
        data_id = r.get("data_id") or Path(r.get("file_name", "")).stem
        state = r.get("state", "unknown")
        zip_url = r.get("full_zip_url", "")
        markdown_content = ""
        error_msg = r.get("err_msg", "")

        if state == "done" and zip_url:
            try:
                markdown_content = download_and_extract_markdown(zip_url) or ""
            except Exception as e:
                state = "download_failed"
                error_msg = str(e)
                logger.warning("下载/解析 MinerU 结果失败 data_id=%s: %s", data_id, e)

        batch_results[data_id] = {
            "state": state,
            "markdown": markdown_content,
            "zip_url": zip_url,
            "err_msg": error_msg,
        }

    return batch_results


def select_tasks_for_batch(
    tasks: list[Task],
    path_map: dict[Path, UniqueImage],
    batch: list[UniqueImage],
) -> list[Task]:
    """选出当前图片批次对应的回写任务。"""
    batch_data_ids = {item.data_id for item in batch}
    selected: list[Task] = []
    for task in tasks:
        item = path_map[task.image_path.resolve()]
        if item.data_id in batch_data_ids:
            selected.append(task)
    return selected


def merge_stats(total: dict[str, int], delta: dict[str, int]) -> None:
    """累加统计信息。"""
    for key, value in delta.items():
        total[key] = total.get(key, 0) + value


def apply_results(
    records: list[dict[str, Any]],
    tasks: list[Task],
    path_map: dict[Path, UniqueImage],
    results: dict[str, dict[str, Any]],
    keep_mineru_meta: bool = False,
) -> dict[str, int]:
    """把识别结果回写到 records。"""
    stats = {
        "question_updated": 0,
        "answer_updated": 0,
        "failed": 0,
    }

    for task in tasks:
        item = path_map[task.image_path.resolve()]
        result = results.get(item.data_id, {})
        state = result.get("state", "missing_result")
        markdown = str(result.get("markdown") or "").strip()
        rec = records[task.record_index]

        if state == "done" and markdown:
            if task.kind == "question":
                rec["question"] = markdown
                stats["question_updated"] += 1
                if keep_mineru_meta:
                    rec["question_image_md5"] = item.file_md5
                    rec["question_mineru_zip_url"] = result.get("zip_url", "")
            elif task.kind == "answer":
                # 注意：这里先保存旧的 reference_answer 图片路径，再用识别文本覆盖 reference_answer。
                if not rec.get("reference_answer_image_path"):
                    rec["reference_answer_image_path"] = task.raw_json_path
                rec["reference_answer"] = markdown
                stats["answer_updated"] += 1
                if keep_mineru_meta:
                    rec["reference_answer_image_md5"] = item.file_md5
                    rec["reference_answer_mineru_zip_url"] = result.get("zip_url", "")
        else:
            stats["failed"] += 1
            field_prefix = "question" if task.kind == "question" else "reference_answer"
            rec[f"{field_prefix}_mineru_state"] = state
            if result.get("err_msg"):
                rec[f"{field_prefix}_mineru_err_msg"] = result["err_msg"]
            # 答案识别失败时，仍保留原 reference_answer 图片路径，避免丢数据。
            if task.kind == "answer" and not rec.get("reference_answer_image_path"):
                rec["reference_answer_image_path"] = task.raw_json_path

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 JSON 中的图片路径，调用 MinerU 识别并回写 JSON")
    parser.add_argument("json_path", help="输入 JSON 文件，格式与 output_qwen3.json 相同")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 文件；不指定时输出 <输入名>.mineru.json")
    parser.add_argument("--inplace", action="store_true", help="原地覆盖输入 JSON；覆盖前自动生成 .bak 备份")
    parser.add_argument("--image_base_dir", default=None, help="图片相对路径的基准目录；默认是 JSON 文件所在目录")
    parser.add_argument("--model_version", default=DEFAULT_MODEL_VERSION, choices=["pipeline", "vlm", "MinerU-HTML"], help="MinerU 模型版本")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="每批上传图片数量")
    parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL, help="轮询间隔秒")
    parser.add_argument("--result_timeout", type=int, default=DEFAULT_RESULT_TIMEOUT, help="单批任务超时秒")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="MinerU API Base URL")
    parser.add_argument("--force", action="store_true", help="强制重新识别，覆盖已有 question/reference_answer 识别结果")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条 JSON 记录；未处理记录原样保留")
    parser.add_argument("--dry-run", action="store_true", help="只检查待识别图片，不调用 MinerU，不写输出")
    parser.add_argument("--keep_mineru_meta", action="store_true", help="保留 zip_url、图片 MD5 等 MinerU 元信息")
    parser.add_argument("--disable_formula", action="store_true", help="关闭公式识别")
    parser.add_argument("--disable_table", action="store_true", help="关闭表格识别")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    json_path = Path(args.json_path).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    base_dir = Path(args.image_base_dir).expanduser().resolve() if args.image_base_dir else json_path.parent

    if args.inplace and args.output:
        raise ValueError("--inplace 和 --output 不能同时使用")

    if args.inplace:
        output_path = json_path
    elif args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = json_path.with_name(f"{json_path.stem}.mineru{json_path.suffix}")

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit 不能小于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须大于 0")

    # 加载 .env：优先当前工作目录，其次脚本目录。
    load_env_file(Path.cwd() / ".env")
    load_env_file(Path(__file__).resolve().parent / ".env")

    records, records_source = load_records_for_run(json_path, output_path, inplace=args.inplace)
    tasks, warnings = collect_tasks(records, base_dir=base_dir, force=args.force, limit=args.limit)
    unique_images, path_map = build_unique_images(tasks)

    logger.info("读取记录数: %d，来源: %s", len(records), records_source)
    if not args.inplace and records_source == output_path:
        logger.info("检测到输出文件已存在，按断点续跑模式跳过已处理记录。")
    if args.limit is not None:
        logger.info("本次限制处理前 %d 条记录", args.limit)
    logger.info("待识别任务数: %d，其中唯一图片数: %d", len(tasks), len(unique_images))
    if warnings:
        logger.warning("发现 %d 条路径/格式警告，下面仅显示前 20 条", len(warnings))
        for msg in warnings[:20]:
            logger.warning(msg)

    if args.dry_run:
        print(json.dumps({
            "json_path": str(json_path),
            "output_path": str(output_path),
            "records_source": str(records_source),
            "resume_from_output": (not args.inplace and records_source == output_path),
            "base_dir": str(base_dir),
            "records": len(records),
            "limit": args.limit,
            "tasks": len(tasks),
            "unique_images": len(unique_images),
            "warnings_count": len(warnings),
            "warnings_sample": warnings[:20],
        }, ensure_ascii=False, indent=2))
        return

    if args.inplace:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(json_path, backup_path)
            logger.info("已备份原文件: %s", backup_path)
        else:
            logger.info("备份文件已存在，本次不重复覆盖: %s", backup_path)

    if not tasks:
        logger.info("没有需要识别的图片。")
        if output_path != records_source:
            write_json_records(records, output_path)
            logger.info("已写出 JSON: %s", output_path)
        return

    token = os.getenv("MINERU_API_TOKEN", "").strip()
    if not token:
        raise ValueError("未设置环境变量 MINERU_API_TOKEN，请先 export MINERU_API_TOKEN='你的 token'")

    client = MinerUClient(token=token, base_url=args.base_url)
    total_stats = {
        "question_updated": 0,
        "answer_updated": 0,
        "failed": 0,
    }
    total = len(unique_images)

    for start in range(0, total, args.batch_size):
        batch = unique_images[start:start + args.batch_size]
        batch_no = start // args.batch_size + 1
        logger.info("=== 第 %d 批: %d/%d，共 %d 张图片 ===", batch_no, start + 1, total, len(batch))

        results = recognize_image_batch(
            client=client,
            batch=batch,
            model_version=args.model_version,
            poll_interval=args.poll_interval,
            result_timeout=args.result_timeout,
            enable_formula=not args.disable_formula,
            enable_table=not args.disable_table,
        )

        batch_tasks = select_tasks_for_batch(tasks=tasks, path_map=path_map, batch=batch)
        batch_stats = apply_results(
            records=records,
            tasks=batch_tasks,
            path_map=path_map,
            results=results,
            keep_mineru_meta=args.keep_mineru_meta,
        )
        merge_stats(total_stats, batch_stats)

        write_json_records(records, output_path)
        logger.info(
            "第 %d 批完成并已保存: %s；本批统计: %s",
            batch_no,
            output_path,
            json.dumps(batch_stats, ensure_ascii=False),
        )

    logger.info("全部完成，输出 JSON: %s", output_path)
    logger.info("累计更新统计: %s", json.dumps(total_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
