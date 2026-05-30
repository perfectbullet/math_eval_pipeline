#!/usr/bin/env python3
"""MinerU 精准解析 API 批量处理脚本：上传本地文件到 MinerU，获取解析结果并保存为 JSONL。

用法:
    # 基本用法
    python extract_mineru_sdk.py --input_dir /path/to/files

    # 断点续跑（跳过已有输出文件中已处理的文件）
    python extract_mineru_sdk.py --input_dir /path/to/files --skip_existing

    # 限制处理数量
    python extract_mineru_sdk.py --input_dir /path/to/files --limit 10

    # 指定模型版本
    python extract_mineru_sdk.py --input_dir /path/to/files --model_version vlm
"""

import argparse
import json
import logging
import os
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://mineru.net"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_BATCH_SIZE = 50
DEFAULT_UPLOAD_TIMEOUT = 30
DEFAULT_RESULT_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
}


class MinerUClient:
    """MinerU 精准解析 API 客户端，支持本地文件批量上传解析。"""

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
        """批量上传本地文件，返回 batch_id。

        流程: POST /api/v4/file-urls/batch 获取上传链接 → PUT 上传每个文件。
        """
        url = f"{self.base_url}/api/v4/file-urls/batch"
        files_payload = []
        for i, fp in enumerate(file_paths):
            item = {"name": fp.name}
            if data_ids and i < len(data_ids):
                item["data_id"] = data_ids[i]
            else:
                item["data_id"] = fp.stem
            files_payload.append(item)

        data = {
            "files": files_payload,
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }

        logger.info("申请 %d 个文件的上传链接...", len(file_paths))
        resp = requests.post(url, headers=self.headers, json=data, timeout=DEFAULT_UPLOAD_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")

        batch_id = result["data"]["batch_id"]
        upload_urls = result["data"]["file_urls"]

        logger.info("上传 %d 个文件...", len(file_paths))
        for i, (fp, upload_url) in enumerate(zip(file_paths, upload_urls)):
            with open(fp, "rb") as f:
                put_resp = requests.put(upload_url, data=f, timeout=120)
                put_resp.raise_for_status()
            logger.debug("  [%d/%d] %s 上传成功", i + 1, len(file_paths), fp.name)

        logger.info("batch_id=%s 全部上传完成，等待服务端自动提交解析", batch_id)
        return batch_id

    def get_batch_result(self, batch_id: str) -> dict:
        """查询批量任务结果。GET /api/v4/extract-results/batch/{batch_id}"""
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        resp = requests.get(url, headers=self.headers, timeout=DEFAULT_UPLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def poll_batch_result(
        self,
        batch_id: str,
        timeout: int = DEFAULT_RESULT_TIMEOUT,
        interval: int = DEFAULT_POLL_INTERVAL,
    ) -> list[dict]:
        """轮询批量任务直到全部完成或超时，返回 extract_result 列表。"""
        start = time.time()
        pbar = tqdm(desc=f"batch {batch_id[:8]}...", unit="poll")

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                pbar.close()
                raise TimeoutError(f"batch {batch_id} 轮询超时 ({timeout}s)")

            result = self.get_batch_result(batch_id)
            if result.get("code") != 0:
                pbar.close()
                raise RuntimeError(f"查询结果失败: {result.get('msg')}")

            extract_result = result["data"].get("extract_result", [])
            done_count = sum(1 for r in extract_result if r["state"] in ("done", "failed"))
            pbar.set_postfix(done=f"{done_count}/{len(extract_result)}")
            pbar.update(1)

            if done_count == len(extract_result):
                pbar.close()
                return extract_result

            time.sleep(interval)


def _download_via_curl(zip_url: str, tmp_path: Path) -> bool:
    """用 curl 子进程下载，绕过 Python SSL 兼容性问题。"""
    import subprocess
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "-o", str(tmp_path), zip_url],
        capture_output=True, timeout=120,
    )
    return r.returncode == 0


def download_and_extract_markdown(zip_url: str, max_retries: int = 3) -> str | None:
    """下载结果 zip 并提取 full.md 内容。优先 curl，失败回退 requests。"""
    import tempfile

    last_err: Exception = RuntimeError("未知下载错误")

    for attempt in range(1, max_retries + 1):
        # 优先用 curl 下载
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ok = _download_via_curl(zip_url, tmp_path)
            if not ok:
                # 回退 requests
                try:
                    no_proxy = {"http": "", "https": ""}
                    resp = requests.get(zip_url, timeout=120, verify=False, proxies=no_proxy)
                    resp.raise_for_status()
                    tmp_path.write_bytes(resp.content)
                except Exception as e:
                    last_err = e
                    continue

            with zipfile.ZipFile(tmp_path) as zf:
                for name in zf.namelist():
                    if name.endswith("full.md"):
                        return zf.read(name).decode("utf-8")
            return None
        except Exception as e:
            last_err = e
        finally:
            tmp_path.unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(attempt * 3)

    raise last_err


def load_existing_output(output_path: Path) -> set[str]:
    """读取已有输出文件，返回已处理的 data_id 集合。"""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("data_id"):
                    done.add(rec["data_id"])
            except json.JSONDecodeError:
                pass
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="MinerU 批量文件解析脚本")
    parser.add_argument("--input_dir", required=True, help="输入文件目录")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument(
        "--model_version", default=DEFAULT_MODEL_VERSION,
        choices=["pipeline", "vlm", "MinerU-HTML"],
        help="模型版本",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="每批文件数")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已处理文件")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL, help="轮询间隔秒")
    parser.add_argument("--result_timeout", type=int, default=DEFAULT_RESULT_TIMEOUT, help="单批超时秒")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 加载 .env（脚本同级目录）
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    # 读取 token
    token = os.getenv("MINERU_API_TOKEN", "").strip()
    if not token:
        raise ValueError("未设置环境变量 MINERU_API_TOKEN")

    client = MinerUClient(token)

    # 扫描输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    all_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info("扫描到 %d 个文件", len(all_files))

    # 跳过已处理
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"
    if args.skip_existing:
        existing = load_existing_output(output_path)
        all_files = [f for f in all_files if f.stem not in existing]
        logger.info("跳过已处理，剩余 %d 个", len(all_files))

    # 限制数量
    if args.limit:
        all_files = all_files[: args.limit]
        logger.info("限制处理 %d 个", len(all_files))

    if not all_files:
        logger.info("没有需要处理的文件")
        return

    # 分批处理
    total = len(all_files)
    processed = 0

    for batch_start in range(0, total, args.batch_size):
        batch_files = all_files[batch_start: batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        logger.info("=== 第 %d 批 (%d/%d): %d 个文件 ===", batch_num, batch_start + 1, total, len(batch_files))

        data_ids = [f.stem for f in batch_files]

        # 上传
        batch_id = client.upload_batch(
            batch_files, data_ids=data_ids, model_version=args.model_version,
        )

        # 轮询结果
        extract_results = client.poll_batch_result(
            batch_id, timeout=args.result_timeout, interval=args.poll_interval,
        )

        # 构建 data_id → result 映射
        result_map: dict[str, dict] = {}
        for r in extract_results:
            did = r.get("data_id") or Path(r.get("file_name", "")).stem
            result_map[did] = r

        # 处理每个文件的结果
        with open(output_path, "a", encoding="utf-8") as out_f:
            for f, did in zip(batch_files, data_ids):
                r = result_map.get(did, {})
                state = r.get("state", "unknown")
                zip_url = r.get("full_zip_url", "")
                markdown_content = None

                if state == "done" and zip_url:
                    try:
                        markdown_content = download_and_extract_markdown(zip_url)
                    except Exception as e:
                        logger.warning("下载/解压 %s 失败: %s", f.name, e)
                        state = "download_failed"

                record = {
                    "file_name": f.name,
                    "data_id": did,
                    "state": state,
                    "markdown_content": markdown_content,
                    "zip_url": zip_url,
                }
                if r.get("err_msg"):
                    record["err_msg"] = r["err_msg"]

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

        logger.info("第 %d 批完成，累计 %d/%d", batch_num, processed, total)

    logger.info("全部完成，结果写入 %s", output_path)


if __name__ == "__main__":
    main()
