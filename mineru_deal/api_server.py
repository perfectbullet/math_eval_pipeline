#!/usr/bin/env python3
"""MinerU 解析 API 服务 — FastAPI 封装 MinerUClient。

用法:
    MINERU_API_TOKEN=xxx uvicorn api_server:app --host 0.0.0.0 --port 8090

    # 测试
    curl -X POST http://localhost:8090/extract -F "file=@test.png"
"""

import logging
import os

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from extract_mineru_sdk import MinerUClient, download_and_extract_markdown

logger = logging.getLogger(__name__)

app = FastAPI(title="MinerU Extract API")

MINERU_API_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI5MDMwNzg4NyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MDA0NDg3MSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNDk5M2VjMDktZDUxNC00MjMwLTlkZGQtYjM2NDVlYjA0ZDg5IiwiZW1haWwiOiIiLCJleHAiOjE3ODc4MjA4NzF9.Qe8cWmDXUWzBzw3aELE1Q6_4XpeIlo2Yh0T3-Ck0adeOonha1arAdagg3K7Xueo4tlOk8QnbyBqCF_XnWsJeuQ"

TOKEN = os.getenv("MINERU_API_TOKEN", MINERU_API_TOKEN)
if not TOKEN:
    TOKEN = MINERU_API_TOKEN

client = MinerUClient(TOKEN)


@app.post("/extract")
async def extract(
    file: UploadFile = File(..., description="上传文件"),
    model_version: str = Form("vlm", description="pipeline / vlm / MinerU-HTML"),
    timeout: int = Form(600, description="轮询超时秒"),
):
    """上传文件到 MinerU 解析，返回 markdown 内容。"""
    upload_dir = Path(__file__).parent / "upload_images"
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / (file.filename or "upload")
    content = await file.read()
    dest.write_bytes(content)

    try:
        batch_id = client.upload_batch(
            [dest],
            data_ids=[dest.stem],
            model_version=model_version,
        )
        extract_results = client.poll_batch_result(batch_id, timeout=timeout)
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    # 构建结果
    result_map = {}
    for r in extract_results:
        did = r.get("data_id") or Path(r.get("file_name", "")).stem
        result_map[did] = r

    r = result_map.get(dest.stem, {})
    state = r.get("state", "unknown")
    zip_url = r.get("full_zip_url", "")
    markdown_content = None

    if state == "done" and zip_url:
        try:
            markdown_content = download_and_extract_markdown(zip_url)
        except Exception as e:
            logger.warning("下载/解压失败: %s", e)
            state = "download_failed"

    record = {
        "file_name": dest.name,
        "data_id": dest.stem,
        "state": state,
        "markdown_content": markdown_content,
        "zip_url": zip_url,
    }
    if r.get("err_msg"):
        record["err_msg"] = r["err_msg"]

    return {"results": [record]}
