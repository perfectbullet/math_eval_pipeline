"""
PRM 评分数据查询 API。

按 id 字段查询 prm_step_scores JSONL 文件中的记录。

用法:
    # 默认加载 results/prm/ 下第一个 *-gaokao-*.jsonl 文件
    python app/math_render_service.py

    # 指定文件（支持多个，用空格分隔）
    python app/math_render_service.py --file results/prm/a.jsonl results/prm/b.jsonl

    # 指定端口（默认 8900）
    python app/math_render_service.py --port 8901

接口:
    GET /scores/{id}    按 id 查询，返回一条 JSON
    GET /scores         id 为空时返回第一条
    GET /scores/        同上
    GET /ids            返回所有 id 列表
    GET /               健康检查
"""

import argparse
import json
import re
import sys
from pathlib import Path

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException


# ── 行内公式空格修复 ─────────────────────────────────────────────────────
# KaTeX 要求行内公式 $...$ 的 $ 与内容之间不能有空格，否则无法渲染。
# 例如 $ m = -\frac{3}{4} $ → $m = -\frac{3}{4}$

_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$]+?)\$")


def _fix_inline_math(text: str) -> str:
    """修复行内公式空格：$ xxx $ → $xxx$（不影响 $$...$$ 块级公式）。"""
    if not text or "$" not in text:
        return text

    # 1. 保护 $$...$$ 块级公式
    placeholders: list[str] = []

    def _save(m):
        placeholders.append(m.group(0))
        return f"%%MATH{len(placeholders) - 1}%%"

    protected = _DISPLAY_MATH_RE.sub(_save, text)

    # 2. 修复行内公式：去掉 $ 紧内侧的首尾空格
    def _trim(m):
        inner = m.group(1)
        stripped = inner.strip()
        return f"${stripped}$" if stripped != inner else m.group(0)

    fixed = _INLINE_MATH_RE.sub(_trim, protected)

    # 3. 恢复块级公式
    for idx, block in enumerate(placeholders):
        fixed = fixed.replace(f"%%MATH{idx}%%", block)

    return fixed


# ── 去掉 <think...>...</think > 思考过程 ──────────────────────────────────
# Qwen3 等模型的输出包含 <think xmlns="...">...</think > 包裹的思考过程，
# 前端不需要展示，加载时去掉。

_THINK_TAG_RE = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """去掉 <think...>...</think > 包裹的思考过程文本。"""
    if not text or "<think" not in text:
        return text
    return _THINK_TAG_RE.sub("", text).strip()


# ── \boxed{} 包裹修复 ───────────────────────────────────────────────────
# 模型输出 \boxed{xxx} 时偶尔没有用 $ 包裹，KaTeX 无法渲染。
# 检测裸露的 \boxed{...}（前面不是 $），自动补上 $...$。

_BOXED_RE = re.compile(r"(?<!\$)(\\boxed\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})")


def _fix_boxed_wrapping(text: str) -> str:
    """为未被 $ 包裹的 \boxed{...} 补上 $...$（跳过已在 $$...$$ 内的）。"""
    if not text or "\\boxed{" not in text:
        return text

    # 1. 保护 $$...$$ 块级公式，避免误包内部 \boxed
    placeholders: list[str] = []

    def _save(m):
        placeholders.append(m.group(0))
        return f"%%DMATH{len(placeholders) - 1}%%"

    protected = _DISPLAY_MATH_RE.sub(_save, text)

    # 2. 对裸露的 \boxed{...} 补 $...$
    def _wrap(m):
        return f"${m.group(1)}$"

    fixed = _BOXED_RE.sub(_wrap, protected)

    # 3. 恢复块级公式
    for idx, block in enumerate(placeholders):
        fixed = fixed.replace(f"%%DMATH{idx}%%", block)

    return fixed

# 全局数据存储 {id: record}
_data: dict[str, dict] = {}
_data_files: list[Path] = []


@asynccontextmanager
async def lifespan(application: FastAPI):
    """启动时加载数据。"""
    global _data, _data_files
    file_paths = application.state.file_paths
    if not file_paths:
        default = _find_default_file()
        if default is None:
            print("错误：未找到 PRM 评分文件，请用 --file 指定", file=sys.stderr)
            sys.exit(1)
        file_paths = [default]
    for p in file_paths:
        print(f"加载数据: {p}")
    _data_files = file_paths
    _data = load_jsonl_multi(file_paths)
    print(f"共 {len(_data)} 条记录")
    yield


app = FastAPI(title="PRM Scores API", version="1.0.0", lifespan=lifespan)


def load_jsonl(path: Path) -> dict[str, dict]:
    """加载单个 JSONL 文件，返回 {id: record} 字典。自动修复公式渲染。"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rid = record.get("id", "")
            if rid:
                _fix_record_math(record)
                result[rid] = record
    return result


def load_jsonl_multi(paths: list[Path]) -> dict[str, dict]:
    """加载多个 JSONL 文件，合并为 {id: record} 字典（后加载的覆盖先加载的）。"""
    result: dict[str, dict] = {}
    for path in paths:
        result.update(load_jsonl(path))
    return result


# 需要修复行内公式空格的顶层文本字段
_TEXT_FIELDS = ("question", "model_output", "reference_answer")


def _fix_record_math(record: dict):
    """对 record 中的文本字段和 steps[].text 修复公式渲染问题。"""
    for key in _TEXT_FIELDS:
        if key in record and isinstance(record[key], str):
            record[key] = _strip_think_tags(record[key])
            record[key] = _fix_boxed_wrapping(record[key])
            record[key] = _fix_inline_math(record[key])
    for step in record.get("steps", []):
        if isinstance(step, dict) and "text" in step and isinstance(step["text"], str):
            step["text"] = _strip_think_tags(step["text"])
            step["text"] = _fix_boxed_wrapping(step["text"])
            step["text"] = _fix_inline_math(step["text"])


def _find_default_file() -> Path | None:
    """在 results/prm/ 下查找第一个 *-gaokao-*.jsonl 文件。"""
    prm_dir = Path(__file__).resolve().parent.parent / "results" / "prm"
    if not prm_dir.exists():
        return None
    for f in sorted(prm_dir.glob("*-gaokao-*.jsonl")):
        return f
    return None


@app.get("/")
def health():
    return {"status": "ok", "total": len(_data)}


@app.get("/ids")
def list_ids():
    """重新加载数据并返回所有 id 列表。"""
    global _data
    if _data_files:
        _data = load_jsonl_multi(_data_files)
    return list(_data.keys())


@app.get("/scores/{item_id}")
def get_score(item_id: str):
    """按 id 查询一条记录。"""
    if item_id not in _data:
        raise HTTPException(status_code=404, detail=f"id '{item_id}' 未找到")
    return _data[item_id]


@app.get("/scores")
@app.get("/scores/")
def get_first():
    """id 为空时返回第一条记录。"""
    if not _data:
        raise HTTPException(status_code=404, detail="无数据")
    first_id = next(iter(_data))
    return _data[first_id]


def main():
    parser = argparse.ArgumentParser(description="PRM 评分数据查询 API")
    parser.add_argument("--file", type=str, nargs="+", default=None, help="JSONL 文件路径（支持多个）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8900, help="监听端口")
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.file] if args.file else []
    app.state.file_paths = file_paths

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
