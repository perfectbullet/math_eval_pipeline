"""
PRM 评分数据查询 API — 带流式处理管线。

处理流程：加载原始数据 → 按需对 model_output 执行
  去思考标签(ThinkTagBuffer) → 断句(SentenceBuffer) → LaTeX修复(normalize_latex_formulas)
→ 保存中间结果 → 返回处理后的数据。

用法:
    python app/math_render_service.py \
      --file ../results/prm/prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl \
      --port 8900

接口:
    GET /scores/{id}    按 id 查询，返回处理后的 JSON
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

# 支持 python app/math_render_service.py 直接启动
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.think_tag_buffer import ThinkTagBuffer
from app.sentence_buffer import SentenceBuffer
from app.latex import normalize_latex_formulas


class MathRecord(BaseModel):
    """JSONL 中的一条数学题模型输出记录。"""

    id: str = Field(..., description="题目或样本 id")
    source: str = Field(..., description="数据来源")
    question: str = Field(..., description="题目文本")
    reference_answer: str = Field(..., description="参考答案")
    model_name: str = Field(..., description="生成结果的模型名称")
    model_output: str = Field(..., description="模型原始输出")
    final_answer_raw: str = Field("", description="原始最终答案")
    metadata: dict = Field(default_factory=dict, description="额外元数据")

    class Config:
        extra = "allow"


class MathRecordUpdate(BaseModel):
    """允许从 API 更新的题目字段。"""

    question: str | None = Field(None, description="题目文本")
    reference_answer: str | None = Field(None, description="参考答案")

    class Config:
        extra = "forbid"


# ── 模拟流式输出 ───────────────────────────────────────────────────────────

class FakeStreamer:
    """将完整文本模拟成流式 chunk 输出。

    特点：
    1. 遇到 <think> 或 </think> 时，整个标签作为一个 token 返回；
    2. 其他内容按 chunk_size 切块返回；
    3. 不打印调试信息；
    4. 校验 chunk_size，避免死循环；
    5. 支持大小写标签，例如 <Think>、</THINK>。
    """

    _THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

    def __init__(self, text: str, chunk_size: int = 1):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        self.text = text or ""
        self.chunk_size = chunk_size
        self.pos = 0
        self.length = len(self.text)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.pos >= self.length:
            raise StopIteration

        # 如果当前位置正好是 <think> 或 </think>，完整返回标签
        match = self._THINK_TAG_RE.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            return match.group(0)

        # 普通文本按 chunk_size 返回，但不能把即将出现的 think 标签切断
        next_tag = self._THINK_TAG_RE.search(self.text, self.pos)

        if next_tag:
            end = min(self.pos + self.chunk_size, next_tag.start())
        else:
            end = min(self.pos + self.chunk_size, self.length)

        # 理论兜底：避免 end == self.pos 导致死循环
        if end <= self.pos:
            end = min(self.pos + self.chunk_size, self.length)

        chunk = self.text[self.pos:end]
        self.pos = end
        return chunk


# ── 处理管线 ───────────────────────────────────────────────────────────────

async def _process_model_output(item_id: str, model_output: str) -> dict:
    """处理流水线：去思考标签 → 断句 → LaTeX修复。"""

    # Step 1: FakeStreamer + ThinkTagBuffer 去思考标签
    streamer = FakeStreamer(model_output)
    think_buffer = ThinkTagBuffer()
    sentence_buffer = SentenceBuffer()

    segments: list[str] = []

    for token in streamer:
        if 'think' in token:
            print('_process_model_output token ', token)
        filtered = think_buffer.add(token)
        if filtered is None:
            continue
        for char in filtered:
            segment = sentence_buffer.add(char)
            if segment:
                segments.append(segment)

    # flush think_buffer 剩余
    remaining = think_buffer.flush()
    if remaining:
        for char in remaining:
            segment = sentence_buffer.add(char)
            if segment:
                segments.append(segment)
    print(f"segments len is {len(segments)}")
    # flush sentence_buffer 剩余
    final_segment = await sentence_buffer.flush(is_final=True)
    if final_segment and final_segment.content:
        segments.append(final_segment.content)

    # Step 2: LaTeX 修复
    fixed_segments = [normalize_latex_formulas(s) for s in segments]
    print(f"fixed_segments len is {len(fixed_segments)}")
    # Step 3: 拼接
    model_output_new = "".join(fixed_segments)

    return {
        "id": item_id,
        "segments": segments,
        "fixed_segments": fixed_segments,
        "model_output_new": model_output_new,
    }


# ── 缓存 & 文件保存 ───────────────────────────────────────────────────────

_processed: dict[str, dict] = {}   # id → 处理结果缓存
_output_dir: Path | None = None
_sentences_file: Path | None = None
_latex_fixed_file: Path | None = None


def _init_output_paths(file_paths: list[Path]):
    """根据 --file 路径推导输出文件路径。"""
    global _output_dir, _sentences_file, _latex_fixed_file
    if not file_paths:
        return
    primary = file_paths[0].resolve()
    _output_dir = primary.parent
    stem = primary.stem
    _sentences_file = _output_dir / f"{stem}_sentences.json"
    _latex_fixed_file = _output_dir / f"{stem}_latex_fixed.json"
    print(f"输出文件: {_sentences_file}")
    print(f"输出文件: {_latex_fixed_file}")


def _save_results():
    """将处理结果保存到 JSON 文件。"""
    if not _processed or not _sentences_file or not _latex_fixed_file:
        return

    # sentences.json: [{id, segments}, ...]
    sentences_data = [
        {"id": v["id"], "segments": v["segments"]}
        for v in _processed.values()
    ]
    _sentences_file.write_text(
        json.dumps(sentences_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # latex_fixed.json: [{id, fixed_segments, model_output_new}, ...]
    latex_data = [
        {
            "id": v["id"],
            "fixed_segments": v["fixed_segments"],
            "model_output_new": v["model_output_new"],
        }
        for v in _processed.values()
    ]
    _latex_fixed_file.write_text(
        json.dumps(latex_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── 数据加载 ───────────────────────────────────────────────────────────────

_data: dict[str, dict] = {}
_data_files: list[Path] = []


def reload_data() -> dict[str, dict]:
    """从当前数据文件重新加载记录，并清空处理缓存。"""
    global _data
    _data = load_jsonl_multi(_data_files)
    _processed.clear()
    return _data


def update_jsonl_record(item_id: str, record: dict) -> list[Path]:
    """将指定 id 的记录写回所有包含该 id 的 JSONL 文件。"""
    updated_files: list[Path] = []
    replacement = json.dumps(record, ensure_ascii=False)

    for path in _data_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        new_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue

            current = json.loads(stripped)
            if current.get("id") == item_id:
                new_lines.append(replacement)
                changed = True
            else:
                new_lines.append(line)

        if changed:
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            updated_files.append(path)

    return updated_files


@asynccontextmanager
async def lifespan(application: FastAPI):
    """启动时加载原始数据（不做修复）。"""
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
    _init_output_paths(file_paths)
    reload_data()
    print(f"共 {len(_data)} 条记录")
    yield


app = FastAPI(title="PRM Scores API", version="2.0.0", lifespan=lifespan)


def load_jsonl(path: Path) -> dict[str, dict]:
    """加载单个 JSONL 文件，返回 {id: record} 字典。保留原始数据，不做修复。"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rid = record.get("id", "")
            if rid:
                result[rid] = record
    return result


def load_jsonl_multi(paths: list[Path]) -> dict[str, dict]:
    """加载多个 JSONL 文件，合并为 {id: record} 字典（后加载的覆盖先加载的）。"""
    result: dict[str, dict] = {}
    for path in paths:
        result.update(load_jsonl(path))
    return result


def _find_default_file() -> Path | None:
    """在 results/prm/ 下查找第一个 *-gaokao-*.jsonl 文件。"""
    prm_dir = Path(__file__).resolve().parent.parent.parent / "results" / "prm"
    if not prm_dir.exists():
        return None
    for f in sorted(prm_dir.glob("*-gaokao-*.jsonl")):
        return f
    return None


def _item_id_sort_key(item_id: str) -> tuple[str, int, str]:
    """按 id 前缀和末尾数字排序，例如 MATH-001 < MATH-247。"""
    match = re.match(r"^(.*?)(\d+)$", item_id)
    if not match:
        return (item_id, -1, item_id)
    prefix, number = match.groups()
    return (prefix, int(number), item_id)


# ── API 端点 ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "total": len(_data),
        "processed": len(_processed),
    }


@app.get("/ids")
def list_ids():
    """重新加载数据并返回所有 id 列表。"""
    reload_data()
    return sorted(_data.keys(), key=_item_id_sort_key)


@app.put("/scores/{item_id}")
def update_score(item_id: str, patch: MathRecordUpdate):
    """按 item_id 更新 question/reference_answer，并标记需要重新推理。"""
    reload_data()
    if item_id not in _data:
        raise HTTPException(status_code=404, detail=f"id '{item_id}' 未找到")

    updates = patch.dict(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="至少需要提供 question 或 reference_answer")

    record = dict(_data[item_id])
    changed = False
    for field_name, value in updates.items():
        if record.get(field_name) != value:
            record[field_name] = value
            changed = True

    if changed:
        record["needs_model_rerun"] = True

    updated_files = update_jsonl_record(item_id, record)
    if not updated_files:
        raise HTTPException(status_code=404, detail=f"id '{item_id}' 未在数据文件中找到")

    reload_data()
    return _data[item_id]


@app.get("/scores/{item_id}")
async def get_score(item_id: str):
    """按 id 查询一条记录，自动处理 model_output。"""
    if item_id not in _data:
        raise HTTPException(status_code=404, detail=f"id '{item_id}' 未找到")

    record = dict(_data[item_id])  # 浅拷贝，不修改原始数据

    # 已缓存则直接用
    if item_id in _processed:
        record["model_output_original"] = _data[item_id].get("model_output", "")
        record["model_output"] = _processed[item_id]["model_output_new"]
        return record

    # 处理流水线
    model_output = record.get("model_output", "")
    if model_output:
        result = await _process_model_output(item_id, model_output)
        _processed[item_id] = result
        _save_results()
        record["model_output_original"] = model_output
        record["model_output"] = result["model_output_new"]
    else:
        record["model_output_original"] = ""

    return record


@app.get("/scores")
@app.get("/scores/")
async def get_first():
    """id 为空时返回第一条记录。"""
    if not _data:
        raise HTTPException(status_code=404, detail="无数据")
    first_id = next(iter(_data))
    return await get_score(first_id)


def main():
    parser = argparse.ArgumentParser(description="PRM 评分数据查询 API（带流式处理管线）")
    parser.add_argument("--file", type=str, nargs="+", default=None, help="JSONL 文件路径（支持多个）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8900, help="监听端口")
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.file] if args.file else []
    app.state.file_paths = file_paths

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
