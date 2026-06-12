# PRM 评分数据查询 API

这是一个用于查看和修复 PRM 评分数据的轻量级 FastAPI 服务。

项目主要面向数学模型输出结果的展示场景，支持从 JSONL 文件中加载 PRM 评分数据，并在查询单条记录时，对 `model_output` 执行流式处理管线：

```text
原始 model_output
    → FakeStreamer 模拟流式输出
    → ThinkTagBuffer 过滤 <think>...</think> 思考内容
    → SentenceBuffer 按句子 / 长度 / 时间切分文本
    → normalize_latex_formulas 修复 LaTeX 公式
    → 缓存处理结果
    → 返回前端可渲染数据
```

---

## 功能特性

- 支持加载一个或多个 JSONL 数据文件。
- 支持按 `id` 查询 PRM 评分记录。
- 支持默认返回第一条记录，方便快速调试。
- 支持过滤模型输出中的 `<think>` 思考标签。
- 支持模拟流式 token/chunk 输出。
- 支持智能断句，避免把 LaTeX 公式切坏。
- 支持修复常见 LaTeX 渲染问题。
- 支持将处理中间结果保存为 JSON 文件。

---

## 项目结构

```text
app/
├── math_render_service.py   # FastAPI 服务入口，负责数据加载、接口暴露和处理管线调度
├── think_tag_buffer.py      # 流式过滤 <think>...</think> 标签
├── sentence_buffer.py       # 流式断句缓冲区，保护 LaTeX 公式完整性
├── latex.py                 # LaTeX 公式规范化和修复工具
└── stream_split.py          # 流式断句模块，当前为待实现占位文件
```

---

## 环境要求

建议使用 Python 3.10 或以上版本。

核心依赖：

```bash
pip install fastapi uvicorn
```

可选依赖：

```bash
pip install loguru
```

如果没有安装 `loguru`，`sentence_buffer.py` 会自动回退到 Python 标准库 `logging`。

---

## 快速启动

在项目根目录执行：

```bash
python app/math_render_service.py \
  --file ../results/prm/prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl \
  --port 8900
```

也可以指定多个 JSONL 文件：

```bash
python app/math_render_service.py \
  --file data/a.jsonl data/b.jsonl \
  --host 0.0.0.0 \
  --port 8900
```

多个文件会合并为一个 `{id: record}` 字典。如果多个文件中存在相同 `id`，后加载的文件会覆盖先加载的记录。

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `--file` | `str[]` | `None` | JSONL 文件路径，支持多个 |
| `--host` | `str` | `0.0.0.0` | 服务监听地址 |
| `--port` | `int` | `8900` | 服务监听端口 |

如果没有指定 `--file`，服务会尝试在 `results/prm/` 目录下查找第一个匹配 `*-gaokao-*.jsonl` 的文件。

---

## 数据格式

输入文件为 JSONL 格式，每一行是一条 JSON 记录。

最低要求字段：

```json
{
  "id": "sample-001",
  "model_output": "模型输出内容"
}
```

常见完整字段示例：

```json
{
  "id": "sample-001",
  "question": "题目内容",
  "model_output": "<think>这里是思考过程</think>这里是最终答案：\\boxed{2}",
  "reference_answer": "参考答案",
  "steps": [
    {
      "text": "步骤内容",
      "score": 1
    }
  ]
}
```

接口返回时会保留原始记录字段，并额外加入或替换：

```json
{
  "model_output_original": "原始模型输出",
  "model_output": "过滤 think 并修复 LaTeX 后的模型输出"
}
```

---

## API 接口

### 健康检查

```http
GET /
```

返回示例：

```json
{
  "status": "ok",
  "total": 120,
  "processed": 3
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `total` | 当前加载的原始记录数量 |
| `processed` | 已经触发处理并缓存的记录数量 |

---

### 查询所有 ID

```http
GET /ids
```

返回示例：

```json
[
  "sample-001",
  "sample-002",
  "sample-003"
]
```

---

### 查询指定记录

```http
GET /scores/{id}
```

示例：

```bash
curl http://127.0.0.1:8900/scores/sample-001
```

处理逻辑：

1. 根据 `id` 查找原始记录。
2. 如果该记录已经处理过，直接返回缓存结果。
3. 如果未处理过，执行处理管线：
   - `FakeStreamer`
   - `ThinkTagBuffer`
   - `SentenceBuffer`
   - `normalize_latex_formulas`
4. 保存中间结果。
5. 返回处理后的记录。

---

### 查询第一条记录

```http
GET /scores
GET /scores/
```

用于没有明确 `id` 时快速预览第一条数据。

---

## 处理管线说明

### 1. FakeStreamer

`FakeStreamer` 用于把完整文本模拟成流式 chunk 输出。

它有两个关键行为：

- 遇到 `<think>` 或 `</think>` 时，整个标签作为一个 token 返回。
- 其他内容按照 `chunk_size` 切块返回。

这样可以模拟真实模型流式输出时，标签作为完整 token 输出的行为。

---

### 2. ThinkTagBuffer

`ThinkTagBuffer` 是一个状态机，用于过滤流式文本中的 think 标签内容。

状态包括：

| 状态 | 说明 |
|---|---|
| `outside` | 正常输出状态 |
| `detect_open` | 检测 `<think>` 开始标签 |
| `inside` | 位于 think 内容内部，丢弃 token |
| `detect_close` | 检测 `</think>` 结束标签 |

它的作用是把：

```text
<think>这里是模型思考过程</think>这里是最终答案
```

过滤为：

```text
这里是最终答案
```

---

### 3. SentenceBuffer

`SentenceBuffer` 用于对流式文本进行智能断句。

优先级大致如下：

1. 句末标点断句，例如 `。！？.!?`
2. 逗号、分号等弱标点断句
3. 超过最大字符数时强制断句
4. 超过等待时间时强制 flush
5. 如果当前位置处于 LaTeX 公式内部，则尽量等待公式闭合后再切分

它会保护以下公式结构：

```text
$...$
$$...$$
\(...\)
\[...\]
\boxed{...}
\begin{...}...\end{...}
```

这样可以避免前端渲染时出现公式被切断的问题。

---

### 4. LaTeX 修复

`latex.py` 提供了多个 LaTeX 修复函数，核心入口是：

```python
normalize_latex_formulas(text: str) -> str
```

主要处理：

- 将 `\(...\)`、`\[...\]` 等定界符规范化为 `$...$` 或 `$$...$$`
- 移除公式定界符内侧多余空格
- 为裸露的 `\boxed{...}` 补充 `$...$`
- 将 `tabular` 转换为 `array`
- 为裸露的 `\begin{...}...\end{...}` 补充 `$$...$$`
- 将 `\text` 替换为 `\mathrm`

---

## 中间结果文件

当某条记录第一次被请求并处理后，服务会把当前已处理结果保存到输入文件所在目录。

假设输入文件为：

```text
prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl
```

则会生成：

```text
prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602_sentences.json
prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602_latex_fixed.json
```

其中：

- `*_sentences.json` 保存断句后的 `segments`
- `*_latex_fixed.json` 保存修复后的 `fixed_segments` 和拼接后的 `model_output_new`

---

## 使用示例

### 启动服务

```bash
python app/math_render_service.py \
  --file ../results/prm/prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl \
  --host 0.0.0.0 \
  --port 8900
```

### 查看服务状态

```bash
curl http://127.0.0.1:8900/
```

### 查看所有 ID

```bash
curl http://127.0.0.1:8900/ids
```

### 查询指定记录

```bash
curl http://127.0.0.1:8900/scores/sample-001
```

### 查询第一条记录

```bash
curl http://127.0.0.1:8900/scores
```

---

## 开发说明

### 当前处理模式

当前实现是“按需处理”：

```text
服务启动
    → 加载原始 JSONL
    → 不处理 model_output

首次请求某个 id
    → 处理该条 model_output
    → 写入内存缓存 _processed
    → 保存中间结果文件

再次请求同一个 id
    → 直接读取 _processed 缓存
```

这种方式启动速度快，但第一次访问某条记录时会有处理开销。

如果数据量不大，或者希望前端查询更快，可以改造成“启动时全部预处理”的模式。

---

## 已知注意点

1. `stream_split.py` 当前还是占位文件，实际断句逻辑集中在 `sentence_buffer.py`。
2. 当前 `_save_results()` 会在每次新记录被处理后保存全部已处理结果；如果数据量很大，频繁请求不同 `id` 时可能产生较多磁盘写入。
3. 当前 `ThinkTagBuffer` 主要面向流式 token 过滤，如果已经拿到完整文本，也可以考虑用正则一次性提取和删除 think 内容。
4. 当前服务使用模块级全局变量保存 `_data` 和 `_processed`，单进程运行没问题；如果使用多 worker 部署，需要注意进程间缓存不共享。
5. 当前代码里仍有少量调试 `print`，生产环境建议替换为 `logging` 或 `loguru`。

---

## 后续优化建议

可以考虑以下改造：

- 启动时预处理全部记录，避免首次查询卡顿。
- 增加 `--rebuild` 参数，用于控制是否重建中间结果。
- 优先读取已有 `*_latex_fixed.json` 缓存文件，减少重复计算。
- 将 `_data`、`_processed` 移入 `app.state`，减少全局变量。
- 将保存逻辑改为定时保存或退出时保存，降低频繁写盘。
- 为 `SentenceBuffer`、`ThinkTagBuffer`、`normalize_latex_formulas` 增加单元测试。
- 补全 `stream_split.py`，或者删除该占位文件，避免模块职责混乱。

---

## License

当前代码未声明许可证。若计划开源，请补充 LICENSE 文件。
