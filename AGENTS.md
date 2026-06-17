# 项目结构

- `scripts/`：主评测流水线脚本，覆盖标准化、模型推理、Math-Verify、PRM、报告生成。
- `tir_math/`：TIR/COT 推理子项目，包含并发推理脚本 `run_model_tir_inference_concurrent.py` 和四模式批量推理脚本 `run_all_mode_lang_inference.sh`。
- `math_render_service/`：FastAPI 子项目，用于处理和渲染 PRM `model_output`。
- `mineru_deal/`：MinerU 接入子项目，负责图片/PDF 解析为 markdown 文本。
- `data/`：输入数据、标准化结果、模型输出等中间目录。
- `results/`：验证结果、PRM 结果、报告输出目录。
- `reports/`：跨步骤汇总指标输出目录。
- `demo_scripts/`：演示、联调与接口测试脚本目录，包含 `test_api_math.py`、`test_api_math_markdown.py`、`test_lx_conficius3_client.py` 等。

# 安装命令

- 主项目环境：`conda activate math-eval`
- 运行脚本默认 Python：`/home/zj/miniconda3/envs/math-eval/bin/python`
- 主项目依赖：`pip install -r requirements.txt`
- 补充依赖：`pip install requests tqdm`
- `math_render_service` 依赖：`pip install -r math_render_service/requirements.txt`

# 运行与测试命令

- Excel 转 JSON：`python scripts/excel_to_json.py ../data/数字人数学场景测试-2026-06-12.xlsx -o ../data/math_qa_275_20260612.json`
- MinerU OCR 转文本：`python mineru_deal/json_image_mineru_ocr.py ../data/math_qa_275_20260612.json -o ../data/math_qa_275_20260612.mineru.json`
- 标准化数据：`python scripts/standardize_datasets.py --input_dir data/raw --output_dir data/standardized --limit 1000`
- 模型推理：`python scripts/run_model_inference.py --input data/standardized/train_or_eval_all.jsonl --output data/model_outputs/model_outputs.jsonl --api_base http://<host>/v1 --model <model> --limit 100 --skip_existing --timeout 600`
- TIR 并发推理，`en`：`cd tir_math && python run_model_tir_inference_concurrent.py --input ../data/math_qa_275_20260612.mineru.json --output ../data/model_outputs/math_qa_275_20260612.jsonl --api_base http://192.168.100.203:8200/v1 --model Qwen3-32B --mode tir --lang en --workers 8`
- COT 并发推理，`en`：`cd tir_math && python run_model_tir_inference_concurrent.py --input ../data/math_qa_275_20260612.mineru.json --output ../data/model_outputs/model_outputs-math_qa_275_20260612_cot_en.jsonl --api_base http://192.168.100.203:8200/v1 --model Qwen3-32B --mode cot --lang en --workers 8`
- TIR 并发推理，`zh`：`cd tir_math && python run_model_tir_inference_concurrent.py --input ../data/math_qa_275_20260612.mineru.json --output ../data/model_outputs/model_outputs-math_qa_275_20260613_tir_zh_qwen3-32b.jsonl --api_base http://192.168.100.203:8200/v1 --model Qwen3-32B --mode tir --lang zh --workers 8`
- COT 并发推理，`zh`：`cd tir_math && python run_model_tir_inference_concurrent.py --input ../data/math_qa_275_20260612.mineru.json --output ../data/model_outputs/model_outputs-math_qa_275_20260613_cot_zh_qwen3-32b.jsonl --api_base http://192.168.100.203:8200/v1 --model Qwen3-32B --mode cot --lang zh --workers 8`
- 运行 Math-Verify：`python scripts/run_math_verify.py --input data/model_outputs/model_outputs.jsonl --output results/verify/verify_results.jsonl --summary results/verify/verify_summary.md`
- 运行 Qwen PRM：`python scripts/run_qwen_prm.py --input results/verify/verify_results.jsonl --output results/prm/prm_step_scores.jsonl --summary results/prm/prm_summary.md --model <prm_model_path> --cache_dir <hf_cache_dir>`
- 生成归因报告：`python scripts/build_error_report.py --verify results/verify/verify_results.jsonl --prm results/prm/prm_step_scores.jsonl --output_md results/reports/error_attribution_report.md --output_jsonl results/reports/error_cases.jsonl`
- 启动 `math_render_service`：`python math_render_service/app/math_render_service.py --file results/prm/<file>.jsonl --port 8900`
- 测试 `math_render_service`：`pytest math_render_service/tests -q`
- 测试切分逻辑：`python scripts/test_split_steps.py --input results/verify/<file>.jsonl --output results/test_split_steps_report.md --limit 50`
- 启动 MinerU 服务：`cd mineru_deal && MINERU_API_TOKEN=<token> uvicorn api_server:app --host 0.0.0.0 --port 8090`

# 常用批处理脚本

## `tir_math/run_all_mode_lang_inference.sh`

- 用途：对同一输入数据批量执行 `cot/tir` 与 `en/zh` 组合推理，输出到指定目录。
- 执行目录：在 `tir_math/` 下运行。
- 默认行为：`start` 会用 `nohup` 后台启动，并把日志写到 `tir_math/logs/`，PID 写到 `tir_math/run_pids/`。
- 输出文件名格式：`model_outputs-<input_stem>_<mode>_<lang>_<model_slug>.jsonl`。
- `input_stem` 会从输入文件名去掉 `.jsonl`、`.json`、`.mineru` 后追加 `RUN_DATE`，`RUN_DATE` 默认当天 `YYYYMMDD`，也可通过环境变量指定。

常用命令：

```bash
cd tir_math
bash run_all_mode_lang_inference.sh start \
  --input ../data/standardized/gaokao_bench.jsonl \
  --output_dir ../data/model_outputs/gaokao_bench \
  --api_base http://192.168.100.203:8200/v1 \
  --model DeepSeek-R1-Distill-Llama-70B \
  --workers 8 \
  --limit 100 \
  --api_key no-key
```

查看最近任务：

```bash
cd tir_math
bash run_all_mode_lang_inference.sh status
```

停止最近任务：

```bash
cd tir_math
bash run_all_mode_lang_inference.sh stop
```

指定日期前缀：

```bash
cd tir_math
RUN_DATE=20260616 bash run_all_mode_lang_inference.sh start \
  --input ../data/standardized/gaokao_bench.jsonl \
  --output_dir ../data/model_outputs/gaokao_bench \
  --api_base http://192.168.100.203:8200/v1 \
  --model DeepSeek-R1-Distill-Llama-70B \
  --workers 8 \
  --limit 100 \
  --api_key no-key
```

## `scripts/run_gaokao_verify_prm_pipeline.sh`

- 用途：对高考 benchmark 的 4 组模型输出依次执行 `run_math_verify.py`、`run_qwen_prm.py`，最后执行 `export_verify_prm_metrics.py` 汇总 CSV/JSON。
- 执行目录：可在项目任意目录调用，脚本内部会切到 `scripts/`。
- 默认 benchmark：`gaokao_bench`。
- 默认模型：`DeepSeek-R1-Distill-Llama-70B`，脚本会自动转为文件名 slug，例如 `deepseek-r1-distill-llama-70b`。
- 默认 case：`tir_en cot_en cot_zh tir_zh`。
- 输入优先匹配：`../data/model_outputs/<benchmark>/model_outputs-<benchmark>_<日期>_<case>_<model_slug>.jsonl`。
- 同时兼容：`../data/model_outputs/<benchmark>/<case>_<model_slug>.jsonl`。
- 重要习惯：先跑 `--dry-run`，确认 input、verify、prm、report 路径后再正式执行。

预览命令：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh --dry-run
```

正式执行：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh
```

换模型：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh \
  --model Qwen2.5-Math-72B \
  --dry-run
```

跳过已有结果：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh --skip-existing
```

只跑指定 case：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh \
  --cases tir_en,cot_en \
  --dry-run
```

## `scripts/sync_rerun_records_to_mineru.py`

- 用途：把接口保存的模型输出 JSONL 中 `needs_model_rerun=True` 的样本同步回 MinerU 数据集。
- 匹配方式：按 `id` 对齐。
- 只覆盖字段：`question`、`reference_answer`、`needs_model_rerun`。
- MinerU 输入要求：顶层必须是 JSON 数组。
- 默认输出：把 `--mineru-input` 文件名中的日期替换成当天 `YYYYMMDD`，生成新的 `.mineru.json` 文件。
- 安全策略：默认不覆盖已有输出；写文件先写 `.tmp`，校验 JSON 后再替换；建议先使用 `--dry-run`。

预览同步：

```bash
python scripts/sync_rerun_records_to_mineru.py --dry-run
```

正式同步：

```bash
python scripts/sync_rerun_records_to_mineru.py
```

指定文件：

```bash
python scripts/sync_rerun_records_to_mineru.py \
  --model-output data/model_outputs/math_qa_275/model_outputs-math_qa_275_20260612_tir_en_qwen3-32b.jsonl \
  --mineru-input data/math_qa_275_20260612.mineru.json \
  --output data/math_qa_275_20260617.mineru.json \
  --dry-run
```

允许覆盖输出：

```bash
python scripts/sync_rerun_records_to_mineru.py --overwrite
```

允许 JSONL 中的部分更新 ID 在 MinerU 输入里不存在：

```bash
python scripts/sync_rerun_records_to_mineru.py --allow-missing --dry-run
```

# 代码风格规则

- 默认使用 Python 3.10+，脚本入口保持 `argparse` 风格。
- Bash 总控脚本使用 `set -euo pipefail`，路径和命令参数必须加引号。
- 涉及批处理、长耗时、会写输出文件的脚本，优先提供 `--dry-run`。
- 路径处理优先使用 `pathlib.Path`，文件读写默认 `utf-8`。
- 保持类型标注、简短函数、中文 docstring/注释风格，与现有脚本一致。
- 流水线脚本优先兼容断点续跑和现有字段命名，不随意变更输出结构。
- 新增命令或脚本时，优先补到已有子项目边界内，不引入额外流程分叉。

# 完成标准

- 变更能准确落在 `scripts/`、`tir_math/`、`math_render_service/`、`mineru_deal/` 之一，边界清晰。
- 涉及运行方式变更时，同步更新对应命令，尤其是 `tir_math` 的 `mode/lang/workers` 用法。
- 不读取或改写数据目录中的 `json/jsonl` 内容，除非任务明确要求处理数据。
- 文档保持短、直接、可执行，优先记录当前常用命令而不是完整教程。
