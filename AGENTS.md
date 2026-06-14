# 项目结构

- `scripts/`：主评测流水线脚本，覆盖标准化、模型推理、Math-Verify、PRM、报告生成。
- `tir_math/`：TIR/COT 推理子项目，包含并发推理脚本 `run_model_tir_inference_concurrent.py`。
- `math_render_service/`：FastAPI 子项目，用于处理和渲染 PRM `model_output`。
- `mineru_deal/`：MinerU 接入子项目，负责图片/PDF 解析为 markdown 文本。
- `data/`：输入数据、标准化结果、模型输出等中间目录。
- `results/`：验证结果、PRM 结果、报告输出目录。
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

# 代码风格规则

- 默认使用 Python 3.10+，脚本入口保持 `argparse` 风格。
- 路径处理优先使用 `pathlib.Path`，文件读写默认 `utf-8`。
- 保持类型标注、简短函数、中文 docstring/注释风格，与现有脚本一致。
- 流水线脚本优先兼容断点续跑和现有字段命名，不随意变更输出结构。
- 新增命令或脚本时，优先补到已有子项目边界内，不引入额外流程分叉。

# 完成标准

- 变更能准确落在 `scripts/`、`tir_math/`、`math_render_service/`、`mineru_deal/` 之一，边界清晰。
- 涉及运行方式变更时，同步更新对应命令，尤其是 `tir_math` 的 `mode/lang/workers` 用法。
- 不读取或改写数据目录中的 `json/jsonl` 内容，除非任务明确要求处理数据。
- 文档保持短、直接、可执行，优先记录当前常用命令而不是完整教程。
