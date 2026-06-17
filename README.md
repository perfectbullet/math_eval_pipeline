# math_eval_pipeline

这是一个数学模型评测流水线项目，主要用于对数学问答模型进行推理、答案校验、PRM 过程评分和指标汇总。

## 主要能力

- 数据整理：把 Excel、MinerU OCR 结果等数据整理成评测输入。
- 模型推理：支持普通 COT 推理，也支持 TIR 模式推理。
- Math-Verify 校验：对模型最终答案做自动正确性校验。
- PRM 评分：使用 `Qwen2.5-Math-PRM-7B` 对错误样本做步骤级评分。
- 报告汇总：导出 verify + PRM 指标到 CSV/JSON。
- 人工修正回写：把接口里修正过、需要重跑的样本同步回 MinerU 数据集。

## 目录说明

```text
scripts/              主评测脚本、校验脚本、汇总脚本
tir_math/             COT/TIR 推理脚本
math_render_service/  PRM 结果查看和人工修正接口
mineru_deal/          MinerU OCR 接入脚本
data/                 输入数据、标准化数据、模型输出
results/              verify 和 PRM 中间结果
reports/              汇总报告输出
```

## 常用流程

### 1. 四模式推理

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

查看后台任务：

```bash
bash run_all_mode_lang_inference.sh status
```

停止后台任务：

```bash
bash run_all_mode_lang_inference.sh stop
```

### 2. 高考数据集 verify + PRM 汇总

先预览命令，不真正执行：

```bash
cd scripts
bash run_gaokao_verify_prm_pipeline.sh --dry-run
```

确认路径无误后正式执行：

```bash
bash run_gaokao_verify_prm_pipeline.sh
```

换模型时只需要改 `--model`：

```bash
bash run_gaokao_verify_prm_pipeline.sh \
  --model Qwen2.5-Math-72B \
  --dry-run
```

### 3. 同步人工修正样本到 MinerU 数据集

先预览：

```bash
python scripts/sync_rerun_records_to_mineru.py --dry-run
```

正式生成新数据集：

```bash
python scripts/sync_rerun_records_to_mineru.py
```

脚本只会同步 `needs_model_rerun=True` 的样本，并且只覆盖：

```text
question
reference_answer
needs_model_rerun
```

## 使用习惯

涉及长耗时任务、批量写文件、PRM 显存任务时，先执行 `--dry-run` 检查路径和命令。确认无误后再正式运行。

## 更多说明

开发和维护约定见：

```text
AGENTS.md
```
