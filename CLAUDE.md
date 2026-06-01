# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言偏好

始终使用中文进行交流和回复。

## 概述

数学评测流水线，五步顺序执行：标准化 → 模型推理 → Math-Verify 答案验证 → PRM 过程评分 → 归因报告。每个步骤对应 `scripts/` 下的一个独立脚本，通过 JSONL 文件串联数据。

## 环境准备

```bash
conda create -n math-eval python=3.10 -y && conda activate math-eval
pip install -r requirements.txt
pip install requests tqdm   # run_model_inference.py 额外依赖
```

PRM 步骤需要 GPU + torch，7B FP16 约 14GB 显存，8bit 约 8GB。

## 执行命令

所有命令在 `math_eval_pipeline/` 目录下执行。每步输入是上一步的输出。

```bash
# 1. 标准化（data/raw 是到 ../../math_dataset_original 的软链）
python scripts/standardize_datasets.py \
  --input_dir data/raw --output_dir data/standardized --limit 1000

# 2. 模型推理（调用 Ollama/OpenAI 兼容 API，支持 --skip_existing 断点续跑）
python scripts/run_model_inference.py \
  --input data/standardized/train_or_eval_all.jsonl \
  --output data/model_outputs/model_outputs.jsonl \
  --api_base http://192.168.8.231:11434/v1 \
  --model qwen2.5-math-Q6_K_L:7b --limit 100

# 3. 答案验证
python scripts/run_math_verify.py \
  --input data/model_outputs/model_outputs.jsonl \
  --output results/verify/verify_results.jsonl \
  --summary results/verify/verify_summary.md

# 4a. PRM 过程评分 - DeepSeek（逐步增量评分）
python scripts/run_deepseek_prm.py \
  --input results/verify/verify_results.jsonl \
  --output results/prm/prm_step_scores.jsonl \
  --summary results/prm/prm_summary.md --limit 50

# 4b. PRM 过程评分 - Qwen（单次前向传播，更高效）
python scripts/run_qwen_prm.py \
  --input results/verify/verify_results.jsonl \
  --output results/prm/prm_step_scores.jsonl \
  --summary results/prm/prm_summary.md --limit 50

# 5. 归因报告
python scripts/build_error_report.py \
  --verify results/verify/verify_results.jsonl \
  --prm results/prm/prm_step_scores.jsonl \
  --output_md results/reports/error_attribution_report.md \
  --output_jsonl results/reports/error_cases.jsonl

# 测试用：用标准答案构造 dummy 模型输出，验证流水线本身
python scripts/build_dummy_model_outputs.py \
  --dataset data/standardized/train_or_eval_all.jsonl \
  --output data/model_outputs/model_outputs_dummy.jsonl --limit 100
```

## 数据流

```
data/raw/ (软链 → math_dataset_original)
  ↓ standardize_datasets.py
data/standardized/{math23k,gaokao_bench,linkwise_cot}.jsonl + train_or_eval_all.jsonl
  ↓ run_model_inference.py
data/model_outputs/model_outputs.jsonl
  ↓ run_math_verify.py
results/verify/verify_results.jsonl + verify_summary.md
  ↓ run_deepseek_prm.py / run_qwen_prm.py (仅错误样本)
results/prm/prm_step_scores.jsonl + prm_summary.md
  ↓ build_error_report.py
results/reports/error_attribution_report.md + error_cases.jsonl
```

## 架构要点

**数据格式**：所有中间产物为 JSONL，每行一个 JSON 对象。标准化后的记录字段：`id`, `source`, `subset`, `question`, `reference_answer`, `reference_solution`, `answer_type`（`expr` / `choice`）, `metadata`。

**支持的数据集**：Math23K（方程类）、GAOKAO-Bench（高考数学，含选择/填空/解答）、LinkWiseCoTDataset（含推理链）。`standardize_datasets.py` 用 `_parse_json_variants()` 处理三种 JSON 格式变体（数组、JSONL、拼接对象）。

**答案验证**（`run_math_verify.py`）：优先用 `math-verify` 库的 LaTeX/表达式解析，不可用时降级为数值比较或字符串匹配。选择题走独立的选项字母提取逻辑。

**PRM 评分**（`run_deepseek_prm.py`）：`PRMScorer` 类加载 DeepSeek-Math PRM 模型，`split_steps()` 按步骤标记/换行/句号三级降级切分解题过程，逐步骤增量评分（0~1），通过 softmax 归一化 yes/no token。

**PRM 评分 - Qwen**（`run_qwen_prm.py`）：`QwenPRMScorer` 类加载 Qwen2.5-Math-PRM-7B 模型（`AutoModel` + `trust_remote_code`），用 `<extra_0>` 特殊 token 分隔步骤，**单次前向传播**获取所有步骤分数。通过 `make_step_rewards()` 在 `<extra_0>` 位置提取 softmax 正类概率（0~1）。比 DeepSeek PRM 更高效（一次前向 vs N 次增量）。支持 `--torch_dtype auto`（自动根据 GPU 选择 bf16/fp16）。

**归因逻辑**（`build_error_report.py`）：`verify_correct=true → pass`；`extracted_prediction 为空 → answer_extract_error`；有 PRM 结果时按 `min_step_score < 0.45 → logic_error`，`avg_score >= 0.70 → answer_or_render_suspect`；无 PRM → `uncertain`。

**推理脚本**（`run_model_inference.py`）：调用 Ollama 的 OpenAI 兼容 API，通过 `MATH_SYSTEM_PROMPT` 要求模型输出 `\boxed{}` 格式答案。支持 `--skip_existing` 断点续跑。
