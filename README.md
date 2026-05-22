# 数学评测流水线

标准化 → 模型推理 → Math-Verify 答案验证 → PRM 过程评分 → 归因报告。

## 目录结构

```
math_eval_pipeline/
├── scripts/                          # 所有脚本
│   ├── standardize_datasets.py       #   数据标准化
│   ├── run_model_inference.py        #   模型推理（Ollama）
│   ├── build_dummy_model_outputs.py  #   构建 dummy 输出（测试用）
│   ├── run_math_verify.py            #   Math-Verify 答案验证
│   ├── run_deepseek_prm.py           #   PRM 过程评分
│   └── build_error_report.py         #   综合归因报告
├── data/
│   ├── raw                           # → 软链到 math_dataset_original（只读）
│   ├── standardized/                 # 标准化后的 JSONL
│   └── model_outputs/                # 模型推理输出
├── results/
│   ├── verify/                       # Math-Verify 结果 + 汇总
│   ├── prm/                          # PRM 逐步骤评分 + 汇总
│   └── reports/                      # 最终归因报告
├── external/                         # 外部工具（Math-Verify、PRM 模型）
├── logs/                             # 运行日志
├── requirements.txt
└── README.md
```

## 环境准备

```bash
conda create -n math-eval python=3.10 -y
conda activate math-eval
pip install -r requirements.txt
pip install requests tqdm   # 推理脚本额外依赖
```

## 快速开始

所有命令在 `math_eval_pipeline/` 目录下执行：

```bash
cd ~/math_model_deployment/math_eval_pipeline
```

### 第一步：标准化数据

```bash
python scripts/standardize_datasets.py \
  --input_dir data/raw \
  --output_dir data/standardized \
  --limit 1000
```

验收：
```bash
wc -l data/standardized/*.jsonl
cat data/standardized/dataset_summary.md
```

### 第二步：模型推理

```bash
python scripts/run_model_inference.py \
  --input data/standardized/train_or_eval_all.jsonl \
  --output data/model_outputs/model_outputs.jsonl \
  --api_base http://192.168.8.231:11434/v1 \
  --model qwen2.5-math-Q6_K_L:7b \
  --limit 100
```

支持 `--skip_existing` 断点续跑。

### 第三步：Math-Verify 验证

```bash
python scripts/run_math_verify.py \
  --input data/model_outputs/model_outputs.jsonl \
  --output results/verify/verify_results.jsonl \
  --summary results/verify/verify_summary.md
```

### 第四步：PRM 过程评分（仅错误样本，需要 GPU）

```bash
python scripts/run_deepseek_prm.py \
  --input results/verify/verify_results.jsonl \
  --output results/prm/prm_step_scores.jsonl \
  --summary results/prm/prm_summary.md \
  --model mukaj/deepseek-math-7b-rl-prm-v0.1 \
  --limit 50
```

### 第五步：归因报告

```bash
python scripts/build_error_report.py \
  --verify results/verify/verify_results.jsonl \
  --prm results/prm/prm_step_scores.jsonl \
  --output_md results/reports/error_attribution_report.md \
  --output_jsonl results/reports/error_cases.jsonl
```

## 归因逻辑

```
Math-Verify 正确 → pass
Math-Verify 错误 + PRM 低分 → logic_error（解题逻辑问题）
Math-Verify 错误 + PRM 高分 → answer_or_render_suspect（答案抽取/渲染问题）
Math-Verify 错误 + 无法提取答案 → answer_extract_error
Math-Verify 错误 + 无 PRM 结果 → uncertain
```

## 外部工具安装

### Math-Verify

```bash
pip install math-verify
# 或从源码：
cd external && git clone https://github.com/huggingface/Math-Verify.git && cd Math-Verify && pip install -e .
```

### DeepSeek-Math PRM

```bash
# 需要 GPU + torch，模型自动从 HuggingFace 下载
# 7B FP16 约 14GB 显存，8bit 约 8GB
```
