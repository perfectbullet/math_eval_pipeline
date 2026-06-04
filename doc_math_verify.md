# Math-Verify 在本项目中的使用

## Math-Verify 是什么

[HuggingFace Math-Verify](https://github.com/huggingface/Math-Verify) 是一个数学表达式评测库，专门用于评估 LLM 在数学任务上的输出。它的核心能力是**从自由文本中抽取数学答案，并判断两个数学表达式是否等价**。

在 MATH 数据集的评测中，Math-Verify 的准确率高于其他评测工具（Harness 0.0802、Qwen 0.1288、Math-Verify 0.1328），原因是它不要求模型输出严格遵循特定格式，能容忍各种 LaTeX 写法变体。

### 核心问题

评测数学模型时，最困难的部分不是"答案对不对"，而是"怎么判断两个答案等价"。例如：

| 标准答案 | 模型输出 | 是否等价 |
|----------|----------|----------|
| `\frac{1}{2}` | `0.5` | 是 |
| `\boxed{100}` | `100.0` | 是 |
| `{1,2,3,4}` | `{1,3} \cup {2,4}` | 是 |
| `x < 2` | `2 > x` | 是 |

如果用简单的字符串匹配，这些都会被判错，严重低估模型真实水平。

### 三个核心 API

```python
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig

# parse() —— 从文本中抽取数学表达式
# verify() —— 判断两组表达式是否等价
```

**parse()** 支持三种抽取配置：

| 配置 | 用途 | 示例 |
|------|------|------|
| `LatexExtractionConfig` | LaTeX 格式 | `\frac{1}{2}`、`\sqrt{2}`、`\boxed{100}` |
| `ExprExtractionConfig` | 纯数学表达式 | `1/2`、`x+1` |
| `StringExtractionConfig` | 字面字符串（选择题） | `A`、`B`、`C`、`D` |

**verify()** 内置多种等价判断策略：数值比较（含精度控制）、符号化简、集合/区间比较、矩阵逐元素比较、不等式翻转等。

## 在本项目中的集成

### 在流水线中的位置

```
模型推理输出 (model_outputs.jsonl)
        ↓
  run_math_verify.py    ← Math-Verify 在这里工作
        ↓
验证结果 (verify_results.jsonl)
        ↓
  PRM 过程评分 → 归因报告
```

Math-Verify 负责流水线的**第三步**：判断模型输出与标准答案是否等价。

### 验证策略

[run_math_verify.py](scripts/run_math_verify.py) 采用**两级策略**：

**第一级：math-verify 端到端验证**（优先）

直接把标准答案和完整的模型原始输出分别传给 `parse()`，由 math-verify 同时完成答案抽取和等价比较：

```python
# 选择题
configs = [StringExtractionConfig(strings=("A","B","C","D","E","F")), LatexExtractionConfig()]

# 表达式
configs = [LatexExtractionConfig(), ExprExtractionConfig()]

gold_parsed = parse(reference_answer, extraction_config=configs)
pred_parsed = parse(model_output, extraction_config=configs)
is_correct = verify(gold_parsed, pred_parsed)
```

这样做的优势是：
- 不需要自己写正则去从模型输出里抠答案，math-verify 能自动识别 `\\boxed{}`、`$$...$$`、`\(...\)` 等各种 LaTeX 环境
- 选择题通过 `StringExtractionConfig` 直接抽取选项字母，准确率更高

**第二级：手工降级**（math-verify 失败时）

当 math-verify 无法解析（如含变量的表达式 `x^2+1`）或抛异常时，自动降级：

```
1. 手工正则抽取答案（\\boxed{}、<final_answer>、中文关键词）
2. 数值比较（容差 1e-3）
3. 字符串比较（去空格/符号后匹配）
```

### 示例：完整调用流程

```
输入: reference_answer="A", model_output="经过分析，本题答案为 A\n最终答案：\boxed{A}"
      answer_type="choice"

→ 第一级: math-verify
  parse("A", configs=[StringExtractionConfig, LatexExtractionConfig]) → ["A"]
  parse("经过分析...\boxed{A}", configs=[...]) → ["A"]
  verify(["A"], ["A"]) → True

输出: (True, "A", "")
```

```
输入: reference_answer="\\frac{1}{2}", model_output="计算得 \\boxed{0.5}"
      answer_type="expr"

→ 第一级: math-verify
  parse("\\frac{1}{2}") → [1/2]
  parse("计算得 \\boxed{0.5}") → [0.5]
  verify([1/2], [0.5]) → True

输出: (True, "1/2", "")
```

## 安装

```bash
# 从本项目 external/ 目录本地安装
pip install -e external/Math-Verify
```

依赖 `latex2sympy2_extended` 和 `antlr4-python3-runtime`，会自动安装。
