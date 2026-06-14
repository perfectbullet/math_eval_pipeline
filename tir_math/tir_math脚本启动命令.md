# 从 JSONL 文件加载 question 作为推荐问题
python tir_math/tir_math.py --inputs data/model_outputs/model_outputs-tester-Qwen3-32B.jsonl

# 多个文件
python tir_math/tir_math.py \
  --inputs data/model_outputs/a.jsonl data/model_outputs/b.jsonl

# 不指定则无推荐问题
python tir_math/tir_math.py



# TIR 模式（默认，工具集成推理）
python tir_math.py

# COT 模式（纯链式推理，不调用工具）
python tir_math.py --mode cot

# 组合使用
python tir_math.py --mode cot --inputs data/xxx.jsonl --history chat_history.jsonl


# 组合使用
python tir_math.py --input ../data/math_questions_31.jsonl --history chat_history_Qwen2-Math-72B-Instruct-W4A16.jsonl


# TIR 模式（默认，工具集成推理）
python tir_math.py

# COT 模式（纯链式推理，不调用工具）
python tir_math.py --mode cot

# 组合使用
python tir_math.py --mode cot --inputs data/xxx.jsonl --history chat_history.jsonl




脚本已创建：tir_math/run_model_tir_inference.py

核心逻辑：

增量保存：启动时读取 output 文件中已有成功 ID，自动跳过；每条推理后立即 flush() 追加写入
TIR 推理：用 TIRMathAgent 初始化 agent，bot.run(messages) 生成器迭代取最后一条 assistant 响应
4 个必传参数：--input、--output、--api_base、--model，加可选 --limit
运行方式：

cd tir_math/
python run_model_tir_inference.py \
  --input ../data/math_questions_31.jsonl \
  --output ../data/model_outputs/model_outputs-math_questions_31-Qwen2.5-Math-72B-Instruct.jsonl \
  --api_base http://192.168.100.203:8200/v1 \
  --model Qwen2.5-Math-72B-Instruct --limit 2








并发版脚本已创建：tir_math/run_model_tir_inference_concurrent.py

与串行版的区别：

--workers 4 参数控制并发数（默认 4）
每个 worker 独立创建 TIRMathAgent（避免线程安全问题）
threading.Lock 保护文件写入，避免多线程输出交错
ThreadPoolExecutor + as_completed 调度，tqdm 进度条照常

cd tir_math/
python run_model_tir_inference_concurrent.py \
  --input ../data/math_questions_31.jsonl \
  --output ../data/model_outputs/model_outputs-concurrent-Qwen3-32B-TIR-EN.jsonl \
  --api_base http://192.168.100.202:8200/v1 \
  --model Qwen3-32B \
  --workers 4 --limit 4


--mode tir|cot（默认 tir）— TIR 工具集成推理 vs COT 纯链式推理
--lang zh|en（默认 zh）— 中文/英文系统提示词
--api_key sk-xxxxx
四种组合自动匹配：

--mode	--lang	提示词
tir	zh	请结合自然语言推理和程序来解决...请全程使用中文作答
tir	en	Please integrate natural language reasoning with programs...
cot	zh	请逐步推理...请全程使用中文作答
cot	en	Please reason step by step...
示例：


python run_model_tir_inference_concurrent.py \
  --input ../data/math_questions_31.jsonl \
  --output ../data/model_outputs/model_outputs-concurrent-Qwen3-32B-TIR-EN-0612.jsonl \
  --api_base http://192.168.100.202:8200/v1 \
  --model Qwen3-32B \
  --mode cot --lang en --workers 4



python run_model_tir_inference_concurrent.py \
  --input ../data/math_qa_275_20260612.json \
  --output ../data/model_outputs/math_qa_275_20260612.jsonl \
  --api_base http://192.168.100.202:8200/v1 \
  --model Qwen3-32B \
  --limit 10 \
  --mode cot --lang en --workers 4