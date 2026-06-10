"""A TIR(tool-integrated reasoning) math agent
```bash
# 默认启动（无推荐问题）
python tir_math.py

# 从 JSONL 文件加载 question 字段作为推荐问题
python tir_math.py --inputs data/model_outputs/model_outputs-tester-Qwen3-32B.jsonl

# 指定多个文件
python tir_math.py --inputs a.jsonl b.jsonl
```
"""

import argparse
import json
import os
from pathlib import Path
from pprint import pprint

from qwen_agent.agents import TIRMathAgent
from qwen_agent.gui import WebUI

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")

# We use the following two systems to distinguish between COT mode and TIR mode
TIR_SYSTEM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
COT_SYSTEM = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)

MODEL_NAME = "Qwen3-32B"

def init_agent_service():
    # Use this to access the qwen2.5-math model deployed on dashscope
    llm_cfg = {
        "model": MODEL_NAME,
        # "model_type": "qwen_dashscope",
        # "generate_cfg": {"top_k": 1},
        "model_server": "http://192.168.100.202:8200/v1",  # base_url，也称为 api_base
        "api_key": "EMPTY",
        # 'model': 'Qwen2.5-7B-Instruct',
        # 'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
        # 'api_key': 'EMPTY',
    }
    bot = TIRMathAgent(llm=llm_cfg, name=MODEL_NAME, system_message=TIR_SYSTEM)
    return bot


# def test(query: str = "斐波那契数列前10个数字"):
#     # Define the agent
#     bot = init_agent_service()

#     # Chat
#     messages = [{"role": "user", "content": query}]
#     for response in bot.run(messages):
#         pprint(response, indent=2)


def load_questions_from_jsonl(paths: list[str]) -> list[str]:
    """从 JSONL 文件列表中读取 question 字段，返回去重后的题目列表。"""
    questions: list[str] = []
    seen: set[str] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] 文件不存在: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = rec.get("question", "").strip()
                if q and q not in seen:
                    questions.append(q)
                    seen.add(q)
    return questions


def app_gui(suggestions: list[str] | None = None):
    bot = init_agent_service()
    chatbot_config = {
        "prompt.suggestions": suggestions or [],
    }
    WebUI(bot, chatbot_config=chatbot_config).run(server_name='0.0.0.0', server_port=8222)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIR 数学推理 Agent")
    parser.add_argument("--inputs", type=str, nargs="+", default=None, help="JSONL 文件路径，读取 question 字段作为推荐问题")
    args = parser.parse_args()

    suggestions = None
    if args.inputs:
        suggestions = load_questions_from_jsonl(args.inputs)
        print(f"从 {len(args.inputs)} 个文件中加载了 {len(suggestions)} 条推荐问题")

    # test()
    app_gui(suggestions=suggestions)
