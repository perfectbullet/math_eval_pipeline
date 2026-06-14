"""A TIR(tool-integrated reasoning) math agent
```bash
# 默认启动（无推荐问题）
python tir_math.py

# 从 JSONL 文件加载 question 字段作为推荐问题
python tir_math.py --inputs data/model_outputs/model_outputs-tester-Qwen3-32B.jsonl

# 指定多个文件
python tir_math.py --inputs a.jsonl b.jsonl

# 启用对话历史保存（每次对话自动追加到 JSONL 文件，重启后自动加载）
python tir_math.py --history chat_history.jsonl
```
"""

import argparse
import json
import os
import re
from pathlib import Path
from pprint import pprint

from qwen_agent.agents import TIRMathAgent
from qwen_agent.gui import WebUI

# 去掉 <think...>...</think > 标签
_THINK_TAG_RE = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL)

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")

# We use the following two systems to distinguish between COT mode and TIR mode
TIR_SYSTEM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
COT_SYSTEM = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)

# MODEL_NAME = "Qwen3-32B"
# MODEL_SERVER = "http://192.168.100.202:8200/v1"
# API_KEY = "no-key"

MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_SERVER = "https://api.siliconflow.cn/v1"
API_KEY = "sk-ujrcmopvnimyhuitcltefpgkaaaaffffkmelrtbhhxxnyhif"

# MODEL_NAME = "Qwen2-Math-72B-Instruct-W4A16"
# MODEL_SERVER = "http://192.168.100.202:8015/v1"
# API_KEY = "no-key"

# MODEL_NAME = "Qwen2.5-Math-72B-Instruct"
# MODEL_SERVER = "http://192.168.100.203:8200/v1"
# API_KEY = "no-key"


def init_agent_service(mode="tir"):
    # Use this to access the qwen2.5-math model deployed on dashscope
    llm_cfg = {
        "model": MODEL_NAME,
        # "model_type": "qwen_dashscope",
        # "generate_cfg": {"top_k": 1},
        # "model_server": "http://192.168.100.202:8200/v1",  # base_url，也称为 api_base
        "model_server": MODEL_SERVER,  # base_url，也称为 api_base
        "api_key": API_KEY,
        "generate_cfg": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 3000,
        },
    }
    system_msg = COT_SYSTEM if mode == "cot" else TIR_SYSTEM
    bot = TIRMathAgent(llm=llm_cfg, name=MODEL_NAME, system_message=system_msg)
    return bot


def test(query: str = "斐波那契数列前10个数字"):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [{"role": "user", "content": query}]
    for response in bot.run(messages):
        pprint(response, indent=2)


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


def app_gui(
    suggestions: list[str] | None = None,
    history_path: str | None = None,
    mode: str = "tir",
):
    bot = init_agent_service(mode=mode)
    chatbot_config = {
        "prompt.suggestions": suggestions or [],
    }

    if history_path:
        ui = PersistWebUI(bot, chatbot_config=chatbot_config, history_path=history_path)
    else:
        ui = WebUI(bot, chatbot_config=chatbot_config)

    ui.run(server_name="0.0.0.0", server_port=8222)


class PersistWebUI(WebUI):
    """支持对话历史持久化的 WebUI 子类。

    每次 agent 回复完成后自动追加到 JSONL 文件，重启时自动加载。
    """

    def __init__(self, *args, history_path: str = "chat_history.jsonl", **kwargs):
        self.history_path = Path(history_path)
        super().__init__(*args, **kwargs)

    def agent_run(self, _chatbot, _history, _agent_selector=None):
        """重写 agent_run：隔离推理 + 追加保存本轮问答。

        数学题之间不应互相污染，所以每次推理只传最后一道题给模型。
        但 _history 保留完整记录用于 Gradio 显示。
        """
        # 记录本轮开始前的消息数，用于推理完成后提取新增部分
        pre_len = len(_history)

        # 找到最后一个用户消息的位置
        last_user_idx = 0
        for i in range(len(_history) - 1, -1, -1):
            if _history[i].get("role") == "user":
                last_user_idx = i
                break

        # 保存历史前缀，截断 _history 使父类 agent_run 只看到最后一轮
        saved_prefix = _history[:last_user_idx]
        _history[:] = _history[last_user_idx:]

        # 调用父类（推理隔离：模型只看到当前这道题）
        gen = super().agent_run(_chatbot, _history, _agent_selector)
        for last in gen:
            yield last

        # 恢复完整历史（Gradio 显示需要）
        _history[:] = saved_prefix + _history[:]

        # 追加保存本轮问答（用户问题 + 模型回复）
        if _history and pre_len > 0:
            round_msgs = _history[max(0, pre_len - 1) :]
            self._append_round(round_msgs)

    def run(
        self,
        messages=None,
        share=False,
        server_name=None,
        server_port=None,
        concurrency_limit=10,
        enable_mention=False,
        **kwargs,
    ):
        """重写 run：启动时从 JSONL 加载历史。"""
        if messages is None and self.history_path.exists():
            messages = self._load_history()
            print(f"从 {self.history_path} 加载了 {len(messages)} 条历史消息")
        super().run(
            messages=messages,
            share=share,
            server_name=server_name,
            server_port=server_port,
            concurrency_limit=concurrency_limit,
            enable_mention=enable_mention,
            **kwargs,
        )

    def _append_round(self, messages: list):
        """追加保存一轮问答（用户问题 + 模型回复）到 JSONL 文件。

        每行是一个 JSON 数组，代表一轮完整的问答。
        多用户并发时互不覆盖——只追加，不回写已有数据。
        """
        cleaned = [m for m in (self._clean_msg(msg) for msg in messages) if m]
        if not cleaned:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        print(f"对话历史已追加到 {self.history_path}（本轮 {len(cleaned)} 条消息）")

    def _load_history(self) -> list:
        """从 JSONL 文件读取所有历史消息，展平为消息列表供 WebUI 显示。

        新格式：每行一个 JSON 数组（一轮问答）。
        旧格式兼容：每行一个 JSON 对象（单条消息）。
        """
        messages = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 新格式：JSON 数组（一轮问答）
                if isinstance(obj, list):
                    for msg in obj:
                        m = self._clean_msg(msg)
                        if m:
                            messages.append(m)
                # 旧格式兼容：单个 JSON 对象
                elif isinstance(obj, dict):
                    m = self._clean_msg(obj)
                    if m:
                        messages.append(m)
        return messages

    @staticmethod
    def _clean_msg(msg: dict) -> dict | None:
        """清洗单条消息：提取文本、去除 think 标签、只保留 role + content。"""
        role = msg.get("role", "")
        content = msg.get("content", "")
        # content 可能是列表 [{"text": "..."}, {"image": "..."}]，提取纯文本
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and "text" in p
            ]
            content = "\n".join(text_parts)
        if role and content:
            # 去掉 <think...>...</think > 思考过程，只保留最终输出
            content = _THINK_TAG_RE.sub("", content).strip()
            if content:
                return {"role": role, "content": content}
        return None


def main_webui():
    parser = argparse.ArgumentParser(description="TIR 数学推理 Agent")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=None,
        help="JSONL 文件路径，读取 question 字段作为推荐问题",
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="对话历史保存路径（如 chat_history.jsonl），启用后自动保存/加载",
    )
    parser.add_argument(
        "--mode",
        choices=["tir", "cot"],
        default="tir",
        help="推理模式：tir=工具集成推理（默认），cot=纯链式推理",
    )
    args = parser.parse_args()

    suggestions = None
    if args.inputs:
        suggestions = load_questions_from_jsonl(args.inputs)
        print(f"从 {len(args.inputs)} 个文件中加载了 {len(suggestions)} 条推荐问题")

    print(f"推理模式: {args.mode.upper()}")
    app_gui(suggestions=suggestions, history_path=args.history, mode=args.mode)


if __name__ == "__main__":
    main_webui()
    # test()
