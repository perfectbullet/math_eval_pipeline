#!/usr/bin/env python3
"""数学模型 / 数学 Agent 工厂。

这个模块提供两类能力：
1. 动态创建 OpenAI 兼容的数学 LLM（ChatOpenAI）
2. 动态创建数学 Agent：
   - cot -> Assistant(function_list=[])
   - tir -> TIRMathAgent

同时提供一个轻量适配器，把 Qwen-Agent 的全量流式输出转换为更接近
OpenAI delta 风格的增量文本流，方便上层服务统一消费。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from qwen_agent.agents import Assistant, TIRMathAgent


TIR_SYSTEM_EN = (
    "Please integrate natural language reasoning with programs to solve "
    "the problem above, and put your final answer within \\boxed{}."
)
COT_SYSTEM_EN = "Please reason step by step, and put your final answer within \\boxed{}."

TIR_SYSTEM_ZH = "请结合自然语言推理和程序来解决上述问题，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"
COT_SYSTEM_ZH = "请逐步推理，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"

SYSTEM_PROMPTS = {
    ("tir", "zh"): TIR_SYSTEM_ZH,
    ("tir", "en"): TIR_SYSTEM_EN,
    ("cot", "zh"): COT_SYSTEM_ZH,
    ("cot", "en"): COT_SYSTEM_EN,
}


@dataclass
class MathModelConfig:
    base_url: str
    model: str
    api_key: str = "dummy-key"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 10240
    streaming: bool = True


class MathAgentFactory:
    """动态创建数学 LLM / Agent。

    用法示例：

    ```python
    cfg = MathModelConfig(
        base_url="http://127.0.0.1:8200/v1",
        model="Qwen3-32B",
    )

    factory = MathAgentFactory(cfg)

    # 1. 直接拿 OpenAI 兼容数学模型
    math_llm = factory.create_chat_openai()

    # 2. 拿数学 agent
    bot = factory.create_agent(mode="cot", lang="en")

    # 3. 以近似 OpenAI delta 的方式消费 agent 输出
    for delta in factory.stream_agent_text(question="1+1=?", mode="cot", lang="en"):
        print(delta, end="")
    ```
    """

    def __init__(self, config: MathModelConfig):
        self.config = config

    def create_chat_openai(self) -> ChatOpenAI:
        """动态创建 OpenAI 兼容的数学模型实例。"""
        return ChatOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
            top_p=self.config.top_p,
        )

    def create_qwen_agent_llm_cfg(self) -> dict:
        """创建给 Qwen-Agent 使用的 llm 配置。"""
        return {
            "model": self.config.model,
            "model_server": self.config.base_url,
            "api_key": self.config.api_key,
            "generate_cfg": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
        }

    def create_agent(self, mode: str = "cot", lang: str = "zh"):
        """动态创建数学 agent。"""
        if mode not in {"cot", "tir"}:
            raise ValueError("mode must be 'cot' or 'tir'")
        if lang not in {"zh", "en"}:
            raise ValueError("lang must be 'zh' or 'en'")

        llm_cfg = self.create_qwen_agent_llm_cfg()
        system_message = SYSTEM_PROMPTS[(mode, lang)]

        if mode == "cot":
            return Assistant(
                llm=llm_cfg,
                name=self.config.model,
                system_message=system_message,
                function_list=[],
            )

        return TIRMathAgent(
            llm=llm_cfg,
            name=self.config.model,
            system_message=system_message,
        )

    def create_streaming_math_interface(self, mode: str = "cot", lang: str = "zh"):
        """创建兼容 `astream/ainvoke` 的数学接口对象。

        - `mode="llm"`：直接返回 ChatOpenAI
        - `mode="cot"` / `mode="tir"`：返回 Qwen-Agent 适配器
        """
        if mode == "llm":
            return self.create_chat_openai()
        return MathAgentStreamingAdapter(factory=self, mode=mode, lang=lang)

    def run_agent_full_text(self, question: str, mode: str = "cot", lang: str = "zh") -> str:
        """返回 agent 的最终完整文本。

        Qwen-Agent 的 bot.run() 默认是全量流式，每轮都会返回累计后的文本。
        因此这里直接取最后一轮 assistant 的 content 作为最终结果。
        """
        bot = self.create_agent(mode=mode, lang=lang)
        messages = [{"role": "user", "content": question}]
        last_response = None

        for response in bot.run(messages):
            last_response = response

        return self._extract_full_text(last_response)

    def stream_agent_text(self, question: str, mode: str = "cot", lang: str = "zh") -> Iterator[str]:
        """把 agent 的全量流式输出转成增量文本流。

        这不是直接拿到底层 OpenAI chunk，而是对 Qwen-Agent 的累计文本做 diff：
        - bot.run() 第 N 轮返回的是“到目前为止的完整文本”
        - 这里改为只 yield 本轮新增部分

        这样更适合接到依赖 OpenAI 风格流式输出的上层服务里。
        """
        bot = self.create_agent(mode=mode, lang=lang)
        messages = [{"role": "user", "content": question}]
        previous_text = ""

        for response in bot.run(messages):
            full_text = self._extract_full_text(response)
            if not full_text:
                continue

            if full_text.startswith(previous_text):
                delta = full_text[len(previous_text):]
            else:
                # 如果 agent 输出被重写/截断，保守退回整段
                delta = full_text

            previous_text = full_text
            if delta:
                yield delta

    @staticmethod
    def _extract_full_text(response) -> str:
        """从 Qwen-Agent 的 response 中提取 assistant 完整文本。"""
        if not response:
            return ""

        if isinstance(response, list):
            for msg in reversed(response):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                    return "\n".join(parts)
                return content or ""

        if isinstance(response, dict):
            return response.get("content", "") or ""

        return ""


class MathAgentStreamingAdapter:
    """把 Qwen-Agent 适配成类似 LangChain ChatModel 的接口。

    目标是兼容这种调用方式：

    ```python
    async for chunk in math_llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
    ```
    """

    def __init__(self, factory: MathAgentFactory, mode: str = "cot", lang: str = "zh"):
        if mode not in {"cot", "tir"}:
            raise ValueError("mode must be 'cot' or 'tir'")
        if lang not in {"zh", "en"}:
            raise ValueError("lang must be 'zh' or 'en'")
        self.factory = factory
        self.mode = mode
        self.lang = lang
        self.model_name = factory.config.model
        self.base_url = factory.config.base_url
        # 尽量兼容上层现有日志字段读取
        self.openai_api_base = factory.config.base_url

    async def astream(self, messages: list[Any]) -> AsyncIterator[AIMessageChunk]:
        """异步增量流式输出。

        内部把同步的 `bot.run()` 放到后台线程，并把增量文本推入 asyncio.Queue。
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

        def worker():
            previous_text = ""
            try:
                bot = self.factory.create_agent(mode=self.mode, lang=self.lang)
                agent_messages = self._to_agent_messages(messages)
                for response in bot.run(agent_messages):
                    full_text = self.factory._extract_full_text(response)
                    if not full_text:
                        continue
                    if full_text.startswith(previous_text):
                        delta = full_text[len(previous_text):]
                    else:
                        delta = full_text
                    previous_text = full_text
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", delta))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:  # pragma: no cover
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event, payload = await queue.get()
            if event == "chunk":
                yield AIMessageChunk(content=payload or "")
            elif event == "error":
                raise RuntimeError(payload or "math agent stream failed")
            elif event == "done":
                break

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        """异步一次性返回最终完整文本。"""
        text = await asyncio.to_thread(self._invoke_sync, messages)
        return AIMessage(content=text)

    def _invoke_sync(self, messages: list[Any]) -> str:
        bot = self.factory.create_agent(mode=self.mode, lang=self.lang)
        agent_messages = self._to_agent_messages(messages)
        last_response = None
        for response in bot.run(agent_messages):
            last_response = response
        return self.factory._extract_full_text(last_response)

    @staticmethod
    def _to_agent_messages(messages: list[Any]) -> list[dict]:
        """把 LangChain / OpenAI 风格消息转成 Qwen-Agent dict 消息。"""
        converted: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif isinstance(msg, HumanMessage):
                role = "user"
                content = msg.content
            elif isinstance(msg, SystemMessage):
                role = "system"
                content = msg.content
            elif isinstance(msg, BaseMessage):
                role = getattr(msg, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = msg.content
            else:
                role = "user"
                content = str(msg)
            converted.append({"role": role, "content": content})
        return converted
