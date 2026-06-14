"""MathLLMClient — 封装 OpenAI 兼容 API 的流式推理客户端。

支持 Qwen3（reasoning_content 字段）和 Confucius3 等模型（<thinktelltale> 标签）。
每次推理自动统计 token 数、首 token 延迟、总耗时、think 长度/耗时，并保存 JSON + Markdown。

用法:
    client = MathLLMClient("Qwen3-32B-AWQ", "http://192.168.100.201:8200/v1")
    result = client.ask("求解方程 x^2 - 3x + 2 = 0")
    print(result["full_content"])
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

DEFAULT_SYSTEM_PROMPT = "请逐步推理，并将最终答案放在 \\boxed{} 内。"


class MathLLMClient:
    """数学推理 LLM 客户端，支持流式推理 + 统计 + 自动保存。"""

    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def ask(
        self,
        question: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.6,
        max_tokens: int = 30720,
        timeout: int = 600,
        stream: bool = True,
        stream_options: dict | None = None,
        output_dir: str = "output/markdown",
        save: bool = True,
    ) -> dict:
        """执行推理，返回结果 dict。

        Returns:
            {
                "question": str,
                "full_content": str,       # 正式回复
                "think_content": str,      # 推理过程
                "total_tokens": int,
                "first_token_latency": float,  # 秒
                "total_time": float,           # 秒
                "think_time": float,           # 秒
                "speed": float,                # tokens/s
                "all_deltas": list[dict],
            }
        """
        if stream_options is None:
            stream_options = {"include_usage": True}

        if stream:
            result = self._stream_and_collect(
                question, system_prompt, temperature, max_tokens,
                timeout, stream_options,
            )
        else:
            result = self._non_stream(
                question, system_prompt, temperature, max_tokens, timeout,
            )

        result["question"] = question
        result["model_name"] = self.model_name
        result["base_url"] = self.base_url

        if save:
            self._save(result, output_dir)

        return result

    # ── 流式推理 ──

    def _stream_and_collect(
        self, question, system_prompt, temperature, max_tokens, timeout, stream_options,
    ) -> dict:
        t_start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options=stream_options,
            timeout=timeout,
        )

        full_content = ""
        think_raw = ""
        first_token_time = None
        think_start_time = None
        think_end_time = None
        total_tokens = 0
        all_deltas = []
        in_think_tag = False

        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            now = time.time()

            # 记录原始 delta
            if hasattr(delta, "model_dump"):
                delta_info = delta.model_dump()
            else:
                delta_info = {}
                for attr in ("content", "reasoning_content", "role", "tool_calls", "function_call"):
                    val = getattr(delta, attr, None)
                    if val is not None:
                        delta_info[attr] = val
            if delta_info:
                all_deltas.append(delta_info)

            # 方式1: reasoning_content（Qwen3）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if first_token_time is None:
                    first_token_time = now
                if think_start_time is None:
                    think_start_time = now
                think_raw += delta.reasoning_content
                print(delta.reasoning_content, end="", flush=True)

            # 方式2: <thinktelltale> 标签（Confucius3 等）+ 正式内容
            if delta.content:
                if first_token_time is None:
                    first_token_time = now

                content = delta.content

                if "<think" in content and not in_think_tag:
                    in_think_tag = True
                    if think_start_time is None:
                        think_start_time = now
                    content = re.sub(r"<think[^>]*>", "", content)

                if "</think" in content and in_think_tag:
                    in_think_tag = False
                    think_end_time = now
                    content = re.sub(r"</think[^>]*>", "", content)

                if in_think_tag:
                    think_raw += content
                    print(content, end="", flush=True)
                else:
                    if think_end_time is None and think_raw:
                        think_end_time = now
                    full_content += content
                    print(content, end="", flush=True)

            # usage
            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = chunk.usage.total_tokens

        t_end = time.time()
        print("\n")

        # 计算指标
        total_time = t_end - t_start
        first_token_latency = (first_token_time - t_start) if first_token_time else 0
        if think_start_time and think_end_time:
            think_time = think_end_time - think_start_time
        elif think_start_time and think_raw:
            think_time = t_end - think_start_time
        else:
            think_time = 0
        gen_time = total_time - first_token_latency
        # vLLM 可能不返回 usage，用内容长度粗估
        if total_tokens == 0 and (full_content or think_raw):
            total_tokens = int(len(full_content + think_raw) * 0.6)
        speed = total_tokens / gen_time if gen_time > 0 else 0

        return {
            "full_content": full_content,
            "think_content": think_raw,
            "total_tokens": total_tokens,
            "first_token_latency": first_token_latency,
            "total_time": total_time,
            "think_time": think_time,
            "speed": speed,
            "all_deltas": all_deltas,
        }

    # ── 非流式推理 ──

    def _non_stream(
        self, question, system_prompt, temperature, max_tokens, timeout,
    ) -> dict:
        t_start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            timeout=timeout,
        )
        t_end = time.time()

        choice = resp.choices[0]
        full_content = choice.message.content or ""
        total_tokens = resp.usage.total_tokens if resp.usage else 0
        total_time = t_end - t_start

        return {
            "full_content": full_content,
            "think_content": "",
            "total_tokens": total_tokens,
            "first_token_latency": 0,
            "total_time": total_time,
            "think_time": 0,
            "speed": total_tokens / total_time if total_time > 0 else 0,
            "all_deltas": [],
        }

    # ── 保存 ──

    def _save(self, result: dict, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_data = {
            "model_name": result["model_name"],
            "base_url": result["base_url"],
            "question": result["question"],
            "full_content": result["full_content"],
            "think_content": result["think_content"],
            "total_tokens": result["total_tokens"],
            "first_token_latency": result["first_token_latency"],
            "total_time": result["total_time"],
            "think_time": result["think_time"],
            "speed": result["speed"],
            "all_deltas": result.get("all_deltas", []),
        }
        json_path = out / f"{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # Markdown
        md_path = out / f"{ts}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"- 模型: {result['model_name']} @ {result['base_url']}\n\n")
            f.write(f"## Question\n\n{result['question']}\n\n")
            think = result["think_content"]
            if think:
                f.write(f"<details><summary>Think ({len(think)} chars, {result['think_time']:.2f}s)</summary>\n\n")
                f.write(think)
                f.write("\n\n</details>\n\n")
            f.write(f"## Answer\n\n{result['full_content']}\n")
            f.write("\n---\n")
            f.write(f"- 总 token: {result['total_tokens']}\n")
            f.write(f"- 首 token 延迟: {result['first_token_latency']:.2f}s\n")
            f.write(f"- 总耗时: {result['total_time']:.2f}s\n")
            f.write(f"- 推理速度: {result['speed']:.1f} tokens/s\n")
            f.write(f"- think 长度: {len(think)} chars\n")
            f.write(f"- think 耗时: {result['think_time']:.2f}s\n")

        print(f"已保存到 {json_path} / {md_path}")

    # ── 打印统计 ──

    @staticmethod
    def print_stats(result: dict):
        print("=== 统计 ===")
        print(f"模型:              {result['model_name']} @ {result['base_url']}")
        print(f"总 token 数:       {result['total_tokens']}")
        print(f"首 token 延迟:     {result['first_token_latency']:.2f}s")
        print(f"总耗时:            {result['total_time']:.2f}s")
        print(f"推理速度:          {result['speed']:.1f} tokens/s")
        print(f"think 长度:        {len(result['think_content'])} 字符")
        print(f"think 耗时:        {result['think_time']:.2f}s")
