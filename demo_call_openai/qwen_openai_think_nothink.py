"""Qwen3 流式推理测试脚本 — 支持 thinking / no-thinking 模式，并打印性能指标。"""

import argparse
import time

from openai import OpenAI


def create_client(api_base: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=api_base)


def streaming_chat(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 7168,
    temperature: float = 0.7,
    top_p: float = 0.8,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    enable_thinking: bool = False,
):
    """流式调用并实时打印，返回性能指标。"""

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        stream=True,
        stream_options={"include_usage": True},  # 请求 usage 统计
        extra_body={
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )

    ttft = None               # 首 token 时间
    token_count = 0            # 生成的 token 数
    reasoning_tokens = 0       # 思考 token 数
    output_tokens = 0          # 输出 token 数
    prompt_tokens = 0          # 输入 token 数
    total_tokens = 0           # 总 token 数
    start_time = time.perf_counter()
    reasoning_buffer = []      # 思考内容缓冲
    content_buffer = []        # 正文内容缓冲
    in_thinking = False        # 是否处于 thinking 块中

    print("\n" + "=" * 60)
    print(f"🧠 思考模式: {'开启' if enable_thinking else '关闭'}")
    print("=" * 60)

    for chunk in stream:
        # ---------- 处理 usage ----------
        if chunk.usage is not None:
            # 部分 API 在最后一个 chunk 带完整 usage
            output_tokens = getattr(chunk.usage, "completion_tokens_details", None)
            if output_tokens and hasattr(output_tokens, "reasoning_tokens"):
                reasoning_tokens = output_tokens.reasoning_tokens
            total_tokens = chunk.usage.total_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0
            prompt_tokens = chunk.usage.prompt_tokens or 0

        # ---------- 处理 delta ----------
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        # 首 token 时间
        if ttft is None and (delta.content or getattr(delta, "reasoning_content", None)):
            ttft = time.perf_counter() - start_time

        # 思考内容 (reasoning_content)
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            if not in_thinking:
                print("\n--- 🧠 思考过程 ---")
                in_thinking = True
            print(reasoning_content, end="", flush=True)
            reasoning_buffer.append(reasoning_content)
            token_count += 1

        # 正文内容
        if delta.content:
            if in_thinking:
                print("\n--- 📝 回答内容 ---")
                in_thinking = False
            print(delta.content, end="", flush=True)
            content_buffer.append(delta.content)
            token_count += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 如果没有拿到 ttft（极端情况），用总时间兜底
    if ttft is None:
        ttft = total_time

    # 如果 usage 没返回 token 数，用本地计数
    if output_tokens == 0:
        output_tokens = token_count

    print("\n\n" + "=" * 60)
    print("📊 性能指标")
    print("=" * 60)
    print(f"  首 token 时间 (TTFT):  {ttft:.3f} s")
    print(f"  总推理时间:            {total_time:.3f} s")
    if total_tokens:
        print(f"  Prompt tokens:         {prompt_tokens}")
    print(f"  生成 tokens:           {output_tokens}"
          + (f"  (含思考: {reasoning_tokens})" if reasoning_tokens else ""))
    print(f"  推理速度:              {output_tokens / total_time:.2f} tokens/s")
    if reasoning_tokens:
        print(f"  思考速度:              {reasoning_tokens / total_time:.2f} tokens/s (思考部分)")
    print("=" * 60)

    return {
        "ttft": ttft,
        "total_time": total_time,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tokens_per_second": output_tokens / total_time if total_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3 流式推理测试")
    parser.add_argument("--api_base", default="http://192.168.8.233:8200/v1", help="API 地址")
    parser.add_argument("--model", default="Qwen3-14B-AWQ", help="模型名称")
    parser.add_argument("--prompt", default="什么是土豆，什么是马铃薯？", help="提问内容")
    parser.add_argument("--max_tokens", type=int, default=7168, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top_p", type=float, default=0.8, help="top_p")
    parser.add_argument("--top_k", type=int, default=20, help="top_k")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="presence penalty")
    parser.add_argument("--think", action="store_true", help="开启 thinking 模式")
    args = parser.parse_args()

    client = create_client(args.api_base)

    # 关闭思考模式
    metrics_no_think = streaming_chat(
        client, args.model, args.prompt,
        max_tokens=args.max_tokens,
        # temperature=args.temperature,
        # top_p=args.top_p,
        # top_k=args.top_k,
        # presence_penalty=args.presence_penalty,
        enable_thinking=False,
    )

    # 开启思考模式
    if args.think:
        metrics_think = streaming_chat(
            client, args.model, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            enable_thinking=True,
        )

        # 对比汇总
        print("\n" + "=" * 60)
        print("📋 模式对比")
        print("=" * 60)
        print(f"  {'指标':<22} {'关闭思考':>12} {'开启思考':>12}")
        print(f"  {'-'*22} {'-'*12} {'-'*12}")
        print(f"  {'TTFT (s)':<22} {metrics_no_think['ttft']:>12.3f} {metrics_think['ttft']:>12.3f}")
        print(f"  {'总时间 (s)':<22} {metrics_no_think['total_time']:>12.3f} {metrics_think['total_time']:>12.3f}")
        print(f"  {'生成 tokens':<22} {metrics_no_think['output_tokens']:>12} {metrics_think['output_tokens']:>12}")
        print(f"  {'速度 (tokens/s)':<22} {metrics_no_think['tokens_per_second']:>12.2f} {metrics_think['tokens_per_second']:>12.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
