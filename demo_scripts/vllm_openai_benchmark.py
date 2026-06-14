import time
from openai import OpenAI

# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://192.168.8.231:8200/v1",
# )
# MODEL = "qwen3-14b"


# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://192.168.100.230:11434/v1",
# )
# MODEL = "qwen3-14b"


# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://192.168.100.201:8200/v1",
# )
# MODEL = "weights/Confucius3-Math"


client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.100.203:8200/v1",
)
MODEL = "Qwen2-Math-72B-Instruct"


QUESTION = "当 $x$ 属于 $[-2, 1]$ 的闭区间时，不等式 $a x^{3} - x^{2} + 4 x + 3 \\geq 0$ 恒成立，则实数 $a$ 的取值范围是多少？"


start_time = time.perf_counter()
first_token_time = None
last_token_time = None

chunk_count = 0
token_count = 0
content_parts = []

if 'Confucius3-Math' in MODEL:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请逐步推理，并将最终答案放在 \\boxed{} 内。"},
            {"role": "user", "content": QUESTION}
        ],
        temperature=0.6,
        max_tokens=10240,
        stream=True,
        timeout=600,
    )
elif 'Qwen2-Math-72B' in MODEL:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请逐步推理，并将最终答案放在 \\boxed{} 内。"},
            {"role": "user", "content": QUESTION}
        ],
        temperature=0.1,
        top_p=0.95,
        max_tokens=3000,
        stream=True,
        timeout=600,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
else:
    # qwen对于思考模式（enable_thinking=True），使用 Temperature=0.6、TopP=0.95、TopK=20 和 MinP=0。不要使用贪婪解码，因为它可能导致性能下降和无尽的重复。
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请逐步推理，并将最终答案放在 \\boxed{} 内。"},
            {"role": "user", "content": QUESTION}
        ],
        max_tokens=3000,
        stream=True,
        timeout=600,
    )
    

print("\n===== STREAM OUTPUT =====\n")

for chunk in stream:
    now = time.perf_counter()
    chunk_count += 1

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta.content

    if delta:
        if first_token_time is None:
            first_token_time = now
            print(f"\n[TTFT: {first_token_time - start_time:.3f}s]\n")

        last_token_time = now
        content_parts.append(delta)

        # OpenAI 流式接口返回的是文本片段，不一定是严格 token
        token_count += 1

        print(delta, end="", flush=True)

end_time = time.perf_counter()

output_text = "".join(content_parts)

total_time = end_time - start_time
ttft = first_token_time - start_time if first_token_time else None
decode_time = last_token_time - first_token_time if first_token_time and last_token_time else 0

avg_chunk_speed = token_count / total_time if total_time > 0 else 0
decode_chunk_speed = token_count / decode_time if decode_time > 0 else 0

print("\n\n===== METRICS =====")
print(f"模型: {MODEL}")
print(f"问题: {QUESTION}")
print(f"总耗时 Total Time: {total_time:.3f}s")

if ttft is not None:
    print(f"首 Token 延迟 TTFT: {ttft:.3f}s")
else:
    print("首 Token 延迟 TTFT: 未收到有效输出")

print(f"Decode 耗时: {decode_time:.3f}s")
print(f"流式 chunk 总数: {chunk_count}")
print(f"有效文本 chunk 数: {token_count}")
print(f"平均 chunk 速度: {avg_chunk_speed:.2f} chunk/s")
print(f"首 Token 后速度: {decode_chunk_speed:.2f} chunk/s")
print(f"输出字符数: {len(output_text)}")