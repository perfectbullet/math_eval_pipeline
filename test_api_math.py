"""Test client for the Qwen3 API server."""

from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://192.168.100.203:8200/v1")
model_name = 'Qwen2.5-Math-72B-Instruct'
# Test streaming
print("=== Streaming Test ===")

q = """# 化简下列各式：

$$
\sqrt [ 3 ]{a ^ {\frac {7}{2}} \cdot \sqrt {a ^ {- 3}}} \div \sqrt [ 3 ]{\sqrt {a ^ {- 3}} \cdot \sqrt {a ^ {- 1}}}. \tag {2}
$$

"""

q2 = "定义在 $\\mathbb{R}$ 上的奇函数 $f(x)$ 满足 $f(2-x) = f(x)$，且在 $[0, 1)$ 的区间上单调递减，其中 $0$ 是闭区间，$1$ 是开区间。若方程 $f(x) = -1$ 在 $[0, 1)$ 上有实数根，则方程 $f(x) = 1$ 在区间 $[-1, 11]$ 的闭区间上，所有实数根之和是多少？"

stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "请逐步推理，并将最终答案放在 \\boxed{} 内。"},
        {"role": "user", "content": q},
    ],
    temperature=0.1,
    top_p=0.95,
    max_tokens=3000,
    stream=True,
    timeout=600,
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print("\n")

