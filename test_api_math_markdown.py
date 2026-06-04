"""Test client for API server — 使用 MathLLMClient。"""

from math_llm_client import MathLLMClient

# model_name = "Qwen3-32B-GPTQ-Int8"
# base_url = "http://192.168.100.202:8000/v1"
model_name = "Qwen3-32B"
base_url = "http://192.168.100.202:8000/v1"
max_tokens = int(10240 * 0.9)

q = r"""# 化简下列各式：

$$
\sqrt [ 3 ]{a ^ {\frac {7}{2}} \cdot \sqrt {a ^ {- 3}}} \div \sqrt [ 3 ]{\sqrt {a ^ {- 3}} \cdot \sqrt {a ^ {- 1}}}. \tag {2}
$$
"""

print("=== Streaming Test ===\n")
print(f"模型: {model_name} @ {base_url}\n")

client = MathLLMClient(model_name, base_url)
result = client.ask(q, max_tokens=max_tokens)

MathLLMClient.print_stats(result)
