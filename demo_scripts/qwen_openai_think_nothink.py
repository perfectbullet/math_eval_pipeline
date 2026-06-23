from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://192.168.100.203:8200/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

DEFAULT_SYSTEM_PROMPT = "请逐步推理，并将最终答案放在 \\boxed{} 内。"

question = r"""# 化简下列各式：
$$
\sqrt [ 3 ]{a ^ {\frac {7}{2}} \cdot \sqrt {a ^ {- 3}}} \div \sqrt [ 3 ]{\sqrt {a ^ {- 3}} \cdot \sqrt {a ^ {- 1}}}. \tag {2}
$$
"""

stream = client.chat.completions.create(
    model="Qwen3-32B",
    messages=[
        {"role": "system", "content": "请逐步推理，并将最终答案放在 \\boxed{} 内。"},
        {"role": "user", "content": question},
    ],
    stream=True,
    max_tokens=30720,
    # temperature=0.7,
    # top_p=0.8,
    # presence_penalty=0,
    extra_body={
        # "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
for trunk in stream:
    content = trunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print("\n")
