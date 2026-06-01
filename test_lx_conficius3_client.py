# import numpy as np

#!/usr/bin/env python3
"""
Qwen2.5-14B-GPTQ API 测试客户端
使用 vLLM OpenAI-compatible API
"""

import requests
import json
import time
import textwrap

def print_model_response(response, t1, width=40):
    """
    结构化打印大模型API返回结果
    """
    # print("=" * 60)
    # print("📊 API 原始结构：")
    # print(json.dumps(response, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("📝 模型回答：")
    print("-"*60)
    
    # 获取回答（适配绝大多数API格式）
    if "choices" in response:
        content = response["choices"][0]["message"]["content"]
        thinking = response["choices"][0]["message"]["reasoning"]
    elif "data" in response:
        content = response["data"]["content"]
    else:
        content = str(response)
    num = len(content) + len(thinking)
    print('use time ever miao is :', num / (time.time() - t1))
    
    # 自动换行打印
    print(textwrap.fill(thinking, width=width))
    print("="*60)
    print(textwrap.fill(content, width=width))
    print("="*60)


def test_models(base_url="http://localhost:8000"):
    """测试模型列表端点"""
    response = requests.get(f"{base_url}/v1/models")
    print("Available Models:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_chat_completion(base_url="http://localhost:8000"):
    """测试聊天完成端点"""
    data = {
        "model": "/home/phi-4-mini-reasoning",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=data)
    print("Chat Completion (中文):")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_math(base_url="http://localhost:8000"):
    """测试数学能力"""
    t1 = time.time()
    data = {
        "model": "Confucius3-Math",
        "messages": [
            {"role": "user", "content": "求 sin(15°) 的精确值"}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=data, timeout=200)
    # print("Math Test:")
    # message = json.dumps(response.json(), indent=2, ensure_ascii=False)
    print_model_response(response.json(), t1)

    

def test_code(base_url="http://localhost:8000"):
    """测试编程能力"""
    data = {
        "model": "/home/phi-4-mini-reasoning",
        "messages": [
            {"role": "user", "content": "用Python写一个快速排序算法"}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=data)
    print("Code Test:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def test_streaming(base_url="http://localhost:8000"):
    """测试流式响应"""
    t1 = time.time()
    word = ''
    data = {
        # "model": "/home/phi-mini-llm/weights/Phi-4-mini-flash-fp8",
        # "model": "phi4-mini-flash-reasoning",
        # "model": "Phi-4-mini-reasoning",
        "model": "Confucius3-Math",
        "messages": [
            {"role": "user", "content": "已知 sin α = 3/5，且 α 为第二象限角，求 cos α 和 tan α 的值"}
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": True
    }
    print("Streaming Response:")
    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=True)
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str != '[DONE]':
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(content, end='', flush=True)
                                word += content
                    except json.JSONDecodeError:
                        pass
    print("\n")
    num = len(word)
    print('use time ever miao is :', num / (time.time() - t1))



if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.100.201:8200"

    print("=" * 50)
    print()

    try:
        # test_models(base_url)
        # test_chat_completion(base_url)
        test_math(base_url)
        # test_code(base_url)
        # test_streaming(base_url)

        print("=" * 50)
        print("所有测试完成!")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器")
    except Exception as e:
        print(f"错误: {e}")

