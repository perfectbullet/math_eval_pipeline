"""A TIR(tool-integrated reasoning) math agent
```bash
python tir_math.py
```
"""

import os
from pprint import pprint

from qwen_agent.agents import TIRMathAgent
from qwen_agent.gui import WebUI

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")

# We use the following two systems to distinguish between COT mode and TIR mode
TIR_SYSTEM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
COT_SYSTEM = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)


def init_agent_service():
    # Use this to access the qwen2.5-math model deployed on dashscope
    llm_cfg = {
        "model": "Qwen3-32B",
        # "model_type": "qwen_dashscope",
        # "generate_cfg": {"top_k": 1},
        "model_server": "http://192.168.100.202:8200/v1",  # base_url，也称为 api_base
        "api_key": "EMPTY",
        # 'model': 'Qwen2.5-7B-Instruct',
        # 'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
        # 'api_key': 'EMPTY',
    }
    bot = TIRMathAgent(llm=llm_cfg, name="Qwen2.5-Math", system_message=TIR_SYSTEM)
    return bot


def test(query: str = "斐波那契数列前10个数字"):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [{"role": "user", "content": query}]
    for response in bot.run(messages):
        pprint(response, indent=2)


def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        "prompt.suggestions": [
            "曲线 $y=2 \\ln (x+1)$ 在点 $(0,0)$ 处的切线方程为 $( )$.",
            "10.（多选）已知 $(1 - x)^{2025} = a_0 + a_1x + a_2x^2 +\\dots +a_{2025}x^{2025}$ ，则()\n\nA. 展开式的各二项式系数的和为0\n\nB. $a_{1}+a_{2}+\\cdots+a_{2025}=-1$\n\nC. $2^{2025}a_{0}+2^{2024}a_{1}+2^{2023}a_{2}+\\cdots+a_{2025}=1$\n\nD. $\\frac{1}{a_1} +\\frac{1}{a_2} +\\dots +\\frac{1}{a_{2025}} = -1$",
            "已知角 $\\alpha$ 的顶点与原点O重合，始边与x轴的非负半轴重合，它的终边过点 $P\\left(-\\frac{3}{5}, -\\frac{4}{5}\\right)$\n\n( I ) 求 $\\sin(\\alpha+\\pi)$ 的值;\n\n(Ⅱ) 若角β满足sin（α+β）= $\\frac{5}{13}$ ，求cosβ的值.",
            "14. 已知 $x = x_{1}$ 和 $x = x_{2}$ 分别是函数 $f(x) = 2a^{x} - ex^{2}$ （ $a > 0$ 且 $a \\neq 1$ ）的极小值点和极大值点。若 $x_{1} < x_{2}$ ，则 $a$ 的取值范围是$( )$",
            "24. 已知角 $\\alpha$ 的顶点与原点O重合，始边与x轴的非负半轴重合，它的终边过点 $P\\left(-\\frac{3}{5}, -\\frac{4}{5}\\right)$\n\n( I ) 求 $\\sin(\\alpha+\\pi)$ 的值;\n\n(Ⅱ) 若角β满足sin（α+β）= $\\frac{5}{13}$ ，求cosβ的值.",
            "6.（多选）下列说法正确的是（）\n\nA. 一组样本数据通过计算得到线性回归方程为 $\\hat{y}=0.95x+a$ ，若 $(\\bar{x},\\bar{y})=(1,1)$ ，则a=0.05  \nB. 有一组数1,2,3,5,这组数的第75百分位数是3  \nC. 随机变量 $X \\sim B(n, p)$ ，若 $E(X) = 60, D(X) = 20$ ，则n = 180  \nD. 在 $\\alpha = 0.01$ 的独立性检验中, 若 $\\chi^2$ 不小于 $\\alpha$ 对应的临界值 $x_{0.01}$ , 可以推断两变量不独立, 该推断犯错误的概率不超过 0.01",
            "10.（多选）若函数 $f(x) = a \\ln x + \\frac{b}{x} + \\frac{c}{x^2} (a \\neq 0)$ 既有极大值也有极小值，则（\n\nA. bc>0\n\nB. $ab > 0$\n\nC. $b^{2}+8ac>0$\n\nD. ac < 0",
            "24. 已知10件不同产品中有4件是次品，现对它们进行一一测试，直至找出所有4件次品为止。\n\n(1) 若恰在第5次测试, 才测试到第一件次品, 第10次才找到最后一件次品, 则这样的不同测试方法数是多少?  \n(2) 若恰在第5次测试后, 就找出了所有4件次品, 则这样的不同测试方法数是多少?",
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run(server_name='0.0.0.0', server_port=8222)


if __name__ == "__main__":
    # test()
    # app_tui()
    app_gui()
