"""
LaTeX 公式处理工具集。

提供 LaTeX 公式的规范化、清理和转义功能。
"""
import re
from typing import List, Optional

def normalize_latex_delimiters(text: str) -> str:
    r"""
    将转义的 LaTeX 定界符替换为标准格式。

    替换规则：
    - \( 或 \\( 或 \\\( → $
    - \) 或 \\) 或 \\\) → $
    - \[ 或 \\[ 或 \\\[ → $$
    - \] 或 \\] 或 \\\] → $$

    Args:
        text: 可能包含转义 LaTeX 定界符的文本

    Returns:
        规范化后的文本
    """
    # 从多反斜杠到单反斜杠依次处理
    text = re.sub(r'\\\\\(', '$', text)
    text = re.sub(r'\\\\\)', '$', text)
    text = re.sub(r'\\\\\[', '$$', text)
    text = re.sub(r'\\\\\]', '$$', text)

    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)

    return text

# 向后兼容别名
_normalize_latex_delimiters = normalize_latex_delimiters

def clean_latex_formula_spaces(text: str) -> str:
    """
    移除 LaTeX 公式定界符内侧的空格。

    处理规则：
    - `$ text $` → `$text$`
    - `$$ text $$` → `$$text$$`
    - 仅移除紧邻定界符的空格，保留公式内部空格
    - 保留外部空格（如 "formula $x^2$ and"）

    Examples:
        >>> clean_latex_formula_spaces("$ S_n = a_1 \\cdot q^{n-1} $")
        '$S_n = a_1 \\cdot q^{n-1}$'
        >>> clean_latex_formula_spaces("$$ \\frac{a}{b} $$")
        '$$\\frac{a}{b}$$'
        >>> clean_latex_formula_spaces("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        定界符内侧空格已移除的文本
    """
    result = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None
    just_entered_formula = False

    while i < len(text):
        # 跳过转义的美元符号 \$
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

        # 处理 $$ 定界符
        if text[i:i+2] == '$$':
            if not in_formula:
                # 进入公式
                in_formula = True
                formula_delimiter = '$$'
                result.append('$$')
                just_entered_formula = True
            elif formula_delimiter == '$$':
                # 退出公式前，移除尾部空格
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$$')
            i += 2
            continue

        # 处理 $ 定界符
        if text[i] == '$':
            if not in_formula:
                # 进入公式
                in_formula = True
                formula_delimiter = '$'
                result.append('$')
                just_entered_formula = True
            elif formula_delimiter == '$':
                # 退出公式前，移除尾部空格
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$')
            i += 1
            continue

        # 刚进入公式时跳过前导空格
        if in_formula and just_entered_formula and text[i] == ' ':
            i += 1
            while i < len(text) and text[i] == ' ':
                i += 1
            just_entered_formula = False
            continue

        just_entered_formula = False
        result.append(text[i])
        i += 1

    return ''.join(result)

def escape_latex_backslashes(text: str) -> str:
    """
    在 LaTeX 公式内部将单反斜杠转义为双反斜杠。

    仅处理 $...$ 或 $$...$$ 定界符内的内容。

    Examples:
        >>> escape_latex_backslashes("$S_n = a_1 \\frac{1-q^n}{1-q}$")
        '$S_n = a_1 \\\\frac{1-q^n}{1-q}$'
        >>> escape_latex_backslashes("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        公式内反斜杠已转义的文本
    """
    result: List[str] = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None

    while i < len(text):
        # 跳过转义的美元符号 \$
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

        # 处理 $$ 定界符
        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$$'
                result.append('$$')
            elif formula_delimiter == '$$':
                in_formula = False
                formula_delimiter = None
                result.append('$$')
            i += 2
            continue

        # 处理 $ 定界符
        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$'
                result.append('$')
            elif formula_delimiter == '$':
                in_formula = False
                formula_delimiter = None
                result.append('$')
            i += 1
            continue

        # 在公式内部转义反斜杠
        if in_formula and text[i] == '\\':
            if i + 1 < len(text) and text[i + 1] == '\\':
                result.append('\\\\')
                i += 2
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)

def wrap_bare_boxed(text: str) -> str:
    r"""
    将裸露的 \boxed{...} 用 $...$ 包裹。

    数学模型（如 vLLM/Qwen）可能输出不带 $ 定界符的 \boxed{(2, 3)}。
    本函数仅包裹不在 $...$ 或 $$...$$ 内部的 \boxed{...}。

    Examples:
        >>> wrap_bare_boxed("答案为 \\boxed{(2, 3)}")
        '答案为 $\\boxed{(2, 3)}$'
        >>> wrap_bare_boxed("$$\\boxed{(2, 3)}$$")
        '$$\\boxed{(2, 3)}$$'

    Args:
        text: 可能包含裸露 \boxed{...} 的文本

    Returns:
        裸露 \boxed{...} 已用 $...$ 包裹的文本
    """
    BOXED_PATTERN = re.compile(r'\\boxed\s*\{')
    result: list[str] = []
    i = 0
    in_formula = False
    formula_delim: Optional[str] = None

    while i < len(text):
        # 处理 $$ 定界符
        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delim = '$$'
            elif formula_delim == '$$':
                in_formula = False
                formula_delim = None
            result.append('$$')
            i += 2
            continue

        # 处理 $ 定界符
        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delim = '$'
            elif formula_delim == '$':
                in_formula = False
                formula_delim = None
            result.append('$')
            i += 1
            continue

        # 已在公式内部 —— 原样复制
        if in_formula:
            result.append(text[i])
            i += 1
            continue

        # 检测裸露的 \boxed{...}
        m = BOXED_PATTERN.match(text, i)
        if m:
            start = i
            i = m.end()
            # 消费花括号组 {...}（匹配嵌套花括号）
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            # 用 $...$ 包裹
            result.append('$')
            result.append(text[start:i])
            result.append('$')
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)

def normalize_latex_formulas(text: str) -> str:
    r"""
    执行完整的 LaTeX 公式规范化。

    按顺序执行以下操作：
    1. 规范化定界符（\( \) \[ \] → $ $$）
    2. 移除定界符内侧空格
    3. 将裸露的 \boxed{...} 用 $...$ 包裹
    4. 将 \text 替换为 \mathrm

    Examples:
        >>> normalize_latex_formulas(r"\( S_n = a_1 \cdot q^{n-1} \)")
        '$S_n = a_1 \\cdot q^{n-1}$'
        >>> normalize_latex_formulas("$$ \\frac{a}{b} $$")
        '$$\\frac{a}{b}$$'
        >>> normalize_latex_formulas("\\boxed{(2, 3)}")
        '$\\boxed{(2, 3)}$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        完整规范化后的文本
    """
    text = normalize_latex_delimiters(text)
    text = clean_latex_formula_spaces(text)
    text = wrap_bare_boxed(text)
    text = re.sub(r'\\text\b', r'\\mathrm', text)
    return text
