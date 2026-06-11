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


# ── 来自 scripts/serve_prm_scores.py 的修复函数（正则版）───────────────────

_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$]+?)\$")


def _fix_inline_math(text: str) -> str:
    """修复行内公式空格：$ xxx $ → $xxx$（不影响 $$...$$ 块级公式）。"""
    if not text or "$" not in text:
        return text

    # 1. 保护 $$...$$ 块级公式
    placeholders: list[str] = []

    def _save(m):
        placeholders.append(m.group(0))
        return f"%%MATH{len(placeholders) - 1}%%"

    protected = _DISPLAY_MATH_RE.sub(_save, text)

    # 2. 修复行内公式：去掉 $ 紧内侧的首尾空格
    def _trim(m):
        inner = m.group(1)
        stripped = inner.strip()
        return f"${stripped}$" if stripped != inner else m.group(0)

    fixed = _INLINE_MATH_RE.sub(_trim, protected)

    # 3. 恢复块级公式
    for idx, block in enumerate(placeholders):
        fixed = fixed.replace(f"%%MATH{idx}%%", block)

    return fixed


_THINK_TAG_RE = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """去掉 <think...>...</think > 包裹的思考过程文本。"""
    if not text or "<think" not in text:
        return text
    return _THINK_TAG_RE.sub("", text).strip()


_BOXED_RE = re.compile(r"(?<!\$)(\\boxed\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})")


def _fix_boxed_wrapping(text: str) -> str:
    """为未被 $ 包裹的 \boxed{...} 补上 $...$（跳过已在 $$...$$ 内的）。"""
    if not text or "\\boxed{" not in text:
        return text

    # 1. 保护 $$...$$ 块级公式，避免误包内部 \boxed
    placeholders: list[str] = []

    def _save(m):
        placeholders.append(m.group(0))
        return f"%%DMATH{len(placeholders) - 1}%%"

    protected = _DISPLAY_MATH_RE.sub(_save, text)

    # 2. 对裸露的 \boxed{...} 补 $...$
    def _wrap(m):
        return f"${m.group(1)}$"

    fixed = _BOXED_RE.sub(_wrap, protected)

    # 3. 恢复块级公式
    for idx, block in enumerate(placeholders):
        fixed = fixed.replace(f"%%DMATH{idx}%%", block)

    return fixed


# 需要修复的文本字段
_TEXT_FIELDS = ("question", "model_output", "reference_answer")


def fix_record_math(record: dict):
    """对 record 中的文本字段和 steps[].text 修复公式渲染问题。

    依次执行：去 think 标签 → 裸 boxed 包裹 → 行内公式空格。
    """
    for key in _TEXT_FIELDS:
        if key in record and isinstance(record[key], str):
            record[key] = _strip_think_tags(record[key])
            record[key] = _fix_boxed_wrapping(record[key])
            record[key] = _fix_inline_math(record[key])
    for step in record.get("steps", []):
        if isinstance(step, dict) and "text" in step and isinstance(step["text"], str):
            step["text"] = _strip_think_tags(step["text"])
            step["text"] = _fix_boxed_wrapping(step["text"])
            step["text"] = _fix_inline_math(step["text"])
