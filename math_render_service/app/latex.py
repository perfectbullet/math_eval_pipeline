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

def _match_command(text: str, pos: int, command: str) -> bool:
    """检查 text[pos:] 是否以 \\command 开头。"""
    if pos + len(command) >= len(text):
        return False
    return text[pos:pos + len(command)] == command


def _consume_brace_group(text: str, pos: int) -> Optional[int]:
    r"""从 text[pos]（应为 '{'）消费一个花括号组，返回闭 '}' 之后的位置，不匹配返回 None。"""
    if pos >= len(text) or text[pos] != '{':
        return None
    depth = 1
    i = pos + 1
    while i < len(text) and depth > 0:
        if text[i] == '\\' and i + 1 < len(text):
            i += 2
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return i if depth == 0 else None


_BEGIN_CMD = r'\begin{'
_END_CMD = r'\end{'


def _strip_inline_dollars(text: str) -> str:
    r"""移除 $...$ 定界符，保留内部内容。

    用于 tabular→array 转换：在 $$\begin{array}...\end{array}$$ 内部，
    原来的 $...$ 是多余的（已经在数学模式中）。
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        # 跳过 $$（安全处理）
        if text[i:i+2] == '$$':
            result.append('$$')
            i += 2
            continue
        if text[i] == '$':
            i += 1  # 跳过开 $
            while i < len(text):
                if text[i:i+2] == '$$':
                    break
                if text[i] == '$':
                    i += 1  # 跳过闭 $
                    break
                result.append(text[i])
                i += 1
            continue
        result.append(text[i])
        i += 1
    return ''.join(result)


def convert_tabular_to_array(text: str) -> str:
    r"""
    将 \begin{tabular}{col}...\end{tabular} 转换为 \begin{array}{col}...\end{array}，
    并剥离内部 $...$ 定界符。

    KaTeX 不支持 tabular 环境。此函数将其转为等效的 array 环境，
    同时移除内部 $...$（在 $$\begin{array}...\end{array}$$ 内已处于数学模式，$...$ 冗余）。

    Examples:
        >>> convert_tabular_to_array(r"\begin{tabular}{|l|l|} a & $b$ \end{tabular}")
        '\\\\begin{array}{|l|l|} a & b \\\\end{array}'

    Args:
        text: 可能包含 tabular 环境的文本

    Returns:
        tabular 已转为 array 的文本
    """
    if not text or '\\begin{tabular}' not in text:
        return text

    def _convert(m):
        col_spec = m.group(1)
        body = _strip_inline_dollars(m.group(2))
        return f'\\begin{{array}}{col_spec}{body}\\end{{array}}'

    return re.sub(
        r'\\begin\{tabular\}(\{[^}]+\})(.*?)\\end\{tabular\}',
        _convert,
        text,
        flags=re.DOTALL,
    )


def wrap_bare_environments(text: str) -> str:
    r"""
    将裸露的 \begin{X}...\end{X} 用 $$...$$ 包裹。

    仅包裹不在 $...$ 或 $$...$$ 内部的 \begin{X}...\end{X} 环境。

    Examples:
        >>> wrap_bare_environments(r"\begin{tabular}{|l|l|} \hline a \\\\ \end{tabular}")
        '$$\\begin{tabular}{|l|l|} \\hline a \\\\\\end{tabular}$$'
        >>> wrap_bare_environments("$$\\begin{array} x \\end{array}$$")
        '$$\\begin{array} x \\end{array}$$'

    Args:
        text: 可能包含裸露 LaTeX 环境的文本

    Returns:
        裸露环境已用 $$...$$ 包裹的文本
    """
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

        # ── 在公式外部：检测 \begin{X} ──
        if text[i:i + len(_BEGIN_CMD)] == _BEGIN_CMD:
            # 提取环境名 \begin{env_name}
            name_start = i + len(_BEGIN_CMD) - 1  # 指向 '{'
            name_end = _consume_brace_group(text, name_start)
            if name_end is None:
                # 不完整，原样保留
                result.append(text[i])
                i += 1
                continue
            env_name = text[name_start + 1:name_end - 1]

            # 在 \end{env_name} 之前的内容（含可能的额外花括号参数）都属于环境体
            # 向后扫描匹配的 \end{env_name}
            end_tag = _END_CMD + env_name + '}'
            search_pos = name_end
            depth = 1  # 嵌套层数
            env_body_start = name_end
            found_end = -1

            while search_pos < len(text) and depth > 0:
                next_begin = text.find(_BEGIN_CMD, search_pos)
                next_end = text.find(end_tag, search_pos)

                if next_end < 0:
                    break  # 没找到闭合标签

                if 0 <= next_begin < next_end:
                    # 嵌套同名的 \begin — 需要检查环境名是否一致
                    nb_name_start = next_begin + len(_BEGIN_CMD) - 1
                    nb_name_end = _consume_brace_group(text, nb_name_start)
                    if nb_name_end and text[nb_name_start + 1:nb_name_end - 1] == env_name:
                        depth += 1
                    search_pos = nb_name_end if nb_name_end else next_begin + 1
                    continue

                depth -= 1
                if depth == 0:
                    found_end = next_end
                search_pos = next_end + len(end_tag)

            if found_end < 0:
                # 未找到闭合 \end{env_name}，原样保留
                result.append(text[i])
                i += 1
                continue

            # 完整匹配：用 $$ 包裹 \begin{X}...\end{X}
            env_end = found_end + len(end_tag)
            result.append('$$')
            result.append(text[i:env_end])
            result.append('$$')
            i = env_end
            continue

        # 普通字符
        result.append(text[i])
        i += 1

    return ''.join(result)


def fix_degree_limit_superscripts(text: str) -> str:
    r"""
    修复角度极限中的连续上标。

    KaTeX/MathJax 会把 `90^\circ^-` 解析为 double superscript。
    在 `\circ` 后补一个空分组，让单侧极限符号绑定到独立空基底。

    Examples:
        >>> fix_degree_limit_superscripts(r"$B \to 90^\circ^-$")
        '$B \to 90^\circ{}^{-}$'
        >>> fix_degree_limit_superscripts(r"$B \to 30^\circ^+$")
        '$B \to 30^\circ{}^{+}$'
    """
    if not text or r'\circ^' not in text:
        return text

    return re.sub(
        r'(\\circ)\s*\^\s*(?:\{?\s*([+-])\s*\}?)',
        r'\1{}^{\2}',
        text,
    )


def normalize_latex_formulas(text: str) -> str:
    r"""
    执行完整的 LaTeX 公式规范化。

    按顺序执行以下操作：
    1. 规范化定界符（\( \) \[ \] → $ $$）
    2. 移除定界符内侧空格
    3. 将裸露的 \boxed{...} 用 $...$ 包裹
    4. 将 tabular 转为 array 并剥离内部 $...$
    5. 将裸露的 \begin{X}...\end{X} 用 $$...$$ 包裹
    6. 将 \text 替换为 \mathrm

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
    text = convert_tabular_to_array(text)
    text = wrap_bare_environments(text)
    text = re.sub(r'\\text\b', r'\\mathrm', text)
    text = fix_degree_limit_superscripts(text)
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
