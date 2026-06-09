#!/usr/bin/env python3
"""
独立 HTML 渲染脚本：从 PRM JSONL 生成可浏览的 HTML 报告。

不加载 torch / transformers，秒级完成。仅依赖 markdown + 标准库。

用法:
    python scripts/render_html_report.py \
        --input results/prm/prm_step_scores-gaokao-Qwen3-32B-GPTQ-Int8-0602.jsonl \
        --output_dir results/html_preview
"""

import argparse
import html
import json
import markdown
import re
from pathlib import Path

# ---------- Markdown + LaTeX 安全渲染 ----------

_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+?\$)", re.DOTALL)


def _safe_markdown(text: str) -> str:
    """安全地将含 LaTeX 的文本转为 HTML：先保护数学块，再 markdown，再恢复。"""
    if not text:
        return ""

    math_blocks = []

    def _replace(m):
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"%%MATH{idx}%%"

    protected = _MATH_BLOCK_RE.sub(_replace, text)

    # 防止行首的 "数字.空格" 被 Markdown 解析为有序列表
    protected = re.sub(r"(^|\n)(\d+)\. ", r"\1\2\\. ", protected)

    result = markdown.markdown(protected, extensions=["fenced_code", "tables"])

    for idx, block in enumerate(math_blocks):
        # HTML 转义数学块内容，防止 < > & 被浏览器解析为 HTML 标签
        escaped_block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        result = result.replace(f"%%MATH{idx}%%", escaped_block)

    return result


# ---------- HTML 模板 ----------

_KATEX_HEAD = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}},
    ]}});"></script>
<style>
body {{ font-family: "Noto Sans SC", sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; padding-bottom: 8px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.tag {{ padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600; color: #fff; }}
.tag-id {{ background: #2c3e50; }}
.tag-source {{ background: #2980b9; }}
.tag-model {{ background: #8e44ad; }}
.tag-category {{ background: #c0392b; }}
.tag-correct {{ background: #27ae60; }}
.tag-wrong {{ background: #e74c3c; }}
.section {{ margin-bottom: 20px; background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
.section-title {{ font-weight: bold; font-size: 15px; color: #2c3e50; margin-bottom: 8px; border-left: 4px solid #3498db; padding-left: 8px; }}
.section-content {{ white-space: pre-wrap; word-break: break-word; line-height: 1.7; font-size: 14px; }}
.section-content.scroll {{ max-height: 400px; overflow-y: auto; }}
.score-bar {{ display: inline-block; height: 14px; border-radius: 7px; }}
.score-bar.green {{ background: #27ae60; }}
.score-bar.red {{ background: #e74c3c; }}
.step-card {{ margin-bottom: 12px; border-radius: 6px; padding: 12px; border-left: 4px solid #bdc3c7; }}
.step-card.wrong {{ border-left-color: #e74c3c; background: #fdf0f0; }}
.step-card.correct {{ border-left-color: #27ae60; background: #f0fdf4; }}
.step-card.first-wrong {{ border-left-color: #e74c3c; background: #fde8e8; box-shadow: 0 0 0 2px #e74c3c; }}
.step-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.step-num {{ font-weight: bold; font-size: 14px; color: #2c3e50; }}
.step-score {{ font-size: 13px; color: #7f8c8d; }}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.summary-item {{ padding: 8px 12px; background: #f8f9fa; border-radius: 4px; }}
.summary-label {{ font-size: 12px; color: #7f8c8d; }}
.summary-value {{ font-size: 16px; font-weight: bold; color: #2c3e50; }}
a {{ color: #2980b9; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{title}</h1>
"""


def _render_sample_html(r: dict) -> str:
    """为单条记录生成完整 HTML。"""
    rid = r.get("id", "unknown")
    title = f"评测样本 - {rid}"
    buf = _KATEX_HEAD.format(title=title)

    # 头部标签
    category = r.get("error_category", "")
    category_map = {
        "logic_error": "解题逻辑错误",
        "answer_or_render_suspect": "疑似渲染问题",
        "answer_extract_error": "答案提取错误",
        "uncertain_logic": "不确定",
        "pass": "通过",
    }
    buf += '<div class="meta">\n'
    buf += f'  <span class="tag tag-id">{html.escape(rid)}</span>\n'
    buf += f'  <span class="tag tag-source">{html.escape(r.get("source", ""))}</span>\n'
    buf += f'  <span class="tag tag-model">{html.escape(r.get("model_name", ""))}</span>\n'
    cat_text = category_map.get(category, category) or "—"
    buf += f'  <span class="tag tag-category">{html.escape(cat_text)}</span>\n'

    # 验证结果
    verify_correct = r.get("verify_correct", None)
    if verify_correct is True:
        buf += '  <span class="tag tag-correct">验证 ✓ 正确</span>\n'
    elif verify_correct is False:
        buf += '  <span class="tag tag-wrong">验证 ✗ 错误</span>\n'
    buf += '</div>\n'

    # 题目（Markdown 渲染，含 LaTeX 保护）
    q_html = _safe_markdown(r.get("question", ""))
    buf += '<div class="section">\n'
    buf += f'  <div class="section-title">题目</div>\n'
    buf += f'  <div class="section-content">{q_html}</div>\n</div>\n'

    # 标准答案（纯文本，KaTeX 浏览器端渲染 $...$）
    a_text = r.get("reference_answer", "")
    a_html = html.escape(a_text)
    buf += '<div class="section">\n'
    buf += '  <div class="section-title">标准答案</div>\n'
    buf += f'  <div class="section-content">{a_html}</div>\n</div>\n'

    # 模型输出（Markdown 渲染，含 LaTeX 保护）
    out_html = _safe_markdown(r.get("model_output", ""))
    buf += '<div class="section">\n'
    buf += '  <div class="section-title">模型输出</div>\n'
    buf += f'  <div class="section-content">{out_html}</div>\n</div>\n'

    buf += "</body></html>"
    return buf


def _render_index_html(records: list[dict], output_dir: Path):
    """生成 index.html 索引页，列出所有样本链接。"""
    title = "评测样本索引"
    buf = _KATEX_HEAD.format(title=title)

    # 统计
    total = len(records)
    correct = sum(1 for r in records if r.get("verify_correct") is True)
    wrong = sum(1 for r in records if r.get("verify_correct") is False)
    none_count = total - correct - wrong

    buf += '<div class="section">\n'
    buf += '  <div class="section-title">统计</div>\n'
    buf += f'  <p>共 <strong>{total}</strong> 条记录'
    if correct or wrong:
        buf += f'：✓ {correct} 正确 / ✗ {wrong} 错误'
    if none_count:
        buf += f' / {none_count} 未标记'
    buf += '</p>\n</div>\n'

    # 列表
    buf += '<div class="section">\n'
    buf += '  <div class="section-title">全部样本</div>\n'
    buf += '  <table style="width:100%;border-collapse:collapse;font-size:14px">\n'
    buf += '    <tr style="background:#f0f0f0"><th style="padding:6px;text-align:left">ID</th><th style="padding:6px">来源</th><th style="padding:6px">验证</th><th style="padding:6px">分类</th></tr>\n'

    category_map = {
        "logic_error": "解题逻辑错误",
        "answer_or_render_suspect": "疑似渲染问题",
        "answer_extract_error": "答案提取错误",
        "uncertain_logic": "不确定",
        "pass": "通过",
    }

    for r in records:
        rid = r.get("id", "unknown")
        source = r.get("source", "")
        verify = r.get("verify_correct", None)
        if verify is True:
            v_html = '<span style="color:#27ae60">✓</span>'
        elif verify is False:
            v_html = '<span style="color:#e74c3c">✗</span>'
        else:
            v_html = "—"
        cat = category_map.get(r.get("error_category", ""), r.get("error_category", "—"))
        row_bg = "#fff" if verify is not False else "#fdf0f0"
        buf += f'    <tr style="background:{row_bg}">'
        buf += f'<td style="padding:6px"><a href="{rid}.html">{html.escape(rid)}</a></td>'
        buf += f'<td style="padding:6px;text-align:center">{html.escape(source)}</td>'
        buf += f'<td style="padding:6px;text-align:center">{v_html}</td>'
        buf += f'<td style="padding:6px;text-align:center">{html.escape(str(cat))}</td>'
        buf += '</tr>\n'

    buf += '  </table>\n</div>\n'
    buf += "</body></html>"

    (output_dir / "index.html").write_text(buf, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="从 JSONL 生成 HTML 报告（无需 GPU）")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_dir", default="results/html_preview", help="HTML 输出目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取全部记录
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"读取 {len(records)} 条记录，开始渲染 HTML …")

    # 逐条渲染
    for r in records:
        rid = r.get("id", "unknown")
        html_content = _render_sample_html(r)
        (output_dir / f"{rid}.html").write_text(html_content, encoding="utf-8")

    # 生成索引页
    _render_index_html(records, output_dir)

    print(f"完成！共生成 {len(records)} 个 HTML 文件 → {output_dir}/")
    print(f"索引页: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
