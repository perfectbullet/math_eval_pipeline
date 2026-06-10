# math_render_service

数学文本渲染服务，为前端提供 LaTeX 公式修复和流式断句能力。

## 功能

- **LaTeX 公式修复**：行内公式空格、`\boxed{}` 包裹、`<think` 标签清理
- **流式断句**：（待实现）

## 运行

```bash
pip install -r requirements.txt
python -m app.serve_prm_scores --port 8900
```

## 测试

```bash
python -m pytest tests/ -v
```
