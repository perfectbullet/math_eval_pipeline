#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEV_PYTHON="/home/zj/miniconda3/envs/math-eval/bin/python"
DEFAULT_SERVER_PYTHON="/home/zenking/miniconda3/envs/math_verify/bin/python"
RUN_DATE="${RUN_DATE:-$(date '+%Y%m%d')}"

PYTHON_BIN=""
MODE=""
LANG=""
INPUT=""
API_BASE=""
MODEL=""
WORKERS=""
LIMIT=""
API_KEY="no-key"
OUTPUT_DIR="../data/model_outputs/math_qa_275"

usage() {
    cat <<'EOF'
用法：
  bash run_single_mode_lang_inference.sh \
    --mode cot|tir \
    --lang en|zh \
    --input <path> \
    --api_base <url> \
    --model <name> \
    --workers <n> \
    --limit <n> \
    [--output_dir <dir>] \
    [--python_bin <path>] \
    [--api_key <key>]

说明：
  1. 自动拼接 output 文件名，不需要手动传 --output
  2. 输出文件名格式：
     model_outputs-<input_stem>_<mode>_<lang>_<model_slug>.jsonl
  3. Python 选择优先级：
     --python_bin > 环境变量 MATH_EVAL_PYTHON > 开发机默认路径 > 服务器默认路径 > python3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --lang)
            LANG="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --api_base)
            API_BASE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python_bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_arg() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "缺少必填参数: $name" >&2
        usage >&2
        exit 1
    fi
}

detect_python_bin() {
    if [[ -n "$PYTHON_BIN" ]]; then
        return 0
    fi

    if [[ -n "${MATH_EVAL_PYTHON:-}" ]]; then
        PYTHON_BIN="$MATH_EVAL_PYTHON"
    elif [[ -x "$DEFAULT_DEV_PYTHON" ]]; then
        PYTHON_BIN="$DEFAULT_DEV_PYTHON"
    elif [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
        PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "错误：未找到可用 Python。请显式传入 --python_bin 或设置 MATH_EVAL_PYTHON" >&2
        exit 1
    fi
}

normalize_input_stem() {
    local input_name stem
    input_name="$(basename "$1")"
    stem="${input_name%.jsonl}"
    stem="${stem%.json}"
    stem="${stem%.mineru}"
    echo "${stem}_${RUN_DATE}"
}

normalize_model_slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

build_output_path() {
    local input_stem model_slug
    input_stem="$(normalize_input_stem "$INPUT")"
    model_slug="$(normalize_model_slug "$MODEL")"
    echo "${OUTPUT_DIR}/model_outputs-${input_stem}_${MODE}_${LANG}_${model_slug}.jsonl"
}

require_arg "--mode" "$MODE"
require_arg "--lang" "$LANG"
require_arg "--input" "$INPUT"
require_arg "--api_base" "$API_BASE"
require_arg "--model" "$MODEL"
require_arg "--workers" "$WORKERS"
require_arg "--limit" "$LIMIT"

if [[ "$MODE" != "cot" && "$MODE" != "tir" ]]; then
    echo "错误：--mode 只能是 cot 或 tir" >&2
    exit 1
fi

if [[ "$LANG" != "en" && "$LANG" != "zh" ]]; then
    echo "错误：--lang 只能是 en 或 zh" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
detect_python_bin

OUTPUT_PATH="$(build_output_path)"
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Python: $PYTHON_BIN"
echo "Mode: $MODE"
echo "Lang: $LANG"
echo "Input: $INPUT"
echo "Output: $OUTPUT_PATH"
echo "API: $API_BASE"
echo "Model: $MODEL"
echo "Workers: $WORKERS"
echo "Limit: $LIMIT"

"$PYTHON_BIN" run_model_tir_inference_concurrent.py \
    --input "$INPUT" \
    --output "$OUTPUT_PATH" \
    --api_base "$API_BASE" \
    --model "$MODEL" \
    --mode "$MODE" \
    --lang "$LANG" \
    --workers "$WORKERS" \
    --limit "$LIMIT" \
    --api_key "$API_KEY"
