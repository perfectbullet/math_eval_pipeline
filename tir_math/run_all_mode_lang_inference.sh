#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$(cd "$SCRIPT_DIR" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="/home/zj/miniconda3/envs/math-eval/bin/python"
INPUT="../data/math_qa_275_20260612.mineru.json"
OUTPUT_DIR="../data/model_outputs/math_qa_275"
API_BASE="http://192.168.100.203:8200/v1"
MODEL="Qwen3-32B"
WORKERS=8
LIMIT=8
API_KEY="no-key"

LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/run_pids"

usage() {
    cat <<'EOF'
用法：
  bash run_all_mode_lang_inference.sh [options]

可选参数：
  --python_bin PATH   Python 可执行文件，默认 /home/zj/miniconda3/envs/math-eval/bin/python
  --input PATH        输入文件，默认 ../data/math_qa_275_20260612.mineru.json
  --output_dir PATH   输出目录，默认 ../data/model_outputs/math_qa_275
  --api_base URL      模型服务地址，默认 http://192.168.100.203:8200/v1
  --model NAME        模型名，默认 Qwen3-32B
  --workers N         并发数，默认 8
  --limit N           限制条数，默认 8
  --api_key KEY       API key，默认 no-key
  --help              显示帮助

说明：
  1. 脚本会按固定顺序依次执行 4 条命令：
     cot/en -> cot/zh -> tir/en -> tir/zh
  2. 脚本本身会以后台方式启动，并把日志写入 tir_math/logs/
  3. 输出文件名格式：
     model_outputs-<input_stem>_<mode>_<lang>_<model_slug>.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python_bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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

mkdir -p "$LOG_DIR" "$PID_DIR"

normalize_input_stem() {
    local input_name stem
    input_name="$(basename "$1")"
    stem="${input_name%.jsonl}"
    stem="${stem%.json}"
    stem="${stem%.mineru}"
    echo "$stem"
}

normalize_model_slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

build_output_path() {
    local mode="$1"
    local lang="$2"
    local input_stem model_slug
    input_stem="$(normalize_input_stem "$INPUT")"
    model_slug="$(normalize_model_slug "$MODEL")"
    echo "${OUTPUT_DIR}/model_outputs-${input_stem}_${mode}_${lang}_${model_slug}.jsonl"
}

run_one() {
    local mode="$1"
    local lang="$2"
    local output_path
    output_path="$(build_output_path "$mode" "$lang")"

    echo "============================================================"
    echo "启动任务: mode=${mode}, lang=${lang}"
    echo "输出文件: ${output_path}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    mkdir -p "$(dirname "$output_path")"

    "$PYTHON_BIN" run_model_tir_inference_concurrent.py \
        --input "$INPUT" \
        --output "$output_path" \
        --api_base "$API_BASE" \
        --model "$MODEL" \
        --mode "$mode" \
        --lang "$lang" \
        --workers "$WORKERS" \
        --limit "$LIMIT" \
        --api_key "$API_KEY"

    echo "完成任务: mode=${mode}, lang=${lang} at $(date '+%Y-%m-%d %H:%M:%S')"
}

if [[ "${RUNNER_MODE:-launcher}" != "worker" ]]; then
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/run_all_mode_lang_${timestamp}.log"
    pid_file="${PID_DIR}/run_all_mode_lang_${timestamp}.pid"

    RUNNER_MODE=worker nohup "$SELF_PATH" "$@" >"$log_file" 2>&1 &
    runner_pid=$!
    echo "$runner_pid" >"$pid_file"

    echo "后台任务已启动"
    echo "PID: $runner_pid"
    echo "日志: $log_file"
    echo "PID 文件: $pid_file"
    echo "执行顺序: cot/en -> cot/zh -> tir/en -> tir/zh"
    exit 0
fi

cd "$SCRIPT_DIR"

echo "后台顺序执行开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $PYTHON_BIN"
echo "Input: $INPUT"
echo "API: $API_BASE"
echo "Model: $MODEL"
echo "Workers: $WORKERS"
echo "Limit: $LIMIT"

run_one cot en
run_one cot zh
run_one tir en
run_one tir zh

echo "全部任务完成: $(date '+%Y-%m-%d %H:%M:%S')"
