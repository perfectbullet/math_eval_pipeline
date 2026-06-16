#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$(cd "$SCRIPT_DIR" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_DEV_PYTHON="/home/zj/miniconda3/envs/math-eval/bin/python"
DEFAULT_SERVER_PYTHON="/home/zenking/miniconda3/envs/math_verify/bin/python"
RUN_DATE="${RUN_DATE:-$(date '+%Y%m%d')}"
PYTHON_BIN=""
INPUT=""
OUTPUT_DIR=""
API_BASE=""
MODEL=""
WORKERS=""
LIMIT=""
API_KEY=""

LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/run_pids"
COMMAND="start"
FORWARD_ARGS=()

usage() {
    cat <<'EOF'
用法：
  bash run_all_mode_lang_inference.sh [start|status|stop] [options]

可选参数：
  --python_bin PATH   Python 可执行文件，优先级最高
  --input PATH        输入文件（必填）
  --output_dir PATH   输出目录（必填）
  --api_base URL      模型服务地址（必填）
  --model NAME        模型名（必填）
  --workers N         并发数（必填）
  --limit N           限制条数（必填）
  --api_key KEY       API key（必填）
  --help              显示帮助

说明：
  1. 脚本会按固定顺序依次执行 4 条命令：
     cot/en -> cot/zh -> tir/en -> tir/zh
  2. 脚本本身会以后台方式启动，并把日志写入 tir_math/logs/
  3. 输出文件名格式：
     model_outputs-<input_stem>_<mode>_<lang>_<model_slug>.jsonl
  4. 子命令：
     start  后台启动任务（默认）
     status 查看最近后台任务状态
     stop   停止最近后台任务
  5. Python 选择优先级：
     --python_bin > 环境变量 MATH_EVAL_PYTHON > 开发机默认路径 > 服务器默认路径 > python3
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        start|status|stop)
            COMMAND="$1"
            shift
            ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python_bin)
            PYTHON_BIN="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --input)
            INPUT="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --api_base)
            API_BASE="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model)
            MODEL="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            FORWARD_ARGS+=("$1" "$2")
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

latest_pid_file() {
    find "$PID_DIR" -maxdepth 1 -type f -name 'run_all_mode_lang_*.pid' | sort | tail -n 1
}

read_pid_from_file() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        tr -d '[:space:]' <"$pid_file"
    fi
}

is_pid_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_pid_and_children() {
    local pid="$1"
    if ! is_pid_running "$pid"; then
        return 0
    fi

    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2

    if is_pid_running "$pid"; then
        pkill -KILL -P "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
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

show_status() {
    local pid_file pid log_file
    pid_file="$(latest_pid_file)"
    if [[ -z "$pid_file" ]]; then
        echo "没有找到后台任务 PID 文件"
        return 0
    fi

    pid="$(read_pid_from_file "$pid_file")"
    log_file="${LOG_DIR}/$(basename "$pid_file" .pid).log"

    echo "PID 文件: $pid_file"
    echo "日志文件: $log_file"
    echo "PID: ${pid:-<empty>}"

    if is_pid_running "$pid"; then
        echo "状态: 运行中"
    else
        echo "状态: 未运行"
    fi
}

stop_runner() {
    local pid_file pid
    pid_file="$(latest_pid_file)"
    if [[ -z "$pid_file" ]]; then
        echo "没有找到可停止的后台任务"
        return 0
    fi

    pid="$(read_pid_from_file "$pid_file")"
    if [[ -z "$pid" ]]; then
        echo "PID 文件为空: $pid_file"
        return 1
    fi

    if is_pid_running "$pid"; then
        stop_pid_and_children "$pid"
        if is_pid_running "$pid"; then
            echo "停止失败，PID 仍在运行: $pid"
            return 1
        fi
        echo "已停止后台任务，PID: $pid"
    else
        echo "后台任务未运行，PID: $pid"
    fi
}

if [[ "${RUNNER_MODE:-launcher}" != "worker" && "$COMMAND" == "status" ]]; then
    show_status
    exit 0
fi

if [[ "${RUNNER_MODE:-launcher}" != "worker" && "$COMMAND" == "stop" ]]; then
    stop_runner
    exit 0
fi

if [[ "$COMMAND" == "start" || "${RUNNER_MODE:-launcher}" == "worker" ]]; then
    require_arg "--input" "$INPUT"
    require_arg "--output_dir" "$OUTPUT_DIR"
    require_arg "--api_base" "$API_BASE"
    require_arg "--model" "$MODEL"
    require_arg "--workers" "$WORKERS"
    require_arg "--limit" "$LIMIT"
    require_arg "--api_key" "$API_KEY"
fi

if [[ "${RUNNER_MODE:-launcher}" != "worker" ]]; then
    detect_python_bin
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/run_all_mode_lang_${timestamp}.log"
    pid_file="${PID_DIR}/run_all_mode_lang_${timestamp}.pid"

    RUNNER_MODE=worker MATH_EVAL_PYTHON="$PYTHON_BIN" nohup "$SELF_PATH" start "${FORWARD_ARGS[@]}" >"$log_file" 2>&1 &
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
detect_python_bin

echo "后台顺序执行开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $PYTHON_BIN"
echo "Input: $INPUT"
echo "API: $API_BASE"
echo "Model: $MODEL"
echo "Workers: $WORKERS"
echo "Limit: $LIMIT"

run_one tir en
run_one cot en
run_one tir zh
run_one cot zh

echo "全部任务完成: $(date '+%Y-%m-%d %H:%M:%S')"
