#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARK="gaokao_bench"
MODEL="DeepSeek-R1-Distill-Llama-70B"
LIMIT="1000"
PYTHON_BIN="${MATH_EVAL_PYTHON:-python}"
PRM_MODEL="/nfs-data/math_model_deployment/math_eval_pipeline/models/Qwen2.5-Math-PRM-7B"
CACHE_DIR="/data/metahuman_work/.hf_cache"
INPUT_DIR=""
VERIFY_DIR=""
PRM_DIR=""
REPORT_DIR=""
DRY_RUN=0
SKIP_EXISTING=0
NO_EXPORT=0
RUN_DATE="${RUN_DATE:-$(date '+%Y%m%d')}"
CASES=("tir_en" "cot_en" "cot_zh" "tir_zh")

usage() {
    cat <<'EOF'
用法：
  bash run_gaokao_verify_prm_pipeline.sh [options]

功能：
  1. 依次对 4 组输出执行 run_math_verify.py
  2. 依次对 verify 输出执行 run_qwen_prm.py
  3. 最后用 export_verify_prm_metrics.py 汇总 CSV/JSON

默认处理 case：
  tir_en, cot_en, cot_zh, tir_zh

常用参数：
  --model NAME          推理模型名，默认 DeepSeek-R1-Distill-Llama-70B
                        会自动转成小写 slug 匹配文件名，例如 deepseek-r1-distill-llama-70b
  --benchmark NAME      benchmark 名，默认 gaokao_bench
  --limit N             传给 run_math_verify.py 的 --limit，默认 1000
  --python-bin PATH     Python 可执行文件，默认使用环境变量 MATH_EVAL_PYTHON，否则 python
  --input-dir PATH      模型输出目录，默认 ../data/model_outputs/<benchmark>
  --verify-dir PATH     verify 输出目录，默认 ../results/verify/<benchmark>
  --prm-dir PATH        PRM 输出目录，默认 ../results/prm/<benchmark>
  --report-dir PATH     汇总报告目录，默认 ../reports/<benchmark>
  --prm-model PATH      Qwen2.5-Math-PRM-7B 模型目录
  --cache-dir PATH      HuggingFace cache_dir
  --cases LIST          逗号分隔 case，例如 tir_en,cot_en,cot_zh,tir_zh
  --dry-run             只打印命令，不执行
  --skip-existing       输出文件已存在且非空时跳过对应步骤
  --no-export           不执行第三步汇总
  -h, --help            显示帮助

输入文件匹配说明：
  脚本优先匹配 run_all_mode_lang_inference.sh 生成的文件名：
    ../data/model_outputs/gaokao_bench/model_outputs-gaokao_bench_<日期>_<case>_<model_slug>.jsonl

  同时兼容简单文件名：
    ../data/model_outputs/gaokao_bench/<case>_<model_slug>.jsonl

日期处理说明：
  不提供 --run-tag 参数。脚本会从实际 input 文件名中保留日期。
  如果 input 文件名不带日期，最终汇总文件会使用 RUN_DATE 环境变量；未设置时使用当天 YYYYMMDD。
EOF
}

normalize_model_slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

parse_cases() {
    local text="$1"
    IFS=',' read -r -a CASES <<< "$text"
    local case_name
    for case_name in "${CASES[@]}"; do
        case_name="$(echo "$case_name" | xargs)"
        case "$case_name" in
            cot_en|cot_zh|tir_en|tir_zh) ;;
            *)
                echo "错误：非法 case: $case_name，可选 cot_en,cot_zh,tir_en,tir_zh" >&2
                exit 1
                ;;
        esac
    done
}

quote_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

run_cmd() {
    echo
    echo "+ $(quote_cmd "$@")"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

strip_input_to_exp_key() {
    local path="$1"
    local name stem
    name="$(basename "$path")"
    stem="${name%.jsonl}"
    stem="${stem#model_outputs-}"
    echo "$stem"
}

report_key_from_exp_key() {
    local exp_key="$1"
    local case_name="$2"
    local rest prefix

    if [[ "$exp_key" == "${case_name}_"* ]]; then
        rest="${exp_key#${case_name}_}"
        echo "${BENCHMARK}_${RUN_DATE}_${rest}"
        return 0
    fi

    if [[ "$exp_key" == *"_${case_name}_"* ]]; then
        prefix="${exp_key%%_${case_name}_*}"
        rest="${exp_key#*_${case_name}_}"
        echo "${prefix}_${rest}"
        return 0
    fi

    echo "${BENCHMARK}_${RUN_DATE}_${MODEL_SLUG}"
}

unique_sorted_paths() {
    awk '!seen[$0]++' | sort
}

find_input_for_case() {
    local case_name="$1"
    local -a candidates=()
    local -a sorted=()
    local path

    shopt -s nullglob
    for path in \
        "${INPUT_DIR}/model_outputs-${BENCHMARK}_"*"_${case_name}_${MODEL_SLUG}.jsonl" \
        "${INPUT_DIR}/model_outputs-"*"_${case_name}_${MODEL_SLUG}.jsonl" \
        "${INPUT_DIR}/${case_name}_${MODEL_SLUG}.jsonl" \
        "${INPUT_DIR}/model_outputs-${case_name}_${MODEL_SLUG}.jsonl"
    do
        [[ -f "$path" ]] && candidates+=("$path")
    done
    shopt -u nullglob

    if [[ "${#candidates[@]}" -eq 0 ]]; then
        echo "错误：未找到 case=${case_name}, model_slug=${MODEL_SLUG} 的 input 文件" >&2
        echo "查找目录：${INPUT_DIR}" >&2
        echo "期望示例：${INPUT_DIR}/model_outputs-${BENCHMARK}_${RUN_DATE}_${case_name}_${MODEL_SLUG}.jsonl" >&2
        echo "或：${INPUT_DIR}/${case_name}_${MODEL_SLUG}.jsonl" >&2
        exit 1
    fi

    mapfile -t sorted < <(printf '%s\n' "${candidates[@]}" | unique_sorted_paths)

    if [[ "${#sorted[@]}" -gt 1 ]]; then
        echo "[WARN] case=${case_name} 找到多个 input，默认使用排序最后一个：" >&2
        printf '       - %s\n' "${sorted[@]}" >&2
    fi

    echo "${sorted[-1]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --python-bin|--python_bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --verify-dir)
            VERIFY_DIR="$2"
            shift 2
            ;;
        --prm-dir)
            PRM_DIR="$2"
            shift 2
            ;;
        --report-dir|--reports-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        --prm-model)
            PRM_MODEL="$2"
            shift 2
            ;;
        --cache-dir|--cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --cases)
            parse_cases "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --no-export)
            NO_EXPORT=1
            shift
            ;;
        -h|--help)
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

MODEL_SLUG="$(normalize_model_slug "$MODEL")"
INPUT_DIR="${INPUT_DIR:-../data/model_outputs/${BENCHMARK}}"
VERIFY_DIR="${VERIFY_DIR:-../results/verify/${BENCHMARK}}"
PRM_DIR="${PRM_DIR:-../results/prm/${BENCHMARK}}"
REPORT_DIR="${REPORT_DIR:-../reports/${BENCHMARK}}"

mkdir -p "$VERIFY_DIR" "$PRM_DIR" "$REPORT_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "错误：input 目录不存在：$INPUT_DIR" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误：Python 不存在或不可执行：$PYTHON_BIN" >&2
    echo "可以使用 --python-bin PATH 或设置 MATH_EVAL_PYTHON" >&2
    exit 1
fi

declare -A INPUT_PATHS
declare -A EXP_KEYS
declare -A VERIFY_PATHS
declare -A VERIFY_SUMMARY_PATHS
declare -A PRM_PATHS
declare -A PRM_SUMMARY_PATHS
declare -A REPORT_KEYS

for case_name in "${CASES[@]}"; do
    input_path="$(find_input_for_case "$case_name")"
    exp_key="$(strip_input_to_exp_key "$input_path")"
    report_key="$(report_key_from_exp_key "$exp_key" "$case_name")"

    INPUT_PATHS["$case_name"]="$input_path"
    EXP_KEYS["$case_name"]="$exp_key"
    VERIFY_PATHS["$case_name"]="${VERIFY_DIR}/verify_${exp_key}.jsonl"
    VERIFY_SUMMARY_PATHS["$case_name"]="${VERIFY_DIR}/verify_summary_${exp_key}.md"
    PRM_PATHS["$case_name"]="${PRM_DIR}/prm_step_${exp_key}.jsonl"
    PRM_SUMMARY_PATHS["$case_name"]="${PRM_DIR}/prm_summary-${exp_key}.md"
    REPORT_KEYS["$case_name"]="$report_key"
done

COMMON_REPORT_KEY="${REPORT_KEYS[${CASES[0]}]}"
for case_name in "${CASES[@]}"; do
    if [[ "${REPORT_KEYS[$case_name]}" != "$COMMON_REPORT_KEY" ]]; then
        echo "错误：4 组 input 推导出的 report key 不一致，可能混用了不同日期或不同模型。" >&2
        for item in "${CASES[@]}"; do
            echo "  ${item}: input=${INPUT_PATHS[$item]} report_key=${REPORT_KEYS[$item]}" >&2
        done
        exit 1
    fi
done

OUTPUT_CSV="${REPORT_DIR}/verify_prm_metrics-${COMMON_REPORT_KEY}.csv"
OUTPUT_JSON="${REPORT_DIR}/verify_prm_metrics-${COMMON_REPORT_KEY}.json"

echo "================================================================================"
echo "Gaokao Verify + PRM pipeline"
echo "scripts dir : $SCRIPT_DIR"
echo "benchmark   : $BENCHMARK"
echo "model       : $MODEL"
echo "model slug  : $MODEL_SLUG"
echo "run date    : $RUN_DATE"
echo "cases       : ${CASES[*]}"
echo "input dir   : $INPUT_DIR"
echo "verify dir  : $VERIFY_DIR"
echo "prm dir     : $PRM_DIR"
echo "report dir  : $REPORT_DIR"
echo "report key  : $COMMON_REPORT_KEY"
echo "dry run     : $DRY_RUN"
echo "================================================================================"

for case_name in "${CASES[@]}"; do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "case        : $case_name"
    echo "input       : ${INPUT_PATHS[$case_name]}"
    echo "exp_key     : ${EXP_KEYS[$case_name]}"
    echo "verify out  : ${VERIFY_PATHS[$case_name]}"
    echo "prm out     : ${PRM_PATHS[$case_name]}"
    echo "--------------------------------------------------------------------------------"

    if [[ "$SKIP_EXISTING" -eq 1 && -s "${VERIFY_PATHS[$case_name]}" ]]; then
        echo "[SKIP] Math-Verify 已存在：${VERIFY_PATHS[$case_name]}"
    else
        run_cmd "$PYTHON_BIN" run_math_verify.py \
            --input "${INPUT_PATHS[$case_name]}" \
            --output "${VERIFY_PATHS[$case_name]}" \
            --summary "${VERIFY_SUMMARY_PATHS[$case_name]}" \
            --limit "$LIMIT"
    fi

    if [[ "$SKIP_EXISTING" -eq 1 && -s "${PRM_PATHS[$case_name]}" ]]; then
        echo "[SKIP] PRM 已存在：${PRM_PATHS[$case_name]}"
    else
        run_cmd "$PYTHON_BIN" run_qwen_prm.py \
            --input "${VERIFY_PATHS[$case_name]}" \
            --output "${PRM_PATHS[$case_name]}" \
            --summary "${PRM_SUMMARY_PATHS[$case_name]}" \
            --model "$PRM_MODEL" \
            --cache_dir "$CACHE_DIR"
    fi
done

if [[ "$NO_EXPORT" -eq 1 ]]; then
    echo
    echo "[DONE] 已跳过第三步汇总。"
    exit 0
fi

EXPORT_CMD=("$PYTHON_BIN" export_verify_prm_metrics.py)
for case_name in "${CASES[@]}"; do
    EXPORT_CMD+=(--case "$case_name" "${VERIFY_PATHS[$case_name]}" "${PRM_PATHS[$case_name]}")
done
EXPORT_CMD+=(--output_csv "$OUTPUT_CSV" --output_json "$OUTPUT_JSON")

run_cmd "${EXPORT_CMD[@]}"

echo
echo "[DONE] 汇总完成："
echo "CSV : $OUTPUT_CSV"
echo "JSON: $OUTPUT_JSON"
