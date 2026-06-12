#!/bin/bash
#
# sync_to_server.sh - 同步本地文件到服务器
# 用法: ./sync_to_server.sh [选项]
#
# 选项:
#   -n, --dry-run    预览模式，显示将要同步的文件但不实际执行
#   -v, --verbose    显示详细输出
#   -d, --delete     删除服务器上存在但本地不存在的文件（谨慎使用）
#

# SERVER="192.168.100.230"
SERVER="192.168.8.233"
REMOTE_DIR="/data/metahuman_work/math_model_deployment/math_eval_pipeline/"
LOCAL_DIR="/home/zj/math_model_deployment/math_eval_pipeline/"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 解析参数
DRY_RUN=""
VERBOSE=""
RSYNC_VERBOSE=""
DELETE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            DRY_RUN="--dry-run"
            echo -e "${YELLOW}[预览模式]${NC} 不会实际同步文件"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            RSYNC_VERBOSE="-v"
            shift
            ;;
        -d|--delete)
            DELETE="--delete"
            echo -e "${RED}[删除模式]${NC} 将删除服务器上不存在于本地的文件"
            shift
            ;;
        *)
            echo "未知选项: $1"
            echo "用法: $0 [-n|--dry-run] [-v|--verbose] [-d|--delete]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== 同步文件到服务器 ===${NC}"
echo "本地目录: $LOCAL_DIR"
echo "远程服务器: $SERVER:$REMOTE_DIR"
echo ""

# 排除项列表
EXCLUDES=(
    "*.jsonl"
    "*.json"
    "results/*"
    "results/prm"
    # model_outputs模型
    "data/model_outputs/"
    # Git 相关
    ".git/"
    ".gitignore"

    # Python 缓存
    "__pycache__/"
    "*.py[cod]"
    "*$py.class"
    ".pytest_cache/"
    ".mypy_cache/"
    "*.so"

    # 虚拟环境
    ".venv/"
    "venv/"
    "env/"
    "phi4-venv/"
    "*-venv/"
    "*.egg-info/"

    # IDE 配置
    ".vscode/"
    ".idea/"
    "*.swp"
    "*.swo"
    "*~"

    # 日志和数据目录（大文件，已存在于服务器）
    "logs/"
    "*.log"

    # 模型文件（大文件，已存在于服务器）
    "models/"
    "*.safetensors"
    "*.bin"
    "*.gguf"

    # Docker 相关（不需要同步）
    "*.pid"

    # 系统文件
    ".DS_Store"
    "Thumbs.db"
    "desktop.ini"

    # 临时文件
    "._____temp/"
    "*.tmp"
    "*.bak"
    "*.swp"

    # Claude 本地记忆
    ".claude/"

    # 符号链接
    "wheels"
)

# 构建 rsync 排除参数
EXCLUDE_ARGS=()
for item in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude="$item")
done

# 构建完整命令
RSYNC_CMD=(rsync -avz --progress)
[[ -n "$DELETE" ]] && RSYNC_CMD+=($DELETE)
[[ -n "$DRY_RUN" ]] && RSYNC_CMD+=($DRY_RUN)
[[ -n "$RSYNC_VERBOSE" ]] && RSYNC_CMD+=($RSYNC_VERBOSE)
RSYNC_CMD+=("${EXCLUDE_ARGS[@]}")
RSYNC_CMD+=("$LOCAL_DIR/" "zenking@$SERVER:$REMOTE_DIR/")

# 执行 rsync
echo -e "${YELLOW}正在同步...${NC}"
echo ""
echo "执行命令:"
echo "  ${RSYNC_CMD[*]}"
echo ""

"${RSYNC_CMD[@]}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 同步完成! ===${NC}"
    if [ -n "$DRY_RUN" ]; then
        echo -e "${YELLOW}注意: 这是预览模式，文件未被实际同步${NC}"
        echo "去掉 -n 参数以执行实际同步"
    fi
else
    echo ""
    echo -e "${RED}=== 同步失败! ===${NC}"
    exit 1
fi
