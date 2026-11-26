#!/bin/bash
# Kronos 每月超大检定时任务
# 每月初 10:00 执行：全面审计 + 模型更换 + 代码重构

PROJECT_DIR="/Users/tangshaofeng/Downloads/my/kronos/kronos"
LOG_DIR="${PROJECT_DIR}/cron_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/monthly_check_$(date +%Y-%m).log"

echo "===== 每月超大检 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

python scripts/self_check/monthly_check.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[WARNING] 每月超大检发现问题 (退出码 $EXIT_CODE)" >> "$LOG_FILE"
fi

echo "===== 完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

exit $EXIT_CODE
