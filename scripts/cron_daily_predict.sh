#!/bin/bash
# Kronos 每日预测定时任务
# 每周一至周五 16:00 执行：更新数据(腾讯源,绕开 baostock) + 验证昨日 + 预测明日

PROJECT_DIR="/Users/tangshaofeng/Downloads/my/kronos/kronos"
LOG_DIR="${PROJECT_DIR}/cron_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d).log"

echo "===== Kronos 每日预测 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

# 直接跑 daily_predict.py 完整流程：内置 update_data() 走腾讯接口，不依赖 baostock
python scripts/daily_predict.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] daily_predict.py 失败" >> "$LOG_FILE"
    exit 1
fi

echo "===== 完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"
