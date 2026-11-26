#!/bin/bash
# Kronos 盘中实时监控定时任务
# 每5分钟触发, 仅交易时段(9:30-11:30, 13:00-15:00)运行
# 非交易时段直接退出, 不启动 python

PROJECT_DIR="/Users/tangshaofeng/Downloads/my/kronos/kronos"
LOG_DIR="${PROJECT_DIR}/cron_logs"
mkdir -p "$LOG_DIR"

# 交易时段判断: 周一到周五, 9:30-11:30 或 13:00-15:00
DOW=$(date +%u)   # 1=周一 ... 7=周日
NOW=$(date +%H%M)
NOW=$((10#$NOW))  # 强制十进制, 避免前导0被当八进制
if [ "$DOW" -ge 6 ]; then exit 0; fi
if { [ "$NOW" -ge 930 ] && [ "$NOW" -le 1130 ]; } || { [ "$NOW" -ge 1300 ] && [ "$NOW" -le 1500 ]; }; then
    :
else
    exit 0
fi

LOG_FILE="${LOG_DIR}/realtime_$(date +%Y-%m-%d).log"
echo "===== 实时监控 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

python scripts/realtime_monitor.py --once >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] realtime_monitor.py 失败" >> "$LOG_FILE"
    exit 1
fi

echo "===== 完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"
