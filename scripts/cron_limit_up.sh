#!/bin/bash
# Kronos 首板涨停分析定时任务
# 每周一至周五 16:30 执行：抓取涨停池/炸板池 → 统计分析 → 推荐次日连板
# (16:00 的 daily_predict 任务之后运行, 确保涨停池数据已定型)

PROJECT_DIR="/Users/tangshaofeng/Downloads/my/kronos/kronos"
LOG_DIR="${PROJECT_DIR}/cron_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/limit_up_$(date +%Y-%m-%d).log"

echo "===== 首板涨停分析 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

python scripts/limit_up_analysis.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] limit_up_analysis.py 失败" >> "$LOG_FILE"
    exit 1
fi

# 高连板全方位观测 (依赖上面的涨停池数据)
python scripts/high_board_analysis.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] high_board_analysis.py 失败" >> "$LOG_FILE"
    exit 1
fi

echo "===== 完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

# ===== 每日自检 (一日小检) =====
SELF_CHECK_LOG="${LOG_DIR}/self_check_$(date +%Y-%m-%d).log"
echo "===== 每日自检 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$SELF_CHECK_LOG"

python scripts/self_check/daily_check.py >> "$SELF_CHECK_LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[WARNING] 每日自检发现问题 (退出码 $?)" >> "$SELF_CHECK_LOG"
fi

echo "===== 自检完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$SELF_CHECK_LOG"
