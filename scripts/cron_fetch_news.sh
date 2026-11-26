#!/bin/bash
# 新闻情绪抓取定时任务 (每交易日 9:00 / 12:30)
# 全部股票抓最近3天新闻 + FinBERT打分, 存档 news_log/

PROJECT_DIR="/Users/tangshaofeng/Downloads/my/kronos/kronos"
LOG_DIR="${PROJECT_DIR}/news_log"
mkdir -p "$LOG_DIR"

echo "===== 新闻抓取 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "${LOG_DIR}/cron.log"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

python scripts/news_sentiment.py --all >> "${LOG_DIR}/cron.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] news_sentiment.py --all 失败" >> "${LOG_DIR}/cron.log"
    exit 1
fi

echo "===== 完成 $(date '+%Y-%m-%d %H:%M:%S') =====" >> "${LOG_DIR}/cron.log"
