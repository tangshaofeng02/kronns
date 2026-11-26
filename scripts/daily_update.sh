#!/bin/bash
# 每日自动更新 finetune_csv/data 下的股票数据
# 工作日15:30后运行（A股收盘后）
# 用法: bash daily_update.sh

cd "$(dirname "$0")/.."
python3 scripts/daily_update.py
