#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自检总调度器 — 根据时间自动选择检查类型

调度规则:
  - 每日 17:00: 一日小检 (daily_check)
  - 每周六 10:00: 一周大检 (weekly_check)
  - 每月初 10:00: 一月超大检 (monthly_check)

用法:
  python scripts/self_check/orchestrator.py              # 自动选择检查类型
  python scripts/self_check/orchestrator.py --type daily # 强制指定类型
  python scripts/self_check/orchestrator.py --dry-run    # 只检查不修复
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.self_check.daily_check import DailyCheck
from scripts.self_check.weekly_check import WeeklyCheck
from scripts.self_check.monthly_check import MonthlyCheck


def should_run_weekly() -> bool:
    """判断是否应该运行周检（周六）"""
    return datetime.now().weekday() == 5  # 周六


def should_run_monthly() -> bool:
    """判断是否应该运行月检（每月 1 号）"""
    return datetime.now().day == 1


def select_check_type() -> str:
    """自动选择检查类型"""
    if should_run_monthly():
        return 'monthly'
    elif should_run_weekly():
        return 'weekly'
    else:
        return 'daily'


def run_check(check_type: str, dry_run: bool = False):
    """运行指定类型的检查"""
    print(f"\n{'='*70}")
    print(f"  自检总调度器 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  检查类型: {check_type.upper()}")
    print(f"{'='*70}")

    if check_type == 'daily':
        check = DailyCheck(dry_run=dry_run)
    elif check_type == 'weekly':
        check = WeeklyCheck(dry_run=dry_run)
    elif check_type == 'monthly':
        check = MonthlyCheck(dry_run=dry_run)
    else:
        print(f"未知检查类型: {check_type}")
        return None

    return check.run()


def main():
    parser = argparse.ArgumentParser(description='自检总调度器')
    parser.add_argument('--type', type=str, default=None,
                        choices=['daily', 'weekly', 'monthly'],
                        help='检查类型 (默认自动选择)')
    parser.add_argument('--dry-run', action='store_true', help='只检查不修复')
    args = parser.parse_args()

    # 选择检查类型
    check_type = args.type or select_check_type()

    # 运行检查
    result = run_check(check_type, dry_run=args.dry_run)

    if result and result.get('success'):
        print(f"\n✓ {check_type.upper()} 检查完成")
        return 0
    else:
        print(f"\n✗ {check_type.upper()} 检查失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
