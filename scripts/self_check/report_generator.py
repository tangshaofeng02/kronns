#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器 — 生成每日/每周/每月检查报告

报告格式: Markdown
存储位置:
  - reports/daily/YYYY-MM-DD.md
  - reports/weekly/YYYY-WW.md
  - reports/monthly/YYYY-MM.md
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')


class ReportGenerator:
    """报告生成器"""

    def __init__(self):
        self.reports_dir = REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)

    def _ensure_dir(self, subdir: str):
        path = os.path.join(self.reports_dir, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def generate_daily_report(self, date_str: str, check_results: Dict[str, Any],
                              changes: List[Dict], metrics: Dict) -> str:
        """生成每日小检报告"""
        output_dir = self._ensure_dir('daily')
        output_file = os.path.join(output_dir, f'{date_str}.md')

        lines = []
        lines.append(f"# 每日小检报告 — {date_str}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 检查摘要
        lines.append("## 检查摘要")
        lines.append("")
        if check_results.get('issues_found'):
            lines.append(f"⚠️ 发现 {len(check_results['issues'])} 个问题")
        else:
            lines.append("✅ 未发现异常")
        lines.append("")

        # 问题列表
        if check_results.get('issues'):
            lines.append("### 发现的问题")
            lines.append("")
            for issue in check_results['issues']:
                lines.append(f"- **{issue['type']}** (级别: {issue.get('level', '未知')}): {issue.get('message', '')}")
            lines.append("")

        # 修复操作
        if changes:
            lines.append("### 执行的修复")
            lines.append("")
            for change in changes:
                lines.append(f"- [{change.get('timestamp', '')}] {change.get('event', '')}: {change.get('details', {})}")
            lines.append("")

        # 关键指标
        lines.append("## 关键指标")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.2%} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

        # 写入文件
        content = '\n'.join(lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_file

    def generate_weekly_report(self, year: int, week: int,
                               weekly_summary: Dict[str, Any],
                               daily_reports: List[str]) -> str:
        """生成每周总结报告"""
        output_dir = self._ensure_dir('weekly')
        output_file = os.path.join(output_dir, f'{year}-W{week:02d}.md')

        lines = []
        lines.append(f"# 每周总结报告 — {year} 年第 {week} 周")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 本周概览
        lines.append("## 本周概览")
        lines.append("")
        lines.append(f"- 检查天数: {weekly_summary.get('total_days', 0)}")
        lines.append(f"- 发现问题: {weekly_summary.get('total_issues', 0)}")
        lines.append(f"- 执行修复: {weekly_summary.get('total_repairs', 0)}")
        lines.append(f"- 回滚次数: {weekly_summary.get('total_rollbacks', 0)}")
        lines.append("")

        # 性能趋势
        lines.append("## 性能趋势")
        lines.append("")
        lines.append(f"| 指标 | 本周 | 上周 | 变化 |")
        lines.append(f"|------|------|------|------|")
        perf = weekly_summary.get('performance', {})
        for key, values in perf.items():
            if isinstance(values, dict):
                curr = values.get('current', 0)
                prev = values.get('previous', 0)
                change = curr - prev
                lines.append(f"| {key} | {curr:.2%} | {prev:.2%} | {change:+.2%} |")
        lines.append("")

        # 变更记录
        lines.append("## 本周变更")
        lines.append("")
        for change in weekly_summary.get('changes', []):
            lines.append(f"- [{change.get('timestamp', '')}] {change.get('event', '')}")
        lines.append("")

        # 下周计划
        lines.append("## 下周计划")
        lines.append("")
        for plan in weekly_summary.get('next_week_plan', []):
            lines.append(f"- {plan}")
        lines.append("")

        content = '\n'.join(lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_file

    def generate_monthly_report(self, year: int, month: int,
                                monthly_summary: Dict[str, Any]) -> str:
        """生成每月总结报告"""
        output_dir = self._ensure_dir('monthly')
        output_file = os.path.join(output_dir, f'{year}-{month:02d}.md')

        lines = []
        lines.append(f"# 每月总结报告 — {year} 年 {month} 月")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 本月概览
        lines.append("## 本月概览")
        lines.append("")
        lines.append(f"- 检查天数: {monthly_summary.get('total_days', 0)}")
        lines.append(f"- 发现问题: {monthly_summary.get('total_issues', 0)}")
        lines.append(f"- 执行修复: {monthly_summary.get('total_repairs', 0)}")
        lines.append(f"- 模型重训练: {monthly_summary.get('model_retrains', 0)}")
        lines.append(f"- 代码重构: {monthly_summary.get('code_refactors', 0)}")
        lines.append("")

        # 性能总结
        lines.append("## 性能总结")
        lines.append("")
        lines.append(f"| 指标 | 月初 | 月末 | 变化 |")
        lines.append(f"|------|------|------|------|")
        perf = monthly_summary.get('performance', {})
        for key, values in perf.items():
            if isinstance(values, dict):
                start = values.get('start', 0)
                end = values.get('end', 0)
                change = end - start
                lines.append(f"| {key} | {start:.2%} | {end:.2%} | {change:+.2%} |")
        lines.append("")

        # 重大变更
        lines.append("## 重大变更")
        lines.append("")
        for change in monthly_summary.get('major_changes', []):
            lines.append(f"- [{change.get('timestamp', '')}] {change.get('description', '')}")
        lines.append("")

        # 下月规划
        lines.append("## 下月规划")
        lines.append("")
        for plan in monthly_summary.get('next_month_plan', []):
            lines.append(f"- {plan}")
        lines.append("")

        content = '\n'.join(lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_file


if __name__ == '__main__':
    gen = ReportGenerator()
    print("报告生成器测试")
    print(f"报告目录: {gen.reports_dir}")
