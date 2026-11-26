#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变更日志 — 记录所有自动修复操作

日志格式: JSON Lines (每行一个 JSON 对象)
存储位置: logs/self_repair_log.jsonl

功能:
  - 记录每次检查、修复、回滚操作
  - 支持按时间、类型、状态查询
  - 生成变更历史报告
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_DIR, 'logs', 'self_repair_log.jsonl')


class ChangeLog:
    """变更日志管理器"""

    def __init__(self):
        self.log_file = LOG_FILE
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def append(self, event: str, check_type: str = 'unknown',
               level: str = 'info', details: Dict[str, Any] = None):
        """追加一条日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'check_type': check_type,
            'event': event,
            'level': level,
            'details': details or {},
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def query(self, days: int = 7, check_type: str = None,
              event: str = None) -> List[Dict[str, Any]]:
        """查询日志"""
        if not os.path.exists(self.log_file):
            return []

        cutoff = datetime.now() - timedelta(days=days)
        results = []

        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time < cutoff:
                        continue
                    if check_type and entry.get('check_type') != check_type:
                        continue
                    if event and entry.get('event') != event:
                        continue
                    results.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue

        return results

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """获取变更摘要"""
        logs = self.query(days=days)

        summary = {
            'total_entries': len(logs),
            'by_type': {},
            'by_level': {},
            'repairs': [],
            'rollbacks': [],
            'errors': [],
        }

        for entry in logs:
            # 按类型统计
            ct = entry.get('check_type', 'unknown')
            summary['by_type'][ct] = summary['by_type'].get(ct, 0) + 1

            # 按级别统计
            level = entry.get('level', 'info')
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1

            # 关键事件
            if entry.get('event') == 'change_committed':
                summary['repairs'].append({
                    'timestamp': entry['timestamp'],
                    'type': ct,
                    'details': entry.get('details', {}),
                })
            elif entry.get('event') == 'rollback_complete':
                summary['rollbacks'].append({
                    'timestamp': entry['timestamp'],
                    'type': ct,
                    'reason': entry.get('details', {}).get('reason', ''),
                })
            elif level == 'error':
                summary['errors'].append({
                    'timestamp': entry['timestamp'],
                    'type': ct,
                    'event': entry.get('event'),
                    'details': entry.get('details', {}),
                })

        return summary

    def get_recent_changes(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近 N 条变更"""
        logs = self.query(days=30)
        return logs[-n:] if len(logs) > n else logs


if __name__ == '__main__':
    log = ChangeLog()

    # 测试写入
    log.append('test_event', 'test', 'info', {'test': True})
    print("✓ 日志写入成功")

    # 测试查询
    logs = log.query(days=1)
    print(f"✓ 查询到 {len(logs)} 条日志")

    # 测试摘要
    summary = log.get_summary(days=1)
    print(f"✓ 摘要: {summary['total_entries']} 条记录")
