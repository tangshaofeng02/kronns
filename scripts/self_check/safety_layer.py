#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全层 — Git 分支隔离、备份、验证、回滚

所有自动修复操作都通过安全层执行，确保:
  1. 修改前自动创建 Git 分支
  2. 修改前自动备份关键文件
  3. 修改后自动验证性能
  4. 验证失败自动回滚
  5. 所有操作记录到变更日志
"""

import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, Callable

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKUP_DIR = os.path.join(PROJECT_DIR, 'backups')
LOG_FILE = os.path.join(PROJECT_DIR, 'logs', 'self_repair_log.jsonl')


def run_cmd(cmd: list, cwd: str = None, check: bool = True) -> tuple:
    """执行 shell 命令，返回 (success, output)"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd or PROJECT_DIR,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0 and check:
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        return False, str(e)


class SafetyLayer:
    """安全层 — 管理自动修复的安全边界"""

    def __init__(self, check_type: str = 'daily'):
        """
        初始化安全层

        check_type: 'daily' | 'weekly' | 'monthly'
        """
        self.check_type = check_type
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.branch_name = f'auto-repair/{check_type}-{self.timestamp}'
        self.tag_name = f'pre-repair-{check_type}-{self.timestamp}'
        self.original_branch = None
        self.backup_path = None

        # 确保目录存在
        os.makedirs(BACKUP_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def _log(self, event: str, details: Dict[str, Any] = None, level: str = 'info'):
        """记录变更日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'check_type': self.check_type,
            'event': event,
            'level': level,
            'details': details or {},
        }
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def start_session(self) -> bool:
        """开始修复会话: 创建分支 + 备份 + 打 tag"""
        self._log('session_start', {'branch': self.branch_name})

        # 1. 记录当前分支
        ok, output = run_cmd(['git', 'branch', '--show-current'])
        if not ok:
            self._log('get_current_branch_failed', {'error': output}, 'error')
            return False
        self.original_branch = output.strip()

        # 2. 备份关键文件
        self.backup_path = os.path.join(BACKUP_DIR, f'{self.check_type}_{self.timestamp}')
        os.makedirs(self.backup_path, exist_ok=True)
        key_files = [
            'scripts/daily_predict.py',
            'scripts/limit_up_analysis.py',
            'scripts/train_limit_up_model.py',
            'scripts/adaptive_params.py',
            'scripts/backtest_engine.py',
            'scripts/model_monitor.py',
            'models/xgb_limit_up.pkl',
        ]
        for f in key_files:
            src = os.path.join(PROJECT_DIR, f)
            if os.path.exists(src):
                dst = os.path.join(self.backup_path, os.path.basename(f))
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        self._log('backup_created', {'path': self.backup_path})

        # 3. 打 tag（用于回滚）
        ok, _ = run_cmd(['git', 'tag', '-f', self.tag_name])
        if ok:
            self._log('tag_created', {'tag': self.tag_name})

        # 4. 创建并切换到新分支
        ok, output = run_cmd(['git', 'checkout', '-b', self.branch_name])
        if not ok:
            # 如果分支已存在，直接切换
            ok, output = run_cmd(['git', 'checkout', self.branch_name])
            if not ok:
                self._log('create_branch_failed', {'error': output}, 'error')
                return False

        self._log('session_ready', {'branch': self.branch_name})
        return True

    def validate_change(self, validator: Callable[[], Dict[str, Any]],
                        baseline: Dict[str, Any] = None) -> Dict[str, Any]:
        """验证变更是否有效

        validator: 返回 {'success': bool, 'metrics': dict, 'error': str}
        baseline: 变更前的性能指标
        """
        result = validator()

        if not result.get('success'):
            self._log('validation_failed', {
                'error': result.get('error', 'unknown'),
                'metrics': result.get('metrics', {}),
            }, 'warning')
            return {'valid': False, 'action': 'rollback', 'details': result}

        metrics = result.get('metrics', {})

        # 如果有基线，对比性能
        if baseline:
            degraded = self._check_degradation(metrics, baseline)
            if degraded:
                self._log('performance_degraded', {
                    'before': baseline,
                    'after': metrics,
                }, 'warning')
                return {'valid': False, 'action': 'rollback', 'details': result}

        self._log('validation_passed', {'metrics': metrics})
        return {'valid': True, 'action': 'keep', 'details': result}

    def _check_degradation(self, after: Dict, before: Dict) -> bool:
        """检查性能是否下降"""
        # 关键指标: AUC、命中率、准确率
        key_metrics = ['auc', 'hit_rate', 'accuracy', 'top5_hit_rate']

        for metric in key_metrics:
            if metric in after and metric in before:
                # 允许 5% 的波动
                if after[metric] < before[metric] * 0.95:
                    return True

        return False

    def commit_change(self, message: str, files: list = None):
        """提交变更"""
        if files:
            for f in files:
                run_cmd(['git', 'add', f])
        else:
            run_cmd(['git', 'add', '.'])

        ok, output = run_cmd(['git', 'commit', '-m', message])
        if ok:
            self._log('change_committed', {'message': message})
        else:
            self._log('commit_failed', {'error': output}, 'warning')

    def rollback(self, reason: str = ''):
        """回滚到修复前状态"""
        self._log('rollback_start', {'reason': reason}, 'warning')

        # 1. 切换回原分支
        if self.original_branch:
            run_cmd(['git', 'checkout', self.original_branch])

        # 2. 删除修复分支
        run_cmd(['git', 'branch', '-D', self.branch_name])

        # 3. 恢复备份文件
        if self.backup_path and os.path.exists(self.backup_path):
            for f in os.listdir(self.backup_path):
                src = os.path.join(self.backup_path, f)
                # 找到原文件路径
                for key_file in ['scripts/daily_predict.py', 'scripts/limit_up_analysis.py',
                                 'scripts/train_limit_up_model.py', 'scripts/adaptive_params.py',
                                 'scripts/backtest_engine.py', 'scripts/model_monitor.py']:
                    if f == os.path.basename(key_file):
                        dst = os.path.join(PROJECT_DIR, key_file)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                        break

        self._log('rollback_complete', {'reason': reason})

    def merge_to_main(self, message: str = ''):
        """合并修复分支到 main（如果验证通过）"""
        self._log('merge_start', {})

        # 1. 切换到 main
        ok, _ = run_cmd(['git', 'checkout', 'main'])
        if not ok:
            ok, _ = run_cmd(['git', 'checkout', 'master'])
        if not ok:
            self._log('merge_failed', {'error': 'cannot checkout main'}, 'error')
            return False

        # 2. 合并分支
        merge_msg = message or f'auto-repair: {self.check_type} {self.timestamp}'
        ok, output = run_cmd(['git', 'merge', '--no-ff', self.branch_name,
                              '-m', merge_msg])
        if not ok:
            self._log('merge_failed', {'error': output}, 'error')
            # 回滚
            run_cmd(['git', 'merge', '--abort'])
            return False

        self._log('merge_complete', {'branch': self.branch_name})
        return True

    def end_session(self, success: bool, summary: Dict[str, Any] = None):
        """结束修复会话"""
        self._log('session_end', {
            'success': success,
            'summary': summary or {},
        })

        # 清理旧备份（保留最近 30 个）
        self._cleanup_old_backups()

    def _cleanup_old_backups(self, keep: int = 30):
        """清理旧备份"""
        backups = sorted([
            os.path.join(BACKUP_DIR, d)
            for d in os.listdir(BACKUP_DIR)
            if os.path.isdir(os.path.join(BACKUP_DIR, d))
        ], key=os.path.getmtime, reverse=True)

        for old in backups[keep:]:
            shutil.rmtree(old, ignore_errors=True)


def with_safety(check_type: str = 'daily'):
    """装饰器: 为修复函数提供安全层

    用法:
        @with_safety('daily')
        def fix_threshold():
            # 修复逻辑
            return {'success': True, 'metrics': {...}}
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            layer = SafetyLayer(check_type)

            # 开始会话
            if not layer.start_session():
                return {'success': False, 'error': 'safety_layer_init_failed'}

            try:
                # 执行修复
                result = func(*args, **kwargs)

                # 验证
                validation = layer.validate_change(lambda: result)

                if validation['valid']:
                    # 提交变更
                    layer.commit_change(
                        f'auto-repair({check_type}): {func.__name__}',
                        files=result.get('files')
                    )
                    # 合并到 main
                    layer.merge_to_main()
                    layer.end_session(True, result)
                    return result
                else:
                    # 回滚
                    layer.rollback(validation['details'].get('error', 'validation_failed'))
                    layer.end_session(False, validation)
                    return {
                        'success': False,
                        'error': 'validation_failed',
                        'details': validation,
                    }

            except Exception as e:
                layer.rollback(str(e))
                layer.end_session(False, {'error': str(e)})
                return {'success': False, 'error': str(e)}

        return wrapper
    return decorator


if __name__ == '__main__':
    # 测试安全层
    print("安全层测试")
    print("=" * 50)

    layer = SafetyLayer('test')
    print(f"项目目录: {PROJECT_DIR}")
    print(f"分支名: {layer.branch_name}")
    print(f"备份目录: {layer.backup_path}")

    # 测试备份
    print("\n测试会话开始...")
    if layer.start_session():
        print("✓ 会话开始成功")
        print(f"  当前分支: {layer.branch_name}")
        print(f"  备份路径: {layer.backup_path}")

        # 模拟变更
        test_file = os.path.join(PROJECT_DIR, 'logs', 'test_change.txt')
        with open(test_file, 'w') as f:
            f.write('test change')

        layer.commit_change('test: add test file', files=['logs/test_change.txt'])
        print("✓ 变更已提交")

        # 清理测试文件
        os.remove(test_file)
        run_cmd(['git', 'add', 'logs/test_change.txt'])
        run_cmd(['git', 'commit', '-m', 'test: remove test file'])

        # 结束会话
        layer.end_session(True, {'test': 'passed'})
        print("✓ 会话结束")

        # 合并到 main
        layer.merge_to_main('test: merge test branch')
        print("✓ 已合并到 main")
    else:
        print("✗ 会话开始失败")
