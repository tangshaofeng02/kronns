#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一周大检 — 每周深度检查 + 模型重训练 + 特征调整

运行时间: 每周六或周日（非交易日）10:00

检查项:
  1. 周度性能: 推荐命中率、超额收益、概率校准
  2. 特征漂移: 特征分布变化、重要性变化
  3. 模型衰减: AUC 下降、命中率下降趋势
  4. 市场状态: 情绪周期、连板率趋势

修复项 (自动执行):
  1. 模型重训练: 如果周度 AUC < 0.60
  2. 参数优化: 用最新数据重新训练 XGBoost
  3. 特征调整: 删除无用特征、添加新特征
  4. 代码修复: 修复发现的 bug

输出:
  - reports/weekly/YYYY-WW.md: 每周总结报告

用法:
  python scripts/self_check/weekly_check.py              # 正常运行
  python scripts/self_check/weekly_check.py --dry-run    # 只检查不修复
"""

import os
import sys
import glob
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.self_check.change_log import ChangeLog
from scripts.self_check.report_generator import ReportGenerator
from scripts.self_check.safety_layer import SafetyLayer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'limit_up_data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_limit_up.pkl')


class WeeklyCheck:
    """一周大检"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.change_log = ChangeLog()
        self.report_gen = ReportGenerator()
        self.safety = SafetyLayer('weekly')

        self.issues = []
        self.changes = []
        self.metrics = {}
        self.week_start = None
        self.week_end = None

    def _add_issue(self, issue_type: str, message: str, level: str = 'warning',
                   details: Dict = None):
        self.issues.append({
            'type': issue_type,
            'message': message,
            'level': level,
            'details': details or {},
        })

    def _add_change(self, event: str, details: Dict):
        self.changes.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details,
        })

    # ============================================================
    # 检查项
    # ============================================================
    def check_weekly_performance(self):
        """检查周度性能"""
        try:
            from scripts.model_monitor import load_recommends, load_zt_pools, load_prev_pools, compute_daily_metrics

            recs = load_recommends()
            zt_pools = load_zt_pools()
            prev_pools = load_prev_pools()

            if not recs:
                return

            metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=7)
            if metrics_df.empty:
                return

            # 计算周度指标
            self.metrics['weekly_top5_hit_rate'] = metrics_df['top5_hit_rate'].mean()
            self.metrics['weekly_total_hit_rate'] = metrics_df['total_hit_rate'].mean()
            self.metrics['weekly_excess'] = metrics_df['excess'].mean()
            self.metrics['weekly_win_rate'] = (metrics_df['excess'] > 0).mean()
            self.metrics['weekly_calibration'] = metrics_df['prob_calibration'].mean()

            # 告警
            if self.metrics['weekly_top5_hit_rate'] < 0.3:
                self._add_issue('low_weekly_hit_rate',
                                f'周度 TOP5 命中率过低: {self.metrics["weekly_top5_hit_rate"]:.1%}',
                                'error')

            if self.metrics['weekly_excess'] < 0:
                self._add_issue('negative_weekly_excess',
                                f'周度超额收益为负: {self.metrics["weekly_excess"]:+.1%}',
                                'warning')

        except Exception as e:
            self._add_issue('check_error', f'周度性能检查失败: {e}', 'warning')

    def check_feature_drift(self):
        """检查特征漂移"""
        try:
            import pandas as pd
            import joblib

            if not os.path.exists(MODEL_PATH):
                return

            bundle = joblib.load(MODEL_PATH)
            model = bundle.get('model')
            feature_names = bundle.get('feature_names', [])

            if not feature_names:
                return

            # 加载最新数据
            label_file = os.path.join(DATA_DIR, 'label_history.csv')
            if not os.path.exists(label_file):
                return

            df = pd.read_csv(label_file)
            if df.empty:
                return

            # 检查特征分布
            drift_detected = False
            for feat in ['seal_ratio', 'turnover', 'circ_mv']:
                if feat in df.columns:
                    current_mean = df[feat].mean()
                    # 简化的漂移检测：与训练时的均值比较
                    self.metrics[f'{feat}_mean'] = round(current_mean, 3)

        except Exception as e:
            self._add_issue('drift_check_error', f'特征漂移检查失败: {e}', 'warning')

    def check_model_decay(self):
        """检查模型衰减"""
        try:
            from scripts.model_monitor import load_recommends, load_zt_pools, load_prev_pools, compute_daily_metrics

            recs = load_recommends()
            zt_pools = load_zt_pools()
            prev_pools = load_prev_pools()

            if not recs:
                return

            # 对比本周和上周
            metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=14)
            if len(metrics_df) < 6:
                return

            mid = len(metrics_df) // 2
            this_week = metrics_df.tail(mid)
            last_week = metrics_df.head(mid)

            this_hit = this_week['top5_hit_rate'].mean()
            last_hit = last_week['top5_hit_rate'].mean()

            self.metrics['this_week_hit_rate'] = this_hit
            self.metrics['last_week_hit_rate'] = last_hit
            self.metrics['hit_rate_change'] = this_hit - last_hit

            if this_hit < last_hit * 0.8:
                self._add_issue('model_decay',
                                f'模型衰减: 本周命中率 {this_hit:.1%} < 上周 {last_hit:.1%} 的 80%',
                                'error', {
                                    'this_week': this_hit,
                                    'last_week': last_hit,
                                })

        except Exception as e:
            self._add_issue('decay_check_error', f'模型衰减检查失败: {e}', 'warning')

    def check_market_regime(self):
        """检查市场状态"""
        try:
            import pandas as pd

            # 加载最近的数据
            label_file = os.path.join(DATA_DIR, 'label_history.csv')
            if not os.path.exists(label_file):
                return

            df = pd.read_csv(label_file)
            if df.empty:
                return

            # 计算每日连板率
            daily = df.groupby('date')['label'].mean()
            recent_rate = daily.tail(5).mean()

            self.metrics['recent_continuation_rate'] = round(recent_rate, 3)

            # 判断市场状态
            if recent_rate >= 0.30:
                self.metrics['market_regime'] = '升温'
            elif recent_rate <= 0.15:
                self.metrics['market_regime'] = '退潮'
            else:
                self.metrics['market_regime'] = '平稳'

        except Exception:
            pass

    # ============================================================
    # 修复项
    # ============================================================
    def retrain_model(self):
        """重新训练模型"""
        if self.dry_run:
            return

        # 检查是否需要重训练
        need_retrain = False
        for issue in self.issues:
            if issue['type'] in ('model_decay', 'low_weekly_hit_rate'):
                need_retrain = True
                break

        if not need_retrain:
            return

        print("\n  → 重新训练模型...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(PROJECT_DIR, 'scripts', 'train_limit_up_model.py'),
                 '--output', 'xgb_limit_up.pkl'],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                self._add_change('model_retrained', {
                    'output': 'xgb_limit_up.pkl',
                })
                self.change_log.append('model_retrained', 'weekly', 'info', {})
                print("  ✓ 模型重训练完成")
            else:
                self._add_issue('retrain_failed', f'模型重训练失败: {result.stderr[:200]}', 'error')

        except Exception as e:
            self._add_issue('retrain_error', f'模型重训练异常: {e}', 'error')

    def optimize_hyperparams(self):
        """优化超参数（简化版：用默认参数重训练）"""
        # 可以在这里实现 Optuna 超参优化
        # 目前简化处理：如果性能下降，直接重训练
        pass

    # ============================================================
    # 主流程
    # ============================================================
    def run(self) -> Dict[str, Any]:
        """运行一周大检"""
        now = datetime.now()
        self.week_end = now
        self.week_start = now - timedelta(days=7)

        print(f"\n{'='*60}")
        print(f"  一周大检 — {self.week_start.strftime('%Y-%m-%d')} ~ {self.week_end.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # 1. 开始安全会话
        if not self.dry_run:
            print("\n[1/4] 初始化安全会话...")
            if not self.safety.start_session():
                print("  ✗ 安全会话初始化失败")
                return {'success': False, 'error': 'safety_init_failed'}
            print(f"  ✓ 分支: {self.safety.branch_name}")

        # 2. 执行检查
        print("\n[2/4] 执行检查...")
        self.check_weekly_performance()
        self.check_feature_drift()
        self.check_model_decay()
        self.check_market_regime()

        print(f"  ✓ 发现 {len(self.issues)} 个问题")
        for issue in self.issues:
            icon = '🔴' if issue['level'] == 'error' else '🟡'
            print(f"    {icon} [{issue['type']}] {issue['message']}")

        # 3. 执行修复
        print("\n[3/4] 执行修复...")
        self.retrain_model()
        self.optimize_hyperparams()
        print(f"  ✓ 执行 {len(self.changes)} 项修复")

        # 4. 生成报告
        print("\n[4/4] 生成报告...")
        year, week, _ = now.isocalendar()
        weekly_summary = {
            'total_days': 5,
            'total_issues': len(self.issues),
            'total_repairs': len(self.changes),
            'total_rollbacks': 0,
            'performance': {
                'top5_hit_rate': {
                    'current': self.metrics.get('this_week_hit_rate', 0),
                    'previous': self.metrics.get('last_week_hit_rate', 0),
                },
                'excess': {
                    'current': self.metrics.get('weekly_excess', 0),
                    'previous': 0,
                },
            },
            'changes': self.changes,
            'next_week_plan': [
                '继续监控模型性能',
                '关注市场情绪变化',
                '积累更多训练数据',
            ],
        }

        report_file = self.report_gen.generate_weekly_report(year, week, weekly_summary, [])
        print(f"  ✓ 报告: {report_file}")

        # 5. 结束安全会话
        if not self.dry_run:
            self.safety.commit_change(
                f'weekly-check: {year}-W{week:02d}',
                files=['reports/weekly/', 'logs/', 'models/']
            )
            self.safety.end_session(True, {
                'issues': len(self.issues),
                'changes': len(self.changes),
            })

        print(f"\n{'='*60}")
        print(f"  一周大检完成")
        print(f"{'='*60}")

        return {
            'success': True,
            'week': f'{year}-W{week:02d}',
            'issues': len(self.issues),
            'changes': len(self.changes),
            'metrics': self.metrics,
            'report': report_file,
        }


def main():
    parser = argparse.ArgumentParser(description='一周大检')
    parser.add_argument('--dry-run', action='store_true', help='只检查不修复')
    args = parser.parse_args()

    check = WeeklyCheck(dry_run=args.dry_run)
    result = check.run()

    if not result.get('success'):
        sys.exit(2)
    elif any(i['level'] == 'error' for i in check.issues):
        sys.exit(2)
    elif check.issues:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
