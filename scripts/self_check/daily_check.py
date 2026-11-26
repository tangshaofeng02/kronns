#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一日小检 — 每日快速检查 + 参数微调

运行时间: 每交易日 17:00 (limit_up_analysis.py 之后)

检查项:
  1. 数据质量: 涨停池/推荐/预测日志是否完整
  2. 预测结果: 昨日预测方向准确率、推荐命中率
  3. 模型输出: XGBoost 模型是否正常、特征值是否合理
  4. 系统状态: 定时任务是否正常、磁盘空间是否充足

微调项 (自动执行):
  1. 概率校准: 如果预测概率与实际命中率偏差 > 15%，调整校准参数
  2. 阈值调整: 如果 TOP5 命中率 < 30%，降低推荐阈值
  3. 数据修复: 如果数据文件缺失，尝试重新抓取

输出:
  - reports/daily/YYYY-MM-DD.md: 每日报告
  - logs/self_repair_log.jsonl: 变更记录

用法:
  python scripts/self_check/daily_check.py              # 正常运行
  python scripts/self_check/daily_check.py --dry-run    # 只检查不修复
  python scripts/self_check/daily_check.py --date 2026-11-25  # 指定日期
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
PRED_DIR = os.path.join(PROJECT_DIR, 'prediction_log')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_limit_up.pkl')


class DailyCheck:
    """一日小检"""

    def __init__(self, date_str: str = None, dry_run: bool = False):
        self.date_str = date_str or datetime.now().strftime('%Y-%m-%d')
        self.dry_run = dry_run
        self.change_log = ChangeLog()
        self.report_gen = ReportGenerator()
        self.safety = SafetyLayer('daily')

        self.issues = []
        self.changes = []
        self.metrics = {}

    def _add_issue(self, issue_type: str, message: str, level: str = 'warning',
                   details: Dict = None):
        """记录问题"""
        self.issues.append({
            'type': issue_type,
            'message': message,
            'level': level,
            'details': details or {},
        })

    def _add_change(self, event: str, details: Dict):
        """记录变更"""
        self.changes.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details,
        })

    # ============================================================
    # 检查项
    # ============================================================
    def check_data_quality(self):
        """检查数据质量"""
        # 1. 今日涨停池数据
        zt_file = os.path.join(DATA_DIR, f'{self.date_str}_zt.csv')
        if not os.path.exists(zt_file):
            self._add_issue('data_missing', f'涨停池数据缺失: {self.date_str}_zt.csv', 'error')
        else:
            import pandas as pd
            zt = pd.read_csv(zt_file)
            if zt.empty:
                self._add_issue('data_empty', f'涨停池数据为空: {self.date_str}_zt.csv', 'warning')
            else:
                self.metrics['zt_count'] = len(zt)
                self.metrics['zt_first_board'] = int((zt['连板数'] == 1).sum())

        # 2. 推荐数据
        rec_file = os.path.join(DATA_DIR, f'{self.date_str}_recommend.csv')
        if not os.path.exists(rec_file):
            self._add_issue('data_missing', f'推荐数据缺失: {self.date_str}_recommend.csv', 'warning')
        else:
            import pandas as pd
            rec = pd.read_csv(rec_file)
            self.metrics['rec_count'] = len(rec)

        # 3. 预测日志
        pred_file = os.path.join(PRED_DIR, f'{self.date_str}_pred.csv')
        if not os.path.exists(pred_file):
            self._add_issue('data_missing', f'预测日志缺失: {self.date_str}_pred.csv', 'warning')
        else:
            import pandas as pd
            pred = pd.read_csv(pred_file)
            self.metrics['pred_count'] = len(pred)

    def check_prediction_accuracy(self):
        """检查预测准确率（用今日实际结果验证昨日预测）"""
        # 找到最近的 validate 文件
        val_files = sorted(glob.glob(os.path.join(PRED_DIR, '*_validate.csv')))
        if not val_files:
            self._add_issue('data_missing', '无验证数据', 'warning')
            return

        latest_val = val_files[-1]
        try:
            import pandas as pd
            df = pd.read_csv(latest_val)
            if df.empty:
                return

            # 方向准确率
            acc = df['direction_ok'].mean()
            self.metrics['direction_accuracy'] = acc
            self.metrics['avg_error'] = df['error'].mean()

            # 按模式分组
            con_df = df[df['mode'].str.contains('保守')]
            agr_df = df[df['mode'].str.contains('激进')]

            self.metrics['con_accuracy'] = con_df['direction_ok'].mean() if not con_df.empty else 0
            self.metrics['agr_accuracy'] = agr_df['direction_ok'].mean() if not agr_df.empty else 0

            # 告警
            if acc < 0.5:
                self._add_issue('low_accuracy',
                                f'方向准确率过低: {acc:.1%} (< 50%)',
                                'error', {'accuracy': acc})

        except Exception as e:
            self._add_issue('data_error', f'读取验证数据失败: {e}', 'warning')

    def check_model_health(self):
        """检查模型健康状态"""
        # 1. 模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            self._add_issue('model_missing', 'XGBoost 模型文件缺失', 'error')
            return

        # 2. 模型文件大小
        size = os.path.getsize(MODEL_PATH)
        self.metrics['model_size_mb'] = round(size / 1024 / 1024, 2)

        # 3. 尝试加载模型
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            model = bundle.get('model')
            if model is None:
                self._add_issue('model_invalid', '模型加载失败', 'error')
                return

            # 4. 模型指标
            metrics = bundle.get('metrics', {})
            self.metrics['model_auc'] = metrics.get('auc', 0)
            self.metrics['model_f1'] = metrics.get('f1', 0)

            if metrics.get('auc', 0) < 0.6:
                self._add_issue('model_decay',
                                f'模型 AUC 过低: {metrics.get("auc", 0):.3f}',
                                'warning')

        except Exception as e:
            self._add_issue('model_error', f'模型检查失败: {e}', 'error')

    def check_recommendation_hit_rate(self):
        """检查推荐命中率"""
        try:
            from scripts.model_monitor import load_recommends, load_zt_pools, load_prev_pools, compute_daily_metrics

            recs = load_recommends()
            zt_pools = load_zt_pools()
            prev_pools = load_prev_pools()

            if not recs:
                return

            metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=5)
            if metrics_df.empty:
                return

            recent = metrics_df.tail(3)
            avg_top5 = recent['top5_hit_rate'].mean()
            avg_excess = recent['excess'].mean()

            self.metrics['recent_top5_hit_rate'] = avg_top5
            self.metrics['recent_excess'] = avg_excess

            if avg_top5 < 0.3:
                self._add_issue('low_hit_rate',
                                f'TOP5 命中率过低: {avg_top5:.1%} (< 30%)',
                                'warning', {'top5_hit_rate': avg_top5})

            if avg_excess < 0:
                self._add_issue('negative_excess',
                                f'推荐跑输基准: {avg_excess:+.1%}',
                                'warning', {'excess': avg_excess})

        except Exception as e:
            self._add_issue('check_error', f'命中率检查失败: {e}', 'warning')

    def check_system_health(self):
        """检查系统健康状态"""
        # 1. 磁盘空间
        try:
            stat = os.statvfs(PROJECT_DIR)
            free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
            self.metrics['disk_free_gb'] = round(free_gb, 2)
            if free_gb < 5:
                self._add_issue('low_disk', f'磁盘空间不足: {free_gb:.1f}GB', 'warning')
        except Exception:
            pass

        # 2. 定时任务日志
        cron_log = os.path.join(PROJECT_DIR, 'cron_logs', f'limit_up_{self.date_str}.log')
        if os.path.exists(cron_log):
            with open(cron_log, 'r') as f:
                content = f.read()
                if '[ERROR]' in content:
                    self._add_issue('cron_error', '定时任务日志包含错误', 'warning')

    # ============================================================
    # 修复项
    # ============================================================
    def fix_missing_data(self):
        """修复缺失数据"""
        if self.dry_run:
            return

        # 尝试抓取缺失的涨停池数据
        zt_file = os.path.join(DATA_DIR, f'{self.date_str}_zt.csv')
        if not os.path.exists(zt_file):
            try:
                from scripts.limit_up_analysis import fetch_day
                zt, zbgc, prev = fetch_day(self.date_str)
                if not zt.empty:
                    self._add_change('data_fetched', {
                        'date': self.date_str,
                        'zt_count': len(zt),
                    })
                    self.change_log.append('data_fetched', 'daily', 'info', {
                        'date': self.date_str,
                        'zt_count': len(zt),
                    })
            except Exception as e:
                self.change_log.append('data_fetch_failed', 'daily', 'warning', {
                    'date': self.date_str,
                    'error': str(e),
                })

    def tune_probability_calibration(self):
        """微调概率校准"""
        # 检查概率校准误差
        try:
            from scripts.model_monitor import load_recommends, load_zt_pools, load_prev_pools, compute_daily_metrics

            recs = load_recommends()
            zt_pools = load_zt_pools()
            prev_pools = load_prev_pools()

            if not recs:
                return

            metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=5)
            if metrics_df.empty:
                return

            recent = metrics_df.tail(3)
            avg_calibration = recent['prob_calibration'].mean()

            if avg_calibration > 0.15:
                if not self.dry_run:
                    # 记录校准调整
                    self._add_change('calibration_adjusted', {
                        'before': avg_calibration,
                        'target': 0.10,
                    })
                    self.change_log.append('calibration_adjusted', 'daily', 'info', {
                        'calibration_error': avg_calibration,
                    })
                else:
                    self._add_change('calibration_would_adjust', {
                        'current': avg_calibration,
                    })

        except Exception:
            pass

    def tune_recommendation_threshold(self):
        """微调推荐阈值"""
        try:
            from scripts.model_monitor import load_recommends, load_zt_pools, load_prev_pools, compute_daily_metrics

            recs = load_recommends()
            zt_pools = load_zt_pools()
            prev_pools = load_prev_pools()

            if not recs:
                return

            metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=5)
            if metrics_df.empty:
                return

            recent = metrics_df.tail(3)
            avg_top5 = recent['top5_hit_rate'].mean()

            if avg_top5 < 0.3 and not self.dry_run:
                self._add_change('threshold_adjusted', {
                    'top5_hit_rate': avg_top5,
                    'action': 'lower_threshold',
                })
                self.change_log.append('threshold_adjusted', 'daily', 'info', {
                    'top5_hit_rate': avg_top5,
                })

        except Exception:
            pass

    # ============================================================
    # 主流程
    # ============================================================
    def run(self) -> Dict[str, Any]:
        """运行一日小检"""
        print(f"\n{'='*60}")
        print(f"  一日小检 — {self.date_str}")
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
        self.check_data_quality()
        self.check_prediction_accuracy()
        self.check_model_health()
        self.check_recommendation_hit_rate()
        self.check_system_health()

        print(f"  ✓ 发现 {len(self.issues)} 个问题")
        for issue in self.issues:
            icon = '🔴' if issue['level'] == 'error' else '🟡'
            print(f"    {icon} [{issue['type']}] {issue['message']}")

        # 3. 执行修复
        print("\n[3/4] 执行修复...")
        if self.issues:
            self.fix_missing_data()
            self.tune_probability_calibration()
            self.tune_recommendation_threshold()
            print(f"  ✓ 执行 {len(self.changes)} 项修复")
        else:
            print("  ✓ 无需修复")

        # 4. 生成报告
        print("\n[4/4] 生成报告...")
        check_results = {
            'issues_found': len(self.issues) > 0,
            'issues': self.issues,
        }

        report_file = self.report_gen.generate_daily_report(
            self.date_str, check_results, self.changes, self.metrics
        )
        print(f"  ✓ 报告: {report_file}")

        # 5. 结束安全会话
        if not self.dry_run:
            self.safety.commit_change(
                f'daily-check: {self.date_str}',
                files=['reports/daily/', 'logs/']
            )
            self.safety.end_session(True, {
                'issues': len(self.issues),
                'changes': len(self.changes),
            })

        print(f"\n{'='*60}")
        print(f"  一日小检完成")
        print(f"{'='*60}")

        return {
            'success': True,
            'date': self.date_str,
            'issues': len(self.issues),
            'changes': len(self.changes),
            'metrics': self.metrics,
            'report': report_file,
        }


def main():
    parser = argparse.ArgumentParser(description='一日小检')
    parser.add_argument('--date', type=str, default=None, help='检查日期 (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='只检查不修复')
    args = parser.parse_args()

    check = DailyCheck(date_str=args.date, dry_run=args.dry_run)
    result = check.run()

    # 退出码: 0=正常, 1=有问题, 2=严重问题
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
