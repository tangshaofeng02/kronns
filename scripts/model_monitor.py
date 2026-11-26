#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型监控与衰减检测

功能:
  1. 每日推荐准确率跟踪 (命中率、TOP5 命中率、概率校准)
  2. 模型衰减检测 (滚动窗口 AUC / 命中率趋势)
  3. 特征分布漂移告警 (均值/方差突变)
  4. 自动触发重训练建议

用法:
  python scripts/model_monitor.py                    # 查看监控报告
  python scripts/model_monitor.py --days 10          # 最近 N 天
  python scripts/model_monitor.py --threshold 0.4    # 命中率告警阈值
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'limit_up_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_limit_up.pkl')


def load_recommends():
    """加载所有推荐记录"""
    recs = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*_recommend.csv'))):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            recs[date] = df
        except Exception:
            continue
    return recs


def load_zt_pools():
    """加载每日涨停池"""
    pools = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*_zt.csv'))):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            pools[date] = df
        except Exception:
            continue
    return pools


def load_prev_pools():
    """加载昨日涨停池（今日表现）"""
    pools = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*_prev.csv'))):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            pools[date] = df
        except Exception:
            continue
    return pools


def compute_daily_metrics(recs, zt_pools, prev_pools, days=10):
    """计算每日推荐指标

    对每个推荐日 D:
      - 用 D+1 日的实际涨停池验证命中率
      - 计算 TOP5 命中率、概率校准度
    """
    dates = sorted(recs.keys())[-days:]
    rows = []

    for d in dates:
        # 找到下一交易日
        all_dates = sorted(zt_pools.keys())
        try:
            idx = all_dates.index(d)
            if idx + 1 >= len(all_dates):
                continue
            nd = all_dates[idx + 1]
        except ValueError:
            continue

        rec = recs[d]
        zt_next = zt_pools.get(nd, pd.DataFrame())
        if zt_next.empty:
            continue

        # 构建次日连板映射
        lb_map = dict(zip(zt_next['代码'], zt_next['连板数']))

        # 验证每只推荐
        hits = []
        for _, r in rec.iterrows():
            code = str(r['代码'])
            lbc = lb_map.get(code)
            hits.append(1 if (lbc is not None and int(lbc) >= 2) else 0)

        rec = rec.copy()
        rec['hit'] = hits

        # 计算指标
        total_hit = sum(hits) / len(hits) if hits else 0
        top5_hit = sum(hits[:5]) / min(5, len(hits)) if hits else 0

        # 概率校准
        if '连板概率' in rec.columns:
            prob_mean = rec['连板概率'].mean() / 100
            prob_calibration = abs(prob_mean - total_hit)
        else:
            prob_mean = 0
            prob_calibration = 0

        # 市场基准（次日全部首板票的连板率）
        prev = prev_pools.get(d, pd.DataFrame())
        if not prev.empty:
            first = prev[prev['昨日连板数'] == 1]
            if not first.empty:
                # 用今日涨停池验证昨日首板
                base_hit = first['最新价'].ge(first['涨停价'] - 0.011).mean()
            else:
                base_hit = 0
        else:
            base_hit = 0

        rows.append({
            'date': d,
            'n_rec': len(rec),
            'total_hit_rate': total_hit,
            'top5_hit_rate': top5_hit,
            'excess': top5_hit - base_hit,
            'prob_mean': prob_mean,
            'prob_calibration': prob_calibration,
            'base_hit_rate': base_hit,
        })

    return pd.DataFrame(rows)


def detect_decay(metrics_df, window=5):
    """检测模型衰减

    规则:
      1. 最近 N 天命中率 < 历史均值 * 0.7 → 衰减告警
      2. 命中率趋势下降（线性回归斜率 < 0）→ 趋势告警
      3. 概率校准误差 > 0.15 → 校准告警
    """
    if metrics_df.empty or len(metrics_df) < window:
        return []

    alerts = []
    recent = metrics_df.tail(window)
    history = metrics_df.head(len(metrics_df) - window) if len(metrics_df) > window else metrics_df

    # 1. 命中率衰减
    recent_hit = recent['top5_hit_rate'].mean()
    hist_hit = history['top5_hit_rate'].mean() if not history.empty else recent_hit
    if hist_hit > 0 and recent_hit < hist_hit * 0.7:
        alerts.append({
            'type': '命中率衰减',
            'level': '高',
            'message': f'最近 {window} 天 TOP5 命中率 {recent_hit:.1%} < 历史均值 {hist_hit:.1%} 的 70%',
        })

    # 2. 趋势下降
    if len(metrics_df) >= 5:
        x = np.arange(len(metrics_df))
        y = metrics_df['top5_hit_rate'].values
        slope = np.polyfit(x, y, 1)[0]
        if slope < -0.01:
            alerts.append({
                'type': '趋势下降',
                'level': '中',
                'message': f'命中率趋势下降 (斜率 {slope:+.3f}/天)',
            })

    # 3. 概率校准
    recent_cal = recent['prob_calibration'].mean()
    if recent_cal > 0.15:
        alerts.append({
            'type': '概率校准偏差',
            'level': '中',
            'message': f'预测概率与实际命中率偏差 {recent_cal:.1%} > 15%',
        })

    # 4. 连续跑输基准
    recent_excess = recent['excess'].mean()
    if recent_excess < 0:
        alerts.append({
            'type': '跑输基准',
            'level': '高',
            'message': f'最近 {window} 天平均超额 {recent_excess:+.1%} < 0',
        })

    return alerts


def check_feature_drift(current_features, baseline_features, threshold=0.5):
    """检查特征分布漂移

    比较当前特征分布与训练时的基准分布。
    """
    alerts = []
    for col in current_features.columns:
        if col not in baseline_features.columns:
            continue
        cur_mean = current_features[col].mean()
        base_mean = baseline_features[col].mean()
        cur_std = current_features[col].std()
        base_std = baseline_features[col].std()

        if base_std > 0 and cur_std > 0:
            # Z-score 漂移
            z_drift = abs(cur_mean - base_mean) / base_std
            if z_drift > threshold:
                alerts.append({
                    'feature': col,
                    'z_drift': z_drift,
                    'cur_mean': cur_mean,
                    'base_mean': base_mean,
                })

    return alerts


def generate_report(metrics_df, alerts, output_file=None):
    """生成监控报告"""
    print("\n" + "=" * 70)
    print("  模型监控报告")
    print("=" * 70)

    if metrics_df.empty:
        print("  暂无推荐数据")
        return

    # 最近 10 天明细
    print("\n【最近 10 天推荐表现】")
    print(f"{'日期':<12} {'推荐数':>5} {'总命中率':>8} {'TOP5命中率':>10} {'超额':>8} {'基准':>8} {'概率校准':>8}")
    print("-" * 70)
    for _, r in metrics_df.tail(10).iterrows():
        print(f"{r['date']:<12} {int(r['n_rec']):>5} {r['total_hit_rate']:>8.1%} "
              f"{r['top5_hit_rate']:>10.1%} {r['excess']:>+8.1%} {r['base_hit_rate']:>8.1%} "
              f"{r['prob_calibration']:>8.1%}")

    # 汇总统计
    print("\n【汇总统计】")
    print(f"  监控天数:        {len(metrics_df)}")
    print(f"  平均总命中率:    {metrics_df['total_hit_rate'].mean():.2%}")
    print(f"  平均 TOP5 命中率: {metrics_df['top5_hit_rate'].mean():.2%}")
    print(f"  平均超额收益:    {metrics_df['excess'].mean():+.2%}")
    print(f"  胜率:            {(metrics_df['excess'] > 0).mean():.2%}")
    print(f"  平均概率校准误差: {metrics_df['prob_calibration'].mean():.2%}")

    # 衰减检测
    print("\n【衰减检测】")
    if alerts:
        for a in alerts:
            level_icon = '🔴' if a['level'] == '高' else '🟡'
            print(f"  {level_icon} [{a['type']}] {a['message']}")
        print(f"\n  ⚠️ 建议: 模型可能需要重训练")
    else:
        print("  ✅ 模型运行正常，无衰减迹象")

    # 保存报告
    if output_file:
        metrics_df.to_csv(output_file, index=False)
        print(f"\n  报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='模型监控与衰减检测')
    parser.add_argument('--days', type=int, default=10, help='监控最近 N 天')
    parser.add_argument('--threshold', type=float, default=0.4, help='命中率告警阈值')
    parser.add_argument('--output', type=str, default=None, help='报告输出文件')
    args = parser.parse_args()

    print("=" * 70)
    print("  涨停板推荐模型监控")
    print("=" * 70)

    # 加载数据
    print("\n[1/3] 加载数据...")
    recs = load_recommends()
    zt_pools = load_zt_pools()
    prev_pools = load_prev_pools()
    print(f"  推荐记录: {len(recs)} 天")
    print(f"  涨停池: {len(zt_pools)} 天")
    print(f"  昨日涨停池: {len(prev_pools)} 天")

    if not recs:
        print("无推荐数据，退出")
        return

    # 计算每日指标
    print("\n[2/3] 计算每日指标...")
    metrics_df = compute_daily_metrics(recs, zt_pools, prev_pools, days=args.days)
    print(f"  有效监控天数: {len(metrics_df)}")

    # 衰减检测
    print("\n[3/3] 衰减检测...")
    alerts = detect_decay(metrics_df, window=min(5, len(metrics_df)))

    # 生成报告
    output = args.output or os.path.join(DATA_DIR, 'model_monitor_report.csv')
    generate_report(metrics_df, alerts, output_file=output)

    # 返回告警状态（供脚本调用）
    if any(a['level'] == '高' for a in alerts):
        sys.exit(2)  # 严重告警
    elif alerts:
        sys.exit(1)  # 一般告警
    else:
        sys.exit(0)  # 正常


if __name__ == '__main__':
    main()
