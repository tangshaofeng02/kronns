#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日预测短期趋势过滤修复 — 历史回测验证
用历史 pred.csv (修复前预测) + STOCKS池近5日数据, 模拟修复后预测, 对比实际
"""
import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.daily_predict import load_df, STOCKS

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prediction_log')


def calc_short_trend(target_date):
    """计算目标日 STOCKS 池近5日平均涨跌幅 + 上涨占比"""
    rets5 = []
    for code in STOCKS:
        try:
            df = load_df(code, skip_indicators=True)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            sub = df[df['timestamps'] <= target_date]
            if len(sub) < 6:
                continue
            rets5.append((sub['close'].iloc[-1] / sub['close'].iloc[-6] - 1) * 100)
        except:
            continue
    if not rets5:
        return 0.0, 0.5
    return float(np.mean(rets5)), sum(1 for r in rets5 if r > 0) / len(rets5)


def apply_fix(chg, short_trend, short_up_ratio):
    """修复逻辑: 短期趋势过滤 (走弱时削弱看涨, 需两条件同时满足)"""
    if short_trend < -1.5 and short_up_ratio < 0.40:
        if chg > 0:
            chg = chg * 0.4 - 0.3
        else:
            chg = chg * 1.3
        return max(-10, min(10, chg)), True
    return chg, False


def main():
    print('=' * 80)
    print('  每日预测短期趋势过滤修复 — 历史回测验证')
    print('=' * 80)

    # 读取所有 pred + validate
    rows = []
    for vf in sorted(glob.glob(os.path.join(LOG_DIR, '*_validate.csv'))):
        date = os.path.basename(vf).split('_')[0]
        if date < '2026-10-26':
            continue
        pred_file = os.path.join(LOG_DIR, f'{date}_pred.csv')
        if not os.path.exists(pred_file):
            continue
        try:
            pred = pd.read_csv(pred_file, dtype={'code': str})
            val = pd.read_csv(vf, dtype={'code': str})
        except:
            continue
        if pred.empty or val.empty:
            continue

        st, sur = calc_short_trend(date)
        filtered_count = 0
        for _, v in val.iterrows():
            code = str(v['code'])
            p_rows = pred[pred['code'] == code]
            if p_rows.empty:
                continue
            # 取保守/激进均值作为综合预测
            pred_chg = p_rows['pred_chg'].mean()
            actual_chg = v['actual_chg']
            fixed_chg, filtered = apply_fix(pred_chg, st, sur)
            if filtered:
                filtered_count += 1
            rows.append({
                'date': date, 'code': code, 'mode': v.get('mode', ''),
                'pred': pred_chg, 'fixed': fixed_chg, 'actual': actual_chg,
                'filtered': filtered, 'short_trend': st, 'short_up_ratio': sur,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print('无数据')
        return

    print(f"\n{'日期':<12}{'短期趋势':>9}{'上涨占比':>8}{'过滤数':>7}"
          f"{'修复前准确率':>12}{'修复后准确率':>12}{'修复前MAE':>10}{'修复后MAE':>10}")
    print('-' * 90)

    dates = sorted(df['date'].unique())
    for d in dates:
        sub = df[df['date'] == d]
        st = sub['short_trend'].iloc[0]
        sur = sub['short_up_ratio'].iloc[0]
        fc = sub['filtered'].sum()
        # 方向准确率
        acc_before = ((sub['pred'] > 0) == (sub['actual'] > 0)).mean() * 100
        acc_after = ((sub['fixed'] > 0) == (sub['actual'] > 0)).mean() * 100
        mae_before = (sub['pred'] - sub['actual']).abs().mean()
        mae_after = (sub['fixed'] - sub['actual']).abs().mean()
        flag = ' ←过滤' if fc > 0 else ''
        print(f'{d:<12}{st:>8.2f}%{sur:>7.0%}{fc:>7}{acc_before:>11.1f}%{acc_after:>11.1f}%'
              f'{mae_before:>10.2f}{mae_after:>10.2f}{flag}')

    print()
    print('=== 汇总 ===')
    for mode in ['保守', '激进']:
        sub = df[df['mode'] == mode]
        if sub.empty:
            continue
        acc_b = ((sub['pred'] > 0) == (sub['actual'] > 0)).mean() * 100
        acc_a = ((sub['fixed'] > 0) == (sub['actual'] > 0)).mean() * 100
        mae_b = (sub['pred'] - sub['actual']).abs().mean()
        mae_a = (sub['fixed'] - sub['actual']).abs().mean()
        print(f'  {mode}: 准确率 {acc_b:.1f}% → {acc_a:.1f}% | MAE {mae_b:.2f} → {mae_a:.2f}')
    # 相关系数
    corr_b = df['pred'].corr(df['actual'])
    corr_a = df['fixed'].corr(df['actual'])
    print(f'  相关系数: {corr_b:.3f} → {corr_a:.3f}')


if __name__ == '__main__':
    main()
