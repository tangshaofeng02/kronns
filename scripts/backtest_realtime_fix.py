#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控趋势过滤修复 — 历史回测验证
对比修复前后: 下跌趋势中的买入信号数量与信号后收益
"""
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prediction_log')


def load_actual():
    """每日实际涨跌幅 {(date, code): actual_chg}"""
    actual = {}
    for f in glob.glob(os.path.join(LOG_DIR, '*_validate.csv')):
        date = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, dtype={'code': str})
        for _, r in df.iterrows():
            actual[(date, str(r['code']))] = r['actual_chg']
    return actual


# 中文列名 → 英文 (兼容历史英文/新中文 csv)
_COLUMN_MAP = {
    '时间': 'time', '代码': 'code', '名称': 'name', '现价': 'price',
    '涨跌幅': 'change_pct', '量比': 'vol_ratio', '换手率': 'turnover',
    '决策': 'decision', '评分': 'score', '置信度': 'confidence',
    '逆势': 'contrarian', '逆势强度': 'contrarian_strength',
    '底背离': 'bullish_divergence', '顶背离': 'bearish_divergence',
    '背离强度': 'divergence_strength', '抄底评分': 'bottom_score', '信号': 'signals',
}


def load_snapshots():
    """每日实时快照 {date: [(time, df), ...]}"""
    byday = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(LOG_DIR, 'realtime_*.csv'))):
        base = os.path.basename(f)
        date = base.replace('realtime_', '').split('_')[0]
        d = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        df = pd.read_csv(f)
        df.rename(columns=_COLUMN_MAP, inplace=True)  # 中文列名转英文
        df['code'] = df['code'].astype(str)
        byday[d].append((base, df))
    return byday


def main():
    actual = load_actual()
    byday = load_snapshots()

    print('=' * 78)
    print('  实时监控趋势过滤修复 — 历史回测验证')
    print('=' * 78)
    print(f"\n{'日期':<12}{'市场趋势':>9}{'修复前买入':>10}{'修复后买入':>10}"
          f"{'前收益':>9}{'后收益':>9}{'改善':>8}")
    print('-' * 78)

    total_before = []
    total_after = []
    for d in sorted(byday.keys()):
        if d < '2026-11-05':
            continue
        files = sorted(byday[d])
        if not files:
            continue
        _, df = files[0]  # 当日首个快照
        if df.empty:
            continue
        market_trend = df['change_pct'].mean()

        # 修复前: 所有买入信号
        buy_before = df[df['decision'].str.contains('买入', na=False)]
        # 修复后: 下跌趋势(market_trend < -1)中买入信号被过滤
        if market_trend < -1.0:
            buy_after = buy_before.iloc[0:0]  # 全部过滤
        else:
            buy_after = buy_before

        def avg_close_ret(sub):
            rets = [actual.get((d, c)) for c in sub['code']]
            rets = [r for r in rets if r is not None and not pd.isna(r)]
            return np.mean(rets) if rets else np.nan

        r_before = avg_close_ret(buy_before)
        r_after = avg_close_ret(buy_after)
        total_before.append(r_before)
        total_after.append(r_after)
        imp = (r_after - r_before) if (not pd.isna(r_before) and not pd.isna(r_after)) else np.nan
        rb = f'{r_before:+.2f}%' if not pd.isna(r_before) else '-'
        ra = f'{r_after:+.2f}%' if not pd.isna(r_after) else '-'
        im = f'{imp:+.2f}%' if not pd.isna(imp) else '-'
        print(f'{d:<12}{market_trend:>8.2f}%{len(buy_before):>10}{len(buy_after):>10}'
              f'{rb:>9}{ra:>9}{im:>8}')

    print()
    valid_b = [x for x in total_before if not pd.isna(x)]
    valid_a = [x for x in total_after if not pd.isna(x)]
    if valid_b:
        print(f'  修复前 买入信号收盘均收益: {np.mean(valid_b):+.2f}% (共{len(valid_b)}日)')
    if valid_a:
        print(f'  修复后 买入信号收盘均收益: {np.mean(valid_a):+.2f}% (共{len(valid_a)}日)')
    print()
    print('  说明: 修复后在下跌趋势(市场平均<-1%)中过滤全部买入信号,')
    print('        避免"超跌+逆势吸筹"的接飞刀陷阱; 上涨/震荡市不受影响。')


if __name__ == '__main__':
    main()
