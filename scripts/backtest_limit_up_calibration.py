#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首板连板概率校准回测
验证分段校准对 30-40% 区间(高估2倍)的改善效果
"""
import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'limit_up_data')


def load_history():
    df = pd.read_csv(os.path.join(DATA_DIR, 'history.csv'), dtype={'代码': str})
    df['date'] = df['date'].astype(str)
    df['连板数'] = pd.to_numeric(df['连板数'], errors='coerce').fillna(1).astype(int)
    return df


def load_recommends():
    recs = {}
    for f in glob.glob(os.path.join(DATA_DIR, '*_recommend.csv')):
        date = os.path.basename(f).split('_')[0]
        try:
            recs[date] = pd.read_csv(f, dtype={'代码': str})
        except:
            pass
    return recs


def calibrate(p):
    """分段校准: 30-40%降级, 20-30%略降"""
    if 0.30 <= p < 0.40:
        return p * 0.5
    if 0.20 <= p < 0.30:
        return p * 0.85
    return p


def bin_label(p):
    if p >= 0.40:
        return '≥40%'
    if p >= 0.30:
        return '30-40%'
    if p >= 0.20:
        return '20-30%'
    return '<20%'


def main():
    hist = load_history()
    recs = load_recommends()
    dates = sorted(hist['date'].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    day_codes = {d: set(hist[hist['date'] == d]['代码']) for d in dates}

    rows = []
    for d in sorted(recs.keys()):
        if d not in date_idx:
            continue
        i = date_idx[d]
        if i + 1 >= len(dates):
            continue
        nd = dates[i + 1]
        for _, r in recs[d].iterrows():
            code = str(r['代码'])
            prob = r['连板概率'] / 100.0
            hit = code in day_codes[nd]
            rows.append({
                'date': d, 'code': code, 'name': r['名称'],
                'prob': prob, 'prob_cal': calibrate(prob), 'hit': hit,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print('无数据')
        return

    print('=' * 70)
    print('  首板连板概率校准回测')
    print('=' * 70)

    print('\n【校准前】')
    print(f"{'区间':<10}{'样本':>6}{'命中':>6}{'实际命中率':>10}{'预测均值':>10}{'偏差':>8}")
    print('-' * 55)
    for b in ['≥40%', '30-40%', '20-30%', '<20%']:
        sub = df[df['prob'].apply(bin_label) == b]
        if sub.empty:
            continue
        n = len(sub)
        h = sub['hit'].sum()
        rate = h / n * 100
        pred = sub['prob'].mean() * 100
        print(f'{b:<10}{n:>6}{h:>6}{rate:>9.1f}%{pred:>9.1f}%{rate-pred:>+7.1f}')

    print('\n【校准后】')
    print(f"{'区间':<10}{'样本':>6}{'命中':>6}{'实际命中率':>10}{'预测均值':>10}{'偏差':>8}")
    print('-' * 55)
    for b in ['≥40%', '30-40%', '20-30%', '<20%']:
        sub = df[df['prob_cal'].apply(bin_label) == b]
        if sub.empty:
            continue
        n = len(sub)
        h = sub['hit'].sum()
        rate = h / n * 100
        pred = sub['prob_cal'].mean() * 100
        print(f'{b:<10}{n:>6}{h:>6}{rate:>9.1f}%{pred:>9.1f}%{rate-pred:>+7.1f}')

    # 排序质量: TOP5 命中率
    print('\n【TOP5 排名质量】')
    for d in sorted(df['date'].unique()):
        sub = df[df['date'] == d].sort_values('prob', ascending=False).head(5)
        sub_cal = df[df['date'] == d].sort_values('prob_cal', ascending=False).head(5)
        hit_before = sub['hit'].sum()
        hit_after = sub_cal['hit'].sum()
        print(f'  {d}: 校准前 {hit_before}/5 | 校准后 {hit_after}/5')

    print(f'\n  累计: 校准前 {df["hit"].sum()}/{len(df)} = {df["hit"].mean()*100:.1f}%')


if __name__ == '__main__':
    main()
