#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续多日涨停板数据预测分析
输入: limit_up_data/ 下的 history.csv / label_history.csv / high_board_history.csv / *_recommend.csv / *_zbgc.csv
输出: 多日涨停结构趋势、推荐模型多日回测、高连板延续性、情绪周期定位与次日预判
"""
import os
import sys
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'limit_up_data')


def load_history():
    df = pd.read_csv(os.path.join(DATA_DIR, 'history.csv'), dtype={'代码': str})
    df['date'] = df['date'].astype(str)
    df['连板数'] = pd.to_numeric(df['连板数'], errors='coerce').fillna(1).astype(int)
    df['换手率'] = pd.to_numeric(df['换手率'], errors='coerce')
    df['炸板次数'] = pd.to_numeric(df['炸板次数'], errors='coerce').fillna(0).astype(int)
    df['流通市值'] = pd.to_numeric(df['流通市值'], errors='coerce')
    return df


def load_zbgc_counts():
    """每日炸板池数量"""
    counts = {}
    for f in glob.glob(os.path.join(DATA_DIR, '*_zbgc.csv')):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f)
            counts[date] = len(df)
        except Exception:
            pass
    return counts


def load_recommends():
    """每日推荐 {date: DataFrame}"""
    recs = {}
    for f in glob.glob(os.path.join(DATA_DIR, '*_recommend.csv')):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            recs[date] = df
        except Exception:
            pass
    return recs


def daily_structure(hist, zbgc_counts):
    """多日涨停结构"""
    rows = []
    dates = sorted(hist['date'].unique())
    for d in dates:
        sub = hist[hist['date'] == d]
        first = sub[sub['连板数'] == 1]
        cont = sub[sub['连板数'] >= 2]
        rows.append({
            'date': d,
            '涨停数': len(sub),
            '首板数': len(first),
            '连板数': len(cont),
            '最高板': int(sub['连板数'].max()),
            '炸板数': zbgc_counts.get(d, 0),
            '炸板率': round(zbgc_counts.get(d, 0) / max(len(sub) + zbgc_counts.get(d, 0), 1) * 100, 1),
            '平均换手': round(sub['换手率'].mean(), 2),
            '平均连板': round(sub['连板数'].mean(), 2),
        })
    return pd.DataFrame(rows)


def compute_first_board_continuation(hist):
    """每日首板票的次日连板率（首板连板率）"""
    dates = sorted(hist['date'].unique())
    date_set = set(dates)
    # 构建 date -> set of codes 映射
    day_codes = {d: set(hist[hist['date'] == d]['代码']) for d in dates}
    # 交易日顺序（按自然日排序即可，跨周末/节假日用下一可用日期）
    rows = []
    for i, d in enumerate(dates[:-1]):
        nd = dates[i + 1]
        first = hist[(hist['date'] == d) & (hist['连板数'] == 1)]
        if len(first) == 0:
            continue
        codes = set(first['代码'])
        cont = codes & day_codes[nd]
        rows.append({
            'date': d,
            '首板数': len(first),
            '次日连板数': len(cont),
            '首板连板率': round(len(cont) / len(first) * 100, 1),
            '次日涨停数': len(day_codes[nd]),
        })
    return pd.DataFrame(rows)


def validate_recommends(hist, recs):
    """多日推荐回测：推荐日 D 的票在下一交易日是否连板"""
    dates = sorted(hist['date'].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    day_codes = {d: set(hist[hist['date'] == d]['代码']) for d in dates}
    day_lb = {d: dict(zip(hist[hist['date'] == d]['代码'], hist[hist['date'] == d]['连板数'])) for d in dates}

    rows = []
    detail = []
    for d in sorted(recs.keys()):
        if d not in date_idx:
            continue
        i = date_idx[d]
        if i + 1 >= len(dates):
            continue
        nd = dates[i + 1]
        rec = recs[d]
        hit = 0
        top5_hit = 0
        for _, r in rec.iterrows():
            code = str(r['代码'])
            is_hit = code in day_codes[nd]
            if is_hit:
                hit += 1
                if int(r['排名']) <= 5:
                    top5_hit += 1
            detail.append({
                '推荐日': d, '代码': code, '名称': r['名称'], '排名': int(r['排名']),
                '预测概率': r['连板概率'], '次日连板': '✓' if is_hit else '✗',
                '次日板数': day_lb[nd].get(code, 0),
            })
        n = len(rec)
        rows.append({
            '推荐日': d, '次日': nd, '推荐数': n,
            '命中数': hit, '命中率': round(hit / n * 100, 1) if n else 0,
            'TOP5命中': top5_hit, 'TOP5命中率': round(top5_hit / min(5, n) * 100, 1) if n else 0,
        })
    return pd.DataFrame(rows), pd.DataFrame(detail)


def high_board_continuation():
    """高连板延续性分析（基于 high_board_history.csv）"""
    f = os.path.join(DATA_DIR, 'high_board_history.csv')
    if not os.path.exists(f):
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(f, dtype={'code': str})
    df['date'] = df['date'].astype(str)
    dates = sorted(df['date'].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    day_lb = {d: dict(zip(df[df['date'] == d]['code'], df[df['date'] == d]['lbc'])) for d in dates}

    rows = []
    for i, d in enumerate(dates[:-1]):
        sub = df[df['date'] == d]
        nd = dates[i + 1]
        cont = sum(1 for c in sub['code'] if day_lb[nd].get(c, 0) >= 2)
        rows.append({
            'date': d, '高标数': len(sub),
            '次日仍涨停': cont,
            '延续率': round(cont / len(sub) * 100, 1) if len(sub) else 0,
        })
    # 分层延续率
    layer_rows = []
    for layer, g in df.groupby('layer'):
        dates_l = sorted(g['date'].unique())
        cont = 0
        total = 0
        for d in dates_l:
            i = date_idx.get(d, -1)
            if i < 0 or i + 1 >= len(dates):
                continue
            nd = dates[i + 1]
            for c in g[g['date'] == d]['code']:
                total += 1
                if day_lb[nd].get(c, 0) >= 2:
                    cont += 1
        layer_rows.append({'层标': layer, '样本数': total, '次日延续数': cont,
                           '延续率': round(cont / total * 100, 1) if total else 0})
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


def sentiment_cycle(struct, cont):
    """情绪周期定位：结合首板连板率与炸板率"""
    if cont.empty:
        return
    m = struct.merge(cont[['date', '首板连板率', '次日连板数']], on='date', how='left')
    print('\n' + '=' * 70)
    print('  多日情绪周期 (首板连板率 × 炸板率)')
    print('=' * 70)
    print(f"  {'日期':<12}{'涨停':>5}{'首板':>5}{'连板':>5}{'最高':>5}{'炸板率':>8}{'首板连板率':>10}")
    print('  ' + '-' * 60)
    for _, r in m.iterrows():
        fb = f"{r['首板连板率']}%" if pd.notna(r['首板连板率']) else '-'
        print(f"  {r['date']:<12}{r['涨停数']:>5}{r['首板数']:>5}{r['连板数']:>5}"
              f"{r['最高板']:>5}{r['炸板率']:>7}%{fb:>10}")
    valid = m.dropna(subset=['首板连板率'])
    if len(valid):
        print(f"\n  首板连板率: 均值 {valid['首板连板率'].mean():.1f}% | "
              f"最高 {valid['首板连板率'].max():.1f}% | 最低 {valid['首板连板率'].min():.1f}%")
        print(f"  炸板率: 均值 {m['炸板率'].mean():.1f}% | "
              f"最高 {m['炸板率'].max():.1f}% | 最低 {m['炸板率'].min():.1f}%")
        # 近5日趋势
        recent = valid.tail(5)
        if len(recent) >= 2:
            trend = recent['首板连板率'].iloc[-1] - recent['首板连板率'].iloc[0]
            dir_txt = '↑ 转暖' if trend > 3 else ('↓ 转冷' if trend < -3 else '→ 持平')
            print(f"  近5日首板连板率趋势: {trend:+.1f}pct ({dir_txt})")


def main():
    print('=' * 70)
    print('  连续多日涨停板数据预测分析')
    print('=' * 70)

    hist = load_history()
    zbgc_counts = load_zbgc_counts()
    recs = load_recommends()

    struct = daily_structure(hist, zbgc_counts)
    cont = compute_first_board_continuation(hist)

    print('\n' + '=' * 70)
    print('  一、多日涨停结构趋势')
    print('=' * 70)
    print(f"  {'日期':<12}{'涨停':>5}{'首板':>5}{'连板':>5}{'最高':>5}{'炸板':>5}{'炸板率':>8}{'均换手':>8}")
    print('  ' + '-' * 60)
    for _, r in struct.iterrows():
        print(f"  {r['date']:<12}{r['涨停数']:>5}{r['首板数']:>5}{r['连板数']:>5}"
              f"{r['最高板']:>5}{r['炸板数']:>5}{r['炸板率']:>7}%{r['平均换手']:>7}%")

    sentiment_cycle(struct, cont)

    # 二、推荐模型多日回测
    print('\n' + '=' * 70)
    print('  二、推荐模型多日回测')
    print('=' * 70)
    if recs:
        val, detail = validate_recommends(hist, recs)
        print(f"  {'推荐日':<12}{'次日':<12}{'推荐':>5}{'命中':>5}{'命中率':>8}{'TOP5命中率':>10}")
        print('  ' + '-' * 55)
        for _, r in val.iterrows():
            print(f"  {r['推荐日']:<12}{r['次日']:<12}{r['推荐数']:>5}{r['命中数']:>5}"
                  f"{r['命中率']:>7}%{r['TOP5命中率']:>9}%")
        if len(val):
            print(f"\n  累计: 推荐 {val['推荐数'].sum()} 只, 命中 {val['命中数'].sum()} 只, "
                  f"整体命中率 {val['命中数'].sum()/val['推荐数'].sum()*100:.1f}%")
            print(f"  TOP5 累计命中率: {val['TOP5命中'].sum()/(val['推荐数'].clip(upper=5).sum())*100:.1f}%")
        # 概率校准
        print('\n  概率校准 (预测概率 vs 实际命中率):')
        detail['概率区间'] = pd.cut(detail['预测概率'], bins=[0, 20, 30, 40, 100],
                                    labels=['<20%', '20-30%', '30-40%', '≥40%'])
        cal = detail.groupby('概率区间', observed=True).agg(
            样本数=('次日连板', 'count'),
            实际命中=('次日连板', lambda s: (s == '✓').sum()),
        )
        cal['实际命中率'] = (cal['实际命中'] / cal['样本数'] * 100).round(1)
        for idx, r in cal.iterrows():
            print(f"    预测 {idx}: 样本 {r['样本数']:>3} | 实际命中 {r['实际命中']:>3} | 实际命中率 {r['实际命中率']}%")
        # 明细
        print('\n  推荐明细:')
        for _, r in detail.iterrows():
            print(f"    {r['推荐日']} #{r['排名']:>2} {r['名称']:<8} 预测{r['预测概率']:>5.1f}% "
                  f"→ {r['次日连板']} (次日{r['次日板数']}板)")
    else:
        print('  无推荐数据')

    # 三、高连板延续性
    print('\n' + '=' * 70)
    print('  三、高连板延续性分析')
    print('=' * 70)
    hb_cont, hb_layer = high_board_continuation()
    if not hb_cont.empty:
        print(f"  {'日期':<12}{'高标数':>6}{'次日仍涨停':>10}{'延续率':>8}")
        print('  ' + '-' * 40)
        for _, r in hb_cont.iterrows():
            print(f"  {r['date']:<12}{r['高标数']:>6}{r['次日仍涨停']:>10}{r['延续率']:>7}%")
        print(f"\n  整体高标延续率: {hb_cont['次日仍涨停'].sum()}/{hb_cont['高标数'].sum()} "
              f"= {hb_cont['次日仍涨停'].sum()/hb_cont['高标数'].sum()*100:.1f}%")
        if not hb_layer.empty:
            print('\n  分层延续率:')
            print(f"    {'层标':<8}{'样本数':>6}{'次日延续':>8}{'延续率':>8}")
            for _, r in hb_layer.iterrows():
                print(f"    {r['层标']:<8}{r['样本数']:>6}{r['次日延续数']:>8}{r['延续率']:>7}%")
        recent = hb_cont.tail(5)
        print(f"\n  近5日高标延续率: {recent['次日仍涨停'].sum()}/{recent['高标数'].sum()} "
              f"= {recent['次日仍涨停'].sum()/max(recent['高标数'].sum(),1)*100:.1f}%")

    # 四、次日预判
    print('\n' + '=' * 70)
    print('  四、下一交易日预判')
    print('=' * 70)
    last = struct.iloc[-1]
    last_cont = cont.iloc[-1] if len(cont) else None
    if last_cont is not None:
        fb_rate = last_cont['首板连板率']
        zbgc_rate = last['炸板率']
        print(f"  最近交易日 {last['date']}: 涨停 {last['涨停数']} 只 (首板 {last['首板数']}), "
              f"炸板率 {zbgc_rate}%, 首板连板率 {fb_rate}%")
        # 简单规则预判
        if fb_rate >= 25 and zbgc_rate <= 30:
            outlook = '偏暖 → 情绪修复期，首板接力成功率较高'
        elif fb_rate < 15 or zbgc_rate > 45:
            outlook = '偏冷 → 情绪退潮期，谨慎打板，关注低位首板'
        else:
            outlook = '中性 → 结构性行情，精选个股'
        print(f"  预判: {outlook}")
        print(f"  历史基准: 首板连板率均值 {cont['首板连板率'].mean():.1f}%, "
              f"炸板率均值 {struct['炸板率'].mean():.1f}%")
        print(f"  当前首板连板率 {fb_rate}% vs 均值 {cont['首板连板率'].mean():.1f}% "
              f"→ {'高于' if fb_rate > cont['首板连板率'].mean() else '低于'}均值 "
              f"{abs(fb_rate - cont['首板连板率'].mean()):.1f}pct")


if __name__ == '__main__':
    main()
