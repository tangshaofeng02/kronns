#!/usr/bin/env python3
"""
相对强弱排名评估

核心指标：TOP5跑赢BOTTOM5的天数占比
如果模型选股有效，TOP5实际均值应系统性地高于BOTTOM5。
这比"方向准确率"更有意义——因为大盘涨跌是所有股票的共同噪声，
而排名能力是模型真正的价值。

用法:
  python scripts/evaluate_ranking.py --days 42
  python scripts/evaluate_ranking.py --plot  # 绘制累计收益曲线
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import STOCKS, load_df, LOG_DIR


def load_actual_returns(date_str, codes):
    """获取一组股票在某天的实际涨跌幅"""
    results = {}
    for code in codes:
        try:
            df = load_df(code, skip_indicators=True)
            target = df[df['timestamps'].dt.strftime('%Y-%m-%d') == date_str]
            if target.empty:
                target = df[df['timestamps'] <= pd.to_datetime(date_str)]
                if target.empty:
                    continue
                target = target.tail(1)
            actual_close = target.iloc[0]['close']
            prev_mask = df['timestamps'] < target.iloc[0]['timestamps']
            prev_df = df[prev_mask]
            if prev_df.empty:
                continue
            prev_close = prev_df.iloc[-1]['close']
            results[code] = (actual_close / prev_close - 1) * 100
        except Exception:
            continue
    return results


def evaluate_ranking(days=42, verbose=True):
    """评估历史排名准确率"""
    pred_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_pred.csv')])

    # 只取有实际数据可验证的文件
    valid_files = []
    for pf in pred_files:
        date_str = pf.replace('_pred.csv', '')
        try:
            # 检查该日期是否有实际数据
            df_check = load_df(list(STOCKS.keys())[0], skip_indicators=True)
            has_data = not df_check[df_check['timestamps'].dt.strftime('%Y-%m-%d') == date_str].empty
            if has_data:
                valid_files.append(pf)
        except:
            continue

    recent_files = valid_files[-days:]

    if verbose:
        print(f"{'='*80}")
        print(f"相对强弱排名评估（最近{len(recent_files)}个交易日）")
        print(f"指标：TOP5实际均值 > BOTTOM5实际均值 的天数占比")
        print(f"{'='*80}")
        print(f'{"日期":12s} {"TOP5预期":>10s} {"TOP5实际":>10s} {"BOTTOM5预期":>12s} {"BOTTOM5实际":>12s} {"价差":>8s} {"结果"}')
        print(f"{'-'*80}")

    records = []
    for pf in recent_files:
        date_str = pf.replace('_pred.csv', '')
        try:
            pred_df = pd.read_csv(os.path.join(LOG_DIR, pf), dtype={'code': str})
            pred_df['code'] = pred_df['code'].astype(str).str.zfill(6)
        except Exception:
            continue

        # 计算综合排名（保守+激进均值）
        ranked = []
        for code in STOCKS:
            rows = pred_df[pred_df['code'] == code]
            if rows.empty:
                continue
            con_chg = rows.iloc[0].get('pred_chg', 0)
            agr_chg = rows.iloc[-1].get('pred_chg', 0) if len(rows) > 1 else con_chg
            avg = (con_chg + agr_chg) / 2
            ranked.append({'code': code, 'name': STOCKS[code], 'pred_avg': avg})

        if len(ranked) < 10:
            continue

        ranked.sort(key=lambda x: x['pred_avg'], reverse=True)

        top5 = ranked[:5]
        bot5 = ranked[-5:]

        top5_codes = [r['code'] for r in top5]
        bot5_codes = [r['code'] for r in bot5]

        actual = load_actual_returns(date_str, top5_codes + bot5_codes)

        top5_actual = [actual.get(c, 0) for c in top5_codes if c in actual]
        bot5_actual = [actual.get(c, 0) for c in bot5_codes if c in actual]

        if not top5_actual or not bot5_actual:
            continue

        top5_actual_mean = np.mean(top5_actual)
        bot5_actual_mean = np.mean(bot5_actual)
        spread = top5_actual_mean - bot5_actual_mean

        top5_pred_mean = np.mean([r['pred_avg'] for r in top5])
        bot5_pred_mean = np.mean([r['pred_avg'] for r in bot5])

        win = spread > 0  # TOP5跑赢BOTTOM5

        records.append({
            'date': date_str,
            'top5_pred': top5_pred_mean,
            'top5_actual': top5_actual_mean,
            'bot5_pred': bot5_pred_mean,
            'bot5_actual': bot5_actual_mean,
            'spread': spread,
            'win': win,
        })

        status = '✓' if win else '✗'
        if verbose:
            print(f"{date_str:12s} {top5_pred_mean:+9.2f}% {top5_actual_mean:+9.2f}% {bot5_pred_mean:+11.2f}% {bot5_actual_mean:+11.2f}% {spread:+7.2f}% {status}")

    if not records:
        print("  无足够数据")
        return

    df_rec = pd.DataFrame(records)
    n = len(df_rec)
    wins = df_rec['win'].sum()
    win_rate = wins / n * 100
    avg_spread = df_rec['spread'].mean()

    if verbose:
        print(f"{'='*80}")
        print(f"汇总统计:")
        print(f"  排名胜率: {wins}/{n} = {win_rate:.1f}%  (TOP5跑赢BOTTOM5的天数占比)")
        print(f"  平均价差: {avg_spread:+.2f}%  (TOP5均值 - BOTTOM5均值)")
        print(f"  最大正价差: {df_rec['spread'].max():+.2f}%")
        print(f"  最大负价差: {df_rec['spread'].min():+.2f}%")
        print(f"  价差标准差: {df_rec['spread'].std():.2f}%")
        print(f"{'='*80}")

        if win_rate >= 60:
            print(f"  🟢 排名能力显著（{win_rate:.0f}% > 60%），模型有选股价值")
        elif win_rate >= 55:
            print(f"  🟡 排名能力弱显著（{win_rate:.0f}%），需继续优化")
        else:
            print(f"  🔴 排名能力不显著（{win_rate:.0f}% < 55%），模型无选股能力")

    return df_rec


def plot_cumulative(days=42):
    """绘制TOP5 vs BOTTOM5累计收益曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_rec = evaluate_ranking(days=days, verbose=False)
    if df_rec is None or df_rec.empty:
        print("无足够数据绘图")
        return

    df_rec = df_rec.sort_values('date')
    df_rec['top5_cum'] = df_rec['top5_actual'].cumsum()
    df_rec['bot5_cum'] = df_rec['bot5_actual'].cumsum()
    df_rec['spread_cum'] = df_rec['spread'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上图：TOP5 vs BOTTOM5 累计收益
    ax1.plot(df_rec['date'], df_rec['top5_cum'], 'g-o', label='TOP5 累计收益', markersize=4)
    ax1.plot(df_rec['date'], df_rec['bot5_cum'], 'r-s', label='BOTTOM5 累计收益', markersize=4)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('累计收益(%)')
    ax1.set_title('相对强弱排名：TOP5 vs BOTTOM5 累计收益')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 下图：每日价差
    colors = ['green' if x > 0 else 'red' for x in df_rec['spread']]
    ax2.bar(df_rec['date'], df_rec['spread'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.plot(df_rec['date'], df_rec['spread_cum'], 'b-^', label='累计价差', markersize=4)
    ax2.set_ylabel('价差(%)')
    ax2.set_title(f'每日价差（TOP5-BOTTOM5），胜率{df_rec["win"].mean()*100:.0f}%')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = './prediction_log/ranking_evaluation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {out_path}")

    return df_rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='相对强弱排名评估')
    parser.add_argument('--days', type=int, default=42, help='评估天数')
    parser.add_argument('--plot', action='store_true', help='绘制累计收益曲线')
    args = parser.parse_args()

    if args.plot:
        plot_cumulative(args.days)
    else:
        evaluate_ranking(args.days)
