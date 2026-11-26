#!/usr/bin/env python3
"""
周预测评估：用历史数据验证"预测5天vs预测1天"的准确率差异

核心思路：用5天前的K线数据预测第5天收盘价，与实际第5天收盘价对比。
如果周预测准确率 > 日预测准确率，说明延长预测周期确实有效。

用法:
  python scripts/weekly_evaluate.py --days 42
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import STOCKS, load_df, LOG_DIR
from model import Kronos, KronosTokenizer, KronosPredictor


def backtest_weekly(days=42, verbose=True):
    """用历史数据回测周预测准确率"""
    tokenizer = KronosTokenizer.from_pretrained('./pretrained/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('./pretrained/Kronos-base')
    predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5.0)

    # 获取所有交易日期
    sample_df = load_df('000001', skip_indicators=True)
    all_dates = sample_df['timestamps'].dt.strftime('%Y-%m-%d').tolist()

    # 取最近N个有完整5天数据的日期
    test_dates = []
    for i in range(len(all_dates) - 6, max(len(all_dates) - 6 - days, 0), -1):
        test_dates.append(all_dates[i])
    test_dates = sorted(test_dates)

    if verbose:
        print(f"{'='*80}")
        print(f"周预测回测（{len(test_dates)}个测试日）")
        print(f"用T-5天的数据预测T日收盘价，对比日预测准确率")
        print(f"{'='*80}")

    results = []
    for date_str in test_dates:
        # 找到T-5天的日期
        idx = all_dates.index(date_str)
        if idx < 5:
            continue
        start_date = all_dates[idx - 5]

        day_ok = 0
        day_total = 0
        week_ok = 0
        week_total = 0

        for code in STOCKS:
            try:
                df = load_df(code, skip_indicators=True)
                # 获取T-5到T-1的数据（5天lookback）
                mask = (df['timestamps'] >= start_date) & (df['timestamps'] < date_str)
                hist = df[mask]

                if len(hist) < 5:
                    continue

                # 实际T日价格
                actual_row = df[df['timestamps'].dt.strftime('%Y-%m-%d') == date_str]
                if actual_row.empty:
                    continue
                actual_close = actual_row.iloc[0]['close']
                prev_close = hist.iloc[-1]['close']
                actual_chg = (actual_close / prev_close - 1) * 100
                actual_dir = 1 if actual_chg > 0 else -1

                # 日预测（pred_len=1，用最近1天）
                try:
                    lb = 120
                    x_df = df.iloc[-lb:-5][['open','high','low','close','volume','amount']].reset_index(drop=True)
                    x_ts = df.iloc[-lb:-5]['timestamps'].reset_index(drop=True)
                    if len(x_df) < 50:
                        continue
                    last_close = x_df['close'].iloc[-1]

                    # 日预测
                    y_ts_d = pd.Series([pd.to_datetime(date_str)])
                    pred_d = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts_d,
                                              pred_len=1, T=0.7, top_p=0.9, sample_count=5, verbose=False)
                    pred_close_d = pred_d['close'].iloc[0]
                    chg_d = (pred_close_d / last_close - 1) * 100
                    dir_d = 1 if chg_d > 0 else -1
                    day_ok += (dir_d == actual_dir)
                    day_total += 1
                except:
                    pass

                # 周预测（pred_len=5）
                try:
                    y_ts_w = pd.date_range(start=pd.to_datetime(start_date), periods=5, freq='B')
                    if len(y_ts_w) < 5:
                        continue
                    pred_w = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(y_ts_w),
                                              pred_len=5, T=0.7, top_p=0.9, sample_count=5, verbose=False)
                    pred_close_w = pred_w['close'].iloc[-1]  # 第5天
                    chg_w = (pred_close_w / last_close - 1) * 100
                    dir_w = 1 if chg_w > 0 else -1
                    week_ok += (dir_w == actual_dir)
                    week_total += 1
                except:
                    pass

            except Exception:
                continue

        if day_total > 0 and week_total > 0:
            day_acc = day_ok / day_total * 100
            week_acc = week_ok / week_total * 100
            results.append({
                'date': date_str,
                'day_acc': day_acc,
                'week_acc': week_acc,
                'n': day_total,
            })
            if verbose:
                better = '🟢周优' if week_acc > day_acc else ('🔴日优' if day_acc > week_acc else '⚪平')
                print(f"  {date_str}: 日={day_acc:.0f}% 周={week_acc:.0f}% ({day_total}只) {better}")

    if not results:
        print("  无足够数据")
        return

    df_res = pd.DataFrame(results)
    avg_day = df_res['day_acc'].mean()
    avg_week = df_res['week_acc'].mean()
    week_better = (df_res['week_acc'] > df_res['day_acc']).sum()

    print(f"{'='*80}")
    print(f"汇总:")
    print(f"  日预测平均准确率: {avg_day:.1f}%")
    print(f"  周预测平均准确率: {avg_week:.1f}%")
    print(f"  周预测更优天数: {week_better}/{len(results)}")
    if avg_week > avg_day + 2:
        print(f"  🟢 周预测显著优于日预测（+{avg_week-avg_day:.1f}%），建议切换到周预测")
    elif avg_week > avg_day:
        print(f"  🟡 周预测略优于日预测（+{avg_week-avg_day:.1f}%）")
    else:
        print(f"  🔴 周预测未优于日预测（{avg_week-avg_day:+.1f}%），不建议切换")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='周预测回测评估')
    parser.add_argument('--days', type=int, default=20, help='测试天数（默认20，减少计算量）')
    args = parser.parse_args()
    backtest_weekly(args.days)
