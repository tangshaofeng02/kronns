#!/usr/bin/env python3
"""
30只股票半个月(10个交易日)滚动预测
逐日预测,将预测结果追加到数据中继续预测下一天

用法:
  python scripts/halfmonth_predict.py
"""

import os, sys, time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.daily_predict import STOCKS, STOCK_TAGS, load_df, FEATURES, PARAMS_CORRECT_CON, PARAMS_CORRECT_AGR

DATA_DIR = './finetune_csv/data'
LOG_DIR = './prediction_log'
N_DAYS = 10  # 半个月≈10个交易日


def get_next_trading_days(last_date, n):
    """获取接下来的n个交易日(跳过周末)"""
    days = []
    d = last_date + timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def predict_one_day(predictor, df, target_date, T, top_p, sample_count, lookback):
    """预测单只股票的一个交易日"""
    lb = min(lookback, len(df))
    recent = df.iloc[-lb:].copy()
    x_df = recent[FEATURES].reset_index(drop=True)
    x_ts = recent['timestamps'].reset_index(drop=True)

    pred = predictor.predict(
        df=x_df, x_timestamp=x_ts,
        y_timestamp=pd.Series([target_date]),
        pred_len=1, T=T, top_p=top_p,
        sample_count=sample_count, verbose=False
    )
    return pred


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f'半个月滚动预测 ({N_DAYS}个交易日) - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'股票池: {len(STOCKS)}只')

    # 加载数据
    all_dfs = {}
    for code in STOCKS:
        try:
            all_dfs[code] = load_df(code)
        except Exception as e:
            print(f'  {STOCKS[code]}({code}): 跳过 - {e}')

    print(f'成功加载: {len(all_dfs)}/{len(STOCKS)}只')

    # 计算预测目标日
    first_code = list(all_dfs.keys())[0]
    last_data_date = all_dfs[first_code]['timestamps'].iloc[-1]
    target_days = get_next_trading_days(last_data_date, N_DAYS)
    print(f'数据截止: {last_data_date.strftime("%Y-%m-%d")}')
    print(f'预测区间: {target_days[0].strftime("%Y-%m-%d")} ~ {target_days[-1].strftime("%Y-%m-%d")}')

    # 加载模型
    print('\n加载模型...')
    tokenizer = KronosTokenizer.from_pretrained('./pretrained/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('./pretrained/Kronos-base')
    predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5.0)
    print('模型加载完成')

    # 结果存储 {code: [{date, pred_close, pred_high, pred_low, pred_chg}, ...]}
    all_predictions = {code: [] for code in all_dfs}

    # 逐日滚动预测
    for day_idx, target_date in enumerate(target_days):
        date_str = target_date.strftime('%Y-%m-%d')
        print(f'\n--- 第{day_idx+1}天 {date_str} ---')

        for code in all_dfs:
            df = all_dfs[code]
            last_close = df['close'].iloc[-1]

            # 用保守参数预测
            params = PARAMS_CORRECT_CON
            try:
                pred = predict_one_day(
                    predictor, df, target_date,
                    T=params['T'], top_p=params['top_p'],
                    sample_count=params['sample_count'], lookback=params['lookback']
                )
                pred_close = pred['close'].iloc[0]
                pred_high = pred['high'].iloc[0]
                pred_low = pred['low'].iloc[0]
                pred_open = pred['open'].iloc[0]
                pred_volume = pred['volume'].iloc[0]
                pred_amount = pred['amount'].iloc[0] if 'amount' in pred.columns else pred_volume * pred_close
                pred_chg = (pred_close / last_close - 1) * 100
                pred_chg = max(-10, min(10, pred_chg))
            except Exception as e:
                print(f'  {STOCKS[code]}({code}): 预测失败 - {e}, 沿用前价')
                pred_close = last_close
                pred_high = last_close
                pred_low = last_close
                pred_open = last_close
                pred_volume = df['volume'].iloc[-1]
                pred_amount = df['amount'].iloc[-1]
                pred_chg = 0.0

            # 存储预测结果
            all_predictions[code].append({
                'date': date_str,
                'pred_open': pred_open,
                'pred_high': pred_high,
                'pred_low': pred_low,
                'pred_close': pred_close,
                'pred_volume': pred_volume,
                'pred_amount': pred_amount,
                'pred_chg': pred_chg,
                'last_close': last_close,
            })

            # 追加到数据中,用于下一天预测
            new_row = pd.DataFrame([{
                'timestamps': pd.Timestamp(target_date),
                'open': pred_open, 'high': pred_high, 'low': pred_low,
                'close': pred_close, 'volume': pred_volume, 'amount': pred_amount,
                'main_net_inflow': 0.0, 'huge_net_inflow': 0.0, 'main_net_inflow_ratio': 0.0,
            }])
            all_dfs[code] = pd.concat([all_dfs[code], new_row], ignore_index=True)

        # 打印当日摘要
        day_results = []
        for code in all_dfs:
            if all_predictions[code]:
                p = all_predictions[code][-1]
                day_results.append((code, STOCKS[code], p['pred_chg'], p['pred_close'], p['last_close']))
        day_results.sort(key=lambda x: x[2], reverse=True)
        top3 = day_results[:3]
        bot3 = day_results[-3:]
        avg_chg = np.mean([r[2] for r in day_results])
        up_count = sum(1 for r in day_results if r[2] > 0)
        print(f'  看涨{up_count}/{len(day_results)}只 平均涨跌{avg_chg:+.2f}%')
        top_str = ', '.join([f'{r[1]}({r[2]:+.2f}%)' for r in top3])
        bot_str = ', '.join([f'{r[1]}({r[2]:+.2f}%)' for r in bot3])
        print(f'  最强: {top_str}')
        print(f'  最弱: {bot_str}')

    # ============================================================
    # 汇总报告
    # ============================================================
    print(f'\n\n{"="*80}')
    print(f'  半个月滚动预测汇总 ({target_days[0].strftime("%m/%d")} ~ {target_days[-1].strftime("%m/%d")})')
    print(f'{"="*80}')

    # 计算每只股票10日累计涨跌
    summary = []
    for code in all_dfs:
        preds = all_predictions[code]
        if not preds:
            continue
        start_price = preds[0]['last_close']
        end_price = preds[-1]['pred_close']
        total_chg = (end_price / start_price - 1) * 100
        max_price = max(p['pred_high'] for p in preds)
        min_price = min(p['pred_low'] for p in preds)
        max_drawdown = (min_price / start_price - 1) * 100
        max_gain = (max_price / start_price - 1) * 100
        avg_daily = np.mean([p['pred_chg'] for p in preds])

        summary.append({
            'code': code, 'name': STOCKS[code], 'tag': STOCK_TAGS.get(code, ''),
            'start': start_price, 'end': end_price,
            'total_chg': total_chg, 'avg_daily': avg_daily,
            'max_gain': max_gain, 'max_drawdown': max_drawdown,
            'preds': preds,
        })

    summary.sort(key=lambda x: x['total_chg'], reverse=True)

    # 表格
    print(f'  {"排名":4s} {"股票":14s} {"板块":6s} {"累计涨跌":>8s} {"日均":>8s} {"最大涨幅":>8s} {"最大回撤":>8s} {"起始价":>8s} {"目标价":>8s}')
    print(f'  {"-"*100}')
    for i, s in enumerate(summary, 1):
        arrow = '↑' if s['total_chg'] > 0 else '↓'
        print(f'  {i:4d} {s["name"]+"("+s["code"]+")":14s} {s["tag"]:6s} {s["total_chg"]:+7.2f}%{arrow} {s["avg_daily"]:+7.2f}% {s["max_gain"]:+7.2f}% {s["max_drawdown"]:+7.2f}% {s["start"]:8.2f} {s["end"]:8.2f}')

    # 逐日明细
    print(f'\n\n{"="*80}')
    print(f'  逐日预测明细')
    print(f'{"="*80}')

    for s in summary[:10]:  # 只看前10名
        print(f'\n  {s["name"]}({s["code"]}) [{s["tag"]}] 累计{s["total_chg"]:+.2f}%')
        for p in s['preds']:
            arrow = '↑' if p['pred_chg'] > 0 else '↓'
            print(f'    {p["date"]}: {p["pred_chg"]:+6.2f}%{arrow} 收{p["pred_close"]:.2f} 高{p["pred_high"]:.2f} 低{p["pred_low"]:.2f}')

    # 保存完整结果
    rows = []
    for s in summary:
        for p in s['preds']:
            rows.append({
                'code': s['code'], 'name': s['name'], 'tag': s['tag'],
                'date': p['date'],
                'pred_open': round(p['pred_open'], 2),
                'pred_high': round(p['pred_high'], 2),
                'pred_low': round(p['pred_low'], 2),
                'pred_close': round(p['pred_close'], 2),
                'pred_chg': round(p['pred_chg'], 2),
                'cum_chg': round((p['pred_close'] / s['start'] - 1) * 100, 2),
            })

    pred_file = os.path.join(LOG_DIR, f'halfmonth_{target_days[0].strftime("%Y%m%d")}_{target_days[-1].strftime("%Y%m%d")}.csv')
    df_out = pd.DataFrame(rows)
    df_out.to_csv(pred_file, index=False)
    print(f'\n\n完整结果已保存: {pred_file}')

    # 保存汇总
    summary_file = os.path.join(LOG_DIR, f'halfmonth_summary_{target_days[0].strftime("%Y%m%d")}.csv')
    df_summary = pd.DataFrame([{
        'code': s['code'], 'name': s['name'], 'tag': s['tag'],
        'start_price': round(s['start'], 2), 'end_price': round(s['end'], 2),
        'total_chg_pct': round(s['total_chg'], 2),
        'avg_daily_pct': round(s['avg_daily'], 2),
        'max_gain_pct': round(s['max_gain'], 2),
        'max_drawdown_pct': round(s['max_drawdown'], 2),
    } for s in summary])
    df_summary.to_csv(summary_file, index=False)
    print(f'汇总已保存: {summary_file}')

    print(f'\n全部完成 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')


if __name__ == '__main__':
    main()
