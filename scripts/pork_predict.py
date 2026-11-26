#!/usr/bin/env python3
"""
猪肉+饲料+种植全产业链板块预测
独立脚本, 不修改 daily_predict.py 配置

用法:
  python scripts/pork_predict.py
"""

import os, sys, time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.fetch_stock_data import TencentStrategy
from scripts.indicators import calc_volume_price_features, calc_volume_price_signals, calc_chanlun_simple, calc_chanlun_signals
from scripts.adaptive_params import load_validation_history, calc_stock_metrics, adaptive_params, calc_market_volatility, signal_correction
from scripts.news_sentiment import get_stock_sentiment, apply_sentiment_filter

# ============================================================
# 猪肉+饲料+种植全产业链
# ============================================================
STOCKS = {
    # 生猪养殖
    '002714': '牧原股份', '300498': '温氏股份', '000876': '新希望',
    '002157': '正邦科技', '000735': '罗牛山',
    # 饲料
    '002311': '海大集团', '002385': '大北农', '600438': '通威股份',
    '603363': '傲农生物', '002548': '金新农',
    # 种植/种子
    '000998': '隆平高科', '600598': '北大荒', '600354': '敦煌种业', '000713': '丰乐种业',
    # 动保
    '002100': '天康生物', '300119': '瑞普生物',
}
STOCK_TAGS = {
    '002714': '生猪', '300498': '生猪', '000876': '生猪',
    '002157': '生猪', '000735': '生猪',
    '002311': '饲料', '002385': '饲料', '600438': '饲料/光伏',
    '603363': '饲料', '002548': '饲料',
    '000998': '种子', '600598': '种植', '600354': '种子', '000713': '种子',
    '002100': '动保', '300119': '动保',
}

DATA_DIR = './finetune_csv/data'
LOG_DIR = './prediction_log'
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']
START_DATE = '20240101'

# ============================================================
# 步骤1: 获取猪肉板块数据
# ============================================================
def fetch_pork_data():
    print('\n' + '='*60)
    print('  步骤1: 获取猪肉+饲料+种植板块数据')
    print('='*60)
    strategy = TencentStrategy()
    today = datetime.now().strftime('%Y%m%d')

    for code, name in STOCKS.items():
        path = os.path.join(DATA_DIR, f'{code}_fundflow.csv')
        try:
            # 腾讯取K线 (641个交易日, 约2.5年)
            df = strategy.fetch_range(code, START_DATE, today)
            if df.empty:
                print(f'  {name}({code}): 无数据')
                continue
            df['timestamps'] = pd.to_datetime(df['timestamps'])

            # 过滤零价格行
            price_zero = (df['close'] == 0) | (df['open'] == 0)
            if price_zero.any():
                df = df[~price_zero].reset_index(drop=True)

            if df.empty:
                print(f'  {name}({code}): 过滤后无数据')
                continue

            # 腾讯数据不含amount, 用volume*均价估算
            m = df['amount'] == 0
            if m.any():
                avg_p = df.loc[m, ['open','high','low','close']].mean(axis=1)
                df.loc[m, 'amount'] = df.loc[m, 'volume'] * avg_p

            # 补齐资金流列(猪肉股用东方财富资金流)
            for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
                if fc not in df.columns:
                    df[fc] = 0.0
                else:
                    df[fc] = df[fc].fillna(0)

            # 尝试获取最近一天资金流
            try:
                import requests
                secid = f'1.{code}' if code.startswith(('6','9')) else f'0.{code}'
                params = {'reportName':'RPT_DMSK_TS_STOCKNEW','columns':'ALL',
                          'filter':f'(SECURITY_CODE="{code}")','pageSize':'1','pageNumber':'1',
                          'sortColumns':'TRADE_DATE','sortTypes':'-1'}
                headers = {'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
                r = requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=params,headers=headers,timeout=10)
                data = r.json().get('result',{}).get('data',[])
                if data:
                    item = data[0]
                    prime = float(item.get('PRIME_INFLOW',0) or 0)
                    super_in = float(item.get('SUPERDEAL_INFLOW',0) or 0)
                    super_out = float(item.get('SUPERDEAL_OUTFLOW',0) or 0)
                    huge_net = super_in - super_out
                    trade_date = str(item.get('TRADE_DATE',''))[:10]
                    mask = df['timestamps'].dt.strftime('%Y-%m-%d') == trade_date
                    if mask.any():
                        idx = mask.idxmax()
                        df.loc[idx,'main_net_inflow'] = prime
                        df.loc[idx,'huge_net_inflow'] = huge_net
                        amt = df.loc[idx,'amount']
                        df.loc[idx,'main_net_inflow_ratio'] = prime/amt*100 if amt > 0 else 0
            except:
                pass

            df = df.sort_values('timestamps').reset_index(drop=True)
            df.to_csv(path, index=False)
            last_date = df['timestamps'].iloc[-1].strftime('%Y-%m-%d')
            print(f'  {name}({code}): {len(df)}条, 截止 {last_date}')
        except Exception as e:
            print(f'  {name}({code}): 失败 - {e}')
        time.sleep(0.3)

    print(f'\n  数据获取完成, 共{len(STOCKS)}只')


# ============================================================
# 步骤2: 加载数据 + 计算指标
# ============================================================
def load_df(code, skip_indicators=False):
    path = os.path.join(DATA_DIR, f'{code}_fundflow.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'数据文件不存在: {path}')
    df = pd.read_csv(path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    for col in FEATURES:
        if col not in df.columns: df[col] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0)
    m = df['amount'] == 0
    if m.any():
        df.loc[m, 'amount'] = df.loc[m, 'volume'] * df.loc[m, ['open','high','low','close']].mean(axis=1)
    for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
        if fc not in df.columns: df[fc] = 0.0
        else: df[fc] = df[fc].fillna(0)
    if not skip_indicators:
        vp_feats = calc_volume_price_features(df)
        chanlun_feats = calc_chanlun_simple(df)
        for col in vp_feats.columns:
            df[col] = vp_feats[col]
        for col in chanlun_feats.columns:
            df[col] = chanlun_feats[col]
    return df


# ============================================================
# 步骤3: 预测
# ============================================================
def predict_pork(predictor):
    print(f'\n{"="*60}')
    print(f'  步骤2: 预测猪肉板块下一交易日')
    print(f'{"="*60}')

    all_dfs = {}
    for code in STOCKS:
        try:
            all_dfs[code] = load_df(code)
        except Exception as e:
            print(f'  {STOCKS[code]}({code}): 加载失败 - {e}')

    market_vol = calc_market_volatility(all_dfs)
    print(f'  板块波动率: {market_vol:.4f}')

    # 预测目标日
    first_code = list(all_dfs.keys())[0]
    data_last_date = all_dfs[first_code]['timestamps'].iloc[-1]
    target_date = data_last_date + timedelta(days=1)
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    print(f'  预测目标日: {target_date.strftime("%Y-%m-%d")}')

    results = []

    for code, name in STOCKS.items():
        if code not in all_dfs:
            continue
        df = all_dfs[code]
        last_close = df['close'].iloc[-1]

        # 动量
        pct_5d = df['close'].iloc[-5:].pct_change().dropna().sum() * 100
        pct_3d = df['close'].iloc[-3:].pct_change().dropna().sum() * 100
        mom_signal = 1 if (pct_5d + pct_3d) > 0 else -1

        fund_ratio = df['main_net_inflow_ratio'].iloc[-1] if 'main_net_inflow_ratio' in df.columns else 0

        # 量价+缠论信号
        vp_signals = calc_volume_price_signals(df)
        chanlun_signals = calc_chanlun_signals(df)

        # 自适应参数 (新股票无验证历史,用默认metrics)
        val_history = load_validation_history(LOG_DIR, n_days=10)
        stock_metrics = calc_stock_metrics(val_history, code)
        stock_params = adaptive_params(stock_metrics, market_vol)

        fd = []
        c = df['timestamps'].iloc[-1] + timedelta(days=1)
        while len(fd) < 1:
            if c.weekday() < 5: fd.append(c)
            c += timedelta(days=1)

        # 保守预测
        con_params = {'T': 0.5, 'top_p': 0.9, 'sample_count': 10, 'lookback': min(120, len(df))}
        lb = con_params['lookback']
        x_df = df.iloc[-lb:][FEATURES].reset_index(drop=True)
        x_ts = df.iloc[-lb:]['timestamps'].reset_index(drop=True)
        pred_con = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
                                     pred_len=1, T=con_params['T'], top_p=con_params['top_p'],
                                     sample_count=con_params['sample_count'], verbose=False)
        chg_con = (pred_con['close'].iloc[0] / last_close - 1) * 100
        chg_con = max(-10, min(10, chg_con))
        chg_con, con_correction = signal_correction(chg_con, vp_signals, chanlun_signals, stock_metrics)

        # 激进预测
        agr_params = {'T': 0.8, 'top_p': 0.93, 'sample_count': 8, 'lookback': min(150, len(df))}
        lb2 = agr_params['lookback']
        x_df2 = df.iloc[-lb2:][FEATURES].reset_index(drop=True)
        x_ts2 = df.iloc[-lb2:]['timestamps'].reset_index(drop=True)
        pred_agr = predictor.predict(df=x_df2, x_timestamp=x_ts2, y_timestamp=pd.Series(fd),
                                     pred_len=1, T=agr_params['T'], top_p=agr_params['top_p'],
                                     sample_count=agr_params['sample_count'], verbose=False)
        chg_agr = (pred_agr['close'].iloc[0] / last_close - 1) * 100
        chg_agr = max(-10, min(10, chg_agr))
        chg_agr, agr_correction = signal_correction(chg_agr, vp_signals, chanlun_signals, stock_metrics)

        con_high = pred_con['high'].iloc[0]
        con_low = pred_con['low'].iloc[0]
        agr_high = pred_agr['high'].iloc[0]
        agr_low = pred_agr['low'].iloc[0]

        # 信号原因
        sig_parts = []
        mom_str = '动量↑' if (pct_5d + pct_3d) > 0 else '动量↓'
        fund_str = ''
        if fund_ratio > 0.5: fund_str = '主力流入'
        elif fund_ratio < -0.5: fund_str = '主力流出'
        if isinstance(con_correction, dict):
            for k, v in con_correction.items():
                if isinstance(v, (int, float)) and abs(v) > 0.05:
                    sig_parts.append(k)
        reason = mom_str + ('+' + fund_str if fund_str else '') + ('+信号[' + '+'.join(sig_parts) + ']' if sig_parts else '')

        results.append({
            'code': code, 'name': name, 'tag': STOCK_TAGS.get(code, ''),
            'last_close': last_close,
            'con_chg': chg_con, 'agr_chg': chg_agr,
            'avg_chg': (chg_con + chg_agr) / 2,
            'con_high': con_high, 'con_low': con_low,
            'agr_high': agr_high, 'agr_low': agr_low,
            'reason': reason,
        })
        arrow = '↑' if (chg_con + chg_agr) > 0 else '↓'
        print(f'  {name}({code}) [{STOCK_TAGS.get(code,"")}]: 保守{chg_con:+.2f}% 激进{chg_agr:+.2f}% 综合{(chg_con+chg_agr)/2:+.2f}%{arrow} | {reason}')
        time.sleep(0.2)

    # 新闻情绪
    print(f'\n{"="*60}')
    print(f'  步骤3: 新闻情绪过滤')
    print(f'{"="*60}')
    for r in results:
        try:
            s = get_stock_sentiment(r['code'])
        except:
            s = {"count": 0, "score": 0.0, "neg_ratio": 0.0, "pos_ratio": 0.0, "negative_titles": []}
        new_chg, tag = apply_sentiment_filter(r['avg_chg'], s)
        if tag:
            r['avg_chg'] = new_chg
            r['reason'] += '+' + tag
        neg_str = '; '.join(s.get('negative_titles', [])[:2])
        print(f'  {r["name"]}({r["code"]}): 情绪分{s["score"]:+.2f} 利空{s["neg_ratio"]:.0%} {tag or ""} {neg_str}')
        time.sleep(0.3)

    # 排名输出
    results.sort(key=lambda x: x['avg_chg'], reverse=True)
    tomorrow_str = target_date.strftime('%Y-%m-%d')

    print(f'\n{"="*60}')
    print(f'  {tomorrow_str} 猪肉+饲料+种植板块综合排名')
    print(f'{"="*60}')
    print(f'  {"排名":4s} {"股票":14s} {"板块":6s} {"综合":>8s} {"保守":>8s} {"激进":>8s} {"预测区间":>18s} {"理由"}')
    print(f'  {"-"*100}')
    for i, r in enumerate(results, 1):
        arrow = '↑' if r['avg_chg'] > 0 else '↓'
        price_range = f'{min(r["con_low"],r["agr_low"]):.2f}~{max(r["con_high"],r["agr_high"]):.2f}'
        print(f'  {i:4d} {r["name"]+"("+r["code"]+")":14s} {r["tag"]:6s} {r["avg_chg"]:+7.2f}%{arrow} {r["con_chg"]:+7.2f}% {r["agr_chg"]:+7.2f}% {price_range:>18s} {r["reason"]}')

    # 保存
    pred_file = os.path.join(LOG_DIR, f'{tomorrow_str}_pork_pred.csv')
    df_pred = pd.DataFrame(results)
    for col in ['con_chg','agr_chg','avg_chg','con_high','con_low','agr_high','agr_low']:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].round(2)
    df_pred.to_csv(pred_file, index=False)
    print(f'\n  预测已保存: {pred_file}')

    return results


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f'猪肉+饲料+种植板块预测 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # 步骤1: 获取数据
    fetch_pork_data()

    # 加载模型
    print('\n加载模型...')
    tokenizer = KronosTokenizer.from_pretrained('./pretrained/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('./pretrained/Kronos-base')
    predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5.0)
    print('模型加载完成')

    # 步骤2: 预测
    predict_pork(predictor)

    print(f'\n全部完成 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')


if __name__ == '__main__':
    main()
