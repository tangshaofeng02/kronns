#!/usr/bin/env python3
"""
每日15:03自动执行: 更新数据 → 验证昨日预测 → 生成报告 → 预测明日

用法:
  python scripts/daily_predict.py          # 完整流程
  python scripts/daily_predict.py --skip-update  # 跳过数据更新
"""

import os, sys, time, argparse, json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.fetch_stock_data import TencentStrategy
from scripts.indicators import calc_volume_price_features, calc_volume_price_signals, calc_chanlun_simple, calc_chanlun_signals
from scripts.adaptive_params import load_validation_history, calc_stock_metrics, adaptive_params, calc_market_volatility, signal_correction, detect_market_regime
from scripts.news_sentiment import get_stock_sentiment, apply_sentiment_filter

# ============================================================
# 配置
# ============================================================
STOCKS = {
    '000001': '平安银行', '000858': '五粮液', '300368': '汇金股份',
    '300430': '诚益通', '300451': '创业惠康', '300678': '中科信息',
    '300782': '卓胜微', '301293': '三博脑科', '600036': '招商银行',
    '600664': '哈药股份',
    '601899': '紫金矿业', '603993': '洛阳钼业', '600011': '华能国际',
    '600900': '长江电力', '601985': '中国核电', '600362': '江西铜业',
    '300624': '万兴科技', '300418': '昆仑万维', '300058': '蓝色光标',
    '600909': '华安证券', '601162': '天风证券', '601211': '国泰海通',
    '002172': '澳洋健康', '300017': '网宿科技',
    '300274': '阳光电源',
    # 生猪养殖
    '002714': '牧原股份', '300498': '温氏股份', '000876': '新希望',
    '002157': '正邦科技', '000735': '罗牛山',
    # 算力/AI应用
    '002261': '拓维信息',
    # 园区开发/科创
    '600895': '张江高科',
    # PCB板块
    '002463': '沪电股份', '600183': '生益科技', '002916': '深南电路',
    '002384': '东山精密', '300476': '胜宏科技',
    # 机器人板块
    '688836': '宇树科技-W', '688017': '绿的谐波', '002747': '埃斯顿',
    '300024': '机器人', '002979': '雷赛智能',
    # 脑机接口
    '002173': '创新医疗', '300007': '汉威科技',
}
STOCK_TAGS = {
    '000001': '银行', '000858': '白酒', '300368': '软件',
    '300430': '软件', '300451': '软件', '300678': '软件',
    '300782': '半导体', '301293': '医药', '600036': '银行',
    '600664': '医药', '601899': '铜金', '603993': '铜钴',
    '600011': '火电', '600900': '水电', '601985': '核电',
    '600362': '铜冶炼',
    '300624': '软件', '300418': '软件', '300058': '传媒',
    '600909': '券商', '601162': '券商', '601211': '券商',
    '002172': '医药', '300017': '软件', '300274': '光伏',
    '002714': '生猪', '300498': '生猪', '000876': '生猪',
    '002157': '生猪', '000735': '生猪',
    '002261': '软件', '600895': '园区',
    '002463': 'PCB', '600183': 'PCB', '002916': 'PCB',
    '002384': 'PCB', '300476': 'PCB',
    '688836': '机器人', '688017': '机器人', '002747': '机器人',
    '300024': '机器人', '002979': '机器人',
    '002173': '脑机接口', '300007': '脑机接口',
}
DATA_DIR = './finetune_csv/data'
LOG_DIR = './prediction_log'
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']

PARAMS_CORRECT_CON = {'T': 0.5, 'top_p': 0.9, 'sample_count': 15, 'lookback': 120}
PARAMS_CORRECT_AGR = {'T': 1.0, 'top_p': 0.95, 'sample_count': 5, 'lookback': 120}
PARAMS_WRONG_ENSEMBLE = [
    {'T': 0.7, 'top_p': 0.92, 'sample_count': 10, 'lookback': 200},
    {'T': 0.6, 'top_p': 0.90, 'sample_count': 12, 'lookback': 200},
    {'T': 0.8, 'top_p': 0.93, 'sample_count': 8, 'lookback': 150},
]
PARAMS_WRONG_AGR = {'T': 1.0, 'top_p': 0.95, 'sample_count': 5, 'lookback': 150}
MOM_WEIGHT_CON = 0.3
MOM_WEIGHT_AGR = 0.45


def load_df(code, skip_indicators=False):
    path = os.path.join(DATA_DIR, f'{code}_fundflow.csv')
    df = pd.read_csv(path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    for col in FEATURES:
        if col not in df.columns: df[col] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0)
    m = df['amount'] == 0
    if m.any():
        df.loc[m, 'amount'] = df.loc[m, 'volume'] * df.loc[m, ['open','high','low','close']].mean(axis=1)
    # 补齐资金流列
    for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
        if fc not in df.columns: df[fc] = 0.0
        else: df[fc] = df[fc].fillna(0)
    # 追加技术指标和缠论特征（用于信号计算和后续微调数据准备）
    if not skip_indicators:
        vp_feats = calc_volume_price_features(df)
        chanlun_feats = calc_chanlun_simple(df)
        for col in vp_feats.columns:
            df[col] = vp_feats[col]
        for col in chanlun_feats.columns:
            df[col] = chanlun_feats[col]
    return df


# ============================================================
# 步骤1: 更新数据
# ============================================================
def update_data():
    print('\n' + '='*60)
    print('  步骤1: 更新数据到最新交易日')
    print('='*60)
    strategy = TencentStrategy()
    for code, name in STOCKS.items():
        try:
            df_old = load_df(code, skip_indicators=True)
            last_date = df_old['timestamps'].iloc[-1]
            today = datetime.now().strftime('%Y%m%d')
            df_new = strategy.fetch_range(code, last_date.strftime('%Y%m%d'), today)
            if df_new.empty:
                print(f'  {name}({code}): 无新数据')
                continue
            df_new['timestamps'] = pd.to_datetime(df_new['timestamps'])
            # 过滤零价格行（腾讯qfq数据可能含price=0的异常行）
            price_zero = (df_new['close'] == 0) | (df_new['open'] == 0)
            if price_zero.any():
                print(f'  {name}({code}): 过滤{price_zero.sum()}条零价格行')
                df_new = df_new[~price_zero].reset_index(drop=True)
            if df_new.empty:
                print(f'  {name}({code}): 过滤后无有效数据')
                continue
            # 合并：只追加旧数据中没有的新日期行，旧数据完全保留
            existing_dates = set(df_old['timestamps'].dt.strftime('%Y-%m-%d'))
            df_new_only = df_new[~df_new['timestamps'].dt.strftime('%Y-%m-%d').isin(existing_dates)].copy()
            if df_new_only.empty:
                print(f'  {name}({code}): 无新日期数据')
                continue
            # 腾讯数据不含amount，用volume*均价估算
            m = df_new_only['amount'] == 0
            if m.any():
                avg_p = df_new_only.loc[m, ['open','high','low','close']].mean(axis=1)
                df_new_only.loc[m, 'amount'] = df_new_only.loc[m, 'volume'] * avg_p
            # 补齐资金流列
            for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
                if fc not in df_new_only.columns:
                    df_new_only[fc] = 0.0
                else:
                    df_new_only[fc] = df_new_only[fc].fillna(0)
            all_cols = df_old.columns.tolist()
            for c in all_cols:
                if c not in df_new_only.columns:
                    df_new_only[c] = 0.0
            df_new_only = df_new_only[all_cols]
            merged = pd.concat([df_old, df_new_only], ignore_index=True)
            merged = merged.sort_values('timestamps').reset_index(drop=True)
            merged.to_csv(os.path.join(DATA_DIR, f'{code}_fundflow.csv'), index=False)
            new_last = merged['timestamps'].iloc[-1].strftime('%Y-%m-%d')
            added = len(df_new_only)
            print(f'  {name}({code}): 更新至 {new_last} (+{added}条)')
        except Exception as e:
            print(f'  {name}({code}): 失败 - {e}')
        time.sleep(0.5)

    # 资金流
    for code, name in STOCKS.items():
        try:
            secid = f'1.{code}' if code.startswith(('6','9')) else f'0.{code}'
            params = {'reportName':'RPT_DMSK_TS_STOCKNEW','columns':'ALL',
                      'filter':f'(SECURITY_CODE="{code}")','pageSize':'1','pageNumber':'1',
                      'sortColumns':'TRADE_DATE','sortTypes':'-1'}
            headers = {'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
            r = requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=params,headers=headers,timeout=15)
            data = r.json().get('result',{}).get('data',[])
            if not data: continue
            item = data[0]
            prime = float(item.get('PRIME_INFLOW',0) or 0)
            super_in = float(item.get('SUPERDEAL_INFLOW',0) or 0)
            super_out = float(item.get('SUPERDEAL_OUTFLOW',0) or 0)
            huge_net = super_in - super_out
            trade_date = str(item.get('TRADE_DATE',''))[:10]

            df = load_df(code, skip_indicators=True)
            mask = df['timestamps'].dt.strftime('%Y-%m-%d') == trade_date
            if mask.any():
                idx = mask.idxmax()
                df.loc[idx,'main_net_inflow'] = prime
                df.loc[idx,'huge_net_inflow'] = huge_net
                amt = df.loc[idx,'amount']
                df.loc[idx,'main_net_inflow_ratio'] = prime/amt*100 if amt > 0 else 0
                df.to_csv(os.path.join(DATA_DIR, f'{code}_fundflow.csv'), index=False)
        except: pass
        time.sleep(0.3)

    # 合并文件
    all_dfs = [load_df(c, skip_indicators=True) for c in STOCKS]
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, 'stock_with_fund_flow.csv'), index=False)
    print(f'  合并文件: {len(combined)}条')


# ============================================================
# 步骤2: 验证昨日预测
# ============================================================
def validate_yesterday(predictor):
    # 从今天往回找最新的、且有实际数据可验证的 pred.csv
    pred_date = None
    pred_file = None
    first_code = list(STOCKS.keys())[0]
    df_check = load_df(first_code, skip_indicators=True)
    for d in range(0, 5):
        candidate = (datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d')
        candidate_file = os.path.join(LOG_DIR, f'{candidate}_pred.csv')
        if not os.path.exists(candidate_file):
            continue
        # 检查该日期是否有实际数据
        if not df_check[df_check['timestamps'].dt.strftime('%Y-%m-%d') == candidate].empty:
            pred_date = candidate
            pred_file = candidate_file
            break

    if not pred_file:
        print(f'\n  未找到可验证的预测文件，跳过验证')
        return {}

    print(f'\n{"="*60}')
    print(f'  步骤2: 验证 {pred_date} 预测')
    print(f'{"="*60}')

    pred_df = pd.read_csv(pred_file, dtype={'code': str})
    # 确保code列是6位字符串（前导零补齐）
    pred_df['code'] = pred_df['code'].astype(str).str.zfill(6)
    results = []
    correct_codes = []
    wrong_codes = []

    for _, row in pred_df.iterrows():
        code = row['code']
        if code not in STOCKS: continue
        name = STOCKS[code]
        df = load_df(code, skip_indicators=True)
        actual_row = df[df['timestamps'].dt.strftime('%Y-%m-%d') == pred_date]
        if actual_row.empty:
            # 取最新一天
            actual_row = df.tail(1)
        actual_close = actual_row.iloc[0]['close']
        prev_close = df[df['timestamps'] < actual_row.iloc[0]['timestamps']]['close'].iloc[-1]
        actual_chg = (actual_close / prev_close - 1) * 100
        pred_chg = float(str(row['pred_chg']).replace('+',''))

        pred_dir = '↑' if pred_chg > 0 else '↓'
        actual_dir = '↑' if actual_chg > 0 else '↓'
        direction_ok = pred_dir == actual_dir
        error = abs(pred_chg - actual_chg)

        results.append({
            'code': code, 'name': name, 'mode': row['mode'],
            'pred_chg': pred_chg, 'actual_chg': actual_chg,
            'direction_ok': direction_ok, 'error': error,
        })

        if row['mode'] == pred_df[pred_df['code']==code].iloc[0]['mode']:
            if direction_ok:
                correct_codes.append(code)
            else:
                wrong_codes.append(code)

        status = '✓' if direction_ok else '✗'
        print(f'  {name}({code}) {row["mode"]:6s}: 预测{pred_chg:+.2f}%{pred_dir} 实际{actual_chg:+.2f}%{actual_dir} {status} 误差{error:.2f}%')

    # 方向准确率
    con_results = [r for r in results if '保守' in r['mode']]
    agr_results = [r for r in results if '激进' in r['mode']]
    con_acc = sum(1 for r in con_results if r['direction_ok']) / len(con_results) * 100 if con_results else 0
    agr_acc = sum(1 for r in agr_results if r['direction_ok']) / len(agr_results) * 100 if agr_results else 0
    con_avg_err = np.mean([r['error'] for r in con_results]) if con_results else 0
    agr_avg_err = np.mean([r['error'] for r in agr_results]) if agr_results else 0

    print(f'\n  保守方向准确率: {con_acc:.0f}% 平均误差: {con_avg_err:.2f}%')
    print(f'  激进方向准确率: {agr_acc:.0f}% 平均误差: {agr_avg_err:.2f}%')

    # 保存验证结果
    val_file = os.path.join(LOG_DIR, f'{pred_date}_validate.csv')
    df_val = pd.DataFrame(results)
    for col in ['pred_chg', 'actual_chg', 'error']:
        if col in df_val.columns:
            df_val[col] = df_val[col].round(2)
    df_val.to_csv(val_file, index=False)

    return {'correct': list(set(correct_codes)), 'wrong': list(set(wrong_codes)),
            'detail': {r['code']: '✓' if r['direction_ok'] else '✗' for r in results}}


# ============================================================
# 步骤3: 预测明日
# ============================================================
def predict_tomorrow(predictor, validation_result):
    print(f'\n{"="*60}')
    print(f'  步骤3: 预测下一交易日 (自适应参数+信号修正)')
    print(f'{"="*60}')

    correct_codes = validation_result.get('correct', [])
    wrong_codes = validation_result.get('wrong', [])
    val_detail = validation_result.get('detail', {})

    # 加载验证历史，用于自适应参数
    val_history = load_validation_history(LOG_DIR, n_days=10)

    def _reason(mom_signal, pct_5d, pct_3d, fund_ratio, is_correct, correction_detail=None):
        parts = []
        parts.append('延续' if is_correct else '调参')
        mom_sum = pct_5d + pct_3d
        parts.append('动量↑' if mom_sum > 0 else '动量↓')
        if fund_ratio > 0.5:
            parts.append('主力流入')
        elif fund_ratio < -0.5:
            parts.append('主力流出')
        # 信号修正详情
        if correction_detail:
            sig_parts = []
            for k, v in correction_detail.items():
                if k in ('低准确率缩小', '高准确率放大'):
                    if v: sig_parts.append(k)
                elif isinstance(v, (int, float)) and abs(v) > 0.05:
                    sig_parts.append(k)
            if sig_parts:
                parts.append('信号[' + '+'.join(sig_parts) + ']')
        # 趋势信息
        if market_regime['regime'] != 'sideways':
            parts.append(f'趋势{market_regime["regime"]}')
        return '+'.join(parts)

    # 计算预测目标日
    first_code = list(STOCKS.keys())[0]
    sample_df = load_df(first_code)
    data_last_date = sample_df['timestamps'].iloc[-1]
    target_date = data_last_date + timedelta(days=1)
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    print(f'  预测目标日: {target_date.strftime("%Y-%m-%d")} (数据截止 {data_last_date.strftime("%Y-%m-%d")})')

    # 计算市场波动率 + 趋势状态 + 板块动量
    all_dfs = {code: load_df(code) for code in STOCKS}
    market_vol = calc_market_volatility(all_dfs)
    market_regime = detect_market_regime(all_dfs)
    # 短期趋势指标 (近5日, 修复 MA20 滞后问题: 11-17 预测时 MA20 仍显示uptrend 但近5日已转弱)
    short_trend = 0.0
    short_up_ratio = 0.5
    try:
        rets5 = [(df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
                 for df in all_dfs.values() if len(df) >= 6]
        if rets5:
            short_trend = float(np.mean(rets5))
            short_up_ratio = sum(1 for r in rets5 if r > 0) / len(rets5)
    except:
        pass
    print(f'  市场波动率: {market_vol:.4f}')
    print(f'  市场趋势: {market_regime["regime"]} (强度{market_regime["strength"]:.2f}, 站上MA20占比{market_regime["up_ratio"]:.0%})')
    print(f'  短期趋势: 近5日{short_trend:+.2f}% (上涨占比{short_up_ratio:.0%})')

    results = []

    def _apply_trend_bias(chg, regime, short_trend=0.0, short_up_ratio=0.5):
        """根据市场趋势叠加偏置（仅明确趋势启用，震荡市禁用）
        short_trend: 近5日市场平均涨跌幅(%), short_up_ratio: 近5日上涨占比

        消融实验结论：趋势偏置在震荡市有害（-0.07% ~ -0.31%），
        仅在趋势强度 >= 0.5 时启用。
        """
        # 门控：震荡市或趋势强度不足时不修正
        if regime['regime'] == 'sideways' or regime['strength'] < 0.5:
            return chg

        # 短期趋势过滤：近5日大幅转弱时提前削弱看涨
        if short_trend < -1.5 and short_up_ratio < 0.40:
            if chg > 0:
                chg = chg * 0.4 - 0.3  # 削弱看涨并施加负偏置
            else:
                chg = chg * 1.3  # 强化看跌
            return max(-10, min(10, chg))

        if regime['regime'] == 'uptrend':
            # 上升趋势：强化看涨、削弱看跌
            if chg > 0:
                chg = chg * (1 + 0.3 * regime['strength'])
            else:
                chg = chg * (1 - 0.3 * regime['strength']) + 0.3 * regime['strength']
        elif regime['regime'] == 'downtrend':
            # 下降趋势：强化看跌、削弱看涨
            if chg < 0:
                chg = chg * (1 + 0.3 * regime['strength'])
            else:
                chg = chg * (1 - 0.3 * regime['strength']) - 0.3 * regime['strength']
        return max(-10, min(10, chg))

    def _calc_confidence(predictor, lb, T, top_p, fd, last_close):
        """多次采样计算置信区间"""
        closes = []
        for _ in range(3):
            x_df = df.iloc[-lb:][FEATURES].reset_index(drop=True)
            x_ts = df.iloc[-lb:]['timestamps'].reset_index(drop=True)
            p = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
                                 pred_len=1, T=T, top_p=top_p, sample_count=3, verbose=False)
            closes.append(p['close'].iloc[0])
            time.sleep(0.1)
        mean_c = np.mean(closes)
        std_c = np.std(closes) if len(closes) > 1 else 0
        stderr_pct = (std_c / last_close) * 100
        confidence = max(0, min(1, 1 - stderr_pct / 2))
        return mean_c, stderr_pct, confidence

    for code, name in STOCKS.items():
        df = all_dfs[code]
        last_close = df['close'].iloc[-1]
        last_date = df['timestamps'].iloc[-1]
        fd = []
        c = last_date + timedelta(days=1)
        while len(fd) < 1:
            if c.weekday() < 5: fd.append(c)
            c += timedelta(days=1)

        # 动量计算
        pct_5d = df['close'].iloc[-5:].pct_change().dropna().sum() * 100
        pct_3d = df['close'].iloc[-3:].pct_change().dropna().sum() * 100
        mom_signal = 1 if (pct_5d + pct_3d) > 0 else -1

        is_correct = code in correct_codes if correct_codes else True

        fund_ratio = df['main_net_inflow_ratio'].iloc[-1] if 'main_net_inflow_ratio' in df.columns else 0

        # ====== 自适应参数 ======
        stock_metrics = calc_stock_metrics(val_history, code)
        stock_params = adaptive_params(stock_metrics, market_vol)

        # ====== 量价+缠论信号 ======
        vp_signals = calc_volume_price_signals(df)
        chanlun_signals = calc_chanlun_signals(df)

        if is_correct:
            # 方向正确: 用自适应保守参数
            con_params = {'T': min(stock_params['T'], 0.6), 'top_p': stock_params['top_p'],
                          'sample_count': stock_params['sample_count'], 'lookback': stock_params['lookback']}
            agr_params = {'T': stock_params['T'] + 0.2, 'top_p': min(stock_params['top_p'] + 0.03, 0.98),
                          'sample_count': max(stock_params['sample_count'] - 5, 3), 'lookback': stock_params['lookback']}

            for mode, params in [('保守', con_params), ('激进', agr_params)]:
                lb = params['lookback']
                x_df = df.iloc[-lb:][FEATURES].reset_index(drop=True)
                x_ts = df.iloc[-lb:]['timestamps'].reset_index(drop=True)
                pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
                                        pred_len=1, T=params['T'], top_p=params['top_p'],
                                        sample_count=params['sample_count'], verbose=False)
                chg = (pred['close'].iloc[0] / last_close - 1) * 100
                chg = max(-10, min(10, chg))
                # 信号修正
                chg, correction_detail = signal_correction(chg, vp_signals, chanlun_signals, stock_metrics)
                # 趋势偏置（仅明确趋势启用）
                chg = _apply_trend_bias(chg, market_regime, short_trend, short_up_ratio)
                chg = max(-10, min(10, chg))
                # 置信区间
                _, stderr, confidence = _calc_confidence(predictor, lb, params['T'], params['top_p'], fd, last_close)
                pred_high = pred['high'].iloc[0]
                pred_low = pred['low'].iloc[0]
                reason = _reason(mom_signal, pct_5d, pct_3d, fund_ratio, is_correct, correction_detail)
                results.append({'code':code,'name':name,'mode':mode,'part':'自适应延续',
                               'last_close':last_close,'pred_close':last_close*(1+chg/100),'pred_chg':chg,
                               'pred_high':pred_high,'pred_low':pred_low,'reason':reason,
                               'stderr':stderr,'confidence':confidence})
                print(f'  {name} {mode}: {chg:+.2f}% (±{stderr:.2f}% conf={confidence:.0%}) (高{pred_high:.2f} 低{pred_low:.2f}) T={params["T"]:.2f}')
                time.sleep(0.2)
        else:
            # 方向错误: 集成+自适应动量修正+信号修正
            ensemble_preds = []
            ensemble_highs = []
            ensemble_lows = []
            for ps in PARAMS_WRONG_ENSEMBLE:
                lb = ps['lookback']
                x_df = df.iloc[-lb:][FEATURES].reset_index(drop=True)
                x_ts = df.iloc[-lb:]['timestamps'].reset_index(drop=True)
                pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
                                        pred_len=1, T=ps['T'], top_p=ps['top_p'],
                                        sample_count=ps['sample_count'], verbose=False)
                chg = (pred['close'].iloc[0] / last_close - 1) * 100
                ensemble_preds.append(chg)
                ensemble_highs.append(pred['high'].iloc[0])
                ensemble_lows.append(pred['low'].iloc[0])
                time.sleep(0.2)

            # 自适应动量权重
            mom_weight = stock_params['mom_weight']
            con_median = np.median(ensemble_preds)
            con_adj = con_median + mom_signal * abs(pct_5d + pct_3d) * mom_weight * 0.1
            con_adj = max(-10, min(10, con_adj))
            # 信号修正
            con_adj, con_correction = signal_correction(con_adj, vp_signals, chanlun_signals, stock_metrics)
            # 趋势偏置（仅明确趋势启用）
            con_adj = _apply_trend_bias(con_adj, market_regime, short_trend, short_up_ratio)
            con_adj = max(-10, min(10, con_adj))
            con_high = np.median(ensemble_highs)
            con_low = np.median(ensemble_lows)

            # 激进: 用自适应参数
            agr_lookback = stock_params['lookback']
            x_df = df.iloc[-agr_lookback:][FEATURES].reset_index(drop=True)
            x_ts = df.iloc[-agr_lookback:]['timestamps'].reset_index(drop=True)
            pred_agr = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
                                        pred_len=1, T=stock_params['T'], top_p=stock_params['top_p'],
                                        sample_count=stock_params['sample_count'], verbose=False)
            agr_raw = (pred_agr['close'].iloc[0] / last_close - 1) * 100
            agr_adj = agr_raw + mom_signal * abs(pct_5d + pct_3d) * (mom_weight + 0.15) * 0.1
            agr_adj = max(-10, min(10, agr_adj))
            # 信号修正
            agr_adj, agr_correction = signal_correction(agr_adj, vp_signals, chanlun_signals, stock_metrics)
            # 趋势偏置（仅明确趋势启用）
            agr_adj = _apply_trend_bias(agr_adj, market_regime, short_trend, short_up_ratio)
            agr_adj = max(-10, min(10, agr_adj))
            # 置信区间
            _, agr_stderr, agr_confidence = _calc_confidence(predictor, agr_lookback, stock_params['T'], stock_params['top_p'], fd, last_close)
            agr_high = pred_agr['high'].iloc[0]
            agr_low = pred_agr['low'].iloc[0]

            reason = _reason(mom_signal, pct_5d, pct_3d, fund_ratio, is_correct, con_correction)
            results.append({'code':code,'name':name,'mode':'保守调参','part':'自适应修正',
                           'last_close':last_close,'pred_close':last_close*(1+con_adj/100),'pred_chg':con_adj,
                           'pred_high':con_high,'pred_low':con_low,'reason':reason,
                           'stderr':0,'confidence':0})
            results.append({'code':code,'name':name,'mode':'激进调参','part':'自适应修正',
                           'last_close':last_close,'pred_close':last_close*(1+agr_adj/100),'pred_chg':agr_adj,
                           'pred_high':agr_high,'pred_low':agr_low,'reason':reason,
                           'stderr':agr_stderr,'confidence':agr_confidence})
            print(f'  {name} 保守调参: {con_adj:+.2f}% | 激进调参: {agr_adj:+.2f}% (±{agr_stderr:.2f}% conf={agr_confidence:.0%}) T={stock_params["T"]:.2f}')
            time.sleep(0.2)

    # ====== 步骤4: 新闻情绪过滤 ======
    print(f'\n{"="*60}')
    print(f'  步骤4: 新闻情绪过滤 (最近3天新闻)')
    print(f'{"="*60}')
    sentiment_rows = []
    for code, name in STOCKS.items():
        try:
            s = get_stock_sentiment(code)
        except Exception:
            s = {"count": 0, "score": 0.0, "neg_ratio": 0.0, "pos_ratio": 0.0,
                 "negative_titles": []}
        tag = None
        for r in results:
            if r['code'] != code: continue
            new_chg, tag = apply_sentiment_filter(r['pred_chg'], s)
            if tag:
                r['pred_chg'] = new_chg
                r['pred_close'] = r['last_close'] * (1 + new_chg / 100)
                r['reason'] = r['reason'] + '+' + tag
        neg_str = '; '.join(s['negative_titles'][:2])
        print(f'  {name}({code}): 情绪分{s["score"]:+.2f} 利空占比{s["neg_ratio"]:.0%} 共{s["count"]}条 {tag or ""}')
        sentiment_rows.append({'code': code, 'name': name, 'news_count': s['count'],
                               'score': round(s['score'], 2), 'neg_ratio': round(s['neg_ratio'], 2),
                               'pos_ratio': round(s['pos_ratio'], 2), 'filter_tag': tag or '',
                               'negative_titles': neg_str})
        time.sleep(0.3)

    # ====== 步骤5: 资金流方向确认 ======
    # 核心逻辑：当预测方向与资金流方向一致时，准确率从54%→66%
    print(f'\n{"="*60}')
    print(f'  步骤5: 资金流方向确认 (预测方向×资金流方向)')
    print(f'{"="*60}')
    for r in results:
        code = r['code']
        pred_dir = 'up' if r['pred_chg'] > 0 else 'down'
        try:
            df_fund = load_df(code, skip_indicators=True)
            fund_ratio = df_fund['main_net_inflow_ratio'].iloc[-1] if 'main_net_inflow_ratio' in df_fund.columns else 0
            fund_dir = 'in' if fund_ratio > 0 else 'out'
        except:
            fund_dir = 'unknown'
            fund_ratio = 0

        # 方向一致性判断
        if pred_dir == 'up' and fund_dir == 'in':
            r['fund_confirm'] = '同向↑'  # 高置信：预测涨+资金流入
            r['fund_confidence'] = 'high'
        elif pred_dir == 'down' and fund_dir == 'out':
            r['fund_confirm'] = '同向↓'  # 高置信：预测跌+资金流出
            r['fund_confidence'] = 'high'
        elif pred_dir == 'up' and fund_dir == 'out':
            r['fund_confirm'] = '冲突↑↓'  # 低置信：预测涨但资金流出
            r['fund_confidence'] = 'low'
        elif pred_dir == 'down' and fund_dir == 'in':
            r['fund_confirm'] = '冲突↓↑'  # 低置信：预测跌但资金流入
            r['fund_confidence'] = 'low'
        else:
            r['fund_confirm'] = '未知'
            r['fund_confidence'] = 'medium'

        r['fund_ratio'] = fund_ratio
        r['fund_dir'] = fund_dir
        name = STOCKS.get(code, '')
        print(f'  {name}({code}): 预测{pred_dir} 资金{fund_dir}({fund_ratio:+.2f}%) → {r["fund_confirm"]}')

    tomorrow_str = fd[0].strftime('%Y-%m-%d')
    senti_file = os.path.join(LOG_DIR, f'{tomorrow_str}_sentiment.csv')
    pd.DataFrame(sentiment_rows).to_csv(senti_file, index=False)
    print(f'  情绪日志已保存: {senti_file}')

    # 保存预测
    pred_file = os.path.join(LOG_DIR, f'{tomorrow_str}_pred.csv')
    df_pred = pd.DataFrame(results)
    for col in ['pred_high', 'pred_low', 'pred_close', 'pred_chg', 'stderr', 'confidence',
                'fund_ratio']:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].round(2)
    df_pred.to_csv(pred_file, index=False)
    print(f'\n  预测已保存: {pred_file}')

    # 加载最近4个交易日的预测和实际数据（前4日，因为最新一天的预测可能还没验证）
    history = {}  # {code: [(date_str, pred_avg, actual_chg, result), ...]}
    pred_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_pred.csv')])[-5:-1]  # 最近4个（不含今天的）
    for pf in pred_files:
        date_str = pf.replace('_pred.csv', '')
        try:
            pdf = pd.read_csv(os.path.join(LOG_DIR, pf), dtype={'code': str})
            pdf['code'] = pdf['code'].astype(str).str.zfill(6)
            for code in STOCKS:
                row_p = pdf[pdf['code'] == code]
                if row_p.empty: continue
                # 取保守和激进两行的pred_chg均值作为综合预测
                pred_con = row_p.iloc[0].get('pred_chg', 0)
                pred_agr = row_p.iloc[-1].get('pred_chg', 0) if len(row_p) > 1 else pred_con
                pred_avg = (pred_con + pred_agr) / 2
                # 从K线数据计算实际涨跌
                actual_chg = None
                try:
                    df_k = load_df(code, skip_indicators=True)
                    target_date = pd.to_datetime(date_str)
                    actual_row = df_k[df_k['timestamps'].dt.strftime('%Y-%m-%d') == date_str]
                    if not actual_row.empty:
                        actual_close = actual_row.iloc[0]['close']
                        prev = df_k[df_k['timestamps'] < actual_row.iloc[0]['timestamps']]['close']
                        if not prev.empty:
                            actual_chg = (actual_close / prev.iloc[-1] - 1) * 100
                except: pass
                # 判断结果
                if actual_chg is not None and not (isinstance(actual_chg,float) and pd.isna(actual_chg)):
                    pred_dir = '↑' if pred_avg > 0 else '↓'
                    actual_dir = '↑' if actual_chg > 0 else '↓'
                    result = '✓' if pred_dir == actual_dir else '✗'
                else:
                    result = '?'
                history.setdefault(code, []).append((date_str[-5:], pred_avg, actual_chg, result))
        except Exception as e:
            pass

    # 汇总排名
    print(f'\n  {tomorrow_str} 综合排名:')
    print(f'  {"排名":4s} {"股票":16s} {"综合":>8s} {"±误差":>7s} {"置信":>5s} {"资金流":>6s} {"保守":>8s} {"激进":>8s} {"趋势":>6s} {"验证":>4s} {"理由"}')
    print(f'  {"-"*120}')
    ranked = []
    for code in STOCKS:
        r_c = [r for r in results if r['code']==code and '保守' in r['mode']]
        r_a = [r for r in results if r['code']==code and '激进' in r['mode']]
        if r_c and r_a:
            avg = (r_c[0]['pred_chg'] + r_a[0]['pred_chg']) / 2
            conf = r_a[0].get('confidence', 0)
            stderr = r_a[0].get('stderr', 0)
            fund_conf = r_c[0].get('fund_confirm', '-')
            ranked.append({'name':STOCKS[code],'code':code,'avg':avg,'con':r_c[0]['pred_chg'],'agr':r_a[0]['pred_chg'],
                           'pred_high':r_a[0].get('pred_high',0),'pred_low':r_a[0].get('pred_low',0),
                           'reason':r_c[0].get('reason',''),'stderr':stderr,'confidence':conf,
                           'fund_confirm':fund_conf,
                           'fund_confidence':r_c[0].get('fund_confidence','')})
    ranked.sort(key=lambda x: x['avg'], reverse=True)
    for i, r in enumerate(ranked, 1):
        arrow = '↑' if r['avg'] > 0 else '↓'
        val_res = val_detail.get(r['code'], '-')
        conf_str = f"{r['confidence']:.0%}" if r['confidence'] > 0 else '-'
        trend_str = market_regime['regime'][:4]
        fund_str = r['fund_confirm'] if r['fund_confirm'] != '未知' else '-'
        # 资金流确认的标记
        star = '★' if r.get('fund_confidence') == 'high' else ' '
        print(f'  {i:4d}{star} {r["name"]+"("+r["code"]+")":16s} {r["avg"]:+7.2f}%{arrow} ±{r["stderr"]:5.2f}% {conf_str:>5s} {fund_str:>6s} {r["con"]:+7.2f}% {r["agr"]:+7.2f}% {trend_str:>6s} {val_res:>4s} {r["reason"]}')
    print(f'\n  ★ = 资金流确认（预测方向与资金流方向一致，历史准确率66.5%）')

    return results


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-update', action='store_true')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f'Kronos 每日预测系统 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # 步骤1: 更新数据
    if not args.skip_update:
        update_data()

    # 加载模型
    print('\n加载模型...')
    tokenizer = KronosTokenizer.from_pretrained('./pretrained/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('./pretrained/Kronos-base')
    predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5.0)
    print('模型加载完成')

    # 步骤2: 验证昨日
    val_result = validate_yesterday(predictor)

    # 步骤3: 预测明日
    predict_tomorrow(predictor, val_result)

    print(f'\n全部完成 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')


if __name__ == '__main__':
    main()
