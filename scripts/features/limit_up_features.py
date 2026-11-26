#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨停板专用特征工程

提供 5 大类特征，用于连板预测模型:
  1. 量价异动特征 (volume_price_features)
  2. 资金流向特征 (fund_flow_features)
  3. 板块联动特征 (sector_features)
  4. 市场情绪特征 (market_sentiment_features)
  5. 技术形态特征 (technical_features)

用法:
  from scripts.features.limit_up_features import build_all_features
  features = build_all_features(stock_df, zt_df, prev_df)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'limit_up_data')


# ============================================================
# 1. 量价异动特征
# ============================================================
def volume_price_features(df):
    """量价异动特征

    输入: 单只股票历史行情 DataFrame (含 close, volume, amount, high, low)
    输出: 特征 dict

    特征:
      - vol_ratio_5d: 5 日量比 (当日 volume / 5 日均量)
      - vol_ratio_20d: 20 日量比
      - price_vol_divergence: 价量背离度 (价格涨但量缩为负)
      - amplitude: 振幅 (high-low)/close
      - gap_up: 跳空缺口 (open/prev_close - 1)
      - consecutive_up: 连续上涨天数
      - volume_climax: 成交量峰值 (当日 volume / 60 日均量 > 2)
    """
    if df is None or len(df) < 5:
        return {}

    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values

    # 量比
    vol_ma5 = np.mean(volume[-6:-1]) if len(volume) >= 6 else np.mean(volume)
    vol_ma20 = np.mean(volume[-21:-1]) if len(volume) >= 21 else np.mean(volume)
    vol_ratio_5d = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1.0
    vol_ratio_20d = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1.0

    # 价量背离: 价格涨但量缩
    price_chg = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0
    vol_chg = (volume[-1] - volume[-2]) / volume[-2] if volume[-2] > 0 else 0
    divergence = price_chg - vol_chg  # 正=量价配合，负=背离

    # 振幅
    amplitude = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0

    # 跳空缺口
    if 'open' in df.columns and len(df) >= 2:
        gap_up = (df['open'].iloc[-1] / close[-2] - 1) if close[-2] > 0 else 0
    else:
        gap_up = 0

    # 连续上涨天数
    consecutive_up = 0
    for i in range(len(close) - 1, 0, -1):
        if close[i] > close[i - 1]:
            consecutive_up += 1
        else:
            break

    # 成交量峰值
    vol_ma60 = np.mean(volume[-61:-1]) if len(volume) >= 61 else np.mean(volume)
    volume_climax = 1 if (vol_ma60 > 0 and volume[-1] / vol_ma60 > 2.0) else 0

    return {
        'vol_ratio_5d': round(float(vol_ratio_5d), 3),
        'vol_ratio_20d': round(float(vol_ratio_20d), 3),
        'price_vol_divergence': round(float(divergence), 4),
        'amplitude': round(float(amplitude), 4),
        'gap_up': round(float(gap_up), 4),
        'consecutive_up': int(consecutive_up),
        'volume_climax': int(volume_climax),
    }


# ============================================================
# 2. 资金流向特征
# ============================================================
def fund_flow_features(df):
    """资金流向特征

    输入: 单只股票历史行情 DataFrame (含 main_net_inflow_ratio 等)
    输出: 特征 dict

    特征:
      - main_net_inflow_ratio: 主力净占比 (%)
      - huge_net_inflow_ratio: 超大单净占比 (%)
      - main_net_inflow_3d: 近 3 日主力净占比均值
      - main_net_inflow_5d: 近 5 日主力净占比均值
      - fund_consistency: 资金一致性 (近 3 日主力净流入方向一致为 1)
      - fund_acceleration: 资金加速度 (今日主力净占比 - 昨日)
    """
    if df is None or len(df) < 3:
        return {}

    # 主力净占比
    if 'main_net_inflow_ratio' in df.columns:
        main_ratio = df['main_net_inflow_ratio'].values
        main_1d = float(main_ratio[-1]) if len(main_ratio) > 0 else 0
        main_3d = float(np.mean(main_ratio[-3:])) if len(main_ratio) >= 3 else main_1d
        main_5d = float(np.mean(main_ratio[-5:])) if len(main_ratio) >= 5 else main_1d
        fund_accel = main_1d - float(main_ratio[-2]) if len(main_ratio) >= 2 else 0

        # 资金一致性: 近 3 日方向一致
        if len(main_ratio) >= 3:
            signs = np.sign(main_ratio[-3:])
            fund_consistency = 1 if (signs == signs[0]).all() else 0
        else:
            fund_consistency = 0
    else:
        main_1d = main_3d = main_5d = fund_accel = 0
        fund_consistency = 0

    # 超大单净占比
    if 'huge_net_inflow' in df.columns and 'amount' in df.columns:
        huge = df['huge_net_inflow'].values
        amt = df['amount'].values
        huge_ratio = (huge[-1] / amt[-1] * 100) if amt[-1] > 0 else 0
    else:
        huge_ratio = 0

    return {
        'main_net_inflow_ratio': round(main_1d, 3),
        'huge_net_inflow_ratio': round(float(huge_ratio), 3),
        'main_net_inflow_3d': round(main_3d, 3),
        'main_net_inflow_5d': round(main_5d, 3),
        'fund_consistency': int(fund_consistency),
        'fund_acceleration': round(fund_accel, 3),
    }


# ============================================================
# 3. 板块联动特征
# ============================================================
def sector_features(code, zt_df, sector=None):
    """板块联动特征

    输入:
      code: 股票代码
      zt_df: 当日涨停池 DataFrame (含 代码, 所属行业, 连板数)
      sector: 所属行业 (可选，如果为 None 则从 zt_df 查找)
    输出: 特征 dict

    特征:
      - sector_zt_count: 板块涨停数
      - sector_max_lb: 板块最高连板
      - sector_lb2_count: 板块高标数 (连板>=2)
      - sector_avg_seal_ratio: 板块平均封板强度
      - is_sector_leader: 是否板块龙头 (连板数最高)
      - sector_continuation_rate: 板块延续率 (连板>=2 占比)
    """
    if zt_df is None or zt_df.empty:
        return {}

    # 查找所属行业
    if sector is None:
        row = zt_df[zt_df['代码'] == code]
        if row.empty:
            return {}
        sector = str(row.iloc[0]['所属行业'])

    # 板块内所有股票
    sector_stocks = zt_df[zt_df['所属行业'] == sector]
    if sector_stocks.empty:
        return {}

    lb_values = sector_stocks['连板数'].astype(int).values

    sector_zt_count = len(sector_stocks)
    sector_max_lb = int(lb_values.max())
    sector_lb2_count = int((lb_values >= 2).sum())
    sector_continuation_rate = sector_lb2_count / sector_zt_count if sector_zt_count > 0 else 0

    # 是否龙头
    is_sector_leader = 1 if code in sector_stocks[sector_stocks['连板数'] == sector_max_lb]['代码'].values else 0

    # 板块平均封板强度
    if '封板资金' in sector_stocks.columns and '流通市值' in sector_stocks.columns:
        seal_ratios = []
        for _, r in sector_stocks.iterrows():
            mv = float(r['流通市值'])
            if mv > 0:
                seal_ratios.append(float(r['封板资金']) / mv * 100)
        sector_avg_seal_ratio = float(np.mean(seal_ratios)) if seal_ratios else 0
    else:
        sector_avg_seal_ratio = 0

    return {
        'sector_zt_count': int(sector_zt_count),
        'sector_max_lb': int(sector_max_lb),
        'sector_lb2_count': int(sector_lb2_count),
        'sector_avg_seal_ratio': round(sector_avg_seal_ratio, 3),
        'is_sector_leader': int(is_sector_leader),
        'sector_continuation_rate': round(sector_continuation_rate, 3),
    }


# ============================================================
# 4. 市场情绪特征
# ============================================================
def market_sentiment_features(prev_df, zt_df, zbgc_df=None):
    """市场情绪特征

    输入:
      prev_df: 昨日涨停池 (今日表现)
      zt_df: 今日涨停池
      zbgc_df: 今日炸板池
    输出: 特征 dict

    特征:
      - prev_first_board_count: 昨日首板数
      - prev_continuation_rate: 昨日首板连板率
      - prev_high_board_continuation: 昨日高位连板延续率
      - today_zt_count: 今日涨停总数
      - today_zbgc_rate: 今日炸板率
      - sentiment_score: 情绪评分 (-1~1, 综合判断)
      - sentiment_regime: 情绪周期 (升温/平稳/退潮)
    """
    if prev_df is None or prev_df.empty:
        return {}

    # 昨日首板连板率
    first = prev_df[prev_df['昨日连板数'] == 1]
    prev_first_board_count = len(first)
    if prev_first_board_count > 0:
        hit = first['最新价'].ge(first['涨停价'] - 0.011).sum()
        prev_continuation_rate = float(hit) / prev_first_board_count
    else:
        prev_continuation_rate = 0

    # 昨日高位连板延续率
    high = prev_df[prev_df['昨日连板数'] >= 2]
    if len(high) > 0:
        hit_h = high['最新价'].ge(high['涨停价'] - 0.011).sum()
        prev_high_board_continuation = float(hit_h) / len(high)
    else:
        prev_high_board_continuation = 0

    # 今日涨停总数
    today_zt_count = len(zt_df) if zt_df is not None and not zt_df.empty else 0

    # 今日炸板率
    if zbgc_df is not None and not zbgc_df.empty:
        today_zbgc_rate = len(zbgc_df) / (today_zt_count + len(zbgc_df)) if (today_zt_count + len(zbgc_df)) > 0 else 0
    else:
        today_zbgc_rate = 0

    # 情绪评分
    sentiment_score = (prev_continuation_rate * 0.5 +
                       prev_high_board_continuation * 0.3 -
                       today_zbgc_rate * 0.2)

    # 情绪周期
    if prev_continuation_rate >= 0.35 and today_zbgc_rate <= 0.30:
        sentiment_regime = '升温'
    elif prev_continuation_rate <= 0.15 or today_zbgc_rate >= 0.45:
        sentiment_regime = '退潮'
    else:
        sentiment_regime = '平稳'

    return {
        'prev_first_board_count': int(prev_first_board_count),
        'prev_continuation_rate': round(prev_continuation_rate, 3),
        'prev_high_board_continuation': round(prev_high_board_continuation, 3),
        'today_zt_count': int(today_zt_count),
        'today_zbgc_rate': round(today_zbgc_rate, 3),
        'sentiment_score': round(sentiment_score, 3),
        'sentiment_regime': sentiment_regime,
    }


# ============================================================
# 5. 技术形态特征
# ============================================================
def technical_features(df):
    """技术形态特征

    输入: 单只股票历史行情 DataFrame
    输出: 特征 dict

    特征:
      - ma5_slope: 5 日均线斜率 (%)
      - ma20_slope: 20 日均线斜率 (%)
      - price_vs_ma20: 价格相对 MA20 偏离 (%)
      - breakout_20d: 是否突破 20 日新高
      - support_distance: 距离支撑位 (MA20) 的幅度 (%)
      - rsi14: RSI(14)
      - macd_hist: MACD 柱状图
    """
    if df is None or len(df) < 20:
        return {}

    close = df['close'].values

    # 均线斜率
    ma5 = np.mean(close[-5:])
    ma5_prev = np.mean(close[-10:-5]) if len(close) >= 10 else ma5
    ma5_slope = (ma5 / ma5_prev - 1) * 100 if ma5_prev > 0 else 0

    ma20 = np.mean(close[-20:])
    ma20_prev = np.mean(close[-25:-5]) if len(close) >= 25 else ma20
    ma20_slope = (ma20 / ma20_prev - 1) * 100 if ma20_prev > 0 else 0

    # 价格相对 MA20
    price_vs_ma20 = (close[-1] / ma20 - 1) * 100 if ma20 > 0 else 0

    # 突破 20 日新高
    breakout_20d = 1 if close[-1] >= np.max(close[-20:]) else 0

    # 距离支撑位
    support_distance = (close[-1] / ma20 - 1) * 100 if ma20 > 0 else 0

    # RSI(14)
    if len(close) >= 15:
        deltas = np.diff(close[-15:])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rsi14 = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100
    else:
        rsi14 = 50

    # MACD
    if len(close) >= 26:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        signal = _ema(macd.reshape(-1, 1)[:, 0], 9) if len(macd) >= 9 else macd
        macd_hist = macd[-1] - signal[-1] if len(signal) > 0 else 0
    else:
        macd_hist = 0

    return {
        'ma5_slope': round(float(ma5_slope), 3),
        'ma20_slope': round(float(ma20_slope), 3),
        'price_vs_ma20': round(float(price_vs_ma20), 3),
        'breakout_20d': int(breakout_20d),
        'support_distance': round(float(support_distance), 3),
        'rsi14': round(float(rsi14), 2),
        'macd_hist': round(float(macd_hist), 4),
    }


def _ema(data, span):
    """指数移动平均"""
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


# ============================================================
# 汇总接口
# ============================================================
def build_all_features(stock_df, zt_df=None, prev_df=None, zbgc_df=None, code=None):
    """构建全部特征

    输入:
      stock_df: 单只股票历史行情
      zt_df: 当日涨停池 (可选)
      prev_df: 昨日涨停池 (可选)
      zbgc_df: 今日炸板池 (可选)
      code: 股票代码 (可选)
    输出: 合并后的特征 dict
    """
    features = {}

    # 量价特征
    features.update(volume_price_features(stock_df))

    # 资金流特征
    features.update(fund_flow_features(stock_df))

    # 板块特征
    if zt_df is not None and code is not None:
        features.update(sector_features(code, zt_df))

    # 市场情绪特征
    sentiment = market_sentiment_features(prev_df, zt_df, zbgc_df)
    # 只取数值特征（排除字符串）
    for k, v in sentiment.items():
        if isinstance(v, (int, float)):
            features[k] = v

    # 技术形态特征
    features.update(technical_features(stock_df))

    return features


def load_stock_data(code, data_dir='./finetune_csv/data'):
    """加载单只股票历史数据"""
    path = os.path.join(data_dir, f'{code}_fundflow.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


if __name__ == '__main__':
    # 测试
    print("涨停板特征库测试")
    print("=" * 50)

    # 加载测试数据
    df = load_stock_data('000001')
    if not df.empty:
        print(f"\n测试股票: 000001 平安银行 ({len(df)} 条)")

        vp = volume_price_features(df)
        print(f"\n量价特征: {vp}")

        tf = technical_features(df)
        print(f"\n技术特征: {tf}")

        ff = fund_flow_features(df)
        print(f"\n资金流特征: {ff}")

    # 测试板块特征
    zt_path = os.path.join(DATA_DIR, '2026-11-25_zt.csv')
    if os.path.exists(zt_path):
        zt = pd.read_csv(zt_path, dtype={'代码': str})
        zt['代码'] = zt['代码'].astype(str).str.zfill(6)
        sc = sector_features('000001', zt)
        print(f"\n板块特征 (000001): {sc}")

    print("\n测试完成")
