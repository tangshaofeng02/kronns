#!/usr/bin/env python3
"""
技术指标计算模块

提供三类计算:
1. calc_volume_price_features(df) — 量价指标列（追加到df，为后续微调准备数据）
2. calc_volume_price_signals(df)  — 量价关系信号（标量，用于后处理修正模型输出）
3. calc_chanlun_simple(df)        — 简化缠论特征（分型+笔方向）
"""

import numpy as np
import pandas as pd


# ============================================================
# 量价指标列
# ============================================================
def calc_volume_price_features(df):
    """计算量价技术指标，返回与df等长的DataFrame（仅含新指标列）。

    指标列表:
      ma5, ma20, ma60       — 收盘价均线
      vol_ma5, vol_ma20    — 成交量均线
      vol_ratio            — 量比 (当日volume / 前日volume)
      price_deviation      — 均线偏离度 (close - ma20) / ma20
      rsi14                — RSI(14)
      macd, macd_signal, macd_hist — MACD(12,26,9)
    """
    close = df['close']
    volume = df['volume']

    out = pd.DataFrame(index=df.index)

    # 均线
    out['ma5'] = close.rolling(5, min_periods=1).mean()
    out['ma20'] = close.rolling(20, min_periods=1).mean()
    out['ma60'] = close.rolling(60, min_periods=1).mean()

    # 量均线
    out['vol_ma5'] = volume.rolling(5, min_periods=1).mean()
    out['vol_ma20'] = volume.rolling(20, min_periods=1).mean()

    # 量比
    prev_vol = volume.shift(1)
    out['vol_ratio'] = np.where(prev_vol > 0, volume / prev_vol, 1.0)

    # 均线偏离度
    out['price_deviation'] = np.where(out['ma20'] != 0, (close - out['ma20']) / out['ma20'], 0.0)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    out['rsi14'] = 100.0 - 100.0 / (1.0 + rs)

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd'] = ema12 - ema26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']

    return out.fillna(0)


# ============================================================
# 量价关系信号（标量，用于后处理修正）
# ============================================================
def calc_volume_price_signals(df):
    """从最近N天数据计算量价关系信号，返回标量字典。

    返回:
      vp_bullish     — 量增价增强度 (0~1)
      vp_bearish     — 量缩价跌强度 (0~1)
      vp_divergence  — 背离信号 (-1~1): 正=底背离(看多), 负=顶背离(看空)
      fund_signal    — 主力资金方向 (-1~1)
      trend_strength — 趋势强度 (0~1), 基于MA5/MA20/MA60排列
    """
    n = min(10, len(df))
    recent = df.tail(n)

    close = recent['close'].values
    volume = recent['volume'].values

    # 价格变化和成交量变化
    price_chg = np.diff(close) / close[:-1] if len(close) > 1 else np.array([0])
    vol_chg = np.diff(volume) / np.maximum(volume[:-1], 1) if len(volume) > 1 else np.array([0])

    # 量增价增强度: 价涨且量增的天数占比，加权近期
    weights = np.arange(1, len(price_chg) + 1, dtype=float)
    bullish_mask = (price_chg > 0) & (vol_chg > 0)
    vp_bullish = np.sum(bullish_mask * weights) / np.sum(weights) if len(price_chg) > 0 else 0.0

    # 量缩价跌强度
    bearish_mask = (price_chg < 0) & (vol_chg < 0)
    vp_bearish = np.sum(bearish_mask * weights) / np.sum(weights) if len(price_chg) > 0 else 0.0

    # 背离检测: 价格创新高但量没创新高 → 顶背离; 反之 → 底背离
    vp_divergence = 0.0
    if len(close) >= 5:
        recent_price = close[-5:]
        recent_vol = volume[-5:]
        price_trend = recent_price[-1] - recent_price[0]
        vol_trend = recent_vol[-1] - recent_vol[0]
        if price_trend > 0 and vol_trend < 0:
            vp_divergence = -0.6  # 顶背离
        elif price_trend < 0 and vol_trend > 0:
            vp_divergence = 0.6   # 底背离

    # 主力资金方向
    fund_signal = 0.0
    if 'main_net_inflow_ratio' in df.columns:
        ratios = df['main_net_inflow_ratio'].tail(3).values
        avg_ratio = np.mean(ratios)
        fund_signal = np.clip(avg_ratio / 5.0, -1, 1)  # 归一化: ±5%映射到±1

    # 趋势强度: MA5 > MA20 > MA60 → 强多头(1), 反之 → 强空头(1), 否则弱趋势
    trend_strength = 0.0
    if 'ma5' not in df.columns:
        feats = calc_volume_price_features(df.tail(60))
        ma5 = feats['ma5'].iloc[-1]
        ma20 = feats['ma20'].iloc[-1]
        ma60 = feats['ma60'].iloc[-1]
    else:
        ma5 = df['ma5'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]
        ma60 = df['ma60'].iloc[-1]

    if ma5 > ma20 > ma60:
        trend_strength = 1.0
    elif ma5 < ma20 < ma60:
        trend_strength = 1.0
    else:
        trend_strength = 0.3

    return {
        'vp_bullish': float(vp_bullish),
        'vp_bearish': float(vp_bearish),
        'vp_divergence': float(vp_divergence),
        'fund_signal': float(fund_signal),
        'trend_strength': float(trend_strength),
    }


# ============================================================
# 简化缠论特征
# ============================================================
def calc_chanlun_simple(df):
    """简化缠论: 分型检测 + 笔方向。

    返回与df等长的DataFrame:
      fractal_type  — 分型 (1=顶分型, -1=底分型, 0=无)
      stroke_dir    — 当前笔方向 (1=向上, -1=向下)
      stroke_length — 当前笔已走K线数
      near_fractal  — 最近分型距当前K线数
    """
    high = df['high'].values
    low = df['low'].values
    n = len(df)

    fractal = np.zeros(n, dtype=int)

    # 分型检测: 顶分型 = high[i] > high[i-1] and high[i] > high[i+1]
    #           底分型 = low[i] < low[i-1] and low[i] < low[i+1]
    for i in range(1, n - 1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            # 确认顶分型: 同时需 low[i] > low[i-1] and low[i] > low[i+1] (简化: 只用high)
            fractal[i] = 1
        elif low[i] < low[i-1] and low[i] < low[i+1]:
            fractal[i] = -1

    # 笔方向: 从最近分型推算
    stroke_dir = np.zeros(n, dtype=int)
    stroke_length = np.zeros(n, dtype=int)
    near_fractal = np.full(n, n)  # 默认很远

    # 找到所有分型位置
    fractal_indices = np.where(fractal != 0)[0]

    for i in range(n):
        # 找最近的分型
        past_fractals = fractal_indices[fractal_indices <= i]
        if len(past_fractals) == 0:
            stroke_dir[i] = 0
            stroke_length[i] = i
            near_fractal[i] = i
            continue

        last_fractal_idx = past_fractals[-1]
        near_fractal[i] = i - last_fractal_idx
        stroke_length[i] = i - last_fractal_idx

        if fractal[last_fractal_idx] == -1:
            # 最近是底分型 → 当前笔向上
            stroke_dir[i] = 1
        elif fractal[last_fractal_idx] == 1:
            # 最近是顶分型 → 当前笔向下
            stroke_dir[i] = -1

    out = pd.DataFrame(index=df.index)
    out['fractal_type'] = fractal
    out['stroke_dir'] = stroke_dir
    out['stroke_length'] = stroke_length
    out['near_fractal'] = near_fractal

    return out


def calc_chanlun_signals(df):
    """从最近数据计算缠论信号，返回标量字典。

    返回:
      stroke_dir     — 当前笔方向 (1/-1/0)
      near_fractal   — 距最近分型K线数 (越小越接近转折)
      fractal_signal — 分型信号: 接近底分型→看多, 接近顶分型→看空
    """
    chanlun = calc_chanlun_simple(df)

    last_dir = int(chanlun['stroke_dir'].iloc[-1])
    last_near = int(chanlun['near_fractal'].iloc[-1])
    last_fractal_type = int(chanlun['fractal_type'].iloc[-1])

    # 分型信号: 距分型越近信号越强
    # 当前是底分型或接近底分型 → 看多
    # 当前是顶分型或接近顶分型 → 看空
    if last_near <= 3:
        # 最近3根K线内有分型
        # 找该分型类型
        recent_fractals = chanlun['fractal_type'].tail(5).values
        for ft in reversed(recent_fractals):
            if ft != 0:
                if ft == -1:
                    fractal_signal = 0.5  # 底分型 → 看多
                else:
                    fractal_signal = -0.5  # 顶分型 → 看空
                break
        else:
            fractal_signal = 0.0
    else:
        fractal_signal = 0.0

    return {
        'stroke_dir': last_dir,
        'near_fractal': last_near,
        'fractal_signal': fractal_signal,
    }
