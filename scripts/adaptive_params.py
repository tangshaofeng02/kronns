#!/usr/bin/env python3
"""
自适应参数模块

基于历史验证记录动态调整预测参数，替换原有的"正确→延续/错误→集成"二值逻辑。

提供:
1. load_validation_history() — 加载验证历史
2. calc_stock_metrics()      — 计算单股指标
3. adaptive_params()         — 连续自适应参数
4. calc_market_volatility()  — 市场波动率
5. signal_correction()       — 信号修正模型输出
"""

import os
import numpy as np
import pandas as pd

try:
    from scripts.short_term_rules import calc_trend_signals
except ImportError:
    try:
        from short_term_rules import calc_trend_signals
    except ImportError:
        calc_trend_signals = None


def load_validation_history(log_dir, n_days=10):
    """从 prediction_log/ 加载最近N天的验证结果。

    返回: dict {code: DataFrame}，每个DataFrame包含列:
      date, mode, pred_chg, actual_chg, direction_ok, error
    """
    validate_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith('_validate.csv')]
    )[-n_days:]

    all_records = []
    for vf in validate_files:
        date_str = vf.replace('_validate.csv', '')
        try:
            df = pd.read_csv(os.path.join(log_dir, vf), dtype={'code': str})
            df['code'] = df['code'].astype(str).str.zfill(6)
            df['date'] = date_str
            all_records.append(df)
        except Exception:
            continue

    if not all_records:
        return {}

    combined = pd.concat(all_records, ignore_index=True)

    result = {}
    for code in combined['code'].unique():
        result[code] = combined[combined['code'] == code].reset_index(drop=True)

    return result


def calc_stock_metrics(history, code):
    """计算单只股票的预测指标。

    参数:
      history — load_validation_history() 返回的dict
      code    — 股票代码

    返回: dict
      direction_accuracy — 方向准确率 [0,1]
      avg_error          — 平均绝对误差
      con_accuracy       — 保守模式准确率
      agr_accuracy       — 激进模式准确率
      recent_trend       — 近期趋势 ('improving'/'worsening'/'stable')
      sample_count       — 可用验证样本数
    """
    if code not in history or history[code].empty:
        return _default_metrics()

    df = history[code]

    # 整体方向准确率
    direction_accuracy = df['direction_ok'].mean()

    # 平均误差
    avg_error = df['error'].mean()

    # 保守/激进分别准确率
    con_df = df[df['mode'].str.contains('保守')]
    agr_df = df[df['mode'].str.contains('激进')]
    con_accuracy = con_df['direction_ok'].mean() if not con_df.empty else 0.5
    agr_accuracy = agr_df['direction_ok'].mean() if not agr_df.empty else 0.5

    # 近期趋势: 比较前半段和后半段误差
    recent_trend = 'stable'
    if len(df) >= 4:
        mid = len(df) // 2
        early_error = df['error'].iloc[:mid].mean()
        late_error = df['error'].iloc[mid:].mean()
        if late_error < early_error * 0.8:
            recent_trend = 'improving'
        elif late_error > early_error * 1.2:
            recent_trend = 'worsening'

    return {
        'direction_accuracy': float(direction_accuracy),
        'avg_error': float(avg_error),
        'con_accuracy': float(con_accuracy),
        'agr_accuracy': float(agr_accuracy),
        'recent_trend': recent_trend,
        'sample_count': len(df),
    }


def _default_metrics():
    return {
        'direction_accuracy': 0.5,
        'avg_error': 2.0,
        'con_accuracy': 0.5,
        'agr_accuracy': 0.5,
        'recent_trend': 'stable',
        'sample_count': 0,
    }


def adaptive_params(metrics, market_volatility=0.02):
    """基于历史指标和市场波动率，计算连续自适应参数。

    参数:
      metrics           — calc_stock_metrics() 返回的dict
      market_volatility — 市场波动率（近5天平均振幅）

    返回: dict {T, top_p, sample_count, lookback, mom_weight}
    """
    accuracy = metrics['direction_accuracy']
    avg_error = metrics['avg_error']
    trend = metrics['recent_trend']

    # 温度: 准确率高→低T(更确定), 准确率低→高T(更多样)
    # 准确率0.8→T=0.44, 准确率0.5→T=0.65, 准确率0.3→T=0.79
    T = 0.3 + (1.0 - accuracy) * 0.7

    # top_p: 准确率高→更保守(低top_p)
    top_p = 0.85 + accuracy * 0.1

    # 采样次数: 误差大→多采样求稳, 误差小→少采样提速
    sample_count = int(np.clip(8 + avg_error * 3, 8, 25))

    # lookback: 准确率很低时缩短(只看近期), 准确率高时正常
    if accuracy < 0.4:
        lookback = 150
    elif accuracy > 0.7:
        lookback = 120
    else:
        lookback = 120

    # 动量修正权重: 趋势恶化时加大动量修正(对冲模型偏差), 趋势改善时减小
    if trend == 'worsening':
        mom_weight = 0.5
    elif trend == 'improving':
        mom_weight = 0.2
    else:
        mom_weight = 0.35

    # 波动率调节: 波动大时增加采样, 降低温度确定性
    if market_volatility > 0.03:
        sample_count = int(sample_count * 1.3)
        T = min(T + 0.1, 1.0)

    return {
        'T': float(T),
        'top_p': float(top_p),
        'sample_count': int(sample_count),
        'lookback': int(lookback),
        'mom_weight': float(mom_weight),
    }


def detect_market_regime(all_dfs):
    """检测市场整体趋势状态（综合均线排列 + 价格位置 + 斜率）。

    参数:
      all_dfs — dict {code: DataFrame}

    返回: dict
      regime — 'uptrend' / 'downtrend' / 'sideways'
      strength — 趋势强度 (0~1)
      up_ratio — 多头股票占比 (兼容原字段)
      bull_ratio / bear_ratio — 多/空头信号>=5的股票占比
    """
    if calc_trend_signals is not None:
        bull_count = 0
        bear_count = 0
        total = 0
        for code, df in all_dfs.items():
            sig = calc_trend_signals(df)
            if sig is None:
                continue
            total += 1
            if sig['bull'] >= 5:
                bull_count += 1
            elif sig['bear'] >= 5:
                bear_count += 1

        if total == 0:
            return {'regime': 'sideways', 'strength': 0.0, 'up_ratio': 0.5,
                    'bull_ratio': 0.5, 'bear_ratio': 0.5}

        bull_ratio = bull_count / total
        bear_ratio = bear_count / total
        up_ratio = bull_ratio  # 兼容原字段
        if bull_ratio > 0.45:
            regime = 'uptrend'
            strength = min((bull_ratio - 0.35) / 0.35, 1.0)
        elif bear_ratio > 0.45:
            regime = 'downtrend'
            strength = min((bear_ratio - 0.35) / 0.35, 1.0)
        else:
            regime = 'sideways'
            strength = 0.0

        return {'regime': regime, 'strength': float(strength),
                'up_ratio': float(up_ratio),
                'bull_ratio': float(bull_ratio), 'bear_ratio': float(bear_ratio)}

    # 兜底: 原 MA20 站位逻辑
    up_count = 0
    total = 0
    for code, df in all_dfs.items():
        if len(df) < 20 or 'ma20' not in df.columns or df['ma20'].iloc[-1] == 0:
            continue
        total += 1
        if df['close'].iloc[-1] > df['ma20'].iloc[-1]:
            up_count += 1

    if total == 0:
        return {'regime': 'sideways', 'strength': 0.0, 'up_ratio': 0.5}

    up_ratio = up_count / total
    if up_ratio > 0.65:
        regime = 'uptrend'
        strength = min((up_ratio - 0.5) / 0.3, 1.0)
    elif up_ratio < 0.35:
        regime = 'downtrend'
        strength = min((0.5 - up_ratio) / 0.3, 1.0)
    else:
        regime = 'sideways'
        strength = 0.0

    return {'regime': regime, 'strength': float(strength), 'up_ratio': float(up_ratio)}


def calc_market_volatility(df_dict):
    """计算市场整体波动率（所有股票近5天平均振幅）。

    参数:
      df_dict — dict {code: DataFrame}，每个df含high/low/close列

    返回: float，平均振幅比率
    """
    volatilities = []
    for code, df in df_dict.items():
        if len(df) < 5:
            continue
        recent = df.tail(5)
        amplitude = (recent['high'] - recent['low']) / recent['close']
        volatilities.append(amplitude.mean())

    return float(np.mean(volatilities)) if volatilities else 0.02


def signal_correction(pred_chg, vp_signals, chanlun_signals, metrics):
    """用量价信号和缠论信号修正模型输出。

    参数:
      pred_chg      — 模型原始预测涨跌幅(%)
      vp_signals    — calc_volume_price_signals() 返回的dict
      chanlun_signals — calc_chanlun_signals() 返回的dict
      metrics       — calc_stock_metrics() 返回的dict

    返回: (corrected_chg, correction_detail)
      corrected_chg  — 修正后涨跌幅(%)
      correction_detail — dict，各信号修正分量（用于reason展示）
    """
    correction = 0.0
    detail = {}

    # 1. 量增价增 + 主力流入 → 增强看多信号
    vp_fund = vp_signals['vp_bullish'] * vp_signals['fund_signal'] * 0.5
    correction += vp_fund
    detail['量价+资金'] = vp_fund

    # 2. 背离信号修正
    divergence_adj = vp_signals['vp_divergence'] * 0.3
    correction += divergence_adj
    detail['背离'] = divergence_adj

    # 3. 缠论分型信号
    fractal_adj = chanlun_signals['fractal_signal'] * 0.4
    correction += fractal_adj
    detail['缠论分型'] = fractal_adj

    # 4. 缠论笔方向 + 趋势强度: 笔方向与预测方向不一致时微调
    stroke_dir = chanlun_signals['stroke_dir']
    if stroke_dir != 0:
        pred_dir = 1 if pred_chg > 0 else -1
        if stroke_dir != pred_dir:
            # 笔方向与预测方向相反，接近分型时修正力度更大
            near_factor = 1.0 / (1.0 + chanlun_signals['near_fractal'] * 0.2)
            stroke_adj = stroke_dir * near_factor * 0.2
            correction += stroke_adj
            detail['缠论笔方向'] = stroke_adj

    # 5. 准确率自适应：准确高→放大修正（信号可信），准确低→缩小修正（信号不可信）
    accuracy = metrics['direction_accuracy']
    if accuracy < 0.4:
        correction *= 0.3  # 准确率很低时大幅缩小修正
        detail['低准确率缩小'] = True
    elif accuracy < 0.5:
        correction *= 0.6
        detail['低准确率缩小'] = True
    elif accuracy > 0.7:
        correction *= 1.2  # 准确率高时适度放大修正
        detail['高准确率放大'] = True

    corrected = pred_chg + correction
    corrected = float(np.clip(corrected, -10, 10))

    return corrected, detail
