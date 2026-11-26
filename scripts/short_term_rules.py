"""短线口诀规则引擎。

将实战短线口诀转化为可执行规则，供 daily_predict.py（每日预测修正）
和 realtime_monitor.py（实时信号评分）共用。

口诀来源：
  上涨趋势：上涨看支撑、缩量回调不必逃、有价无量必须跑、连续小涨是真涨、
            连续大涨要离场、低点不破——买、回调至5日线反包买点、主升加速要见顶
  下跌趋势：放量下跌要减仓、缩量新低是底象、增量回升是关键、回头确认要进场、
            三连阴大跌箭穿心、下跌常态不要预设支撑
  通用纪律：横盘不交易、顺势而为、不冲高不卖不跳水不买

设计：软修正（单规则 ±0.3~0.8，总限幅 ±1.5），趋势门控（只在对应趋势下启用）。
"""
import numpy as np
import pandas as pd

# ============ 开关与参数 ============
ENABLE_SHORT_TERM_RULES = True
MAX_TOTAL_ADJ = 1.5  # 口诀修正总量限幅（防止覆盖模型核心判断）

# 上涨趋势规则修正量
SHRINK_PULLBACK_ADJ = 0.4       # 缩量回调不必逃
VOL_PRICE_DIVERGE_ADJ = -0.8    # 有价无量必须跑
SMALL_RISE_ADJ = 0.3            # 连续小涨是真涨
BIG_RISE_ADJ = -0.6             # 连续大涨要离场
LOW_NOT_BROKEN_ADJ = 0.4        # 低点不破——买
MA5_RETEST_ADJ = 0.5            # 回调至5日线反包买点
ACCELERATE_ADJ = -0.4           # 主升加速要见顶

# 下跌趋势规则修正量
VOL_DROP_ADJ = -0.6             # 放量下跌要减仓
SHRINK_NEWLOW_ADJ = 0.4         # 缩量新低是底象
VOL_RISE_ADJ = 0.6              # 增量回升是关键
CONFIRM_PULLBACK_ADJ = 0.8      # 回头确认要进场
THREE_YIN_ADJ = -0.6            # 三连阴大跌箭穿心

# 通用规则
SIDEWAYS_SCORE_PENALTY = -1.0   # 横盘不交易（实时评分惩罚）
SIDEWAYS_CONF_FACTOR = 0.8      # 横盘不交易（预测置信度折减）

# 阈值参数
THRESH_VOL_SHRINK = 0.9         # 缩量阈值（量比 < 0.9）
THRESH_VOL_EXPAND = 1.2         # 放量阈值（量比 > 1.2）
THRESH_VOL_DEEP_SHRINK = 0.8    # 深度缩量（量比 < 0.8）
THRESH_SMALL_RISE_DAILY = 3.0   # 连续小涨单日涨幅上限 %
THRESH_BIG_RISE_3D = 12.0       # 连续大涨3日累计 %
THRESH_BIG_RISE_1D = 7.0        # 连续大涨单日 %
THRESH_ACCEL_RATIO = 2.5        # 主升加速斜率比
THRESH_ACCEL_GAIN = 10.0        # 主升加速累计涨幅 %
THRESH_AMPLITUDE_SIDEWAYS = 1.5 # 横盘振幅阈值 %
THRESH_REBOUND_PCT = 1.0        # 增量回升涨幅阈值 %


def calc_trend_signals(df):
    """综合趋势信号计算（均线排列 + 价格位置 + 斜率 + 回踩确认）。

    上涨趋势特征: 价格在均线上方 + 多头排列(MA5>MA10>MA20>MA60) + 均线向上 + 回踩企稳
    下跌趋势特征: 价格在均线下方 + 空头排列(MA5<MA10<MA20<MA60) + 均线向下 + 反弹受阻

    返回 dict: {
        'bull': int (多头信号数 0~7),
        'bear': int (空头信号数 0~7),
        'ma5', 'ma10', 'ma20', 'ma60', 'close': float,
        'ma5_slope', 'ma20_slope': float (5日斜率%),
        'signals': list (命中信号描述),
    }
    """
    if df is None or len(df) < 25:
        return None

    close = df['close']
    c = float(close.iloc[-1])

    # 优先复用已算好的均线列，没有则现算
    if all(col in df.columns for col in ['ma5', 'ma10', 'ma20', 'ma60']):
        ma5 = float(df['ma5'].iloc[-1])
        ma10 = float(df['ma10'].iloc[-1])
        ma20 = float(df['ma20'].iloc[-1])
        ma60 = float(df['ma60'].iloc[-1])
        ma5_prev = float(df['ma5'].iloc[-6]) if len(df) >= 6 else np.nan
        ma20_prev = float(df['ma20'].iloc[-6]) if len(df) >= 6 else np.nan
    else:
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma5_prev = float(close.rolling(5).mean().iloc[-6]) if len(df) >= 6 else np.nan
        ma20_prev = float(close.rolling(20).mean().iloc[-6]) if len(df) >= 6 else np.nan

    if np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20) or np.isnan(ma60):
        return None

    bull = 0
    bear = 0
    sigs = []

    # 1. 价格位置 (close vs MA20)
    if c > ma20:
        bull += 1; sigs.append('价格>MA20')
    else:
        bear += 1; sigs.append('价格<MA20')

    # 2. 均线排列 (多头: MA5>MA10>MA20>MA60)
    if ma5 > ma10:
        bull += 1; sigs.append('MA5>MA10')
    else:
        bear += 1; sigs.append('MA5<MA10')
    if ma10 > ma20:
        bull += 1; sigs.append('MA10>MA20')
    else:
        bear += 1; sigs.append('MA10<MA20')
    if ma20 > ma60:
        bull += 1; sigs.append('MA20>MA60')
    else:
        bear += 1; sigs.append('MA20<MA60')

    # 3. 均线斜率 (5日变化率)
    ma5_slope = (ma5 - ma5_prev) / ma5_prev * 100 if ma5_prev and ma5_prev > 0 else 0.0
    ma20_slope = (ma20 - ma20_prev) / ma20_prev * 100 if ma20_prev and ma20_prev > 0 else 0.0
    if ma5_slope > 0.3:
        bull += 1; sigs.append(f'MA5向上{ma5_slope:+.1f}%')
    elif ma5_slope < -0.3:
        bear += 1; sigs.append(f'MA5向下{ma5_slope:+.1f}%')
    if ma20_slope > 0.2:
        bull += 1; sigs.append(f'MA20向上{ma20_slope:+.1f}%')
    elif ma20_slope < -0.2:
        bear += 1; sigs.append(f'MA20向下{ma20_slope:+.1f}%')

    # 4. 回踩确认 (近5日: 曾跌破/突破 MA20 后当前收复/受阻)
    if 'ma20' in df.columns:
        recent = df.tail(5)
        if (recent['close'] < recent['ma20']).any() and c > ma20:
            bull += 1; sigs.append('回踩MA20企稳')
        if (recent['close'] > recent['ma20']).any() and c < ma20:
            bear += 1; sigs.append('反弹MA20受阻')

    return {
        'bull': bull, 'bear': bear,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'close': c, 'ma5_slope': ma5_slope, 'ma20_slope': ma20_slope,
        'signals': sigs,
    }


def detect_trend_state(df):
    """判断趋势状态（综合均线排列 + 价格位置 + 斜率 + 回踩确认）。

    判断规则: 多头信号 >=5 → uptrend; 空头信号 >=5 → downtrend; 其他 → sideways

    返回 dict: {'state': str, 'bull': int, 'bear': int, 'ma5', 'ma10', 'ma20',
                'ma60', 'close': float, 'ma5_slope', 'ma20_slope': float,
                'ma60_up': bool, 'signals': list}
    """
    sig = calc_trend_signals(df)
    if sig is None:
        return {'state': 'sideways', 'bull': 0, 'bear': 0,
                'ma5': np.nan, 'ma10': np.nan, 'ma20': np.nan, 'ma60': np.nan,
                'close': np.nan, 'ma5_slope': 0.0, 'ma20_slope': 0.0,
                'ma60_up': False, 'signals': []}

    if sig['bull'] >= 5:
        state = 'uptrend'
    elif sig['bear'] >= 5:
        state = 'downtrend'
    else:
        state = 'sideways'

    return {
        'state': state,
        'bull': sig['bull'], 'bear': sig['bear'],
        'ma5': sig['ma5'], 'ma10': sig['ma10'], 'ma20': sig['ma20'], 'ma60': sig['ma60'],
        'close': sig['close'],
        'ma5_slope': sig['ma5_slope'], 'ma20_slope': sig['ma20_slope'],
        'ma60_up': sig['ma20_slope'] > 0,
        'signals': sig['signals'],
    }


def _get_vol_ratio(df, realtime_ctx=None):
    """获取最新量比。实时模式优先用实时量比。"""
    if realtime_ctx and realtime_ctx.get('vol_ratio') is not None:
        return realtime_ctx['vol_ratio']
    if 'vol_ratio' in df.columns and len(df) > 0:
        return float(df['vol_ratio'].iloc[-1])
    # 兜底：用 5 日均量现算
    if len(df) >= 6:
        return float(df['volume'].iloc[-1] / df['volume'].iloc[-6:-1].mean())
    return np.nan


def _daily_pct(df):
    """每日涨跌幅序列（%）。"""
    return df['close'].pct_change() * 100


def _is_yin_candle(df, i):
    """判断第 i 根 K 线是否为阴线（close < open）。"""
    return df['close'].iloc[i] < df['open'].iloc[i]


def calc_kouju_rules(df, pred_chg=0.0, trend_state=None, realtime_ctx=None):
    """计算口诀规则命中情况。

    参数:
      df            — 历史行情（含 ma5/ma20/ma60/vol_ratio 等列，来自 load_df）
      pred_chg      — 模型预测涨跌幅（%），用于方向一致性参考（可忽略）
      trend_state   — detect_trend_state() 返回值；为 None 则自动判断
      realtime_ctx  — 实时上下文 {price, change_pct, vol_ratio, prev_close}
                       非 None 时启用实时规则（用实时价格/量比覆盖历史末值）

    返回 dict: {规则名: {'hit': bool, 'adj': float, 'desc': str, 'trend': 'up/down/common'}}
    """
    if not ENABLE_SHORT_TERM_RULES:
        return {}

    if trend_state is None:
        trend_state = detect_trend_state(df)
    state = trend_state['state']

    if df is None or len(df) < 25:
        return {}

    rules = {}
    vr = _get_vol_ratio(df, realtime_ctx)
    pct = _daily_pct(df)
    close = df['close']

    # 实时上下文
    rt_price = realtime_ctx.get('price') if realtime_ctx else None
    rt_chg = realtime_ctx.get('change_pct') if realtime_ctx else None
    rt_vr = realtime_ctx.get('vol_ratio') if realtime_ctx else None
    cur_price = rt_price if rt_price is not None else float(close.iloc[-1])

    # ============ 上涨趋势规则（顺势：找支撑、辨健康回调） ============
    if state == 'uptrend':
        # 1. 缩量回调不必逃：近3日累计跌 且 量比<0.9 → 健康回调
        pct_3d = float(pct.iloc[-3:].sum()) if len(pct) >= 4 else 0.0
        if pct_3d < -0.5 and not np.isnan(vr) and vr < THRESH_VOL_SHRINK:
            rules['缩量回调'] = {'hit': True, 'adj': SHRINK_PULLBACK_ADJ,
                                 'desc': f'健康(3日{pct_3d:+.1f}%缩量x{vr:.1f})', 'trend': 'up'}

        # 2. 有价无量必须跑：创20日新高 且 量比<0.8 → 买盘衰竭
        high_20 = float(close.iloc[-20:].max())
        if cur_price >= high_20 - 0.001 and not np.isnan(vr) and vr < THRESH_VOL_DEEP_SHRINK:
            rules['有价无量'] = {'hit': True, 'adj': VOL_PRICE_DIVERGE_ADJ,
                                 'desc': f'20日新高缩量x{vr:.1f}', 'trend': 'up'}

        # 3. 连续小涨是真涨：近5日每日涨幅 0~3%
        if len(pct) >= 6:
            last5 = pct.iloc[-5:]
            if all(0 < x < THRESH_SMALL_RISE_DAILY for x in last5):
                rules['连续小涨'] = {'hit': True, 'adj': SMALL_RISE_ADJ,
                                     'desc': f'5日连阳(均{float(last5.mean()):.1f}%)', 'trend': 'up'}

        # 4. 连续大涨要离场：近3日累计>12% 或 单日>7%
        pct_3d_sum = float(pct.iloc[-3:].sum())
        pct_1d = float(pct.iloc[-1])
        if pct_3d_sum > THRESH_BIG_RISE_3D or pct_1d > THRESH_BIG_RISE_1D:
            rules['连续大涨'] = {'hit': True, 'adj': BIG_RISE_ADJ,
                                 'desc': f'3日{pct_3d_sum:+.1f}%', 'trend': 'up'}

        # 5. 低点不破——买：回踩到前低附近(3%内)且未跌破（右侧确认）
        if len(df) >= 10:
            low_recent = float(df['low'].iloc[-5:].min())
            low_prev = float(df['low'].iloc[-10:-5].min())
            if low_prev <= low_recent <= low_prev * 1.03:
                rules['低点不破'] = {'hit': True, 'adj': LOW_NOT_BROKEN_ADJ,
                                     'desc': f'回踩{low_recent:.2f}不破{low_prev:.2f}', 'trend': 'up'}

        # 6. 回调至5日线反包买点：close 在 MA5±1% 且 昨日破今日稳
        ma5 = trend_state.get('ma5')
        if ma5 and len(close) >= 2:
            c0, c1 = float(close.iloc[-1]), float(close.iloc[-2])
            ma5_prev = float(df['ma5'].iloc[-2]) if 'ma5' in df.columns else ma5
            if abs(c0 / ma5 - 1) < 0.01 and c1 < ma5_prev and c0 >= ma5:
                rules['回踩MA5'] = {'hit': True, 'adj': MA5_RETEST_ADJ,
                                    'desc': f'反包({c0:.2f} vs MA5 {ma5:.2f})', 'trend': 'up'}

        # 7. 主升加速要见顶：5日斜率/20日斜率>2.5 且 累计涨幅>10%
        if len(close) >= 21:
            slope_5 = float(close.iloc[-1] - close.iloc[-6]) / 5
            slope_20 = float(close.iloc[-1] - close.iloc[-21]) / 20
            gain_5 = float(close.iloc[-1] / close.iloc[-6] - 1) * 100
            if slope_20 > 0 and slope_5 / slope_20 > THRESH_ACCEL_RATIO and gain_5 > THRESH_ACCEL_GAIN:
                rules['主升加速'] = {'hit': True, 'adj': ACCELERATE_ADJ,
                                     'desc': f'斜率比{slope_5/slope_20:.1f}', 'trend': 'up'}

    # ============ 下跌趋势规则（顺势：看阻力、等确认、不猜底） ============
    elif state == 'downtrend':
        pct_3d = float(pct.iloc[-3:].sum()) if len(pct) >= 4 else 0.0

        # 1. 放量下跌要减仓：近3日跌>3% 且 量比>1.2
        if pct_3d < -3.0 and not np.isnan(vr) and vr > THRESH_VOL_EXPAND:
            rules['放量下跌'] = {'hit': True, 'adj': VOL_DROP_ADJ,
                                 'desc': f'3日{pct_3d:+.1f}%放量x{vr:.1f}', 'trend': 'down'}

        # 2. 缩量新低是底象：创20日新低 且 量比<0.8 → 观察不下调
        low_20 = float(close.iloc[-20:].min())
        if cur_price <= low_20 + 0.001 and not np.isnan(vr) and vr < THRESH_VOL_DEEP_SHRINK:
            rules['缩量新低'] = {'hit': True, 'adj': SHRINK_NEWLOW_ADJ,
                                 'desc': f'20日新低缩量x{vr:.1f}', 'trend': 'down'}

        # 3. 增量回升是关键：今日涨>1% 且 量比>1.2（实时模式优先用实时涨跌）
        chg_today = rt_chg if rt_chg is not None else float(pct.iloc[-1])
        vr_today = rt_vr if rt_vr is not None else vr
        if chg_today > THRESH_REBOUND_PCT and not np.isnan(vr_today) and vr_today > THRESH_VOL_EXPAND:
            rules['增量回升'] = {'hit': True, 'adj': VOL_RISE_ADJ,
                                 'desc': f'放量回升{chg_today:+.1f}%', 'trend': 'down'}

        # 4. 回头确认要进场：近3日回升 且 回踩不破前5日低
        if len(df) >= 8:
            rebound_3d = float(pct.iloc[-3:].sum())
            low_recent = float(df['low'].iloc[-3:].min())
            low_prev = float(df['low'].iloc[-8:-3].min())
            if rebound_3d > 0.5 and low_recent >= low_prev:
                rules['回头确认'] = {'hit': True, 'adj': CONFIRM_PULLBACK_ADJ,
                                     'desc': f'回踩确认(3日{rebound_3d:+.1f}%)', 'trend': 'down'}

        # 5. 三连阴大跌箭穿心：连续3根阴线
        if len(df) >= 3 and all(_is_yin_candle(df, i) for i in range(-3, 0)):
            rules['三连阴'] = {'hit': True, 'adj': THREE_YIN_ADJ,
                               'desc': '连续3阴', 'trend': 'down'}

        # 6. 下跌常态不要预设支撑（门控，非加减分）：
        #     downtrend 中只允许"增量回升/回头确认/缩量新低"等确认信号上调，
        #    禁止单纯"跌幅大/超跌"类上调。本规则不产成分值，仅作为纪律标记。
        rules['不预设支撑'] = {'hit': False, 'adj': 0.0,
                               'desc': 'downtrend中禁止超跌猜底', 'trend': 'down'}

    # ============ 通用规则 ============
    # 横盘不交易：20日平均振幅<1.5% 或 近5日振幅<3%
    if len(df) >= 20:
        amplitude = (df['high'] - df['low']) / df['close'] * 100
        amp_20 = float(amplitude.iloc[-20:].mean())
        amp_5 = float(amplitude.iloc[-5:].mean())
        if amp_20 < THRESH_AMPLITUDE_SIDEWAYS or amp_5 < THRESH_AMPLITUDE_SIDEWAYS:
            rules['横盘不交易'] = {'hit': True, 'adj': 0.0,
                                   'desc': f'振幅{amp_5:.1f}%', 'trend': 'common'}
            # 横盘模式：方向性规则全部失效（清空已命中项）
            for k in list(rules.keys()):
                if rules[k]['trend'] in ('up', 'down'):
                    rules.pop(k)

    return rules


def apply_kouju_correction(chg, rules, max_total=MAX_TOTAL_ADJ):
    """汇总口诀规则修正量并限幅。

    参数:
      chg   — 当前预测涨跌幅（%）
      rules — calc_kouju_rules() 返回的 dict
      max_total — 总修正限幅

    返回 (corrected_chg, detail_dict)
      detail_dict: {规则名: adj}，仅包含命中项（abs(adj) > 0.05），供 reason 展示
    """
    if not rules:
        return chg, {}

    total = 0.0
    detail = {}
    for name, rule in rules.items():
        if not rule.get('hit'):
            continue
        adj = rule.get('adj', 0.0)
        if abs(adj) < 0.05:
            continue
        total += adj
        detail[name] = adj

    # 限幅
    total = float(np.clip(total, -max_total, max_total))
    corrected = float(np.clip(chg + total, -10, 10))

    # 横盘不交易：仅标记，不影响 chg（置信度折减由调用方处理）
    if '横盘不交易' in rules and rules['横盘不交易'].get('hit'):
        detail['横盘不交易'] = 0.0

    return corrected, detail


def get_kouju_penalty(rules):
    """实时模式：获取口诀规则对评分的净惩罚/奖励（含横盘惩罚）。"""
    if not rules:
        return 0.0, []
    score_delta = 0.0
    hits = []
    for name, rule in rules.items():
        if not rule.get('hit'):
            continue
        adj = rule.get('adj', 0.0)
        score_delta += adj
        if abs(adj) >= 0.05:
            hits.append(f"{name}({rule.get('desc','')})")
    # 横盘不交易：额外惩罚
    if '横盘不交易' in rules and rules['横盘不交易'].get('hit'):
        score_delta += SIDEWAYS_SCORE_PENALTY
        hits.append(f"横盘不交易({rules['横盘不交易'].get('desc','')})")
    return float(np.clip(score_delta, -3.0, 3.0)), hits


def get_sideways_conf_factor(rules):
    """预测模式：横盘不交易时的置信度折减系数。"""
    if rules and rules.get('横盘不交易', {}).get('hit'):
        return SIDEWAYS_CONF_FACTOR
    return 1.0
