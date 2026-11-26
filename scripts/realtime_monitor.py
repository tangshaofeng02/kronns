#!/usr/bin/env python3
"""
盘中实时监控系统 — 每5分钟抓取30只关注股实时数据, 给出决策建议

交易时段: 9:30-11:30, 13:00-15:00
数据源: 腾讯财经实时行情 + 量价/缠论信号 + 资金流 + 新闻情绪

决策等级:
  🟢 强烈买入  — 多信号共振看涨
  🟢 买入      — 偏多信号
  ⚪ 持有/观望  — 信号中性
  🔴 卖出      — 偏空信号
  🔴 强烈卖出  — 多信号共振看空

用法:
  python scripts/realtime_monitor.py          # 持续监控(交易时段内)
  python scripts/realtime_monitor.py --once   # 只跑一次
  python scripts/realtime_monitor.py --test   # 非交易时段也能跑(测试)
"""

import os, sys, time, json, requests
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import STOCKS, STOCK_TAGS, load_df, FEATURES
from scripts.indicators import calc_volume_price_features, calc_volume_price_signals, calc_chanlun_simple, calc_chanlun_signals
from scripts.short_term_rules import detect_trend_state, calc_kouju_rules, get_kouju_penalty

# ============================================================
# 配置
# ============================================================
INTERVAL = 300  # 5分钟 = 300秒
LOG_DIR = './prediction_log'

# 腾讯实时行情API
TENCENT_QUOTE_URL = 'http://qt.gtimg.cn/q='


def fetch_realtime_quote(code):
    """获取单只股票实时行情"""
    prefix = 'sh' if code.startswith(('6', '9')) else 'sz'
    url = f'{TENCENT_QUOTE_URL}{prefix}{code}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.encoding = 'gbk'
        text = r.text.strip()

        # 解析腾讯行情字符串
        # 格式: v_sz000001="1~平安银行~000001~当前价~昨收~今开~...~...~...~";
        if '~' not in text:
            return None

        data = text.split('~')
        if len(data) < 50:
            return None

        return {
            'name': data[1],
            'code': data[2],
            'price': float(data[3]) if data[3] else 0,
            'prev_close': float(data[4]) if data[4] else 0,
            'open': float(data[5]) if data[5] else 0,
            'volume': float(data[6]) if data[6] else 0,  # 手
            'buy_vol': float(data[7]) if data[7] else 0,
            'sell_vol': float(data[8]) if data[8] else 0,
            'bid1_price': float(data[9]) if data[9] else 0,
            'bid1_vol': float(data[10]) if data[10] else 0,
            'ask1_price': float(data[19]) if data[19] else 0,
            'ask1_vol': float(data[20]) if data[20] else 0,
            'high': float(data[33]) if data[33] else 0,
            'low': float(data[34]) if data[34] else 0,
            'amount': float(data[37]) if data[37] else 0,  # 万元
            'change_pct': float(data[32]) if data[32] else 0,
            'change': float(data[31]) if data[31] else 0,
            'turnover': float(data[38]) if data[38] else 0,  # 换手率%
            'amplitude': float(data[47]) if len(data) > 47 and data[47] else 0,  # 振幅%
            'pe': float(data[39]) if len(data) > 39 and data[39] else 0,
            'circulating_value': float(data[44]) if len(data) > 44 and data[44] else 0,  # 流通市值亿
            'total_value': float(data[45]) if len(data) > 45 and data[45] else 0,  # 总市值亿
            'pb': float(data[46]) if len(data) > 46 and data[46] else 0,
            'limit_up': float(data[47]) if len(data) > 47 and data[47] else 0,
            'limit_down': float(data[48]) if len(data) > 48 and data[48] else 0,
        }
    except Exception:
        return None


def fetch_all_realtime(codes):
    """批量获取实时行情(一次请求多只)"""
    results = {}
    # 分批(每批最多30只)
    for i in range(0, len(codes), 30):
        batch = codes[i:i+30]
        prefixes = ['sh' if c.startswith(('6', '9')) else 'sz' for c in batch]
        symbols = ','.join(f'{p}{c}' for p, c in zip(prefixes, batch))
        url = f'{TENCENT_QUOTE_URL}{symbols}'
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.encoding = 'gbk'
            lines = r.text.strip().split(';')

            for line in lines:
                line = line.strip()
                if '~' not in line or '"' not in line:
                    continue
                try:
                    # 提取代码
                    prefix_code = line.split('="')[0].split('_')[-1]
                    code = prefix_code[-6:]
                    data = line.split('~')

                    if len(data) >= 50 and data[2]:
                        results[code] = {
                            'name': data[1],
                            'code': data[2],
                            'price': float(data[3]) if data[3] else 0,
                            'prev_close': float(data[4]) if data[4] else 0,
                            'open': float(data[5]) if data[5] else 0,
                            'volume': float(data[6]) if data[6] else 0,
                            'high': float(data[33]) if data[33] else 0,
                            'low': float(data[34]) if data[34] else 0,
                            'amount': float(data[37]) if data[37] else 0,
                            'change_pct': float(data[32]) if data[32] else 0,
                            'change': float(data[31]) if data[31] else 0,
                            'turnover': float(data[38]) if data[38] else 0,
                            'bid1_price': float(data[9]) if data[9] else 0,
                            'ask1_price': float(data[19]) if data[19] else 0,
                            'circulating_value': float(data[44]) if data[44] else 0,
                            'total_value': float(data[45]) if data[45] else 0,
                        }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f'  批量请求失败: {e}')

        time.sleep(0.2)

    return results


def calc_volume_ratio(code, current_volume, current_seconds):
    """计算量比: 当前成交量 / 同时段5日均量
    返回 -1 表示非交易时段, 无法计算
    """
    if current_seconds <= 0:
        return -1  # 非交易时段
    try:
        df = load_df(code, skip_indicators=True)
        if len(df) < 6:
            return -1

        # 5天平均日成交量
        avg_daily_vol = df['volume'].iloc[-5:].mean()
        if avg_daily_vol == 0:
            return -1

        # 交易总秒数 = 4h * 3600 = 14400秒
        total_trading_seconds = 4 * 3600
        # 同时段5日均量 = 日均量 * (已过秒数 / 总秒数)
        expected_vol = avg_daily_vol * (current_seconds / total_trading_seconds)
        if expected_vol <= 0:
            return -1

        ratio = current_volume / expected_vol
        return round(ratio, 2)
    except:
        return -1


def fetch_realtime_fund_flow(code):
    """获取实时资金流(交易日内返回当日实时数据)"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from scripts.fetch_fund_flow import fetch_fund_flow_latest
        data = fetch_fund_flow_latest(code)
        if data:
            return data.get('main_net_inflow', 0)
    except:
        pass
    return None


def get_trading_seconds():
    """计算当前交易时段已过秒数"""
    now = datetime.now()
    current_seconds = (now.hour * 3600 + now.minute * 60 + now.second)

    morning_start = 9 * 3600 + 30 * 60  # 9:30
    morning_end = 11 * 3600 + 30 * 60   # 11:30
    afternoon_start = 13 * 3600          # 13:00
    afternoon_end = 15 * 3600            # 15:00

    if current_seconds < morning_start:
        return 0
    elif current_seconds <= morning_end:
        return current_seconds - morning_start
    elif current_seconds < afternoon_start:
        return morning_end - morning_start  # 盘间固定
    elif current_seconds <= afternoon_end:
        return (morning_end - morning_start) + (current_seconds - afternoon_start)
    else:
        return 4 * 3600  # 收盘后


def is_trading_time():
    """判断当前是否在交易时段"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current = now.hour * 3600 + now.minute * 60 + now.second
    return (9*3600+30*60 <= current <= 11*3600+30*60) or (13*3600 <= current <= 15*3600)


def calc_sector_summary(quotes, all_dfs, stock_tags):
    """计算各板块实时涨跌均值 + T-1资金流均值。

    参数:
      quotes     — dict {code: quote}
      all_dfs    — dict {code: DataFrame}
      stock_tags — dict {code: tag}

    返回: dict {tag: {'avg_change': float, 'avg_fund': float, 'count': int}}
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {'changes': [], 'funds': []})
    for code, quote in quotes.items():
        tag = stock_tags.get(code)
        if not tag:
            continue
        if quote and quote.get('price', 0) > 0:
            agg[tag]['changes'].append(quote.get('change_pct', 0))
        df = all_dfs.get(code)
        if df is not None and 'main_net_inflow_ratio' in df.columns and len(df) > 0:
            agg[tag]['funds'].append(df['main_net_inflow_ratio'].iloc[-1])

    return {
        tag: {
            'avg_change': float(np.mean(d['changes'])) if d['changes'] else 0.0,
            'avg_fund': float(np.mean(d['funds'])) if d['funds'] else 0.0,
            'count': len(d['changes']),
            'max_change': float(max(d['changes'])) if d['changes'] else 0.0,
        }
        for tag, d in agg.items()
    }


def generate_signal(code, quote, df, sector_summary=None, sentiment_bias=0.0, market_trend=0.0):
    """综合多维度信号生成决策
    market_trend: 市场趋势指标(关注池平均涨跌幅%), < -1 视为下跌趋势, 触发抄底过滤
    """
    signals = []
    score = 0  # 正=看多, 负=看空

    if quote is None or df is None or len(df) < 20:
        return {'decision': '⚪ 数据不足', 'score': 0, 'signals': [], 'confidence': 0,
                'price': 0, 'change_pct': 0, 'vol_ratio': 0, 'turnover': 0}

    price = quote['price']
    prev_close = quote['prev_close']
    if price == 0 or prev_close == 0:
        return {'decision': '⚪ 未开盘', 'score': 0, 'signals': [], 'confidence': 0,
                'price': 0, 'change_pct': 0, 'vol_ratio': 0, 'turnover': 0}

    change_pct = quote['change_pct']

    # ===== 信号1: 涨跌幅 (修正: 反转逻辑, 奖励超跌) =====
    if change_pct > 3:
        signals.append(f'大涨{change_pct:+.1f}%')
        score -= 2  # 追高风险, 扣分
    elif change_pct > 1:
        signals.append(f'上涨{change_pct:+.1f}%')
        score -= 1  # 追高, 扣分
    elif change_pct < -3:
        signals.append(f'超跌{change_pct:+.1f}%')
        score += 2 if market_trend >= -1.0 else -1  # 下跌趋势中超跌=接飞刀风险
    elif change_pct < -1:
        signals.append(f'下跌{change_pct:+.1f}%')
        score += 1 if market_trend >= -1.0 else 0  # 下跌趋势中不抄底
    else:
        signals.append(f'微动{change_pct:+.1f}%')

    # ===== 信号2: 量比 (修复: 非交易时段跳过, 盘后接口volume异常) =====
    trading_secs = get_trading_seconds()
    vol_ratio = 0
    if is_trading_time() and quote['volume'] > 0:
        vol_ratio = calc_volume_ratio(code, quote['volume'], trading_secs)
        if vol_ratio > 0:  # 有效量比
            if vol_ratio > 2.5:
                signals.append(f'放量x{vol_ratio:.1f}')
                score += 2 if change_pct < 0 else -1  # 下跌放量=吸筹(加分), 上涨放量=出货(扣分)
            elif vol_ratio > 1.5:
                signals.append(f'温和放量x{vol_ratio:.1f}')
                score += 1 if change_pct < 0 else 0
            elif vol_ratio < 0.5:
                signals.append(f'缩量x{vol_ratio:.1f}')
                score -= 0.3
            else:
                signals.append(f'量比{vol_ratio:.1f}')
        else:
            signals.append('量比N/A')

    # ===== 信号3: 量价信号 =====
    try:
        vp = calc_volume_price_signals(df)
        if not vp.empty:
            last = vp.iloc[-1]
            bullish_count = 0
            bearish_count = 0
            for col in vp.columns:
                if col == 'timestamps':
                    continue
                val = last[col]
                if val > 0:
                    bullish_count += 1
                elif val < 0:
                    bearish_count += 1

            if bullish_count >= 3:
                signals.append(f'量价看多({bullish_count})')
                score += 2
            elif bearish_count >= 3:
                signals.append(f'量价看空({bearish_count})')
                score -= 2
            elif bullish_count > bearish_count:
                signals.append('量价偏多')
                score += 1
            elif bearish_count > bullish_count:
                signals.append('量价偏空')
                score -= 1
    except:
        pass

    # ===== 信号4: 缠论信号 =====
    try:
        cl = calc_chanlun_signals(df)
        if not cl.empty:
            last_cl = cl.iloc[-1]
            cl_bull = 0
            cl_bear = 0
            for col in cl.columns:
                if col == 'timestamps':
                    continue
                val = last_cl[col]
                if val > 0:
                    cl_bull += 1
                elif val < 0:
                    cl_bear += 1

            if cl_bull >= 2:
                signals.append(f'缠论底分型')
                score += 2
            elif cl_bear >= 2:
                signals.append(f'缠论顶分型')
                score -= 2
    except:
        pass

    # ===== 信号5: 资金流 (修复: 交易时段用实时数据) =====
    fund_source = 'T-1'
    fund_ratio = 0
    try:
        fund_ratio = df['main_net_inflow_ratio'].iloc[-1]
        # 交易时段尝试获取实时资金流
        if is_trading_time():
            rt_flow = fetch_realtime_fund_flow(code)
            if rt_flow is not None and rt_flow != 0:
                # 将实时资金流转换为占比(估算)
                price_at_signal = quote['price'] if quote['price'] > 0 else df['close'].iloc[-1]
                avg_price = df['close'].iloc[-5:].mean() if len(df) >= 5 else df['close'].iloc[-1]
                amount_today = quote['amount'] * 10000 if quote.get('amount', 0) > 0 else df['amount'].iloc[-1]
                if amount_today > 0:
                    fund_ratio_pct = (rt_flow / amount_today) * 100
                    fund_ratio = fund_ratio_pct
                    fund_source = '实时'
    except:
        pass

    if fund_ratio > 2:
        signals.append(f'主力大幅流入{fund_ratio:.1f}%({fund_source})')
        score += 6 if market_trend >= -1.0 else 2  # 下跌趋势中防拆单陷阱, 减半
    elif fund_ratio > 0.5:
        signals.append(f'主力流入{fund_ratio:.1f}%({fund_source})')
        score += 3 if market_trend >= -1.0 else 1
    elif fund_ratio < -2:
        signals.append(f'主力大幅流出{fund_ratio:.1f}%({fund_source})')
        score -= 6
    elif fund_ratio < -0.5:
        signals.append(f'主力流出{fund_ratio:.1f}%({fund_source})')
        score -= 3

    # ===== 信号6: 日内位置 (修正: 低位+资金流入=最佳买点) =====
    day_range = quote['high'] - quote['low']
    if day_range > 0:
        position = (price - quote['low']) / day_range
        if position > 0.85:
            signals.append(f'日内高位{position:.0%}')
            score -= 1  # 高位风险, 扣分
        elif position < 0.15:
            signals.append(f'日内低位{position:.0%}')
            score += 1 if market_trend >= -1.0 else 0  # 下跌趋势中低位不加分

    # ===== 信号7: 换手率 =====
    turnover = quote.get('turnover', 0)
    if turnover > 10:
        signals.append(f'高换手{turnover:.1f}%')
        score += 1 if change_pct > 0 else -1
    elif turnover < 1:
        signals.append(f'低换手{turnover:.1f}%')

    # ===== 信号8: 逆势买入检测 =====
    # 核心逻辑: 股价下跌 + 主力大幅流入 = 主力压盘吸筹
    contrarian = False
    contrarian_strength = 0
    try:
        fund_ratio = df['main_net_inflow_ratio'].iloc[-1]
        # 股价下跌超0.5% 且 主力流入超2%
        if change_pct < -0.5 and fund_ratio > 2.0:
            contrarian = True
            # 背离程度 = 主力流入强度 + 股价下跌幅度
            contrarian_strength = fund_ratio + abs(change_pct)
            signals.append(f'逆势吸筹({fund_ratio:+.1f}%/{change_pct:+.1f}%)')
        # 股价微跌/平盘 且 主力大幅流入超5%
        elif change_pct < 0.5 and fund_ratio > 5.0:
            contrarian = True
            contrarian_strength = fund_ratio + abs(change_pct)
            signals.append(f'逆势吸筹({fund_ratio:+.1f}%/{change_pct:+.1f}%)')
    except:
        pass

    # ===== 信号9: 底背离/顶背离检测 =====
    # 底背离: 价格创新低 + 资金流改善 = 底部信号
    # 顶背离: 价格创新高 + 资金流恶化 = 顶部信号
    bullish_divergence = False
    bearish_divergence = False
    divergence_strength = 0
    try:
        if len(df) >= 20:
            # 计算价格均线
            ma5 = df['close'].iloc[-5:].mean()
            ma10 = df['close'].iloc[-10:].mean()
            ma20 = df['close'].iloc[-20:].mean()

            # 计算资金流均线
            fund_5d = df['main_net_inflow_ratio'].iloc[-5:].mean()
            fund_10d = df['main_net_inflow_ratio'].iloc[-10:].mean()
            fund_20d = df['main_net_inflow_ratio'].iloc[-20:].mean()
            fund_current = df['main_net_inflow_ratio'].iloc[-1]

            # 计算成交量均线
            vol_5d = df['volume'].iloc[-5:].mean()
            vol_current = quote['volume'] if quote['volume'] > 0 else df['volume'].iloc[-1]

            # ===== 底背离检测 =====
            # 条件1: 价格在MA20下方(中长期下跌趋势)
            # 条件2: 当前资金流 > 5日均值(资金开始改善)
            # 条件3: 5日均值 > 20日均值(资金改善趋势确立)
            price_below_ma20 = price < ma20
            fund_improving = fund_current > fund_5d > fund_20d
            # 或者: 价格连续下跌但资金流连续3天为正
            price_declining = price < ma5 < ma10
            fund_3d_positive = all(df['main_net_inflow_ratio'].iloc[-3:] > 0)

            if price_below_ma20 and fund_improving:
                bullish_divergence = True
                divergence_strength = (fund_current - fund_20d) + abs((price / ma20 - 1) * 10)
                signals.append(f'底背离(价↓资↑{fund_current:.1f}>{fund_20d:.1f})')
            elif price_declining and fund_3d_positive:
                bullish_divergence = True
                divergence_strength = sum(df['main_net_inflow_ratio'].iloc[-3:])
                signals.append(f'底背离(价跌资连3日流入)')

            # 量价底背离: 价格下跌但成交量放大(恐慌中有人接盘)
            if change_pct < -1 and vol_current > vol_5d * 1.5:
                if not bullish_divergence:
                    bullish_divergence = True
                    divergence_strength = vol_current / vol_5d
                signals.append(f'量价底背离(放量接盘)')

            # ===== 顶背离检测 =====
            # 条件1: 价格在MA20上方(中长期上涨趋势)
            # 条件2: 当前资金流 < 5日均值(资金开始恶化)
            # 条件3: 5日均值 < 20日均值(资金恶化趋势确立)
            price_above_ma20 = price > ma20
            fund_worsening = fund_current < fund_5d < fund_20d
            # 或者: 价格连续上涨但资金流连续3天为负
            price_rising = price > ma5 > ma10
            fund_3d_negative = all(df['main_net_inflow_ratio'].iloc[-3:] < 0)

            if price_above_ma20 and fund_worsening:
                bearish_divergence = True
                divergence_strength = (fund_20d - fund_current) + abs((price / ma20 - 1) * 10)
                signals.append(f'顶背离(价↑资↓{fund_current:.1f}<{fund_20d:.1f})')
            elif price_rising and fund_3d_negative:
                bearish_divergence = True
                divergence_strength = abs(sum(df['main_net_inflow_ratio'].iloc[-3:]))
                signals.append(f'顶背离(价涨资连3日流出)')

            # 量价顶背离: 价格上涨但缩量(买盘不足)
            if change_pct > 1 and vol_current < vol_5d * 0.7:
                if not bearish_divergence:
                    bearish_divergence = True
                    divergence_strength = vol_5d / vol_current
                signals.append(f'量价顶背离(缩量上涨)')
    except:
        pass

    # ===== 信号10: 板块联动 =====
    if sector_summary:
        tag = STOCK_TAGS.get(code)
        sec = sector_summary.get(tag) if tag else None
        if sec and sec.get('count', 0) > 0:
            sec_change = sec['avg_change']
            sec_fund = sec['avg_fund']
            rel = change_pct - sec_change  # 个股相对板块

            if sec_change > 1 and change_pct > 0:
                signals.append(f'板块强势{sec_change:+.1f}%')
                score += 2
            elif sec_change < -1 and change_pct < 0:
                signals.append(f'板块弱势{sec_change:+.1f}%')
                score -= 2
            elif sec_change > 1:
                signals.append(f'板块强{sec_change:+.1f}%未跟涨')
                score -= 1
            elif sec_change < -1:
                signals.append(f'板块弱{sec_change:+.1f}%未跟跌')
                score += 1

            if rel > 2:
                signals.append(f'龙头(强于板块{rel:+.1f}%)')
                score += 1
            elif rel < -2:
                signals.append(f'弱于板块{rel:+.1f}%')
                score -= 1

            # 板块内最强(龙头确认)
            if sec.get('count', 0) >= 2 and change_pct >= sec.get('max_change', 0) - 0.1:
                signals.append('板块内最强')
                score += 1

            if sec_fund > 1:
                signals.append(f'板块资金流入{sec_fund:+.1f}%(T-1)')
                score += 1
            elif sec_fund < -1:
                signals.append(f'板块资金流出{sec_fund:+.1f}%(T-1)')
                score -= 1

    # ===== 信号11: 均线趋势 (MA5/MA20/MA60 排列 + 金叉死叉) =====
    try:
        if df is not None and len(df) >= 20:
            ma5 = df['close'].iloc[-5:].mean()
            ma20 = df['close'].iloc[-20:].mean()
            ma60 = df['close'].iloc[-60:].mean() if len(df) >= 60 else ma20

            if price > ma20:
                signals.append(f'站上MA20({ma20:.2f})')
                score += 1
            else:
                signals.append(f'跌破MA20({ma20:.2f})')
                score -= 1

            if ma5 > ma20 > ma60:
                signals.append('均线多头排列')
                score += 2
            elif ma5 < ma20 < ma60:
                signals.append('均线空头排列')
                score -= 2

            # 金叉/死叉: MA5 与 MA20 的前后关系
            prev_ma5 = df['close'].iloc[-6:-1].mean()
            prev_ma20 = df['close'].iloc[-21:-1].mean()
            if prev_ma5 <= prev_ma20 and ma5 > ma20:
                signals.append('金叉(MA5上穿MA20)')
                score += 2
            elif prev_ma5 >= prev_ma20 and ma5 < ma20:
                signals.append('死叉(MA5下穿MA20)')
                score -= 2
    except:
        pass

    # ===== 信号11+: 短线口诀规则 (趋势门控软修正) =====
    try:
        if df is not None and len(df) >= 25:
            ts = detect_trend_state(df)
            rt_ctx = {
                'price': price, 'change_pct': change_pct,
                'vol_ratio': vol_ratio if vol_ratio and vol_ratio > 0 else None,
                'prev_close': prev_close,
            }
            kouju_rules = calc_kouju_rules(df, 0.0, ts, realtime_ctx=rt_ctx)
            penalty, hits = get_kouju_penalty(kouju_rules)
            if hits:
                score += penalty
                signals.extend(hits)
    except Exception:
        pass

    # ===== 信号12: 换手率衰竭/恐慌 (Volume Climax) =====
    if turnover > 15:
        if change_pct > 5:
            signals.append(f'天量高潮(换手{turnover:.1f}%+)')
            score -= 2  # 高潮反转风险
        elif change_pct < -5:
            signals.append(f'恐慌抛售(换手{turnover:.1f}%+)')
            score += 2  # 恐慌见底机会

    # ===== 信号13: 盘子大小 (小盘弹性/大盘稳健) =====
    circ = quote.get('circulating_value', 0)
    if 0 < circ < 100:
        signals.append(f'小盘({circ:.0f}亿)')
        score += 1  # 小盘易拉, 短线弹性
    elif circ >= 500:
        signals.append(f'大盘({circ:.0f}亿)')
        score -= 0.5

    # ===== 信号15: 分歧转一致/一致转分歧 (短线博主核心: 炒股养家) =====
    # 量价本质: 放量=分歧(多空博弈), 缩量=一致(惜售/无人问津)
    try:
        if df is not None and len(df) >= 7 and vol_ratio > 0:
            vol_yesterday = df['volume'].iloc[-2]
            vol_avg5 = df['volume'].iloc[-7:-2].mean()  # 近5日(不含昨日)
            if vol_avg5 > 0:
                vr_yesterday = vol_yesterday / vol_avg5
                # 昨日放量分歧 + 今日缩量一致 → 转一致(看涨)
                if vr_yesterday > 1.5 and vol_ratio < 0.8:
                    signals.append(f'分歧转一致(昨放{vr_yesterday:.1f}→今缩{vol_ratio:.1f})')
                    score += 2
                # 昨日缩量一致 + 今日放量分歧 → 转分歧(看跌)
                elif vr_yesterday < 0.8 and vol_ratio > 1.5:
                    signals.append(f'一致转分歧(昨缩{vr_yesterday:.1f}→今放{vol_ratio:.1f})')
                    score -= 2
    except:
        pass

    # ===== 信号16: 弱转强/强转弱 (短线涨停战法) =====
    try:
        if df is not None and len(df) >= 2:
            prev_chg = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            # 昨日弱(跌) + 今日强(涨) → 弱转强(买点)
            if prev_chg < -1 and change_pct > 1:
                signals.append(f'弱转强(昨{prev_chg:+.1f}%→今{change_pct:+.1f}%)')
                score += 2
            # 昨日强(涨) + 今日弱(跌) → 强转弱(卖点)
            elif prev_chg > 1 and change_pct < -1:
                signals.append(f'强转弱(昨{prev_chg:+.1f}%→今{change_pct:+.1f}%)')
                score -= 2
    except:
        pass

    # ===== 信号17: 高位缩量加速/天量见顶 (短线风险) =====
    # 游资经验: 高位不换手缩量加速, 一旦分歧易连续跌停
    try:
        if df is not None and len(df) >= 6 and vol_ratio > 0:
            pct_5d = (price / df['close'].iloc[-6] - 1) * 100
            # 高位(5日涨幅>15%) + 缩量(vol_ratio<0.7) + 仍在涨 → 缩量加速风险
            if pct_5d > 15 and vol_ratio < 0.7 and change_pct > 0:
                signals.append(f'高位缩量加速(5日{pct_5d:+.0f}%)')
                score -= 2
            # 高位 + 高换手(>15%) → 天量见顶
            if pct_5d > 15 and turnover > 15:
                signals.append(f'高位天量(5日{pct_5d:+.0f}%,换手{turnover:.0f}%)')
                score -= 2
    except:
        pass

    # ===== 信号18: 冲高回落 (止损纪律: 不及预期坚决离场) =====
    try:
        if quote.get('high', 0) > 0 and quote.get('open', 0) > 0:
            drawdown = (quote['high'] - price) / quote['high'] * 100
            # 从日内高点回落>3% 且 跌破开盘价 → 冲高回落
            if drawdown > 3 and price < quote['open']:
                signals.append(f'冲高回落(回撤{drawdown:.1f}%)')
                score -= 2
    except:
        pass

    # ===== 抄底评分 (独立体系) =====
    bottom_score = 0
    try:
        fund_ratio = df['main_net_inflow_ratio'].iloc[-1]
        bottom_score += max(fund_ratio, 0) * 4  # 主力流入×4
        if bullish_divergence:
            bottom_score += divergence_strength * 3  # 背离度×3
        # 缠论底分型加分
        try:
            cl = calc_chanlun_signals(df)
            if not cl.empty:
                last_cl = cl.iloc[-1]
                cl_bull = sum(1 for col in cl.columns if col != 'timestamps' and last_cl[col] > 0)
                if cl_bull >= 2:
                    bottom_score += cl_bull * 2
        except:
            pass
        # 量价底背离加分
        if any('量价底背离' in s for s in signals):
            bottom_score += 4
        # 超跌加分
        if change_pct < -2:
            bottom_score += 3
        elif change_pct < -1:
            bottom_score += 1
        # 日内低位加分
        day_range = quote['high'] - quote['low']
        if day_range > 0:
            position = (price - quote['low']) / day_range
            if position < 0.2:
                bottom_score += 2
    except:
        pass

    # ===== 全局市场情绪偏置 =====
    score += sentiment_bias

    # ===== 市场趋势过滤 (下跌趋势禁用抄底, 避免接飞刀) =====
    if market_trend < -1.0 and (contrarian or bullish_divergence):
        signals.append(f'趋势过滤(下跌市{market_trend:+.1f}%)')
        contrarian = False
        bullish_divergence = False
        if score > 0:
            score = -1  # 强制转空

    # ===== 综合决策 (修复: 解决买入卖出冲突) =====
    confidence = min(abs(score) / 8, 1.0)  # 最高8分=100%置信度

    # 逆势+底背离 → 自动升级为买入, 且屏蔽卖出信号
    buy_override = False
    if contrarian and bullish_divergence:
        decision = '🟢 强烈买入'
        buy_override = True
    elif contrarian or bullish_divergence:
        decision = '🟢 买入'
        buy_override = True
    elif score >= 3:
        decision = '🟢 强烈买入'
    elif score >= 1:
        decision = '🟢 买入'
    elif score <= -6:
        decision = '🔴 强烈卖出'
    elif score <= -3:
        decision = '🔴 卖出'
    else:
        decision = '⚪ 持有'

    # 抄底评分≥10 → 强制升级为买入 (下跌趋势禁用, 避免接飞刀)
    if bottom_score >= 10 and decision not in ['🟢 强烈买入', '🟢 买入']:
        if market_trend >= -1.0:
            decision = '🟢 买入'
            signals.append(f'抄底评分{bottom_score:.0f}')
            buy_override = True
        else:
            signals.append(f'抄底评分{bottom_score:.0f}(下跌市禁用)')

    # 修复冲突: 如果买入升级触发, 移除矛盾的卖出信号
    if buy_override and '🔴' in decision:
        decision = '⚪ 持有'  # 矛盾时保守处理
        signals.append('多空冲突→观望')

    # 修复冲突: 如果同时触发底背离和顶背离, 以资金流方向为准
    if bullish_divergence and bearish_divergence:
        if fund_ratio > 0:
            bearish_divergence = False  # 资金流入, 顶背离不可靠
            signals = [s for s in signals if '顶背离' not in s]
        elif fund_ratio < 0:
            bullish_divergence = False  # 资金流出, 底背离不可靠
            signals = [s for s in signals if '底背离' not in s]

    return {
        'decision': decision,
        'score': score,
        'signals': signals,
        'confidence': confidence,
        'change_pct': change_pct,
        'price': price,
        'vol_ratio': vol_ratio if trading_secs > 0 and quote['volume'] > 0 else 0,
        'turnover': turnover,
        'contrarian': contrarian,
        'contrarian_strength': contrarian_strength,
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence,
        'divergence_strength': divergence_strength,
        'bottom_score': bottom_score,
    }


def display_results(results, elapsed, sentiment_label=''):
    """格式化显示结果"""
    now = datetime.now().strftime('%H:%M:%S')
    sentiment = f' | {sentiment_label}' if sentiment_label else ''
    print(f'\n{"="*100}')
    print(f'  📊 实时监控 {now} | 耗时{elapsed:.1f}s | 共{len(results)}只{sentiment}')
    print(f'{"="*100}')

    # 按决策强度排序
    order = {'🟢 强烈买入': 0, '🟢 买入': 1, '⚪ 持有': 2, '⚪ 数据不足': 3, '⚪ 未开盘': 3, '🔴 卖出': 4, '🔴 强烈卖出': 5}
    results.sort(key=lambda x: (order.get(x['decision'], 2), -abs(x['score'])))

    # 汇总统计
    decisions = [r['decision'] for r in results]
    strong_buy = sum(1 for d in decisions if '强烈买入' in d)
    buy = sum(1 for d in decisions if '买入' in d) - strong_buy
    hold = sum(1 for d in decisions if '持有' in d) + sum(1 for d in decisions if '数据不足' in d) + sum(1 for d in decisions if '未开盘' in d)
    sell = sum(1 for d in decisions if '卖出' in d) - strong_buy
    strong_sell = sum(1 for d in decisions if '强烈卖出' in d)

    print(f'  🟢强烈买入:{strong_buy}  🟢买入:{buy}  ⚪持有:{hold}  🔴卖出:{sell}  🔴强烈卖出:{strong_sell}')
    print()

    # 表格
    print(f'  {"股票":14s} {"板块":6s} {"现价":>8s} {"涨跌":>8s} {"量比":>6s} {"换手":>6s} {"决策":10s} {"置信":>5s}  {"信号"}')
    print(f'  {"-"*100}')

    for r in results:
        name_code = f'{r["name"]}({r["code"]})'
        tag = STOCK_TAGS.get(r['code'], '')
        price = f'{r["price"]:.2f}' if r['price'] > 0 else '-'
        chg = f'{r["change_pct"]:+.2f}%' if r['price'] > 0 else '-'
        vr = f'{r["vol_ratio"]:.1f}' if r['vol_ratio'] > 0 else 'N/A'
        tr = f'{r["turnover"]:.1f}%' if r['turnover'] > 0 else '-'
        conf = f'{r["confidence"]:.0%}' if r['confidence'] > 0 else '-'
        sigs = ' | '.join(r['signals'][:4]) if r['signals'] else '-'

        print(f'  {name_code:14s} {tag:6s} {price:>8s} {chg:>8s} {vr:>6s} {tr:>6s} {r["decision"]:10s} {conf:>5s}  {sigs}')

    # 逆势买入信号
    contrarian = [r for r in results if r.get('contrarian')]
    if contrarian:
        contrarian.sort(key=lambda x: x.get('contrarian_strength', 0), reverse=True)
        print(f'\n  {"="*100}')
        print(f'  🔄 逆势买入信号: 股价下跌 + 主力大幅流入 (共{len(contrarian)}只)')
        print(f'  {"="*100}')
        for r in contrarian:
            sigs = ' | '.join(r['signals'][:3])
            strength = r.get('contrarian_strength', 0)
            print(f'     {r["name"]:6s}({r["code"]}) 现价{r["price"]:.2f} 涨跌{r["change_pct"]:+.2f}%  背离度{strength:.1f}  {sigs}')

    # 底背离/顶背离信号
    bullish_div = [r for r in results if r.get('bullish_divergence')]
    bearish_div = [r for r in results if r.get('bearish_divergence')]
    if bullish_div or bearish_div:
        print(f'\n  {"="*100}')
        print(f'  📊 背离信号检测')
        print(f'  {"="*100}')
        if bullish_div:
            bullish_div.sort(key=lambda x: x.get('divergence_strength', 0), reverse=True)
            print(f'  🟢 底背离 (价格↓ + 资金↑ = 底部信号, 共{len(bullish_div)}只):')
            for r in bullish_div:
                sigs = ' | '.join([s for s in r['signals'] if '背离' in s][:2])
                strength = r.get('divergence_strength', 0)
                print(f'     {r["name"]:6s}({r["code"]}) 现价{r["price"]:.2f} 涨跌{r["change_pct"]:+.2f}%  强度{strength:.1f}  {sigs}')
        if bearish_div:
            bearish_div.sort(key=lambda x: x.get('divergence_strength', 0), reverse=True)
            print(f'  🔴 顶背离 (价格↑ + 资金↓ = 顶部信号, 共{len(bearish_div)}只):')
            for r in bearish_div:
                sigs = ' | '.join([s for s in r['signals'] if '背离' in s][:2])
                strength = r.get('divergence_strength', 0)
                print(f'     {r["name"]:6s}({r["code"]}) 现价{r["price"]:.2f} 涨跌{r["change_pct"]:+.2f}%  强度{strength:.1f}  {sigs}')

    # 重点提示
    print(f'\n  {"-"*100}')
    action_items = [r for r in results if '买入' in r['decision'] or '卖出' in r['decision']]
    if action_items:
        print(f'  ⚡ 重点关注:')
        for r in action_items:
            sigs = ' | '.join(r['signals'][:3])
            print(f'     {r["decision"]} {r["name"]}({r["code"]}) {r["change_pct"]:+.2f}% — {sigs}')
    else:
        print(f'  ⚡ 当前无强烈信号, 建议观望')


def run_once(test_mode=False, contrarian=False):
    """执行一次监控"""
    start = time.time()

    if not test_mode and not is_trading_time():
        now = datetime.now()
        print(f'{now.strftime("%H:%M:%S")} 非交易时段 (交易时间: 9:30-11:30, 13:00-15:00)')
        print('提示: 加 --test 参数可强制运行')
        return

    # 获取实时行情
    codes = list(STOCKS.keys())
    quotes = fetch_all_realtime(codes)

    if not quotes:
        print('❌ 获取实时行情失败, 请检查网络')
        return

    # 先加载全部历史数据, 聚合板块统计
    all_dfs = {}
    for code in STOCKS:
        try:
            all_dfs[code] = load_df(code)
        except:
            all_dfs[code] = None

    sector_summary = calc_sector_summary(quotes, all_dfs, STOCK_TAGS)

    # 市场情绪温度 + 趋势判断 (基于关注股)
    valid_quotes = [q for q in quotes.values() if q and q.get('price', 0) > 0]
    up_count = sum(1 for q in valid_quotes if q.get('change_pct', 0) > 0)
    total = len(valid_quotes)
    avg_change = float(np.mean([q['change_pct'] for q in valid_quotes])) if valid_quotes else 0.0
    market_trend = avg_change  # 市场趋势指标: 关注池平均涨跌幅
    sentiment_bias = 0.0
    sentiment_label = ''
    if total > 0:
        up_ratio = up_count / total
        if avg_change < -1.5:
            sentiment_bias = -4.0
            sentiment_label = f'🔻强势下跌({avg_change:+.1f}%,上涨{up_ratio:.0%})'
        elif avg_change < -0.5:
            sentiment_bias = -2.0
            sentiment_label = f'🔻下跌({avg_change:+.1f}%,上涨{up_ratio:.0%})'
        elif avg_change > 1.5:
            sentiment_bias = 1.0
            sentiment_label = f'🔺强势上涨({avg_change:+.1f}%,上涨{up_ratio:.0%})'
        elif avg_change > 0.5:
            sentiment_bias = 0.5
            sentiment_label = f'🔺上涨({avg_change:+.1f}%,上涨{up_ratio:.0%})'
        else:
            sentiment_label = f'➖震荡({avg_change:+.1f}%,上涨{up_ratio:.0%})'

    # 生成信号
    results = []
    for code in STOCKS:
        quote = quotes.get(code)
        df = all_dfs.get(code)

        signal = generate_signal(code, quote, df, sector_summary, sentiment_bias, market_trend)
        signal['code'] = code
        signal['name'] = STOCKS[code]
        results.append(signal)

    elapsed = time.time() - start
    display_results(results, elapsed, sentiment_label)

    # 逆势模式: 只显示逆势信号 + 背离信号
    if contrarian:
        contrarian_signals = [r for r in results if r.get('contrarian')]
        bullish_div = [r for r in results if r.get('bullish_divergence')]

        if contrarian_signals:
            contrarian_signals.sort(key=lambda x: x.get('contrarian_strength', 0), reverse=True)
            print(f'\n  🔄 逆势买入信号 ({len(contrarian_signals)}只):')
            for r in contrarian_signals:
                sigs = ' | '.join(r['signals'][:3])
                print(f'     {r["name"]:6s}({r["code"]}) 现价{r["price"]:.2f} 涨跌{r["change_pct"]:+.2f}%  背离度{r.get("contrarian_strength",0):.1f}  {sigs}')
        else:
            print(f'\n  当前无逆势买入信号')

        if bullish_div:
            bullish_div.sort(key=lambda x: x.get('divergence_strength', 0), reverse=True)
            print(f'\n  🟢 底背离信号 ({len(bullish_div)}只):')
            for r in bullish_div:
                sigs = ' | '.join([s for s in r['signals'] if '背离' in s][:2])
                print(f'     {r["name"]:6s}({r["code"]}) 现价{r["price"]:.2f} 涨跌{r["change_pct"]:+.2f}%  强度{r.get("divergence_strength",0):.1f}  {sigs}')

        if not contrarian_signals and not bullish_div:
            print(f'\n  当前无逆势/底背离信号')
        return results

    # 保存日志
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f'realtime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    rows = []
    for r in results:
        rows.append({
            '时间': datetime.now().strftime('%H:%M:%S'),
            '代码': r['code'], '名称': r['name'],
            '现价': r.get('price', 0),
            '涨跌幅': r.get('change_pct', 0),
            '量比': r.get('vol_ratio', 0),
            '换手率': r.get('turnover', 0),
            '决策': r['decision'],
            '评分': r['score'],
            '置信度': r.get('confidence', 0),
            '逆势': r.get('contrarian', False),
            '逆势强度': r.get('contrarian_strength', 0),
            '底背离': r.get('bullish_divergence', False),
            '顶背离': r.get('bearish_divergence', False),
            '背离强度': r.get('divergence_strength', 0),
            '抄底评分': r.get('bottom_score', 0),
            '信号': '|'.join(r.get('signals', [])),
        })
    pd.DataFrame(rows).to_csv(log_file, index=False)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='盘中实时监控系统')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    parser.add_argument('--test', action='store_true', help='测试模式(非交易时段也可运行)')
    parser.add_argument('--interval', type=int, default=INTERVAL, help=f'刷新间隔秒(默认{INTERVAL})')
    parser.add_argument('--contrarian', action='store_true', help='仅显示逆势买入信号')
    args = parser.parse_args()

    print(f'📈 盘中实时监控系统')
    print(f'   股票池: {len(STOCKS)}只 | 刷新间隔: {args.interval}秒')
    print(f'   交易时段: 9:30-11:30, 13:00-15:00')
    if args.contrarian:
        print(f'   模式: 🔄 逆势买入 (股价下跌+主力流入)')

    if args.once or args.test:
        run_once(test_mode=args.test, contrarian=args.contrarian)
    else:
        # 持续监控
        print(f'\n开始持续监控 (Ctrl+C 停止)...')
        while True:
            try:
                run_once(test_mode=False, contrarian=args.contrarian)
                if is_trading_time():
                    remaining = args.interval
                    time.sleep(remaining)
                else:
                    # 非交易时段, 等30秒再检查
                    time.sleep(30)
            except KeyboardInterrupt:
                print('\n\n监控已停止')
                break
            except Exception as e:
                print(f'错误: {e}')
                time.sleep(10)


if __name__ == '__main__':
    main()
