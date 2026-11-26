#!/usr/bin/env python3
"""
AI 端侧应用板块全面分析
- 覆盖 AI芯片/SoC、智能终端、边缘计算、AI模组、端侧视觉AI、AI应用等细分赛道
- 实时行情 + 技术面 + 资金面 + 板块联动分析
"""

import os, sys, time, requests, pandas as pd, numpy as np
from datetime import datetime
from io import StringIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.fetch_stock_data import TencentStrategy
from scripts.indicators import (
    calc_volume_price_features, calc_volume_price_signals,
    calc_chanlun_simple, calc_chanlun_signals
)

# ============================================================
# AI 端侧应用板块股票池（按细分赛道分类）
# ============================================================
SECTORS = {
    'AI芯片/SoC': {
        '688256': '寒武纪', '688041': '海光信息', '300474': '景嘉微',
        '603893': '瑞芯微', '300458': '全志科技', '300613': '富瀚微',
        '300672': '国科微', '688049': '炬芯科技', '688018': '乐鑫科技',
    },
    '智能终端/端侧AI': {
        '002230': '科大讯飞', '300496': '中科创达', '688088': '虹软科技',
        '688400': '凌云光',
    },
    '边缘计算基础设施': {
        '000977': '浪潮信息', '601138': '工业富联', '603019': '中科曙光',
        '300017': '网宿科技', '300249': '依米康',
    },
    'AI模组/IoT': {
        '603236': '移远通信', '300638': '广和通', '002881': '美格智能',
    },
    '端侧视觉AI': {
        '688475': '萤石网络', '688039': '当虹科技',
    },
    'AI应用/软件': {
        '300624': '万兴科技', '300418': '昆仑万维', '300058': '蓝色光标',
        '002261': '拓维信息',
    },
    'AI手机/终端硬件': {
        '002456': '欧菲光', '300433': '蓝思科技', '002241': '歌尔股份',
        '688036': '传音控股',
    },
    'AI PCB/精密制造': {
        '002384': '东山精密', '002938': '鹏鼎控股',
    },
    'PCB/印制电路': {
        '002463': '沪电股份', '600183': '生益科技', '002916': '深南电路',
        '002384': '东山精密', '300476': '胜宏科技',
    },
    '机器人': {
        '688836': '宇树科技-W', '688017': '绿的谐波', '002747': '埃斯顿',
        '300024': '机器人', '002979': '雷赛智能',
    },
    '脑机接口': {
        '301293': '三博脑科', '002173': '创新医疗', '300007': '汉威科技',
    },
}

ALL_STOCKS = {}
for sector, stocks in SECTORS.items():
    for code, name in stocks.items():
        ALL_STOCKS[code] = {'name': name, 'sector': sector}

TENCENT_QUOTE_URL = 'http://qt.gtimg.cn/q='
DATA_DIR = './finetune_csv/data'


def fetch_all_realtime(codes):
    """批量获取实时行情"""
    results = {}
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
                            'circulating_value': float(data[44]) if data[44] else 0,
                            'total_value': float(data[45]) if data[45] else 0,
                            'amplitude': float(data[47]) if len(data) > 47 and data[47] else 0,
                        }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f'  批量请求失败: {e}')
        time.sleep(0.3)
    return results


def fetch_history(code, days=60):
    """通过腾讯接口获取历史数据"""
    try:
        strategy = TencentStrategy()
        end = datetime.now()
        start = end - pd.Timedelta(days=days+30)
        df = strategy.fetch_range(code, start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))
        if df is None or df.empty:
            return None
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        price_zero = (df['close'] == 0) | (df['open'] == 0)
        if price_zero.any():
            df = df[~price_zero].reset_index(drop=True)
        if df.empty:
            return None
        m = df['amount'] == 0
        if m.any():
            avg_p = df.loc[m, ['open','high','low','close']].mean(axis=1)
            df.loc[m, 'amount'] = df.loc[m, 'volume'] * avg_p
        for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
            if fc not in df.columns:
                df[fc] = 0.0
            else:
                df[fc] = df[fc].fillna(0)
        # 计算技术指标
        vp = calc_volume_price_features(df)
        cl = calc_chanlun_simple(df)
        for col in vp.columns:
            df[col] = vp[col]
        for col in cl.columns:
            df[col] = cl[col]
        return df
    except Exception as e:
        print(f'  {code} 历史数据获取失败: {e}')
        return None


def analyze_stock(code, quote, df):
    """单只股票全面分析"""
    result = {
        'code': code,
        'name': quote['name'] if quote else ALL_STOCKS[code]['name'],
        'sector': ALL_STOCKS[code]['sector'],
        'price': quote['price'] if quote else 0,
        'change_pct': quote['change_pct'] if quote else 0,
        'turnover': quote['turnover'] if quote else 0,
        'amount': quote['amount'] if quote else 0,
        'circulating_value': quote['circulating_value'] if quote else 0,
        'total_value': quote['total_value'] if quote else 0,
        'amplitude': quote.get('amplitude', 0) if quote else 0,
    }

    if df is None or len(df) < 20:
        result['score'] = 0
        result['signals'] = []
        return result

    last = df.iloc[-1]
    price = result['price'] if result['price'] > 0 else last['close']

    # ===== 趋势分析 =====
    ma5 = df['close'].iloc[-5:].mean()
    ma10 = df['close'].iloc[-10:].mean()
    ma20 = df['close'].iloc[-20:].mean()
    ma60 = df['close'].iloc[-60:].mean() if len(df) >= 60 else df['close'].mean()

    trend = '多头' if price > ma5 > ma10 > ma20 else \
            '空头' if price < ma5 < ma10 < ma20 else '震荡'

    # ===== 动量 =====
    pct_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
    pct_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

    # ===== 量能 =====
    vol_5d = df['volume'].iloc[-5:].mean()
    vol_20d = df['volume'].iloc[-20:].mean()
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1

    # ===== 资金流 =====
    fund_ratio = last.get('main_net_inflow_ratio', 0)
    fund_5d = df['main_net_inflow_ratio'].iloc[-5:].mean() if 'main_net_inflow_ratio' in df.columns else 0

    # ===== 技术指标信号 =====
    signals = []
    score = 0

    # 量价信号
    try:
        vp = calc_volume_price_signals(df)
        if not vp.empty:
            last_vp = vp.iloc[-1]
            bull = sum(1 for c in vp.columns if c != 'timestamps' and last_vp[c] > 0)
            bear = sum(1 for c in vp.columns if c != 'timestamps' and last_vp[c] < 0)
            if bull >= 3:
                signals.append(f'量价看多({bull})')
                score += 2
            elif bear >= 3:
                signals.append(f'量价看空({bear})')
                score -= 2
    except:
        pass

    # 缠论信号
    try:
        cl = calc_chanlun_signals(df)
        if not cl.empty:
            last_cl = cl.iloc[-1]
            cl_bull = sum(1 for c in cl.columns if c != 'timestamps' and last_cl[c] > 0)
            cl_bear = sum(1 for c in cl.columns if c != 'timestamps' and last_cl[c] < 0)
            if cl_bull >= 2:
                signals.append('缠论底分型')
                score += 2
            elif cl_bear >= 2:
                signals.append('缠论顶分型')
                score -= 2
    except:
        pass

    # MACD
    try:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
            signals.append('MACD金叉')
            score += 2
        elif dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]:
            signals.append('MACD死叉')
            score -= 2
        elif dif.iloc[-1] > dea.iloc[-1]:
            signals.append('MACD多头')
            score += 1
        else:
            signals.append('MACD空头')
            score -= 1
    except:
        pass

    # RSI
    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        if rsi_val < 30:
            signals.append(f'RSI超卖{rsi_val:.0f}')
            score += 2
        elif rsi_val > 70:
            signals.append(f'RSI超买{rsi_val:.0f}')
            score -= 2
        elif rsi_val > 55:
            signals.append(f'RSI偏强{rsi_val:.0f}')
            score += 1
        elif rsi_val < 45:
            signals.append(f'RSI偏弱{rsi_val:.0f}')
            score -= 1
    except:
        pass

    # 资金流评分
    if fund_ratio > 3:
        signals.append(f'主力大幅流入{fund_ratio:.1f}%')
        score += 3
    elif fund_ratio > 0.5:
        signals.append(f'主力流入{fund_ratio:.1f}%')
        score += 1
    elif fund_ratio < -3:
        signals.append(f'主力大幅流出{fund_ratio:.1f}%')
        score -= 3
    elif fund_ratio < -0.5:
        signals.append(f'主力流出{fund_ratio:.1f}%')
        score -= 1

    # 趋势评分
    if trend == '多头':
        score += 2
    elif trend == '空头':
        score -= 2

    # 超跌/超涨
    if pct_5d < -5:
        score += 1
        signals.append(f'5日超跌{pct_5d:.1f}%')
    elif pct_5d > 8:
        score -= 1
        signals.append(f'5日超涨{pct_5d:.1f}%')

    result['score'] = score
    result['signals'] = signals
    result['trend'] = trend
    result['ma5'] = ma5
    result['ma10'] = ma10
    result['ma20'] = ma20
    result['ma60'] = ma60
    result['pct_5d'] = pct_5d
    result['pct_20d'] = pct_20d
    result['vol_ratio'] = vol_ratio
    result['fund_ratio'] = fund_ratio
    result['fund_5d'] = fund_5d
    return result


def main():
    print('='*120)
    print(f'  AI 端侧应用板块全面分析 | {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*120)

    # 获取实时行情
    codes = list(ALL_STOCKS.keys())
    print(f'\n📊 共 {len(codes)} 只股票，{len(SECTORS)} 个细分赛道')
    print('\n获取实时行情...')
    quotes = fetch_all_realtime(codes)
    print(f'  获取成功: {len(quotes)}/{len(codes)}')

    # 获取历史数据并分析
    print('\n获取历史数据并分析...')
    results = []
    for i, code in enumerate(codes):
        info = ALL_STOCKS[code]
        quote = quotes.get(code)
        # 先检查本地是否有数据
        local_path = os.path.join(DATA_DIR, f'{code}_fundflow.csv')
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path)
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                if 'amount' in df.columns:
                    m = df['amount'] == 0
                    if m.any():
                        df.loc[m, 'amount'] = df.loc[m, 'volume'] * df.loc[m, ['open','high','low','close']].mean(axis=1)
                for fc in ['main_net_inflow','huge_net_inflow','main_net_inflow_ratio']:
                    if fc not in df.columns:
                        df[fc] = 0.0
                    else:
                        df[fc] = df[fc].fillna(0)
                vp = calc_volume_price_features(df)
                cl = calc_chanlun_simple(df)
                for col in vp.columns:
                    df[col] = vp[col]
                for col in cl.columns:
                    df[col] = cl[col]
            except:
                df = fetch_history(code)
        else:
            df = fetch_history(code)

        r = analyze_stock(code, quote, df)
        results.append(r)
        status = f'{r["score"]:+.0f}'
        print(f'  [{i+1:2d}/{len(codes)}] {r["name"]:8s}({code}) {r["sector"]:10s} 现价{r["price"]:>8.2f} 涨跌{r["change_pct"]:>+7.2f}% 评分{status:>4s}')
        time.sleep(0.3)

    # ===== 按板块汇总 =====
    print(f'\n\n{"="*120}')
    print('  板块汇总分析')
    print('='*120)

    sector_stats = {}
    for r in results:
        s = r['sector']
        if s not in sector_stats:
            sector_stats[s] = []
        sector_stats[s].append(r)

    for sector, stocks in sector_stats.items():
        avg_chg = np.mean([s['change_pct'] for s in stocks])
        avg_score = np.mean([s['score'] for s in stocks])
        avg_fund = np.mean([s['fund_ratio'] for s in stocks if 'fund_ratio' in s])
        up_count = sum(1 for s in stocks if s['change_pct'] > 0)
        total = len(stocks)
        print(f'\n  【{sector}】 平均涨跌{avg_chg:+.2f}% 平均评分{avg_score:+.1f} 平均资金流{avg_fund:+.2f}% 上涨{up_count}/{total}')
        for s in sorted(stocks, key=lambda x: x['score'], reverse=True):
            sigs = ' | '.join(s['signals'][:3]) if s['signals'] else '-'
            print(f'    {s["name"]:8s}({s["code"]}) {s["price"]:>8.2f} {s["change_pct"]:>+7.2f}% 评分{s["score"]:>+3.0f} 资金{s.get("fund_ratio",0):>+6.2f}%  {sigs}')

    # ===== 综合排名 =====
    print(f'\n\n{"="*120}')
    print('  综合排名 TOP 15（按评分排序）')
    print('='*120)
    print(f'  {"排名":4s} {"股票":16s} {"板块":12s} {"现价":>8s} {"涨跌":>8s} {"5日":>7s} {"20日":>7s} {"资金流":>7s} {"评分":>5s} {"信号"}')
    print(f'  {"-"*120}')

    ranked = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(ranked[:15], 1):
        sigs = ' | '.join(r['signals'][:3]) if r['signals'] else '-'
        print(f'  {i:4d} {r["name"]+"("+r["code"]+")":16s} {r["sector"]:12s} {r["price"]:>8.2f} {r["change_pct"]:>+7.2f}% {r.get("pct_5d",0):>+6.1f}% {r.get("pct_20d",0):>+6.1f}% {r.get("fund_ratio",0):>+6.2f}% {r["score"]:>+4.0f}  {sigs}')

    print(f'\n  {"-"*120}')
    print('  综合排名 BOTTOM 10')
    print(f'  {"-"*120}')
    for i, r in enumerate(ranked[-10:], 1):
        sigs = ' | '.join(r['signals'][:3]) if r['signals'] else '-'
        print(f'  {i:4d} {r["name"]+"("+r["code"]+")":16s} {r["sector"]:12s} {r["price"]:>8.2f} {r["change_pct"]:>+7.2f}% {r.get("pct_5d",0):>+6.1f}% {r.get("pct_20d",0):>+6.1f}% {r.get("fund_ratio",0):>+6.2f}% {r["score"]:>+4.0f}  {sigs}')

    # ===== 板块强弱 =====
    print(f'\n\n{"="*120}')
    print('  板块强弱排序（按平均评分）')
    print('='*120)
    sector_rank = sorted(sector_stats.items(), key=lambda x: np.mean([s['score'] for s in x[1]]), reverse=True)
    for i, (sector, stocks) in enumerate(sector_rank, 1):
        avg_score = np.mean([s['score'] for s in stocks])
        avg_chg = np.mean([s['change_pct'] for s in stocks])
        avg_fund = np.mean([s.get('fund_ratio', 0) for s in stocks])
        print(f'  {i}. {sector:14s} 评分{avg_score:>+5.1f} 涨跌{avg_chg:>+6.2f}% 资金{avg_fund:>+6.2f}%')

    print(f'\n{"="*120}')
    print('  分析完成')
    print('='*120)


if __name__ == '__main__':
    main()
