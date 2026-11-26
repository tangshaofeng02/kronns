#!/usr/bin/env python3
"""
高连板票全方位观测分析 (资金 / 换手 / 情绪 / 板块)

观测对象: 连板数 ≥ min_lb 的票 (默认 2, 即"2板以上"; 报表按 2板/3板/4板+ 分层)
数据来源:
  - limit_up_data/history.csv        涨停池历史 (封板资金/换手/市值/行业/连板数/封板时间)
  - 东方财富 push2his 资金流接口       主力/大单/超大单净流入 (逐票缓存)

产出: limit_up_data/
  high_board_obs_{date}.csv          逐票观测明细 (资金/换手/情绪/板块四维)
  high_board_report_{date}.txt       综合分析报告
  high_board_history.csv             观测样本累计 (供后续分析)
  fundflow_cache/{code}.csv          个股资金流缓存

用法:
  python scripts/high_board_analysis.py                 # 最近交易日
  python scripts/high_board_analysis.py --date 2026-11-11
  python scripts/high_board_analysis.py --min-lb 3       # 只看 3板以上高标
  python scripts/high_board_analysis.py --all            # 重建全部历史观测
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.limit_up_analysis import (DATA_DIR, fbt_bin, _fbt_display, board_type,
                                      _parse_zt_stat, load_zt)

# ============================================================
# 配置
# ============================================================
FUND_FLOW_URL = 'https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get'
# 兜底数据源: 只返回最新一天 (push2his 限流时使用)
DC_URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
FUND_CACHE_DIR = os.path.join(DATA_DIR, 'fundflow_cache')
HISTORY_FILE = os.path.join(DATA_DIR, 'high_board_history.csv')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
           'Referer': 'https://data.eastmoney.com/'}

# push2his 资金流字段 (已实测: [11]=收盘价, [12]=涨跌幅)
FF_COLS = ['date', 'main_net', 'small_net', 'mid_net', 'big_net', 'huge_net',
           'main_ratio', 'small_ratio', 'mid_ratio', 'big_ratio', 'huge_ratio',
           'close', 'pct_chg']
FUND_DAYS = 120       # 拉取资金流历史天数
MA5_DAYS = 5          # 近5日主力资金合计


def _secid(code):
    """0.002578 / 1.600664"""
    return f'1.{code}' if code.startswith(('6', '9')) else f'0.{code}'


# ============================================================
# 资金流数据 (逐票缓存)
# ============================================================
def fetch_fundflow(code, days=FUND_DAYS, retries=3):
    """拉取个股日频资金流, 返回 DataFrame (失败返回空; 带重试防限流)"""
    params = {
        'lmt': '0', 'klt': '101', 'secid': _secid(code),
        'fields1': 'f1,f2,f3,f7',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
    }
    klines = []
    for attempt in range(retries):
        try:
            r = requests.get(FUND_FLOW_URL, params=params, headers=HEADERS, timeout=15)
            klines = (r.json().get('data') or {}).get('klines') or []
            if klines:
                break
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))   # 限流退避
    if not klines:
        return pd.DataFrame()
    rows = [k.split(',') for k in klines]
    df = pd.DataFrame(rows)
    df = df.iloc[:, :13]
    df.columns = FF_COLS
    for c in FF_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = df['date'].astype(str)
    return df.tail(days).reset_index(drop=True)


def fetch_fundflow_dc(code, retries=2):
    """datacenter 兜底: 只取最新一天资金流 (无历史), 返回 dict 或 None"""
    params = {'reportName': 'RPT_DMSK_TS_STOCKNEW', 'columns': 'ALL',
              'filter': f'(SECURITY_CODE="{code}")', 'pageSize': '1', 'pageNumber': '1',
              'sortColumns': 'TRADE_DATE', 'sortTypes': '-1'}
    for attempt in range(retries):
        try:
            r = requests.get(DC_URL, params=params, headers=HEADERS, timeout=15)
            rows = (r.json().get('result') or {}).get('data') or []
            if rows:
                it = rows[0]
                return {
                    'date': str(it.get('TRADE_DATE', ''))[:10],
                    'main_net': float(it.get('PRIME_INFLOW') or 0),
                    'huge_net': float(it.get('SUPERDEAL_INFLOW') or 0) - float(it.get('SUPERDEAL_OUTFLOW') or 0),
                    'big_net': float(it.get('BIGDEAL_INFLOW') or 0) - float(it.get('BIGDEAL_OUTFLOW') or 0),
                }
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    return None


def load_fundflow(code, date_str=None, refresh=False):
    """读取资金流缓存; 缓存缺失或未覆盖 date_str 时重新拉取"""
    os.makedirs(FUND_CACHE_DIR, exist_ok=True)
    path = os.path.join(FUND_CACHE_DIR, f'{code}.csv')
    df = pd.DataFrame()
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, dtype={'code': str})
    need = df.empty
    if not need and date_str is not None:
        need = df['date'].max() < date_str   # 缓存未覆盖目标日期, 需刷新
    if need:
        new = fetch_fundflow(code)
        if new.empty:
            # push2his 限流/失败 → datacenter 兜底 (仅最新一天, 占比后续用成交额换算)
            dc = fetch_fundflow_dc(code)
            if dc:
                new = pd.DataFrame([{
                    'date': dc['date'], 'main_net': dc['main_net'], 'small_net': 0.0,
                    'mid_net': 0.0, 'big_net': dc['big_net'], 'huge_net': dc['huge_net'],
                    'main_ratio': 0.0, 'small_ratio': 0.0, 'mid_ratio': 0.0,
                    'big_ratio': 0.0, 'huge_ratio': 0.0, 'close': 0.0, 'pct_chg': 0.0,
                }])
        if not new.empty:
            new.to_csv(path, index=False)
            df = new
            time.sleep(0.5)   # 请求间隔, 降低限流概率
    return df


def _num(v, default=0.0):
    """安全转 float (NaN/None/非法 → default)"""
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def fund_metrics_on(df, date_str, amount=0.0):
    """从资金流序列提取当日 + 近5日指标; 占比缺失时用成交额换算"""
    m = {'main_net': 0.0, 'main_ratio': 0.0, 'huge_ratio': 0.0,
         'ma5_main_net': 0.0, 'ma5_main_ratio': 0.0}
    if df is None or df.empty:
        return m
    row = df[df['date'] == date_str]
    if not row.empty:
        r = row.iloc[-1]
        main_net = _num(r.get('main_net'))
        huge_net = _num(r.get('huge_net'))
        m['main_net'] = main_net
        m['huge_ratio'] = _num(r.get('huge_ratio')) or (huge_net / amount * 100 if amount else 0.0)
        m['main_ratio'] = _num(r.get('main_ratio')) or (main_net / amount * 100 if amount else 0.0)
    recent = df[df['date'] <= date_str].tail(MA5_DAYS)
    if not recent.empty:
        m['ma5_main_net'] = float(recent['main_net'].sum())
        ratios = recent['main_ratio'].replace(0.0, float('nan')).dropna()
        m['ma5_main_ratio'] = float(ratios.mean()) if not ratios.empty else 0.0
    return m


# ============================================================
# 观测样本构建
# ============================================================
def layer_of(lbc):
    """连板分层"""
    if lbc >= 4:
        return '4板+'
    if lbc == 3:
        return '3板'
    return '2板'


def build_observations(date_str, min_lb=2, fetch_fund=True):
    """构建某日高连板票的四维观测明细"""
    zt = load_zt(date_str)
    if zt.empty:
        return pd.DataFrame()
    pool = zt[zt['连板数'] >= min_lb].copy()
    if pool.empty:
        return pd.DataFrame()

    # 板块上下文: 当日全市场涨停
    sector_cnt = zt['所属行业'].value_counts().to_dict()
    sector_max = zt.groupby('所属行业')['连板数'].max().to_dict()
    sector_lb2 = zt[zt['连板数'] >= 2].groupby('所属行业').size().to_dict()

    rows = []
    for _, r in pool.iterrows():
        code = str(r['代码']).zfill(6)
        days, ct = _parse_zt_stat(r['涨停统计'])
        circ_mv = float(r['流通市值']) / 1e8
        seal_ratio = (float(r['封板资金']) / float(r['流通市值']) * 100
                      if float(r['流通市值']) > 0 else 0)
        fm = fund_metrics_on(load_fundflow(code, date_str), date_str,
                             amount=float(r['成交额'])) if fetch_fund else {
            'main_net': 0.0, 'main_ratio': 0.0, 'huge_ratio': 0.0,
            'ma5_main_net': 0.0, 'ma5_main_ratio': 0.0}
        sector = str(r['所属行业'])
        rows.append({
            'date': date_str, 'code': code, 'name': r['名称'], 'sector': sector,
            'lbc': int(r['连板数']), 'layer': layer_of(int(r['连板数'])),
            'pct': round(float(r['涨跌幅']), 2),
            # 资金维度
            'seal_fund': round(float(r['封板资金']) / 1e4, 0),        # 封板资金(万)
            'seal_ratio': round(seal_ratio, 3),                       # 封板强度(%)
            'amount': round(float(r['成交额']) / 1e8, 2),             # 成交额(亿)
            'main_net': round(fm['main_net'] / 1e4, 0),               # 主力净流入(万)
            'main_ratio': round(fm['main_ratio'], 2),                 # 主力净占比(%)
            'huge_ratio': round(fm['huge_ratio'], 2),                 # 超大单净占比(%)
            'ma5_main_net': round(fm['ma5_main_net'] / 1e4, 0),       # 近5日主力净额(万)
            'ma5_main_ratio': round(fm['ma5_main_ratio'], 2),         # 近5日主力净占比均值(%)
            # 换手维度
            'turnover': round(float(r['换手率']), 2),
            'circ_mv': round(circ_mv, 1),
            # 情绪维度
            'fbt': _fbt_display(r['首次封板时间']),
            'fbt_bin': fbt_bin(r['首次封板时间']),
            'zbc': int(r['炸板次数']),
            'zt_stat': f'{days}天{ct}板',
            # 板块维度
            'sector_zt_cnt': int(sector_cnt.get(sector, 0)),
            'sector_max_lb': int(sector_max.get(sector, 0)),
            'sector_lb2_cnt': int(sector_lb2.get(sector, 0)),
        })
    return pd.DataFrame(rows)


def append_obs_history(obs):
    """累计观测样本到 high_board_history.csv"""
    if obs.empty:
        return
    old = (pd.read_csv(HISTORY_FILE, dtype={'code': str})
           if os.path.exists(HISTORY_FILE) else pd.DataFrame())
    new = obs.copy()
    if not old.empty:
        old = old[~((old['date'] == new['date'].iloc[0]) & (old['code'].isin(new['code'])))]
    pd.concat([old, new], ignore_index=True).to_csv(HISTORY_FILE, index=False)


# ============================================================
# 分层 / 板块统计
# ============================================================
LAYER_ORDER = ['2板', '3板', '4板+']


def layer_stats(obs):
    """按连板分层聚合"""
    if obs.empty:
        return pd.DataFrame()
    g = obs.groupby('layer')
    out = pd.DataFrame({
        '数量': g.size(),
        '平均换手%': g['turnover'].mean().round(2),
        '中位换手%': g['turnover'].median().round(2),
        '平均封板强度%': g['seal_ratio'].mean().round(2),
        '平均主力净占比%': g['main_ratio'].mean().round(2),
        '主力净流入为正占比%': g['main_ratio'].apply(lambda s: (s > 0).mean() * 100).round(1),
        '近5日主力净额均值(万)': g['ma5_main_net'].mean().round(0),
        '平均炸板次数': g['zbc'].mean().round(2),
        '早盘封板占比%': g['fbt_bin'].apply(
            lambda s: s.isin(['集合竞价(≤9:25)', '开盘(9:25-9:30)', '早盘(9:30-10:30)']).mean() * 100).round(1),
        '平均流通市值(亿)': g['circ_mv'].mean().round(1),
    })
    return out.reindex([l for l in LAYER_ORDER if l in out.index])


def sector_stats(obs, top_n=10):
    """板块联动统计"""
    if obs.empty:
        return pd.DataFrame()
    g = obs.groupby('sector')
    out = pd.DataFrame({
        '高标数': g.size(),
        '最高连板': g['lbc'].max(),
        '平均连板': g['lbc'].mean().round(2),
        '主力净额合计(万)': g['main_net'].sum().round(0),
        '平均换手%': g['turnover'].mean().round(2),
    })
    return out.sort_values(['高标数', '最高连板'], ascending=False).head(top_n)


def fund_extremes(obs, n=5):
    """资金流入/流出极值"""
    if obs.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = obs.nlargest(n, 'main_net')[['name', 'code', 'layer', 'sector', 'lbc',
                                      'main_net', 'main_ratio', 'turnover']]
    bot = obs.nsmallest(n, 'main_net')[['name', 'code', 'layer', 'sector', 'lbc',
                                        'main_net', 'main_ratio', 'turnover']]
    return top, bot


def turnover_layers(obs):
    """换手分层分布"""
    if obs.empty:
        return {}
    bins = [(0, 3, '<3%'), (3, 8, '3-8%'), (8, 15, '8-15%'), (15, 100, '≥15%')]
    out = {}
    for lo, hi, label in bins:
        out[label] = int(((obs['turnover'] >= lo) & (obs['turnover'] < hi)).sum())
    return out


# ============================================================
# 跨日规律分析 (为后续分析做准备)
# ============================================================
def _zt_lb_map(date_str):
    """{code: 连板数}"""
    zt = load_zt(date_str)
    if zt.empty:
        return {}
    return {str(c).zfill(6): int(l) for c, l in zip(zt['代码'], zt['连板数'])}


def compute_continuation():
    """为历史观测补充次日延续标签 cont (1=次日继续封板)"""
    h = (pd.read_csv(HISTORY_FILE, dtype={'code': str})
         if os.path.exists(HISTORY_FILE) else pd.DataFrame())
    if h.empty:
        return h
    dates = sorted(h['date'].unique())
    next_day = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    cache = {}
    conts = []
    for _, r in h.iterrows():
        nd = next_day.get(r['date'])
        if nd is None:
            conts.append(None)
            continue
        if nd not in cache:
            cache[nd] = _zt_lb_map(nd)
        nlb = cache[nd].get(r['code'])
        conts.append(1 if (nlb is not None and nlb >= int(r['lbc']) + 1) else 0)
    h['cont'] = conts
    return h


def _rank_corr(a, b):
    """秩相关 (无需 scipy)"""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def build_pattern_report(h):
    """跨日全方面规律报告"""
    h = compute_continuation()
    L = []
    L.append('=' * 68)
    L.append('  高连板票跨日规律分析 (为后续分析做准备)')
    L.append('=' * 68)

    L.append('')
    L.append('【一、样本总览】')
    L.append(f'  观测样本: {len(h)} 条 | 交易日: {h["date"].nunique()} | 股票: {h["code"].nunique()}')
    L.append(f'  分层: ' + ' | '.join(f'{k} {v}' for k, v in h['layer'].value_counts().items()))
    L.append(f'  连板分布: ' + ' | '.join(f'{k}板 {v}' for k, v in h['lbc'].value_counts().sort_index().items()))

    L.append('')
    L.append('【二、连板高度 与 四维指标的秩相关】')
    for col, name in [('seal_ratio', '封板强度'), ('seal_fund', '封板资金'),
                      ('main_ratio', '主力净占比'), ('huge_ratio', '超大单净占比'),
                      ('ma5_main_net', '近5日主力净额'), ('turnover', '换手率'),
                      ('circ_mv', '流通市值'), ('sector_lb2_cnt', '板块高标数'),
                      ('zbc', '炸板次数')]:
        L.append(f'  {name:<12s} {_rank_corr(h["lbc"], h[col]):+.3f}')

    L.append('')
    L.append('【三、分层全方面对比】')
    g = h.groupby('layer')
    ls = pd.DataFrame({
        '数量': g.size(),
        '平均换手%': g['turnover'].mean().round(2),
        '平均封板强度%': g['seal_ratio'].mean().round(2),
        '平均主力净占比%': g['main_ratio'].mean().round(2),
        '主力为正占比%': g['main_ratio'].apply(lambda s: (s > 0).mean() * 100).round(1),
        '近5日主力均值(亿)': (g['ma5_main_net'].mean() / 1e4).round(2),
        '平均炸板': g['zbc'].mean().round(2),
        '零炸板占比%': g['zbc'].apply(lambda s: (s == 0).mean() * 100).round(1),
        '早盘封板占比%': g['fbt_bin'].apply(
            lambda s: s.isin(['集合竞价(≤9:25)', '开盘(9:25-9:30)', '早盘(9:30-10:30)']).mean() * 100).round(1),
        '平均流通市值(亿)': g['circ_mv'].mean().round(1),
        '平均板块高标数': g['sector_lb2_cnt'].mean().round(2),
    }).reindex([l for l in LAYER_ORDER if l in g.groups])
    L.append('  层标   数量  换手%  封板强度 主力占比 主力为正 近5日主力 炸板 零炸板 早盘  流通市值 板块高标')
    for layer, r in ls.iterrows():
        L.append(f'  {layer:<6s} {int(r["数量"]):>3d}  {r["平均换手%"]:>6.2f}  {r["平均封板强度%"]:>6.2f}  '
                 f'{r["平均主力净占比%"]:>6.2f}  {r["主力为正占比%"]:>6.1f}%  {r["近5日主力均值(亿)"]:>6.2f}亿  '
                 f'{r["平均炸板"]:>4.2f}  {r["零炸板占比%"]:>5.1f}%  {r["早盘封板占比%"]:>5.1f}%  '
                 f'{r["平均流通市值(亿)"]:>6.1f}亿  {r["平均板块高标数"]:>5.2f}')

    sub = h[h['cont'].notna()]
    if not sub.empty:
        L.append('')
        L.append('【四、高标延续性 (次日是否继续封板)】')
        L.append(f'  整体延续率: {sub["cont"].mean()*100:.1f}% (样本 {len(sub)})')
        L.append('  按层:')
        for layer, r in sub.groupby('layer')['cont'].agg(['mean', 'count']).iterrows():
            L.append(f'    {layer:<6s} {r["mean"]*100:>5.1f}%  (n={int(r["count"])})')
        L.append('  按连板数:')
        for lbc, r in sub.groupby('lbc')['cont'].agg(['mean', 'count']).iterrows():
            L.append(f'    {int(lbc)}板→{int(lbc)+1}板  {r["mean"]*100:>5.1f}%  (n={int(r["count"])})')

        L.append('')
        L.append('【五、延续 vs 断板 特征对比 (核心规律)】')
        gg = sub.groupby('cont')
        cmp_df = pd.DataFrame({
            '数量': gg.size(),
            '平均换手%': gg['turnover'].mean().round(2),
            '平均封板强度%': gg['seal_ratio'].mean().round(2),
            '平均主力净占比%': gg['main_ratio'].mean().round(2),
            '平均炸板': gg['zbc'].mean().round(2),
            '早盘封板占比%': gg['fbt_bin'].apply(
                lambda s: s.isin(['集合竞价(≤9:25)', '开盘(9:25-9:30)', '早盘(9:30-10:30)']).mean() * 100).round(1),
        })
        L.append('  分组     数量  换手%  封板强度 主力占比 炸板 早盘封板')
        for idx in [1, 0]:
            r = cmp_df.loc[idx]
            tag = '延续' if idx == 1 else '断板'
            L.append(f'  {tag:<6s} {int(r["数量"]):>3d}  {r["平均换手%"]:>6.2f}  {r["平均封板强度%"]:>6.2f}  '
                     f'{r["平均主力净占比%"]:>6.2f}  {r["平均炸板"]:>4.2f}  {r["早盘封板占比%"]:>6.1f}%')
        # 差异最大的维度 (排除数量列)
        d = (cmp_df.loc[1] - cmp_df.loc[0]).drop('数量').abs().sort_values(ascending=False)
        L.append(f'  区分度最大维度: ' + ', '.join(d.head(3).index.tolist()))

    L.append('')
    L.append('【六、板块联动 TOP (按高标数)】')
    ss = h.groupby('sector').agg(
        高标数=('code', 'size'), 最高板=('lbc', 'max'),
        主力净额合计万=('main_net', 'sum'), 平均换手=('turnover', 'mean')).round(2)
    ss = ss.sort_values(['高标数', '最高板'], ascending=False).head(12)
    L.append('  行业        高标数  最高板  主力净额合计  平均换手')
    for sec, r in ss.iterrows():
        L.append(f'  {str(sec):<12s} {int(r["高标数"]):>4d}  {int(r["最高板"]):>5d}  '
                 f'{r["主力净额合计万"]:>10.0f}万  {r["平均换手"]:>6.2f}%')

    L.append('')
    L.append('  注: 延续标签 = 次日该票再次封板且连板数+1; 样本不足(连板≥6)的结论请谨慎参考')
    L.append('')
    return '\n'.join(L)


# ============================================================
# 报告
# ============================================================
def build_report(date_str, obs, ls, ss, top_in, top_out, tl):
    L = []
    L.append('=' * 66)
    L.append(f'  {date_str} 高连板票全方位观测 (连板≥2)')
    L.append('=' * 66)

    L.append('')
    L.append('【一、概览】')
    L.append(f'  高标数量: {len(obs)} 只 | 最高连板: {int(obs["lbc"].max())} 板 | '
             f'涉及行业: {obs["sector"].nunique()} 个')
    L.append(f'  分层: ' + ' | '.join(f'{k} {v}只' for k, v in
                                      obs['layer'].value_counts().reindex(LAYER_ORDER).dropna().items()))
    L.append(f'  换手分层: ' + ' | '.join(f'{k} {v}只' for k, v in tl.items()))

    L.append('')
    L.append('【二、分层对比 (资金/换手/封板质量)】')
    L.append('  层标   数量  平均换手  封板强度  主力净占比  主力为正  近5日主力   炸板  早盘封板  流通市值')
    for layer, r in ls.iterrows():
        L.append(f'  {layer:<6s} {int(r["数量"]):>3d}  {r["平均换手%"]:>7.2f}%  '
                 f'{r["平均封板强度%"]:>7.2f}%  {r["平均主力净占比%"]:>8.2f}%  '
                 f'{r["主力净流入为正占比%"]:>7.1f}%  {r["近5日主力净额均值(万)"]:>9.0f}万  '
                 f'{r["平均炸板次数"]:>5.2f}  {r["早盘封板占比%"]:>7.1f}%  {r["平均流通市值(亿)"]:>7.1f}亿')

    L.append('')
    L.append('【三、资金分析】')
    L.append(f'  主力净流入合计: {obs["main_net"].sum()/1e4:.2f} 亿 | '
             f'为正: {(obs["main_net"]>0).sum()}/{len(obs)} 只 ({(obs["main_net"]>0).mean()*100:.0f}%)')
    L.append(f'  近5日主力净流入合计: {obs["ma5_main_net"].sum()/1e4:.2f} 亿')
    L.append('  主力净流入 TOP5:')
    for _, r in top_in.iterrows():
        L.append(f'    {r["name"]}({r["code"]}) {r["layer"]} 主力{r["main_net"]:+.0f}万 '
                 f'占比{r["main_ratio"]:+.2f}% 换手{r["turnover"]:.1f}% {r["sector"]}')
    L.append('  主力净流出 TOP5:')
    for _, r in top_out.iterrows():
        L.append(f'    {r["name"]}({r["code"]}) {r["layer"]} 主力{r["main_net"]:+.0f}万 '
                 f'占比{r["main_ratio"]:+.2f}% 换手{r["turnover"]:.1f}% {r["sector"]}')

    L.append('')
    L.append('【四、换手与封板质量】')
    L.append(f'  平均换手: {obs["turnover"].mean():.2f}% | 中位: {obs["turnover"].median():.2f}% | '
             f'最高: {obs["turnover"].max():.2f}%')
    L.append(f'  封板强度(封板资金/流通市值) 均值: {obs["seal_ratio"].mean():.2f}% | '
             f'最高: {obs["seal_ratio"].max():.2f}%')
    L.append(f'  炸板次数: 平均 {obs["zbc"].mean():.2f} | 0次 {(obs["zbc"]==0).sum()}只 | '
             f'≥3次 {(obs["zbc"]>=3).sum()}只')
    L.append('  封板时段分布: ' + ' | '.join(
        f'{k} {v}只' for k, v in obs['fbt_bin'].value_counts()
        .reindex(['集合竞价(≤9:25)', '开盘(9:25-9:30)', '早盘(9:30-10:30)',
                  '午前(10:30-11:30)', '午后(13:00-14:00)', '尾盘(≥14:00)']).dropna().items()))

    L.append('')
    L.append('【五、板块联动】')
    L.append('  行业      高标数  最高板  平均连板  主力净额合计  平均换手')
    for sec, r in ss.iterrows():
        L.append(f'  {str(sec):<10s} {int(r["高标数"]):>4d}  {int(r["最高连板"]):>5d}  '
                 f'{r["平均连板"]:>7.2f}  {r["主力净额合计(万)"]:>10.0f}万  {r["平均换手%"]:>7.2f}%')

    L.append('')
    L.append('【六、观测样本明细】')
    L.append('  名称(代码)        层标  行业      连板 封板  炸板 换手   封板强度 主力净额  主力占比 近5日主力')
    for _, r in obs.sort_values(['lbc', 'main_net'], ascending=[False, False]).iterrows():
        L.append(f'  {r["name"]+"("+r["code"]+")":<16s} {r["layer"]:<4s} {str(r["sector"]):<8s} '
                 f'{int(r["lbc"]):>3d}  {r["fbt"]:<5s} {int(r["zbc"]):>3d}  '
                 f'{r["turnover"]:>5.2f}% {r["seal_ratio"]:>6.2f}% '
                 f'{r["main_net"]:>7.0f}万 {r["main_ratio"]:>6.2f}% {r["ma5_main_net"]:>8.0f}万')
    L.append('')
    L.append('  注: 资金流来自东方财富 push2his (主力/超大单口径); 观测样本累计于 high_board_history.csv')
    L.append('')
    return '\n'.join(L)


# ============================================================
# 主流程
# ============================================================
def default_date():
    today = datetime.now()
    if today.weekday() < 5:
        return today.strftime('%Y-%m-%d')
    d = today - timedelta(days=today.weekday() - 4)
    return d.strftime('%Y-%m-%d')


def run_date(date_str, min_lb, fetch_fund=True):
    """分析单个交易日"""
    obs = build_observations(date_str, min_lb, fetch_fund)
    if obs.empty:
        print(f'  {date_str}: 无连板≥{min_lb} 的票')
        return None
    append_obs_history(obs)
    obs.to_csv(os.path.join(DATA_DIR, f'high_board_obs_{date_str}.csv'), index=False)
    ls = layer_stats(obs)
    ss = sector_stats(obs)
    top_in, top_out = fund_extremes(obs)
    tl = turnover_layers(obs)
    report = build_report(date_str, obs, ls, ss, top_in, top_out, tl)
    print(report)
    with open(os.path.join(DATA_DIR, f'high_board_report_{date_str}.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'  已保存: high_board_obs_{date_str}.csv / high_board_report_{date_str}.txt')
    return obs


def main():
    parser = argparse.ArgumentParser(description='高连板票全方位观测分析')
    parser.add_argument('--date', help='指定交易日 YYYY-MM-DD')
    parser.add_argument('--min-lb', type=int, default=2, help='最低连板数 (默认 2)')
    parser.add_argument('--all', action='store_true', help='重建全部历史观测')
    parser.add_argument('--patterns', action='store_true', help='跨日规律分析')
    parser.add_argument('--no-fund', action='store_true', help='跳过资金流抓取')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_fund = not args.no_fund

    if args.patterns:
        if not os.path.exists(HISTORY_FILE):
            print('无观测样本, 请先运行 --all')
            return
        report = build_pattern_report(pd.read_csv(HISTORY_FILE, dtype={'code': str}))
        print(report)
        out = os.path.join(DATA_DIR, 'high_board_patterns.txt')
        with open(out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'  已保存: {out}')
        return

    if args.all:
        dates = sorted({f.replace('_zt.csv', '') for f in os.listdir(DATA_DIR)
                        if f.endswith('_zt.csv')})
        print(f'重建 {len(dates)} 个交易日的观测 (连板≥{args.min_lb})')
        for ds in dates:
            run_date(ds, args.min_lb, fetch_fund)
            time.sleep(0.3)
        print(f'\n完成, 累计观测: {len(pd.read_csv(HISTORY_FILE, dtype={"code": str}))} 条')
        return

    date_str = args.date or default_date()
    print(f'Kronos 高连板观测 - {datetime.now().strftime("%Y-%m-%d %H:%M")} (连板≥{args.min_lb})')
    run_date(date_str, args.min_lb, fetch_fund)


if __name__ == '__main__':
    main()
