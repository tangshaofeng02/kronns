#!/usr/bin/env python3
"""
每日首板涨停票统计分析与次日连板推荐

数据来源: 东方财富涨停板行情 (akshare 优先, 失败回退 push2ex 直连)
  - 涨停股池   stock_zt_pool_em          当日收盘涨停 (连板数/封板时间/封板资金/行业)
  - 炸板股池   stock_zt_pool_zbgc_em     当日触板未封住
  - 昨日涨停池 stock_zt_pool_previous_em 昨日涨停股今日表现 (连板标签来源)

产出: limit_up_data/
  {date}_zt.csv / {date}_zbgc.csv / {date}_prev.csv   每日快照
  {date}_recommend.csv                                次日连板推荐
  {date}_report.txt                                   统计分析报告
  history.csv                                         涨停池历史合并
  label_history.csv                                   首板→次日连板 标签样本

推荐模型: 分箱条件概率 (m-估计平滑 + log-odds 叠加 + 温度压缩)
  对每只今日首板票, 按 封板时段/炸板次数/板块类型/流通市值/换手率/
  封板资金强度/近期涨停/行业 分别查历史连板率, 以 log-odds 偏离叠加到
  先验上 (×0.6 温度压缩校准), 得到次日连板概率。
  先验 = 历史首板连板率, 并按昨日首板连板率/炸板率做情绪平移。

用法:
  python scripts/limit_up_analysis.py                # 最近交易日: 抓取+分析+推荐
  python scripts/limit_up_analysis.py --date 2026-11-11
  python scripts/limit_up_analysis.py --backfill     # 回填全部可得历史(近30日)
  python scripts/limit_up_analysis.py --skip-fetch   # 只用本地数据分析
注意: 涨停池数据收盘后才定型, 建议 15:00 后运行 (盘中运行需加 --force)
"""

import os
import math
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

# ============================================================
# 配置
# ============================================================
DATA_DIR = './limit_up_data'
TOP_N = 10            # 推荐数量
ALPHA = 8.0           # 贝叶斯平滑强度 (越大越依赖全局均值)
TEMPER = 0.6          # log-odds 温度压缩 (回测校准: 0.6 时 TOP10 预测≈实际)
DEFAULT_RATE = 0.35   # 无历史数据时的首板连板率先验
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

ZT_URL = 'https://push2ex.eastmoney.com/getTopicZTPool'
ZBG_URL = 'https://push2ex.eastmoney.com/getTopicZBPool'
PREV_URL = 'https://push2ex.eastmoney.com/getYesterdayZTPool'
PUSH2EX_PARAMS = {'ut': '7eea3edcaed734bea9cbfc24409ed989', 'dpt': 'wz.ztzt',
                  'Pageindex': '0', 'pagesize': '10000'}


# ============================================================
# 数据抓取
# ============================================================
def _push2ex(url, date8, sort='fbt:asc'):
    """直连 push2ex 接口, 返回原始 pool 列表"""
    params = dict(PUSH2EX_PARAMS)
    params['sort'] = sort
    params['date'] = date8
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        data = r.json().get('data') or {}
        return data.get('pool', []) or []
    except Exception:
        return []


def _parse_zt_stat_field(zttj):
    """涨停统计字段 → (天数, 板数)"""
    if isinstance(zttj, dict):
        return zttj.get('days', 0) or 0, zttj.get('ct', 0) or 0
    return 0, 0


def _parse_zt_pool(pool):
    """涨停池原始数据 → DataFrame"""
    rows = []
    for it in pool:
        days, ct = _parse_zt_stat_field(it.get('zttj') or {})
        rows.append({
            '代码': str(it.get('c', '')).zfill(6),
            '名称': it.get('n', ''),
            '涨跌幅': float(it.get('zdp', 0) or 0),
            '最新价': float(it.get('p', 0) or 0) / 1000,
            '成交额': float(it.get('amount', 0) or 0),
            '流通市值': float(it.get('ltsz', 0) or 0),
            '总市值': float(it.get('tshare', 0) or 0),
            '换手率': float(it.get('hs', 0) or 0),
            '封板资金': float(it.get('fund', 0) or 0),
            '首次封板时间': str(it.get('fbt', '')).zfill(6),
            '最后封板时间': str(it.get('lbt', '')).zfill(6),
            '炸板次数': int(it.get('zbc', 0) or 0),
            '涨停统计': f'{days}/{ct}',
            '连板数': int(it.get('lbc', 0) or 0),
            '所属行业': it.get('hybk', ''),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, '序号', range(1, len(df) + 1))
    return df


def _parse_zbgc_pool(pool):
    """炸板池原始数据 → DataFrame"""
    rows = []
    for it in pool:
        days, ct = _parse_zt_stat_field(it.get('zttj') or {})
        rows.append({
            '代码': str(it.get('c', '')).zfill(6),
            '名称': it.get('n', ''),
            '涨跌幅': float(it.get('zdp', 0) or 0),
            '最新价': float(it.get('p', 0) or 0) / 1000,
            '涨停价': float(it.get('ztp', 0) or 0) / 1000,
            '成交额': float(it.get('amount', 0) or 0),
            '流通市值': float(it.get('ltsz', 0) or 0),
            '总市值': float(it.get('tshare', 0) or 0),
            '换手率': float(it.get('hs', 0) or 0),
            '首次封板时间': str(it.get('fbt', '')).zfill(6),
            '炸板次数': int(it.get('zbc', 0) or 0),
            '涨停统计': f'{days}/{ct}',
            '所属行业': it.get('hybk', ''),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, '序号', range(1, len(df) + 1))
    return df


def _parse_prev_pool(pool):
    """昨日涨停池原始数据 → DataFrame"""
    rows = []
    for it in pool:
        days, ct = _parse_zt_stat_field(it.get('zttj') or {})
        rows.append({
            '代码': str(it.get('c', '')).zfill(6),
            '名称': it.get('n', ''),
            '涨跌幅': float(it.get('zdp', 0) or 0),
            '最新价': float(it.get('p', 0) or 0) / 1000,
            '涨停价': float(it.get('ztp', 0) or 0) / 1000,
            '成交额': float(it.get('amount', 0) or 0),
            '流通市值': float(it.get('ltsz', 0) or 0),
            '总市值': float(it.get('tshare', 0) or 0),
            '换手率': float(it.get('hs', 0) or 0),
            '昨日封板时间': str(it.get('yfbt', '')).zfill(6),
            '昨日连板数': int(it.get('ylbc', 0) or 0),
            '涨停统计': f'{days}/{ct}',
            '所属行业': it.get('hybk', ''),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, '序号', range(1, len(df) + 1))
    return df


def fetch_zt_pool(date_str):
    """涨停股池: akshare 优先, 失败回退 push2ex 直连"""
    date8 = date_str.replace('-', '')
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date=date8)
        if df is not None and len(df):
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df
    except Exception:
        pass
    return _parse_zt_pool(_push2ex(ZT_URL, date8))


def fetch_zbgc_pool(date_str):
    """炸板股池: akshare 优先, 失败回退 push2ex 直连"""
    date8 = date_str.replace('-', '')
    try:
        import akshare as ak
        df = ak.stock_zt_pool_zbgc_em(date=date8)
        if df is not None and len(df):
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df
    except Exception:
        pass
    return _parse_zbgc_pool(_push2ex(ZBG_URL, date8))


def fetch_prev_pool(date_str):
    """昨日涨停股池(今日表现): akshare 优先, 失败回退 push2ex 直连"""
    date8 = date_str.replace('-', '')
    try:
        import akshare as ak
        df = ak.stock_zt_pool_previous_em(date=date8)
        if df is not None and len(df):
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df
    except Exception:
        pass
    return _parse_prev_pool(_push2ex(PREV_URL, date8, sort='zs:desc'))


def fetch_day(date_str):
    """抓取某日三个池子并存档, 同时追加历史与标签"""
    print(f'\n{'='*60}')
    print(f'  抓取 {date_str} 涨停板数据')
    print(f'{'='*60}')
    zt = fetch_zt_pool(date_str)
    zbgc = fetch_zbgc_pool(date_str)
    prev = fetch_prev_pool(date_str)
    print(f'  涨停池: {len(zt)} 只 | 炸板池: {len(zbgc)} 只 | 昨日涨停池: {len(prev)} 只')

    if zt.empty and zbgc.empty and prev.empty:
        print(f'  {date_str} 无数据 (可能为非交易日)')
        return zt, zbgc, prev

    zt.to_csv(os.path.join(DATA_DIR, f'{date_str}_zt.csv'), index=False)
    zbgc.to_csv(os.path.join(DATA_DIR, f'{date_str}_zbgc.csv'), index=False)
    prev.to_csv(os.path.join(DATA_DIR, f'{date_str}_prev.csv'), index=False)

    append_history(date_str, zt)
    build_labels(date_str, prev)
    return zt, zbgc, prev


# ============================================================
# 历史数据积累
# ============================================================
HISTORY_COLS = ['date', '代码', '名称', '涨跌幅', '最新价', '成交额', '流通市值', '总市值',
                '换手率', '封板资金', '首次封板时间', '最后封板时间', '炸板次数',
                '涨停统计', '连板数', '所属行业']


def append_history(date_str, zt):
    """把当日涨停池追加到 history.csv"""
    if zt.empty:
        return
    path = os.path.join(DATA_DIR, 'history.csv')
    old = pd.read_csv(path, dtype={'代码': str}) if os.path.exists(path) else pd.DataFrame()
    new = zt[[c for c in HISTORY_COLS if c != 'date']].copy()
    new.insert(0, 'date', date_str)
    if not old.empty:
        old = old[~((old['date'] == date_str) & (old['代码'].isin(new['代码'])))]
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(path, index=False)


def _parse_zt_stat(s):
    """涨停统计 'X/Y' → (X天数, Y板数)"""
    try:
        x, y = str(s).split('/')
        return int(x), int(y)
    except Exception:
        return 0, 0


def _prev_trading_day(date_str):
    """返回 date_str 的前一个交易日 (只跳过周末)"""
    d = datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def build_labels(date_str, prev):
    """用昨日涨停池(今日表现)构建首板→次日连板标签, 追加到 label_history.csv

    prev 中「昨日连板数==1」的行 = 前一交易日的首板票,
    标签 = 今日收盘是否涨停 (最新价 >= 涨停价)。
    特征取自前一交易日的涨停池快照。
    """
    if prev.empty:
        return
    prev_date_str = _prev_trading_day(date_str)
    zt_prev = load_zt(prev_date_str)
    if zt_prev.empty:
        return

    feat_map = {r['代码']: r for _, r in zt_prev.iterrows()}
    path = os.path.join(DATA_DIR, 'label_history.csv')
    old = pd.read_csv(path, dtype={'code': str}) if os.path.exists(path) else pd.DataFrame()

    rows = []
    for _, r in prev.iterrows():
        if int(r.get('昨日连板数', 0)) != 1:
            continue
        code = str(r['代码']).zfill(6)
        f = feat_map.get(code)
        if f is None:
            continue
        is_zt = 1 if float(r['最新价']) >= float(r['涨停价']) - 0.011 else 0
        days, ct = _parse_zt_stat(f['涨停统计'])
        circ_mv = float(f['流通市值']) / 1e8
        seal_ratio = (float(f['封板资金']) / float(f['流通市值']) * 100
                      if float(f['流通市值']) > 0 else 0)
        rows.append({
            'date': prev_date_str, 'code': code, 'name': r['名称'], 'sector': f['所属行业'],
            'fbt': str(f['首次封板时间']).zfill(6), 'zbc': int(f['炸板次数']),
            'board_type': board_type(float(f['涨跌幅'])),
            'circ_mv': round(circ_mv, 2), 'turnover': round(float(f['换手率']), 2),
            'seal_ratio': round(seal_ratio, 3), 'zt_days': days, 'zt_count': ct,
            'label': is_zt,
        })
    if not rows:
        return
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~((old['date'] == prev_date_str) & (old['code'].isin(new['code'])))]
    merged = pd.concat([old, new], ignore_index=True)
    merged.to_csv(path, index=False)
    rate = new['label'].mean() * 100
    print(f'  标签样本: {len(new)} 条 ({prev_date_str} 首板 → {date_str} 连板率 {rate:.1f}%)')


# ============================================================
# 特征工程
# ============================================================
def board_type(pct):
    """板块类型: 按涨停幅度区分 10cm/20cm/30cm"""
    if pct >= 25:
        return '30cm'
    if pct >= 15:
        return '20cm'
    return '10cm'


def _fbt_display(fbt):
    """首次封板时间 HHMMSS → 'HH:MM'"""
    s = str(fbt).zfill(6)
    if not s.isdigit():
        return '--:--'
    return f'{s[0:2]}:{s[2:4]}'


def fbt_minutes(fbt):
    """首次封板时间 HHMMSS → 分钟数 (从 0:00 起)"""
    s = str(fbt).zfill(6)
    if not s.isdigit():
        return 15 * 60
    h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
    return h * 60 + m + (1 if sec > 0 else 0)


def fbt_bin(fbt):
    """封板时段分箱"""
    m = fbt_minutes(fbt)
    if m <= 9 * 60 + 25:
        return '集合竞价(≤9:25)'
    if m <= 9 * 60 + 30:
        return '开盘(9:25-9:30)'
    if m <= 10 * 60 + 30:
        return '早盘(9:30-10:30)'
    if m <= 11 * 60 + 30:
        return '午前(10:30-11:30)'
    if m <= 14 * 60:
        return '午后(13:00-14:00)'
    return '尾盘(≥14:00)'


def zbc_bin(zbc):
    """炸板次数分箱"""
    z = int(zbc)
    return '0次' if z == 0 else ('1次' if z == 1 else '2次+')


def circ_mv_bin(circ_mv):
    """流通市值(亿)分箱"""
    v = float(circ_mv)
    if v < 30:
        return '<30亿'
    if v < 60:
        return '30-60亿'
    if v < 100:
        return '60-100亿'
    if v < 200:
        return '100-200亿'
    return '≥200亿'


def turnover_bin(t):
    """换手率(%)分箱"""
    v = float(t)
    if v < 3:
        return '<3%'
    if v < 8:
        return '3-8%'
    if v < 15:
        return '8-15%'
    return '≥15%'


def seal_ratio_bin(r):
    """封板资金强度(封板资金/流通市值 %)分箱"""
    v = float(r)
    if v < 0.5:
        return '<0.5%'
    if v < 1.5:
        return '0.5-1.5%'
    if v < 3:
        return '1.5-3%'
    if v < 6:
        return '3-6%'
    return '≥6%'


def zt_count_bin(ct):
    """近期涨停板数(涨停统计分子)分箱"""
    c = int(ct)
    if c <= 1:
        return '1板(冷启动)'
    if c == 2:
        return '2板'
    return '3板+'


def extract_features(row):
    """从涨停池行提取特征 dict"""
    days, ct = _parse_zt_stat(row['涨停统计'])
    circ_mv = float(row['流通市值']) / 1e8
    seal_ratio = (float(row['封板资金']) / float(row['流通市值']) * 100
                  if float(row['流通市值']) > 0 else 0)
    return {
        '封板时段': fbt_bin(row['首次封板时间']),
        '炸板次数': zbc_bin(row['炸板次数']),
        '板块类型': board_type(float(row['涨跌幅'])),
        '流通市值': circ_mv_bin(circ_mv),
        '换手率': turnover_bin(row['换手率']),
        '封板强度': seal_ratio_bin(seal_ratio),
        '近期涨停': zt_count_bin(ct),
        '行业': str(row['所属行业']),
    }


FEATURE_KEYS = ['封板时段', '炸板次数', '板块类型', '流通市值', '换手率', '封板强度', '近期涨停', '行业']


# ============================================================
# 评分模型 (朴素贝叶斯 log-odds)
# ============================================================
def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


class LBScorer:
    """首板→连板 概率评分器

    对每个特征分箱统计历史连板率, 用贝叶斯平滑, 以 log-odds 形式累加偏离。
    """

    def __init__(self, label_df, prior):
        self.prior = min(max(prior, 0.05), 0.8)
        self.logit0 = logit(self.prior)
        self.sentiment_shift = 0.0   # 情绪平移 (在 train_scorer 中设置)
        self.stats = {}   # {feat: {bin: (success, total)}}
        if label_df is None or label_df.empty:
            return
        for feat in FEATURE_KEYS:
            groups = {}
            for _, r in label_df.iterrows():
                b = self._bin(feat, r)
                s, t = groups.get(b, (0, 0))
                groups[b] = (s + int(r['label']), t + 1)
            self.stats[feat] = groups

    @staticmethod
    def _bin(feat, r):
        """把标签样本的原始字段映射到分箱 (与 extract_features 的分箱一致)"""
        if feat == '封板时段':
            return fbt_bin(r.get('fbt', ''))
        if feat == '炸板次数':
            return zbc_bin(r.get('zbc', 0))
        if feat == '板块类型':
            return r.get('board_type', '') or '10cm'
        if feat == '流通市值':
            return circ_mv_bin(r.get('circ_mv', 0))
        if feat == '换手率':
            return turnover_bin(r.get('turnover', 0))
        if feat == '封板强度':
            return seal_ratio_bin(r.get('seal_ratio', 0))
        if feat == '近期涨停':
            return zt_count_bin(r.get('zt_count', 0))
        if feat == '行业':
            return r.get('sector', '') or '未知'
        return ''

    def _adj(self, feat, b):
        """单个特征分箱的 log-odds 偏离 (m-估计平滑后的连板率 vs 先验)"""
        groups = self.stats.get(feat)
        if not groups:
            return 0.0
        s, t = groups.get(b, (0, 0))
        rate = (s + ALPHA * self.prior) / (t + ALPHA)
        return logit(rate) - self.logit0

    def score(self, features):
        """返回 (概率, 评分明细列表[(特征, 分箱, 偏离)])"""
        logit_sum = self.logit0 + self.sentiment_shift
        detail = []
        for feat in FEATURE_KEYS:
            b = features.get(feat, '')
            adj = self._adj(feat, b) * TEMPER
            logit_sum += adj
            detail.append((feat, b, adj))
        return sigmoid(logit_sum), detail


def train_scorer(label_df, prev_rate, zbgc_rate, default_rate):
    """训练评分器

    分箱统计基于训练数据基准率, 情绪因子(昨日首板连板率/炸板率)只作最终
    log-odds 平移, 避免先验与分箱率基准不一致导致的系统性偏差。
    """
    if label_df is not None and not label_df.empty:
        global_rate = float(label_df['label'].mean())
    else:
        global_rate = default_rate
    scorer = LBScorer(label_df, global_rate)

    # 情绪调整: 相对训练基准的 log-odds 平移
    adjusted = global_rate
    if prev_rate is not None:
        adjusted = 0.6 * global_rate + 0.4 * prev_rate
        if prev_rate > global_rate * 1.15:
            adjusted *= 1.10
        elif prev_rate < global_rate * 0.85:
            adjusted *= 0.90
    if zbgc_rate is not None:
        if zbgc_rate > 0.35:
            adjusted *= 0.85
        elif zbgc_rate < 0.20:
            adjusted *= 1.08
    adjusted = min(max(adjusted, 0.05), 0.8)
    scorer.sentiment_shift = logit(adjusted) - logit(global_rate)
    return scorer, global_rate, adjusted


# ============================================================
# XGBoost 评分器 (替代 LBScorer)
# ============================================================
XGB_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'models', 'xgb_limit_up.pkl')


def extract_features_xgb(row, prev_rate=None, prev_rate_3d=None, prev_n=None):
    """从涨停池行提取数值特征 (供 XGBoost 使用)

    返回 dict: 数值特征 + 类别特征 (与训练脚本一致)
    """
    days, ct = _parse_zt_stat(row['涨停统计'])
    circ_mv = float(row['流通市值']) / 1e8
    seal_ratio = (float(row['封板资金']) / float(row['流通市值']) * 100
                  if float(row['流通市值']) > 0 else 0)
    fbt_min = fbt_minutes(row['首次封板时间'])

    return {
        'fbt_minutes': fbt_min,
        'fbt_bin': _fbt_bin_short(fbt_min),
        'zbc': int(row['炸板次数']),
        'board_type': board_type(float(row['涨跌幅'])),
        'circ_mv': round(circ_mv, 2),
        'turnover': round(float(row['换手率']), 2),
        'seal_ratio': round(seal_ratio, 3),
        'seal_fund_wan': round(float(row['封板资金']) / 1e4, 0),
        'zt_days': days,
        'zt_count': ct,
        'sector': str(row['所属行业']),
        # 市场情绪特征 (如果提供)
        'prev_rate': prev_rate if prev_rate is not None else 0.2,
        'prev_rate_2': prev_rate if prev_rate is not None else 0.2,
        'prev_rate_3d': prev_rate_3d if prev_rate_3d is not None else 0.2,
        'prev_n': prev_n if prev_n is not None else 50,
    }


def _fbt_bin_short(m):
    """封板时段分箱 (简短版)"""
    if m <= 9 * 60 + 25:
        return '集合竞价'
    if m <= 9 * 60 + 30:
        return '开盘'
    if m <= 10 * 60 + 30:
        return '早盘'
    if m <= 11 * 60 + 30:
        return '午前'
    if m <= 14 * 60:
        return '午后'
    return '尾盘'


class XGBScorer:
    """XGBoost 连板概率评分器

    加载 train_limit_up_model.py 训练的模型，输出连板概率。
    接口与 LBScorer 兼容 (score 方法返回 (概率, 明细))。
    """

    def __init__(self, model_path=None, sentiment_shift=0.0):
        self.model = None
        self.feature_names = None
        self.sentiment_shift = sentiment_shift
        self.numeric_features = ['fbt_minutes', 'zbc', 'circ_mv', 'turnover', 'seal_ratio',
                                 'zt_days', 'zt_count', 'seal_fund_wan',
                                 'prev_rate', 'prev_rate_2', 'prev_rate_3d', 'prev_n']
        self.categorical_features = ['board_type', 'fbt_bin', 'sector']

        path = model_path or XGB_MODEL_PATH
        if os.path.exists(path):
            try:
                import joblib
                bundle = joblib.load(path)
                self.model = bundle.get('model')
                self.feature_names = bundle.get('feature_names', None)
                if self.numeric_features is None:
                    self.numeric_features = bundle.get('numeric_features', self.numeric_features)
                if self.categorical_features is None:
                    self.categorical_features = bundle.get('categorical_features', self.categorical_features)
            except Exception as e:
                print(f"  XGB 模型加载失败: {e}")
        else:
            print(f"  XGB 模型不存在: {path} (将回退 LBScorer)")

    def is_available(self):
        return self.model is not None

    def _build_x(self, features):
        """将特征 dict 转换为模型输入向量"""
        # 数值特征
        x = {}
        for col in self.numeric_features:
            x[col] = features.get(col, 0.0)

        # 衍生特征 (与训练脚本一致)
        x['seal_ratio_x_turnover'] = features.get('seal_ratio', 0) * features.get('turnover', 0)
        x['log_circ_mv'] = np.log1p(features.get('circ_mv', 0))
        x['is_early'] = 1 if features.get('fbt_minutes', 999) <= 10 * 60 + 30 else 0
        x['is_zt_2board'] = 1 if features.get('zt_count', 0) >= 2 else 0

        # 类别特征 One-Hot
        for col in self.categorical_features:
            val = features.get(col, '')
            x[f'{col}_{val}'] = 1

        # 按特征名对齐，返回 DataFrame (保留 feature names 避免 warning)
        if self.feature_names:
            vec = [x.get(f, 0) for f in self.feature_names]
            return pd.DataFrame([vec], columns=self.feature_names)
        else:
            return np.array(list(x.values()), dtype=np.float32).reshape(1, -1)

    def score(self, features):
        """返回 (概率, 明细)"""
        if self.model is None:
            return 0.5, []

        x = self._build_x(features)
        prob = float(self.model.predict_proba(x)[0, 1])

        # 情绪平移
        if self.sentiment_shift != 0:
            prob = sigmoid(logit(prob) + self.sentiment_shift)

        # 明细: 返回关键特征值供展示
        detail = [
            ('封板强度', features.get('seal_ratio', 0), features.get('seal_ratio', 0) * 0.1),
            ('封板时段', features.get('fbt_bin', ''), 0),
            ('板块类型', features.get('board_type', ''), 0),
        ]
        return prob, detail


# ============================================================
# 统计分析
# ============================================================
def market_sentiment(zt, zbgc, prev):
    """市场情绪指标"""
    sentiment = {}
    if not prev.empty:
        first = prev[prev['昨日连板数'] == 1]
        sentiment['昨日涨停总数'] = len(prev)
        sentiment['昨日首板数'] = len(first)
        if len(first):
            hit = (first['最新价'] >= first['涨停价'] - 0.011).sum()
            sentiment['昨日首板连板数'] = int(hit)
            sentiment['昨日首板连板率'] = float(hit) / len(first)
        # 昨日连板股(高位)继续封板率
        high = prev[prev['昨日连板数'] >= 2]
        if len(high):
            hit_h = (high['最新价'] >= high['涨停价'] - 0.011).sum()
            sentiment['昨日高位连板延续率'] = float(hit_h) / len(high)
    if not zt.empty and not zbgc.empty:
        sentiment['今日炸板率'] = len(zbgc) / (len(zt) + len(zbgc))
    return sentiment


def analyze_structure(zt, zbgc):
    """当日涨停结构统计"""
    st = {}
    if zt.empty:
        return st
    st['涨停总数'] = len(zt)
    st['首板数'] = int((zt['连板数'] == 1).sum())
    st['连板数'] = int((zt['连板数'] >= 2).sum())
    st['最高连板'] = int(zt['连板数'].max())
    st['炸板数'] = len(zbgc)
    st['炸板率'] = len(zbgc) / (len(zt) + len(zbgc)) if (len(zt) + len(zbgc)) else 0
    st['行业分布'] = zt['所属行业'].value_counts().head(8).to_dict()
    st['封板时段分布'] = zt['首次封板时间'].apply(fbt_bin).value_counts().to_dict()
    st['市值分布'] = zt['流通市值'].apply(lambda x: circ_mv_bin(float(x) / 1e8)).value_counts().to_dict()
    st['连板分布'] = zt['连板数'].value_counts().sort_index().to_dict()
    return st


# ============================================================
# 推荐
# ============================================================
def recommend(zt, scorer, top_n):
    """对今日首板票打分排名"""
    if zt.empty:
        return pd.DataFrame()
    first = zt[zt['连板数'] == 1].copy()
    rows = []
    for _, r in first.iterrows():
        # 根据评分器类型选择特征提取方式
        if isinstance(scorer, XGBScorer):
            feats = extract_features_xgb(r)
        else:
            feats = extract_features(r)
        prob, detail = scorer.score(feats)
        # 分段概率校准 (回测50只样本: 30-40%区间高估2倍, 20-30%略高估)
        if 0.30 <= prob < 0.40:
            prob = prob * 0.5
        elif 0.20 <= prob < 0.30:
            prob = prob * 0.85
        # 理由: 取贡献最大的 3 个正向特征
        pos = sorted([d for d in detail if d[2] > 0.02], key=lambda x: -x[2])[:3]
        neg = sorted([d for d in detail if d[2] < -0.02], key=lambda x: x[2])[:1]
        reason = ' '.join(f'{f}:{b}({a:+.2f})' for f, b, a in pos)
        if neg:
            reason += f' | 风险 {neg[0][0]}:{neg[0][1]}({neg[0][2]:+.2f})'
        days, ct = _parse_zt_stat(r['涨停统计'])
        rows.append({
            '代码': r['代码'], '名称': r['名称'], '行业': r['所属行业'],
            '涨跌幅': round(float(r['涨跌幅']), 2), '最新价': round(float(r['最新价']), 2),
            '换手率': round(float(r['换手率']), 2),
            '流通市值(亿)': round(float(r['流通市值']) / 1e8, 1),
            '封板时间': _fbt_display(r['首次封板时间']),
            '炸板次数': int(r['炸板次数']),
            '封板资金(万)': round(float(r['封板资金']) / 1e4, 0),
            '近期涨停': f'{days}天{ct}板',
            '连板概率': round(prob * 100, 1),
            '理由': reason,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values('连板概率', ascending=False).reset_index(drop=True)
    df.insert(0, '排名', range(1, len(df) + 1))
    # 评分明细单独存一份, 便于排查
    return df.head(top_n)


# ============================================================
# 验证昨日推荐
# ============================================================
def validate_yesterday(date_str):
    """用今日实际表现验证昨日推荐命中率"""
    prev_date = _prev_trading_day(date_str)
    rec_file = os.path.join(DATA_DIR, f'{prev_date}_recommend.csv')
    if not os.path.exists(rec_file):
        return
    try:
        rec = pd.read_csv(rec_file, dtype={'代码': str})
    except Exception:
        return
    if rec.empty:
        return
    zt = load_zt(date_str)
    if zt.empty:
        return
    lb_map = dict(zip(zt['代码'].astype(str).str.zfill(6), zt['连板数']))
    hits = []
    for _, r in rec.iterrows():
        code = str(r['代码']).zfill(6)
        lbc = lb_map.get(code)
        hits.append(1 if (lbc is not None and int(lbc) >= 2) else 0)
    rec['连板成功'] = hits
    n = len(hits)
    top5 = sum(hits[:5])
    top10 = sum(hits[:10])
    print(f'\n{'='*60}')
    print(f'  昨日推荐验证 ({prev_date} 推荐 → {date_str} 实际)')
    print(f'{'='*60}')
    print(f'  推荐 {n} 只, 今日连板 {sum(hits)} 只, 命中率 {sum(hits)/n*100:.1f}%')
    print(f'  TOP5 命中率 {top5}/5 = {top5/5*100:.0f}% | TOP10 命中率 {top10}/10 = {top10/10*100:.0f}%')
    for _, r in rec.iterrows():
        mark = '✓' if r['连板成功'] else '✗'
        print(f'    {mark} {r["名称"]}({str(r["代码"]).zfill(6)}) 概率{r["连板概率"]:.1f}%')


# ============================================================
# 报告输出
# ============================================================
def _bar(n, total, width=20):
    if total <= 0:
        return ''
    filled = int(round(n / total * width))
    return '█' * filled + '·' * (width - filled)


def build_report(date_str, st, sentiment, rec, scorer, prior, global_rate):
    """生成文本报告"""
    L = []
    L.append('=' * 64)
    L.append(f'  {date_str} 首板涨停统计分析与次日连板推荐')
    L.append('=' * 64)

    L.append('')
    L.append('【一、当日涨停结构】')
    L.append(f'  涨停总数: {st.get("涨停总数", 0)} 只 | 首板: {st.get("首板数", 0)} 只 | '
             f'连板(≥2板): {st.get("连板数", 0)} 只 | 最高连板: {st.get("最高连板", 0)} 板')
    L.append(f'  炸板: {st.get("炸板数", 0)} 只 | 炸板率: {st.get("炸板率", 0)*100:.1f}%')
    if st.get('行业分布'):
        L.append('')
        L.append('  行业分布 TOP8:')
        total = st['涨停总数']
        for k, v in list(st['行业分布'].items())[:8]:
            L.append(f'    {k:<10s} {v:3d} ({v/total*100:4.1f}%) {_bar(v, total)}')
    if st.get('封板时段分布'):
        L.append('')
        L.append('  封板时段分布:')
        total = st['涨停总数']
        order = ['集合竞价(≤9:25)', '开盘(9:25-9:30)', '早盘(9:30-10:30)',
                 '午前(10:30-11:30)', '午后(13:00-14:00)', '尾盘(≥14:00)']
        for k in order:
            v = st['封板时段分布'].get(k, 0)
            L.append(f'    {k:<16s} {v:3d} ({v/total*100:4.1f}%) {_bar(v, total)}')
    if st.get('市值分布'):
        L.append('')
        L.append('  流通市值分布:')
        total = st['涨停总数']
        for k in ['<30亿', '30-60亿', '60-100亿', '100-200亿', '≥200亿']:
            v = st['市值分布'].get(k, 0)
            L.append(f'    {k:<10s} {v:3d} ({v/total*100:4.1f}%) {_bar(v, total)}')

    L.append('')
    L.append('【二、昨日涨停表现 (市场情绪)】')
    if sentiment:
        L.append(f'  昨日涨停: {sentiment.get("昨日涨停总数", 0)} 只 | '
                 f'昨日首板: {sentiment.get("昨日首板数", 0)} 只')
        if '昨日首板连板率' in sentiment:
            L.append(f'  今日连板: {sentiment.get("昨日首板连板数", 0)} 只 | '
                     f'首板连板率: {sentiment["昨日首板连板率"]*100:.1f}%')
        if '昨日高位连板延续率' in sentiment:
            L.append(f'  昨日高位连板延续率: {sentiment["昨日高位连板延续率"]*100:.1f}%')
        if '今日炸板率' in sentiment:
            L.append(f'  今日炸板率: {sentiment["今日炸板率"]*100:.1f}%')
        rate = sentiment.get('昨日首板连板率')
        if rate is not None:
            if rate >= 0.45:
                tag = '🔥 情绪偏暖 (打板溢价高)'
            elif rate >= 0.30:
                tag = '🌤 情绪中性'
            else:
                tag = '❄️ 情绪偏冷 (打板谨慎)'
            L.append(f'  情绪评级: {tag} (历史首板连板率均值 {global_rate*100:.1f}%)')
    else:
        L.append('  无昨日涨停池数据')

    L.append('')
    L.append(f'【三、次日连板推荐 TOP{len(rec) if not rec.empty else 0}】')
    L.append(f'  模型先验: 全局连板率 {global_rate*100:.1f}% → 调整后 {prior*100:.1f}%')
    if rec.empty:
        L.append('  今日无首板票')
    else:
        L.append('  排名 | 名称(代码)      | 行业     | 封板   | 炸板 | 流通市值 | 换手   | 概率  | 理由')
        L.append('  ' + '-' * 110)
        for _, r in rec.iterrows():
            name = f'{r["名称"]}({r["代码"]})'
            mv = f'{r["流通市值(亿)"]}亿'
            hs = f'{r["换手率"]}%'
            L.append(f'  {r["排名"]:<4d} | {name:<14s} | {str(r["行业"]):<8s} | {str(r["封板时间"]):<6s} | '
                     f'{r["炸板次数"]:<4d} | {mv:<8s} | {hs:<6s} | {r["连板概率"]:<5.1f} | {r["理由"]}')
        avg = rec['连板概率'].mean()
        L.append(f'  TOP{len(rec)} 平均连板概率: {avg:.1f}%')

    L.append('')
    L.append('【四、模型说明】')
    L.append(f'  评分 = 先验连板率 + 各特征分箱的历史连板率 log-odds 偏离 × {TEMPER} (贝叶斯平滑 α={ALPHA:g})')
    L.append('  特征: 封板时段/炸板次数/板块类型/流通市值/换手率/封板强度/近期涨停/行业')
    L.append('  先验调整: 全局率与昨日首板连板率加权, 按炸板率修正')
    L.append('  注意: 模型基于历史统计, 不构成投资建议; 涨停池数据仅保留近30日, 需每日积累')
    L.append('')
    return '\n'.join(L)


def save_report(date_str, text):
    path = os.path.join(DATA_DIR, f'{date_str}_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'  报告已保存: {path}')


# ============================================================
# 本地数据读取
# ============================================================
def load_zt(date_str):
    path = os.path.join(DATA_DIR, f'{date_str}_zt.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={'代码': str})
    if not df.empty:
        df['代码'] = df['代码'].astype(str).str.zfill(6)
    return df


def load_zbgc(date_str):
    path = os.path.join(DATA_DIR, f'{date_str}_zbgc.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={'代码': str})


def load_prev(date_str):
    path = os.path.join(DATA_DIR, f'{date_str}_prev.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={'代码': str})


def load_label_history():
    path = os.path.join(DATA_DIR, 'label_history.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={'code': str})


# ============================================================
# 回填历史
# ============================================================
def rebuild_labels():
    """从本地全部 prev 快照重建 label_history.csv (回填后调用)"""
    path = os.path.join(DATA_DIR, 'label_history.csv')
    if os.path.exists(path):
        os.remove(path)
    prev_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('_prev.csv'))
    for f in prev_files:
        date_str = f.replace('_prev.csv', '')
        prev = load_prev(date_str)
        if not prev.empty:
            build_labels(date_str, prev)
    print(f'  标签重建完成: 处理 {len(prev_files)} 个昨日涨停快照')


def backfill():
    """回填最近30个自然日内全部可得交易日的数据"""
    print('=' * 60)
    print('  回填涨停板历史数据 (近30日)')
    print('=' * 60)
    today = datetime.now()
    fetched, skipped = 0, 0
    for i in range(30):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime('%Y-%m-%d')
        if not load_zt(date_str).empty:
            skipped += 1
            continue
        zt, zbgc, prev = fetch_day(date_str)
        if zt.empty and zbgc.empty and prev.empty:
            skipped += 1
        else:
            fetched += 1
        time.sleep(0.5)
    # 回填按新→旧遍历, 标签依赖前一日快照, 需统一重建
    rebuild_labels()
    print(f'\n回填完成: 新增 {fetched} 日, 跳过 {skipped} 日')


# ============================================================
# 主流程
# ============================================================
def default_date():
    """默认日期: 今日(周一至周五), 否则上周五"""
    today = datetime.now()
    if today.weekday() < 5:
        return today.strftime('%Y-%m-%d')
    d = today - timedelta(days=today.weekday() - 4)
    return d.strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser(description='每日首板涨停统计分析与次日连板推荐')
    parser.add_argument('--date', help='指定交易日 YYYY-MM-DD, 默认最近交易日')
    parser.add_argument('--backfill', action='store_true', help='回填全部可得历史(近30日)')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过数据抓取, 只用本地数据')
    parser.add_argument('--force', action='store_true', help='盘中强制运行')
    parser.add_argument('--top', type=int, default=TOP_N, help=f'推荐数量 (默认 {TOP_N})')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.backfill:
        backfill()
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    date_str = args.date or default_date()

    # 盘中保护: 涨停池数据收盘后才定型
    if date_str == today_str and not args.force and datetime.now().hour < 15:
        print(f'⚠️ 当前为盘中 ({datetime.now():%H:%M}), 涨停池数据尚未定型, 建议 15:00 后运行。')
        print(f'   如确需运行请加 --force')
        return

    print(f'Kronos 首板涨停分析 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # 抓取 (若指定日期无数据则自动回退到最近有数据的交易日)
    if not args.skip_fetch:
        zt, zbgc, prev = fetch_day(date_str)
        if zt.empty and not args.date:
            for i in range(1, 6):
                d = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
                zt, zbgc, prev = fetch_day(d)
                if not zt.empty:
                    date_str = d
                    break
    else:
        zt, zbgc, prev = load_zt(date_str), load_zbgc(date_str), load_prev(date_str)

    if zt.empty:
        print(f'无 {date_str} 涨停数据, 退出')
        return

    # 分析
    st = analyze_structure(zt, zbgc)
    sentiment = market_sentiment(zt, zbgc, prev)
    # 只用推荐日之前的标签样本, 避免前视偏差 (标签需要次日才能确认)
    label_df = load_label_history()
    label_df = label_df[label_df['date'] < date_str]
    zbgc_rate = sentiment.get('今日炸板率')

    # 优先使用 XGBoost 模型 (效果优于朴素贝叶斯), 不可用时回退 LBScorer
    xgb_scorer = XGBScorer()
    if xgb_scorer.is_available():
        prev_rate = sentiment.get('昨日首板连板率')
        xgb_scorer.sentiment_shift = 0.0  # 模型已含市场情绪特征，无需额外平移
        scorer = xgb_scorer
        global_rate = float(label_df['label'].mean()) if not label_df.empty else DEFAULT_RATE
        prior = global_rate
        print(f'  使用 XGBoost 模型 (基准连板率 {global_rate:.1%})')
    else:
        scorer, global_rate, prior = train_scorer(
            label_df, sentiment.get('昨日首板连板率'), zbgc_rate, DEFAULT_RATE)
        print(f'  使用 LBScorer (基准连板率 {global_rate:.1%})')

    # 推荐
    rec = recommend(zt, scorer, args.top)
    if not rec.empty:
        rec.to_csv(os.path.join(DATA_DIR, f'{date_str}_recommend.csv'), index=False)
        print(f'  推荐已保存: {os.path.join(DATA_DIR, f"{date_str}_recommend.csv")}')

    # 报告
    report = build_report(date_str, st, sentiment, rec, scorer, prior, global_rate)
    print(report)
    save_report(date_str, report)

    # 验证昨日推荐
    validate_yesterday(date_str)

    print(f'\n完成 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')


if __name__ == '__main__':
    main()