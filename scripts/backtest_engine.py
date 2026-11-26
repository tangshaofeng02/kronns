#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨停板策略回测引擎

功能:
  1. Walk-forward 回测 (滚动训练窗口, 避免前视偏差)
  2. 推荐组合收益计算 (TOP-N 等权)
  3. 风险指标: 最大回撤、夏普比率、年化收益
  4. 基准对比: 随机推荐、全市场首板、总是下跌

用法:
  python scripts/backtest_engine.py                          # 默认回测
  python scripts/backtest_engine.py --window 10 --top 5      # 10日滚动窗口, TOP5
  python scripts/backtest_engine.py --start 2026-10-22       # 起始日期
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'limit_up_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
LABEL_FILE = os.path.join(DATA_DIR, 'label_history.csv')


def load_label_data():
    """加载标签数据"""
    if not os.path.exists(LABEL_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LABEL_FILE, dtype={'code': str})
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['_date'] = pd.to_datetime(df['date'])
    return df.sort_values('_date').reset_index(drop=True)


def load_daily_pools():
    """加载每日涨停池快照"""
    pools = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*_zt.csv'))):
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            pools[date] = df
        except Exception:
            continue
    return pools


def walk_forward_backtest(df, train_window=10, test_window=1, top_n=5,
                          model_path=None, verbose=True):
    """Walk-forward 回测

    对每个测试日:
      1. 用前 train_window 日的数据训练模型
      2. 对测试日的首板股打分
      3. 选 TOP-N 推荐
      4. 用实际标签验证命中率

    返回: 每日回测结果 DataFrame + 汇总指标
    """
    dates = sorted(df['_date'].unique())
    if len(dates) < train_window + test_window:
        print(f"数据不足: 需要 {train_window + test_window} 天, 只有 {len(dates)} 天")
        return pd.DataFrame(), {}

    # 加载模型 (如果有)
    model = None
    feature_names = None
    if model_path and os.path.exists(model_path):
        try:
            import joblib
            bundle = joblib.load(model_path)
            model = bundle.get('model')
            feature_names = bundle.get('feature_names')
        except Exception:
            pass

    results = []
    for i in range(train_window, len(dates) - test_window + 1):
        train_dates = dates[i - train_window:i]
        test_dates = dates[i:i + test_window]

        train_df = df[df['_date'].isin(train_dates)]
        test_df = df[df['_date'].isin(test_dates)]

        if train_df.empty or test_df.empty:
            continue

        # 训练模型
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            X_train = _build_features(train_df)
            y_train = train_df['label'].values
            feature_columns = X_train.columns.tolist()

            model_i = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                min_samples_leaf=20, l2_regularization=0.1,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=42,
                class_weight='balanced',
            )
            model_i.fit(X_train, y_train)
        except Exception as e:
            if verbose:
                print(f"  训练失败 {test_dates[0]}: {e}")
            continue

        # 对测试日打分 (用训练集的特征列对齐)
        for d in test_dates:
            day_df = test_df[test_df['_date'] == d]
            if day_df.empty:
                continue

            X_test = _build_features(day_df, feature_columns=feature_columns)
            probs = model_i.predict_proba(X_test)[:, 1]

            day_df = day_df.copy()
            day_df['prob'] = probs
            day_df = day_df.sort_values('prob', ascending=False)

            # TOP-N 推荐
            top = day_df.head(top_n)
            hit_rate = top['label'].mean() if len(top) else 0
            # 全市场基准
            base_rate = day_df['label'].mean()
            # 随机基准 (期望)
            rand_rate = base_rate

            results.append({
                'date': str(d)[:10],
                'n_total': len(day_df),
                'n_top': len(top),
                'top_hit_rate': hit_rate,
                'base_rate': base_rate,
                'excess': hit_rate - base_rate,
                'top_avg_prob': top['prob'].mean() if len(top) else 0,
                'actual_continuation': int(top['label'].sum()),
            })

            if verbose:
                print(f"  {str(d)[:10]}: TOP{top_n} 命中率 {hit_rate:.1%} vs 基准 {base_rate:.1%} ({hit_rate-base_rate:+.1%})")

    df_res = pd.DataFrame(results)

    # 汇总指标
    metrics = {}
    if not df_res.empty:
        metrics['total_days'] = len(df_res)
        metrics['avg_top_hit_rate'] = df_res['top_hit_rate'].mean()
        metrics['avg_base_rate'] = df_res['base_rate'].mean()
        metrics['avg_excess'] = df_res['excess'].mean()
        metrics['win_rate'] = (df_res['excess'] > 0).mean()
        metrics['avg_actual_continuation'] = df_res['actual_continuation'].mean()

        # 最大回撤 (基于累计超额)
        cum = (1 + df_res['excess']).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        metrics['total_return'] = cum.iloc[-1] - 1

        # 夏普比率 (日度)
        if df_res['excess'].std() > 0:
            metrics['sharpe'] = df_res['excess'].mean() / df_res['excess'].std() * np.sqrt(252)
        else:
            metrics['sharpe'] = 0

    return df_res, metrics


def _build_features(df, feature_columns=None):
    """构建特征矩阵 (与训练脚本一致)

    feature_columns: 如果提供，则按此顺序对齐列 (缺失填 0，多余丢弃)
    """
    feat = pd.DataFrame()
    feat['fbt_minutes'] = df['fbt'].apply(lambda x: _fbt_to_min(str(x)))
    feat['zbc'] = df['zbc'].astype(float)
    feat['circ_mv'] = df['circ_mv'].astype(float)
    feat['turnover'] = df['turnover'].astype(float)
    feat['seal_ratio'] = df['seal_ratio'].astype(float)
    feat['zt_days'] = df['zt_days'].astype(float)
    feat['zt_count'] = df['zt_count'].astype(float)

    # 衍生
    feat['seal_ratio_x_turnover'] = feat['seal_ratio'] * feat['turnover']
    feat['log_circ_mv'] = np.log1p(feat['circ_mv'])
    feat['is_early'] = (feat['fbt_minutes'] <= 630).astype(int)
    feat['is_zt_2board'] = (feat['zt_count'] >= 2).astype(int)

    # 市场情绪
    for col in ['prev_rate', 'prev_rate_2', 'prev_rate_3d', 'prev_n']:
        if col in df.columns:
            feat[col] = df[col].astype(float)
        else:
            feat[col] = 0.2

    # 类别 One-Hot
    for col in ['board_type', 'fbt_bin', 'sector']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            feat = pd.concat([feat, dummies], axis=1)

    feat = feat.fillna(0)

    # 按特征列对齐
    if feature_columns is not None:
        feat = feat.reindex(columns=feature_columns, fill_value=0)

    return feat


def _fbt_to_min(fbt):
    s = str(fbt).zfill(6)
    if not s.isdigit():
        return 900
    return int(s[:2]) * 60 + int(s[2:4]) + (1 if int(s[4:6]) > 0 else 0)


def main():
    parser = argparse.ArgumentParser(description='涨停板策略回测引擎')
    parser.add_argument('--window', type=int, default=10, help='训练窗口 (交易日)')
    parser.add_argument('--test-window', type=int, default=1, help='测试窗口 (交易日)')
    parser.add_argument('--top', type=int, default=5, help='推荐数量')
    parser.add_argument('--model', type=str, default=None, help='预训练模型路径')
    args = parser.parse_args()

    print("=" * 70)
    print("  涨停板策略回测引擎 (Walk-Forward)")
    print("=" * 70)

    # 加载数据
    print("\n[1/3] 加载数据...")
    df = load_label_data()
    if df.empty:
        print("无标签数据，退出")
        return

    # 添加市场情绪特征
    from scripts.train_limit_up_model import add_market_sentiment
    df = add_market_sentiment(df)
    print(f"  样本数: {len(df)} (正样本 {df['label'].sum()})")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")

    # 回测
    print(f"\n[2/3] Walk-Forward 回测 (窗口={args.window}, TOP{args.top})...")
    df_res, metrics = walk_forward_backtest(
        df, train_window=args.window, test_window=args.test_window,
        top_n=args.top, model_path=args.model, verbose=True)

    # 汇总
    print(f"\n[3/3] 回测汇总")
    print("=" * 70)
    if not metrics:
        print("无回测结果")
        return

    print(f"  回测天数:        {metrics['total_days']}")
    print(f"  TOP{args.top} 平均命中率:  {metrics['avg_top_hit_rate']:.2%}")
    print(f"  市场基准命中率:   {metrics['avg_base_rate']:.2%}")
    print(f"  平均超额收益:     {metrics['avg_excess']:+.2%}")
    print(f"  胜率:            {metrics['win_rate']:.2%}")
    print(f"  平均连板数:       {metrics['avg_actual_continuation']:.2f}")
    print()
    print(f"  累计收益:        {metrics['total_return']:+.2%}")
    print(f"  最大回撤:        {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率:        {metrics['sharpe']:.2f}")

    # 保存结果
    if not df_res.empty:
        out_file = os.path.join(DATA_DIR, f'backtest_walkforward_top{args.top}.csv')
        df_res.to_csv(out_file, index=False)
        print(f"\n  回测明细已保存: {out_file}")

    print("\n完成！")


if __name__ == '__main__':
    main()
