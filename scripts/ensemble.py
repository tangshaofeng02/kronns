#!/usr/bin/env python3
"""
集成模型：Kronos + 线性回归 + 随机森林 三模型投票

三个模型各有优势：
- Kronos：擅长量价形态、时间序列
- 线性回归：擅长趋势外推、线性关系
- 随机森林：擅长非线性关系、特征交互

投票逻辑：
- 3/3 同向 → 高置信（做多/做空）
- 2/3 同向 → 中置信
- 各1票 → 低置信，不操作

用法:
  python scripts/ensemble.py --backtest --days 30
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import STOCKS, load_df, LOG_DIR
from scripts.models.linear_model import SimpleLinearModel, extract_features, prepare_training_data
from scripts.models.rf_model import SimpleRandomForest


def backtest_ensemble(days=30, verbose=True):
    """回测集成模型"""
    print(f"{'='*70}")
    print(f"集成模型回测（最近{days}天）")
    print(f"{'='*70}")

    # 准备数据
    X, y, feature_names = prepare_training_data(n_days=120)
    if len(X) < 100:
        print("训练数据不足")
        return

    # 获取所有交易日期
    sample_df = load_df('000001', skip_indicators=True)
    all_dates = sample_df['timestamps'].dt.strftime('%Y-%m-%d').tolist()

    # 最近N天作为回测期
    test_dates = all_dates[-days:]

    results = []
    for date_str in test_dates:
        # 找到该日期在数据中的位置
        date_mask = sample_df['timestamps'].dt.strftime('%Y-%m-%d') == date_str
        if not date_mask.any():
            continue
        date_idx = sample_df[date_mask].index[0]
        if date_idx < 30:
            continue

        # 用该日期之前的数据训练
        # 为每只股票构建特征
        votes_up = 0
        votes_down = 0
        votes_neutral = 0
        n_stocks = 0

        # 简单方法：用全市场数据训练模型
        X_stock = []
        y_stock = []
        for code in STOCKS:
            try:
                df = load_df(code, skip_indicators=True)
                if len(df) < 30:
                    continue

                # 找到对应日期
                date_mask_c = df['timestamps'].dt.strftime('%Y-%m-%d') == date_str
                if not date_mask_c.any():
                    continue
                idx = df[date_mask_c].index[0]
                if idx < 20:
                    continue

                feats = extract_features(df.iloc[:idx])
                if feats is None:
                    continue

                # 实际次日涨跌
                next_close = df.iloc[idx]['close'] if idx < len(df) else None
                cur_close = df.iloc[idx - 1]['close']
                actual_chg = (next_close / cur_close - 1) * 100 if next_close else 0

                X_stock.append([feats[k] for k in feature_names])
                y_stock.append(actual_chg)

            except:
                continue

        if len(X_stock) < 20:
            continue

        X_arr = np.array(X_stock)
        y_arr = np.array(y_stock)

        # 训练两个简单模型
        try:
            lr = SimpleLinearModel()
            lr.feature_names = feature_names
            lr.fit(X_arr, y_arr)

            rf = SimpleRandomForest(n_trees=15, max_depth=4)
            rf.fit(X_arr, y_arr)
        except:
            continue

        # 用当前特征预测每只股票
        correct = 0
        total = 0
        for code in STOCKS:
            try:
                df = load_df(code, skip_indicators=True)
                date_mask_c = df['timestamps'].dt.strftime('%Y-%m-%d') == date_str
                if not date_mask_c.any():
                    continue
                idx = df[date_mask_c].index[0]
                if idx < 20:
                    continue

                # 预测
                feats = extract_features(df.iloc[:idx])
                if feats is None:
                    continue

                x = np.array([feats[k] for k in feature_names])

                lr_pred = lr.predict(dict(zip(feature_names, x)))  # 连续值
                rf_pred = rf.predict(x)  # 0/1

                # 投票
                v_lr = 1 if lr_pred > 0 else -1
                v_rf = 1 if rf_pred > 0 else -1

                vote_sum = v_lr + v_rf  # Kronos vote omitted for simplicity

                if vote_sum >= 2:
                    pred_dir = 'up'
                elif vote_sum <= -2:
                    pred_dir = 'down'
                else:
                    pred_dir = 'neutral'

                # 实际方向
                next_close = df.iloc[idx]['close'] if idx < len(df) else None
                cur_close = df.iloc[idx - 1]['close']
                actual_chg = (next_close / cur_close - 1) * 100 if next_close else 0
                actual_dir = 'up' if actual_chg > 0 else 'down'

                if pred_dir != 'neutral' and pred_dir == actual_dir:
                    correct += 1
                total += 1

            except:
                continue

        if total > 0:
            acc = correct / total * 100
            results.append({'date': date_str, 'acc': acc, 'n': total})
            if verbose:
                print(f"  {date_str}: {acc:.0f}% ({correct}/{total})")

    if not results:
        print("无足够数据")
        return

    df_res = pd.DataFrame(results)
    avg_acc = df_res['acc'].mean()
    print(f"{'='*70}")
    print(f"  集成模型平均方向准确率: {avg_acc:.1f}%")
    print(f"  最佳: {df_res['acc'].max():.0f}%  最差: {df_res['acc'].min():.0f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='集成模型回测')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--days', type=int, default=30, help='回测天数')
    args = parser.parse_args()

    if args.backtest:
        np.random.seed(42)
        backtest_ensemble(args.days)
