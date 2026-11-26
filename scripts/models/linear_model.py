#!/usr/bin/env python3
"""
线性回归模型 — 集成学习的基模型之一

特征：动量、资金流、RSI、换手率等技术指标
目标：次日涨跌幅（连续值）
用途：与Kronos、随机森林组成集成模型

训练方式：滚动窗口在线学习（每天用最近N天数据重新训练）
"""

import numpy as np
import pandas as pd
from scripts.daily_predict import STOCKS, load_df


def extract_features(df):
    """从K线数据中提取特征向量

    返回: dict，包含以下特征
    """
    if len(df) < 20:
        return None

    close = df['close'].values
    volume = df['volume'].values

    # 动量特征
    mom_1d = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
    mom_3d = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0
    mom_5d = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
    mom_10d = (close[-1] / close[-11] - 1) * 100 if len(close) >= 11 else 0
    mom_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    # 均线偏离度
    ma5 = np.mean(close[-5:])
    ma20 = np.mean(close[-20:])
    dev_ma5 = (close[-1] / ma5 - 1) * 100
    dev_ma20 = (close[-1] / ma20 - 1) * 100

    # RSI(14)
    deltas = np.diff(close[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    if avg_loss < 1e-8:
        rsi = 70.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)

    # 成交量特征
    vol_ratio = volume[-1] / np.mean(volume[-5:]) if np.mean(volume[-5:]) > 0 else 1
    vol_trend = np.mean(volume[-3:]) / np.mean(volume[-10:]) if np.mean(volume[-10:]) > 0 else 1

    # 资金流特征
    if 'main_net_inflow_ratio' in df.columns:
        fund_1d = df['main_net_inflow_ratio'].iloc[-1]
        fund_3d = df['main_net_inflow_ratio'].iloc[-3:].mean() if len(df) >= 3 else 0
    else:
        fund_1d = 0
        fund_3d = 0

    # 振幅
    high = df['high'].values[-5:]
    low = df['low'].values[-5:]
    amplitude = np.mean((high - low) / low) * 100

    # 换手率
    turnover = df['turnover'].iloc[-1] if 'turnover' in df.columns else 0

    return {
        'mom_1d': mom_1d,
        'mom_3d': mom_3d,
        'mom_5d': mom_5d,
        'mom_10d': mom_10d,
        'mom_20d': mom_20d,
        'dev_ma5': dev_ma5,
        'dev_ma20': dev_ma20,
        'rsi': rsi,
        'vol_ratio': vol_ratio,
        'vol_trend': vol_trend,
        'fund_1d': fund_1d,
        'fund_3d': fund_3d,
        'amplitude': amplitude,
        'turnover': turnover,
    }


def prepare_training_data(n_days=60):
    """准备训练数据

    用最近n天的数据作为训练集，
    每只股票每个交易日作为一个样本。

    返回: X (features), y (target), feature_names
    """
    feature_list = []
    targets = []
    feature_names = None

    for code in STOCKS:
        try:
            df = load_df(code, skip_indicators=True)
            if len(df) < n_days + 20:
                continue

            # 滑动窗口创建样本
            for i in range(20, len(df) - 1):
                window = df.iloc[:i]
                feats = extract_features(window)
                if feats is None:
                    continue

                # 目标：次日涨跌幅
                next_close = df.iloc[i]['close']
                cur_close = df.iloc[i - 1]['close']
                target = (next_close / cur_close - 1) * 100

                if feature_names is None:
                    feature_names = list(feats.keys())

                feature_list.append([feats[k] for k in feature_names])
                targets.append(target)

        except Exception:
            continue

    X = np.array(feature_list)
    y = np.array(targets)

    return X, y, feature_names


class SimpleLinearModel:
    """简单线性回归（最小二乘法，无依赖）"""

    def __init__(self):
        self.weights = None
        self.bias = 0
        self.feature_names = None

    def fit(self, X, y):
        """训练：w = (X^T X)^-1 X^T y"""
        # 添加偏置列
        X_b = np.c_[np.ones(X.shape[0]), X]

        # 正则化最小二乘
        lambda_reg = 0.01
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0  # 不正则化偏置

        try:
            theta = np.linalg.inv(X_b.T @ X_b + lambda_reg * I) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
        except:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0

    def predict(self, features):
        """预测单只股票"""
        if self.weights is None:
            return 0
        x = np.array([features.get(k, 0) for k in self.feature_names])
        return float(self.weights @ x + self.bias)


def train_and_predict():
    """训练并预测"""
    print("训练线性回归模型...")
    X, y, feature_names = prepare_training_data(n_days=100)

    if len(X) < 50:
        print("  训练数据不足")
        return None

    # 时间序列分割：最近10天作为测试集
    split = len(X) - 30  # 留30个样本做测试
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SimpleLinearModel()
    model.feature_names = feature_names
    model.fit(X_train, y_train)

    # 测试集准确率
    pred = np.array([model.predict(dict(zip(feature_names, x))) for x in X_test])
    pred_dir = pred > 0
    actual_dir = y_test > 0
    acc = (pred_dir == actual_dir).mean() * 100

    print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"  线性回归方向准确率: {acc:.1f}%")

    # 输出特征重要性
    importance = list(zip(feature_names, model.weights))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top5重要特征:")
    for name, w in importance[:5]:
        print(f"    {name}: {w:+.4f}")

    return model


if __name__ == '__main__':
    model = train_and_predict()
