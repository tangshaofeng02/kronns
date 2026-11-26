#!/usr/bin/env python3
"""
随机森林模型 — 集成学习的基模型之一

特征：动量、资金流、RSI、换手率、缠论信号
目标：次日涨跌方向（分类：0=跌, 1=涨）
用途：与Kronos、线性回归组成集成模型

注意：使用纯Python实现（无sklearn依赖），基于决策树投票
"""

import numpy as np
from scripts.daily_predict import STOCKS, load_df
from scripts.models.linear_model import extract_features, prepare_training_data


class SimpleDecisionTree:
    """简单决策树（基于特征阈值分割）"""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None

    def fit(self, X, y, depth=0):
        """递归构建决策树"""
        if depth >= self.max_depth or len(y) < 5:
            self.prediction = 1 if np.mean(y) > 0 else 0
            return

        best_gain = -1
        n_features = X.shape[1]

        # 随机选择特征子集
        n_try = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_try, replace=False)

        for idx in feature_indices:
            values = X[:, idx]
            # 尝试几个阈值
            for pct in [25, 50, 75]:
                threshold = np.percentile(values, pct)
                left_mask = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue

                # 信息增益（基尼不纯度）
                gain = _gini_gain(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = idx
                    self.threshold = threshold

        if best_gain <= 0 or self.feature_idx is None:
            self.prediction = 1 if np.mean(y) > 0 else 0
            return

        # 递归构建子树
        left_mask = X[:, self.feature_idx] <= self.threshold
        self.left = SimpleDecisionTree(max_depth=self.max_depth - 1)
        self.right = SimpleDecisionTree(max_depth=self.max_depth - 1)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

    def predict(self, x):
        """预测单一样本"""
        if self.prediction is not None:
            return self.prediction
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


def _gini_gain(y, left_mask, right_mask):
    """计算基尼不纯度增益"""
    def gini(mask):
        if mask.sum() == 0:
            return 0
        p = y[mask].mean()
        return 1 - p**2 - (1-p)**2

    n = len(y)
    n_left = left_mask.sum()
    n_right = right_mask.sum()

    if n_left == 0 or n_right == 0:
        return 0

    return gini(np.ones(n, dtype=bool)) - (n_left/n * gini(left_mask) + n_right/n * gini(right_mask))


class SimpleRandomForest:
    """简单随机森林（多棵决策树投票）"""

    def __init__(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """训练：每棵树用随机样本子集"""
        self.trees = []
        y_binary = (y > 0).astype(int)

        for _ in range(self.n_trees):
            # 自助采样（bootstrap）
            indices = np.random.choice(len(y), len(y), replace=True)
            X_boot = X[indices]
            y_boot = y_binary[indices]

            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, x):
        """预测：投票"""
        votes = [tree.predict(x) for tree in self.trees]
        return 1 if np.mean(votes) > 0.5 else 0

    def predict_proba(self, x):
        """预测概率"""
        votes = [tree.predict(x) for tree in self.trees]
        return np.mean(votes)


def train_and_predict():
    """训练并预测"""
    print("训练随机森林模型...")
    X, y, feature_names = prepare_training_data(n_days=100)

    if len(X) < 50:
        print("  训练数据不足")
        return None

    # 时间序列分割
    split = len(X) - 30
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SimpleRandomForest(n_trees=15, max_depth=4)
    model.fit(X_train, y_train)

    # 测试集准确率
    pred = np.array([model.predict(x) for x in X_test])
    actual_dir = (y_test > 0).astype(int)
    acc = (pred == actual_dir).mean() * 100

    print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"  随机森林方向准确率: {acc:.1f}%")

    return model


if __name__ == '__main__':
    np.random.seed(42)
    model = train_and_predict()
