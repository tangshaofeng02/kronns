#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨停板连板预测模型 — XGBoost 训练脚本

替代原有朴素贝叶斯 LBScorer，用 XGBoost 直接预测"次日连板概率"。

数据来源:
  - limit_up_data/label_history.csv  (1039 条, 20 个交易日)
  - limit_up_data/high_board_history.csv (高连板观测)
  - limit_up_data/*_zt.csv (每日涨停池, 用于扩展数据)

特征工程:
  - 数值特征: fbt_minutes, zbc, circ_mv, turnover, seal_ratio, zt_days, zt_count
  - 类别特征: board_type, fbt_bin, sector (One-Hot)
  - 衍生特征: seal_fund_ma5, main_ratio, huge_ratio, sector_zt_cnt

用法:
  python scripts/train_limit_up_model.py                    # 默认训练
  python scripts/train_limit_up_model.py --days 60          # 扩展 60 天数据
  python scripts/train_limit_up_model.py --output model.pkl # 指定输出
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
HISTORY_FILE = os.path.join(DATA_DIR, 'history.csv')

# 特征列
NUMERIC_FEATURES = ['fbt_minutes', 'zbc', 'circ_mv', 'turnover', 'seal_ratio',
                    'zt_days', 'zt_count', 'seal_fund_wan', 'main_ratio',
                    'huge_ratio', 'ma5_main_ratio']
CATEGORICAL_FEATURES = ['board_type', 'fbt_bin', 'sector']


def _fbt_to_minutes(fbt):
    """首次封板时间 HHMMSS → 分钟数"""
    s = str(fbt).zfill(6)
    if not s.isdigit():
        return 15 * 60
    h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
    return h * 60 + m + (1 if sec > 0 else 0)


def _fbt_bin(m):
    """封板时段分箱"""
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


def _board_type(pct):
    """板块类型"""
    if pct >= 25:
        return '30cm'
    if pct >= 15:
        return '20cm'
    return '10cm'


def load_label_data():
    """加载标签数据"""
    if not os.path.exists(LABEL_FILE):
        print(f"标签文件不存在: {LABEL_FILE}")
        return pd.DataFrame()
    df = pd.read_csv(LABEL_FILE, dtype={'code': str})
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def load_history_data():
    """加载涨停池历史数据"""
    if not os.path.exists(HISTORY_FILE):
        print(f"历史文件不存在: {HISTORY_FILE}")
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE, dtype={'代码': str})
    df['代码'] = df['代码'].astype(str).str.zfill(6)
    return df


def extend_labels_from_zt(days=60):
    """从每日涨停池数据扩展标签

    对每个交易日的涨停池股票，查找其在下一交易日是否继续涨停（连板）。
    """
    # 加载所有日度涨停池文件
    zt_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_zt.csv')))
    if not zt_files:
        return pd.DataFrame()

    # 构建每日涨停池
    daily_pools = {}
    for f in zt_files:
        date = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f, dtype={'代码': str})
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            df = df[['代码', '名称', '涨跌幅', '流通市值', '换手率', '封板资金',
                     '首次封板时间', '炸板次数', '涨停统计', '连板数', '所属行业']].copy()
            daily_pools[date] = df
        except Exception:
            continue

    if not daily_pools:
        return pd.DataFrame()

    dates = sorted(daily_pools.keys())
    print(f"  涨停池数据: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")

    # 为每个股票生成标签
    rows = []
    for i, d in enumerate(dates[:-1]):
        nd = dates[i + 1]
        today_pool = daily_pools[d]
        next_pool = daily_pools[nd]
        next_lb = dict(zip(next_pool['代码'], next_pool['连板数']))

        for _, r in today_pool.iterrows():
            code = r['代码']
            # 只考虑首板票（连板数==1），预测次日是否连板
            if int(r['连板数']) != 1:
                continue
            # 标签：次日是否连板
            label = 1 if code in next_lb else 0

            # 特征
            try:
                days_stat, ct = _parse_zt_stat(r['涨停统计'])
            except Exception:
                days_stat, ct = 1, 1

            circ_mv = float(r['流通市值']) / 1e8 if float(r['流通市值']) > 0 else 0
            seal_fund = float(r['封板资金']) / 1e4 if pd.notna(r['封板资金']) else 0
            seal_ratio = (seal_fund * 1e4 / float(r['流通市值']) * 100
                          if float(r['流通市值']) > 0 else 0)

            rows.append({
                'date': d,
                'code': code,
                'name': r['名称'],
                'sector': str(r['所属行业']),
                'fbt': r['首次封板时间'],
                'fbt_minutes': _fbt_to_minutes(r['首次封板时间']),
                'fbt_bin': _fbt_bin(_fbt_to_minutes(r['首次封板时间'])),
                'zbc': int(r['炸板次数']),
                'board_type': _board_type(float(r['涨跌幅'])),
                'circ_mv': round(circ_mv, 2),
                'turnover': round(float(r['换手率']), 2),
                'seal_ratio': round(seal_ratio, 3),
                'seal_fund_wan': round(seal_fund, 0),
                'zt_days': days_stat,
                'zt_count': ct,
                'label': label,
            })

    df = pd.DataFrame(rows)
    print(f"  扩展标签: {len(df)} 条 (正样本 {df['label'].sum()}, 负样本 {(df['label']==0).sum()})")
    return df


def _parse_zt_stat(s):
    """解析涨停统计 'X/Y' → (X, Y)"""
    try:
        parts = str(s).split('/')
        return int(parts[0]), int(parts[1])
    except Exception:
        return 1, 1


def add_market_sentiment(df):
    """添加市场情绪特征（每日连板率、市场宽度）

    关键洞察：每日连板率从 6% 到 52% 波动，市场情绪是连板的核心驱动。
    用 T-1 日及前几日的连板率作为市场情绪代理。
    """
    df = df.copy()
    df['_date'] = pd.to_datetime(df['date'])
    df = df.sort_values('_date').reset_index(drop=True)

    # 计算每日连板率（包含未来信息仅用于历史日；当日 T 的标签在预测时未知，
    # 所以只用 T-1 及更早的数据做情绪特征）
    daily = df.groupby('_date')['label'].agg(['mean', 'count']).reset_index()
    daily.columns = ['_date', 'daily_rate', 'daily_n']
    daily = daily.sort_values('_date')

    # T-1 日连板率（前一日市场情绪）
    daily['prev_rate'] = daily['daily_rate'].shift(1)
    # T-2 日连板率
    daily['prev_rate_2'] = daily['daily_rate'].shift(2)
    # 3 日均值
    daily['prev_rate_3d'] = daily['daily_rate'].shift(1).rolling(3, min_periods=1).mean()
    # 市场宽度：当日涨停总数（分母）
    daily['prev_n'] = daily['daily_n'].shift(1)

    df = df.merge(daily[['_date', 'prev_rate', 'prev_rate_2', 'prev_rate_3d', 'prev_n']],
                  on='_date', how='left')

    # 填充首日缺失
    global_rate = df['label'].mean()
    for col in ['prev_rate', 'prev_rate_2', 'prev_rate_3d']:
        df[col] = df[col].fillna(global_rate)
    df['prev_n'] = df['prev_n'].fillna(df['prev_n'].median())

    return df


def build_features(df):
    """构建特征矩阵

    数值特征: 直接使用
    类别特征: One-Hot 编码
    """
    # 基础数值特征
    feat_df = df[NUMERIC_FEATURES].copy()

    # 衍生特征
    feat_df['seal_ratio_x_turnover'] = df['seal_ratio'] * df['turnover']
    feat_df['log_circ_mv'] = np.log1p(df['circ_mv'])
    feat_df['is_early'] = (df['fbt_minutes'] <= 10 * 60 + 30).astype(int)
    feat_df['is_zt_2board'] = (df['zt_count'] >= 2).astype(int)

    # 市场情绪特征
    for col in ['prev_rate', 'prev_rate_2', 'prev_rate_3d', 'prev_n']:
        if col in df.columns:
            feat_df[col] = df[col]

    # 类别特征 One-Hot
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            feat_df = pd.concat([feat_df, dummies], axis=1)

    # 填充缺失值
    feat_df = feat_df.fillna(0)

    return feat_df


def train_xgb_model(X_train, y_train, X_val, y_val, params=None, backend='auto'):
    """训练梯度提升模型

    backend: 'xgboost' | 'sklearn' | 'auto'
      auto 优先 xgboost，不可用则回退 sklearn HistGradientBoosting
    """
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 0.01)

    # 尝试 XGBoost
    if backend in ('auto', 'xgboost'):
        try:
            import xgboost as xgb
            if params is None:
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'n_estimators': 300,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': scale_pos_weight,
                    'min_child_weight': 5,
                    'gamma': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                }
            print(f"\n[后端] XGBoost")
            # XGBoost 2.0+: early_stopping_rounds 通过构造函数传递
            model = xgb.XGBClassifier(early_stopping_rounds=30, **params)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
            print(f"  最佳迭代: {model.best_iteration}")
            print(f"  最佳 AUC: {model.best_score:.4f}")
            return model
        except Exception as e:
            if backend == 'xgboost':
                raise
            print(f"  XGBoost 不可用 ({type(e).__name__})，回退 sklearn")

    # 回退: sklearn HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    print(f"\n[后端] sklearn HistGradientBoosting")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )
    # sklearn 不支持 scale_pos_weight，用 class_weight 近似
    model.set_params(class_weight='balanced')
    model.fit(X_train, y_train)
    print(f"  实际迭代: {model.n_iter_}")
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """评估模型效果"""
    from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                                 f1_score, accuracy_score, classification_report,
                                 confusion_matrix)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("  模型评估")
    print("=" * 60)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  AUC:        {auc:.4f}")
    print(f"  准确率:     {acc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1:         {f1:.4f}")

    print("\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不连板', '连板']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵:")
    print(f"    TN={cm[0][0]} FP={cm[0][1]}")
    print(f"    FN={cm[1][0]} TP={cm[1][1]}")

    # 概率校准检查
    print("\n  概率校准 (预测概率 vs 实际连板率):")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df_cal = pd.DataFrame({'prob': y_prob, 'actual': y_test})
    df_cal['bin'] = pd.cut(df_cal['prob'], bins=bins)
    cal = df_cal.groupby('bin', observed=True).agg(
        n=('actual', 'count'),
        pred_mean=('prob', 'mean'),
        actual_mean=('actual', 'mean'),
    )
    for idx, r in cal.iterrows():
        if r['n'] > 0:
            print(f"    {idx}: 样本{int(r['n']):>3} | 预测{r['pred_mean']:.2%} vs 实际{r['actual_mean']:.2%}")

    # 特征重要性
    print("\n  TOP15 特征重要性:")
    # 尝试多种重要性来源
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance()
    else:
        # sklearn HistGradientBoosting: 用 permutation importance 近似
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X_test, y_test, n_repeats=5,
                                       random_state=42, scoring='roc_auc')
            importances = r.importances_mean
        except Exception:
            importances = np.zeros(len(feature_names))

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    for i, (_, r) in enumerate(importance.head(15).iterrows(), 1):
        print(f"    {i:2d}. {r['feature']:<30} {r['importance']:.4f}")

    return {
        'auc': auc, 'accuracy': acc, 'precision': precision,
        'recall': recall, 'f1': f1,
        'y_prob': y_prob, 'y_pred': y_pred,
    }


def main():
    parser = argparse.ArgumentParser(description='涨停板连板预测 XGBoost 训练')
    parser.add_argument('--days', type=int, default=0,
                        help='从涨停池扩展 N 天数据 (0=只用现有 label_history)')
    parser.add_argument('--output', type=str, default='xgb_limit_up.pkl',
                        help='模型输出文件名')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例')
    args = parser.parse_args()

    print("=" * 70)
    print("  涨停板连板预测 — XGBoost 模型训练")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df_label = load_label_data()
    print(f"  label_history.csv: {len(df_label)} 条")

    # 扩展数据
    if args.days > 0:
        print(f"\n  从涨停池扩展 {args.days} 天数据...")
        df_ext = extend_labels_from_zt(days=args.days)
        if not df_ext.empty:
            # 合并数据（去重）
            df_label = pd.concat([df_label, df_ext], ignore_index=True)
            df_label = df_label.drop_duplicates(subset=['date', 'code'], keep='last')
            print(f"  合并后: {len(df_label)} 条")

    if df_label.empty:
        print("无训练数据，退出")
        return

    # 2. 特征工程（先按日期排序，保证时间序列一致性）
    print("\n[2/5] 特征工程...")
    df_label['_date'] = pd.to_datetime(df_label['date'])
    df_label = df_label.sort_values('_date').reset_index(drop=True)

    # 补充衍生字段
    if 'fbt_minutes' not in df_label.columns:
        df_label['fbt_minutes'] = df_label['fbt'].apply(_fbt_to_minutes)
    if 'fbt_bin' not in df_label.columns:
        df_label['fbt_bin'] = df_label['fbt_minutes'].apply(_fbt_bin)
    if 'seal_fund_wan' not in df_label.columns:
        df_label['seal_fund_wan'] = df_label.get('seal_ratio', 0) * df_label.get('circ_mv', 0)

    # 确保数值特征存在
    for col in NUMERIC_FEATURES:
        if col not in df_label.columns:
            df_label[col] = 0.0

    # 添加市场情绪特征（需按日期排序后计算）
    df_label = add_market_sentiment(df_label)

    X = build_features(df_label)
    y = df_label['label'].values
    feature_names = X.columns.tolist()
    print(f"  特征维度: {X.shape[1]}")
    print(f"  样本数: {len(y)} (正样本 {y.sum()}, 负样本 {len(y)-y.sum()})")
    print(f"  正样本率: {y.mean():.2%}")

    # 3. 时间序列切分
    print("\n[3/5] 时间序列切分...")

    n = len(y)
    test_size = int(n * args.test_ratio)
    val_size = int(n * args.val_ratio)
    train_size = n - test_size - val_size

    X_train, y_train = X.iloc[:train_size], y[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y[train_size+val_size:]

    print(f"  训练集: {len(y_train)} (正样本率 {y_train.mean():.2%})")
    print(f"  验证集: {len(y_val)} (正样本率 {y_val.mean():.2%})")
    print(f"  测试集: {len(y_test)} (正样本率 {y_test.mean():.2%})")

    # 4. 训练模型
    print("\n[4/5] 训练 XGBoost 模型...")
    model = train_xgb_model(X_train, y_train, X_val, y_val)

    # 5. 评估
    print("\n[5/5] 模型评估...")
    results = evaluate_model(model, X_test, y_test, feature_names)

    # 6. 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, args.output)

    import joblib
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'metrics': {k: v for k, v in results.items() if k not in ('y_prob', 'y_pred')},
    }, output_path)
    print(f"\n模型已保存: {output_path}")

    # 对比基准
    print("\n" + "=" * 60)
    print("  基准对比")
    print("=" * 60)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"  基准准确率（多数类）: {baseline_acc:.2%}")
    print(f"  模型准确率:          {results['accuracy']:.2%}")
    print(f"  提升:                {results['accuracy']-baseline_acc:+.2%}")
    print(f"  基准 AUC:            0.5000")
    print(f"  模型 AUC:            {results['auc']:.4f}")
    print(f"  AUC 提升:            {results['auc']-0.5:+.4f}")

    print("\n完成！")


if __name__ == '__main__':
    main()
