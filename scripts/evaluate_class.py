#!/usr/bin/env python3
"""
分类准确率评估

将回归预测转换为三分类：涨(>0.5%) / 平(-0.5%~0.5%) / 跌(<-0.5%)
评估分类准确率，对比回归方向准确率。

核心假设：模型不需要精确预测幅度，只需判断方向+幅度是否超过噪声区间。
"平"的类别让模型不再强迫对噪声做无意义的精确预测。

用法:
  python scripts/evaluate_class.py --days 42
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import LOG_DIR


THRESHOLD = 0.5  # 涨/跌阈值(%)


def classify(val, threshold=THRESHOLD):
    """将连续值分类为 涨/平/跌"""
    if val > threshold:
        return '涨'
    elif val < -threshold:
        return '跌'
    return '平'


def evaluate_class(days=42, verbose=True):
    """评估分类准确率"""
    val_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_validate.csv')])[-days:]

    three_class_correct = 0
    three_class_total = 0
    two_class_correct = 0
    two_class_total = 0
    reg_correct = 0
    reg_total = 0

    # 按类别统计
    class_stats = {'涨': {'correct': 0, 'total': 0}, '平': {'correct': 0, 'total': 0}, '跌': {'correct': 0, 'total': 0}}

    for vf in val_files:
        df = pd.read_csv(os.path.join(LOG_DIR, vf), dtype={'code': str})
        df['code'] = df['code'].astype(str).str.zfill(6)

        for _, row in df.iterrows():
            pred = row['pred_chg']
            actual = row['actual_chg']

            pred_class = classify(pred)
            actual_class = classify(actual)

            # 三分类准确率
            three_class_total += 1
            if pred_class == actual_class:
                three_class_correct += 1

            # 二分类准确率（涨 vs 非涨）
            if pred_class != '平' and actual_class != '平':
                two_class_total += 1
                if pred_class == actual_class:
                    two_class_correct += 1

            # 回归方向准确率（对比基线）
            if pred * actual > 0:
                reg_correct += 1
            reg_total += 1

            # 按预测类别统计
            class_stats[pred_class]['total'] += 1
            if pred_class == actual_class:
                class_stats[pred_class]['correct'] += 1

    three_acc = three_class_correct / three_class_total * 100 if three_class_total else 0
    two_acc = two_class_correct / two_class_total * 100 if two_class_total else 0
    reg_acc = reg_correct / reg_total * 100 if reg_total else 0

    if verbose:
        print(f"{'='*60}")
        print(f"分类准确率评估（{len(val_files)}天）")
        print(f"{'='*60}")
        print(f"  回归方向准确率（基线）: {reg_correct}/{reg_total} = {reg_acc:.1f}%")
        print(f"  三分类准确率（涨/平/跌）: {three_class_correct}/{three_class_total} = {three_acc:.1f}%")
        print(f"  二分类准确率（涨vs跌）: {two_class_correct}/{two_class_total} = {two_acc:.1f}%")
        print()
        print(f"  按预测类别:")
        for cls in ['涨', '平', '跌']:
            s = class_stats[cls]
            if s['total'] > 0:
                pct = s['correct'] / s['total'] * 100
                print(f"    预测{cls}: {s['correct']}/{s['total']} = {pct:.1f}%")
        print()
        if three_acc > reg_acc:
            print(f"  🟢 分类准确率 > 回归方向准确率 (+{three_acc-reg_acc:.1f}%)")
        else:
            print(f"  🟡 分类准确率 <= 回归方向准确率 ({three_acc-reg_acc:+.1f}%)")
        print(f"{'='*60}")

    return {'reg_acc': reg_acc, 'three_acc': three_acc, 'two_acc': two_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分类准确率评估')
    parser.add_argument('--days', type=int, default=42, help='评估天数')
    args = parser.parse_args()
    evaluate_class(args.days)
