#!/usr/bin/env python3
"""
资金流方向验证

验证"预测方向+资金流方向"的联合准确率：
- 预测涨 + 次日主力流入 → 实际涨的概率 = ?
- 预测涨 + 次日主力流出 → 实际涨的概率 = ?
- 预测跌 + 次日主力流出 → 实际跌的概率 = ?
- 预测跌 + 次日主力流入 → 实际跌的概率 = ?

如果"预测涨+主力流入"组实际涨的概率显著高于"预测涨+主力流出"组，
说明资金流方向可以作为预测置信度的有效信号。

用法:
  python scripts/validate_fund.py --days 42
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.daily_predict import STOCKS, load_df, LOG_DIR


def validate_fund(days=42, verbose=True):
    """验证预测方向+资金流方向的联合准确率"""
    val_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('_validate.csv')])[-days:]

    groups = {
        'pred_up_fund_in': [],    # 预测涨 + 主力流入
        'pred_up_fund_out': [],   # 预测涨 + 主力流出
        'pred_down_fund_out': [], # 预测跌 + 主力流出
        'pred_down_fund_in': [],  # 预测跌 + 主力流入
    }

    for vf in val_files:
        date_str = os.path.basename(vf).replace('_validate.csv', '')
        try:
            val_df = pd.read_csv(os.path.join(LOG_DIR, vf), dtype={'code': str})
            val_df['code'] = val_df['code'].astype(str).str.zfill(6)
        except:
            continue

        for _, row in val_df.iterrows():
            code = row['code']
            if code not in STOCKS:
                continue
            # 只取每只股票的第一行（保守模式）
            if '激进' in str(row['mode']):
                continue

            pred_dir = 'up' if row['pred_chg'] > 0 else 'down'
            actual_up = row['actual_chg'] > 0

            # 获取该股票该日期的主力资金流方向
            try:
                df = load_df(code, skip_indicators=True)
                target = df[df['timestamps'].dt.strftime('%Y-%m-%d') == date_str]
                if target.empty:
                    continue
                idx = target.index[0]
                if idx == 0:
                    continue
                fund_ratio = df.loc[idx, 'main_net_inflow_ratio'] if 'main_net_inflow_ratio' in df.columns else 0
                fund_dir = 'in' if fund_ratio > 0 else 'out'
            except:
                continue

            key = f'pred_{pred_dir}_fund_{fund_dir}'
            if key in groups:
                groups[key].append(actual_up)

    if verbose:
        print(f"{'='*70}")
        print(f"预测方向 × 资金流方向 联合验证（{len(val_files)}天）")
        print(f"{'='*70}")
        print(f"{'分组':25s} {'次数':>6s} {'实际涨占比':>10s} {'解读'}")
        print(f"{'-'*70}")

    labels = {
        'pred_up_fund_in': '预测涨+主力流入',
        'pred_up_fund_out': '预测涨+主力流出',
        'pred_down_fund_out': '预测跌+主力流出',
        'pred_down_fund_in': '预测跌+主力流入',
    }

    results = {}
    for key, label in labels.items():
        data = groups[key]
        if data:
            pct = np.mean(data) * 100
            results[key] = pct
            n = len(data)
            print(f"{label:25s} {n:6d} {pct:9.1f}%")
        else:
            print(f"{label:25s} {'无数据':>15s}")

    print(f"{'-'*70}")

    # 对比分析
    if 'pred_up_fund_in' in results and 'pred_up_fund_out' in results:
        diff_up = results['pred_up_fund_in'] - results['pred_up_fund_out']
        print(f"预测涨时: 流入组涨{results['pred_up_fund_in']:.1f}% vs 流出组涨{results['pred_up_fund_out']:.1f}% (差{diff_up:+.1f}%)")
        if diff_up > 10:
            print(f"  🟢 资金流方向对预测涨有显著增强作用（+{diff_up:.1f}%）")
        elif diff_up > 5:
            print(f"  🟡 资金流方向有一定增强作用（+{diff_up:.1f}%）")
        else:
            print(f"  🔴 资金流方向对预测涨无显著增强（{diff_up:+.1f}%）")

    if 'pred_down_fund_out' in results and 'pred_down_fund_in' in results:
        # 注意：pred_down组中 actual_up=False 表示预测正确
        diff_down = (100 - results['pred_down_fund_out']) - (100 - results['pred_down_fund_in'])
        print(f"预测跌时: 流出组跌{100-results['pred_down_fund_out']:.1f}% vs 流入组跌{100-results['pred_down_fund_in']:.1f}% (差{diff_down:+.1f}%)")

    # 综合：同向组 vs 反向组
    same_dir = groups['pred_up_fund_in'] + groups['pred_down_fund_out']  # 预测与资金流同向
    opposite_dir = groups['pred_up_fund_out'] + groups['pred_down_fund_in']  # 预测与资金流反向

    if same_dir and opposite_dir:
        # 对于同向组：up+in时actual_up应为True，down+out时actual_up应为False
        same_correct = (np.mean(groups['pred_up_fund_in']) * len(groups['pred_up_fund_in']) +
                       (1 - np.mean(groups['pred_down_fund_out'])) * len(groups['pred_down_fund_out'])) / len(same_dir)

        opp_correct = (np.mean(groups['pred_up_fund_out']) * len(groups['pred_up_fund_out']) +
                      (1 - np.mean(groups['pred_down_fund_in'])) * len(groups['pred_down_fund_in'])) / len(opposite_dir)

        print(f"\n综合对比:")
        print(f"  预测与资金流同向: 准确率 {same_correct*100:.1f}% ({len(same_dir)}次)")
        print(f"  预测与资金流反向: 准确率 {opp_correct*100:.1f}% ({len(opposite_dir)}次)")
        if same_correct > opp_correct:
            print(f"  🟢 资金流方向可提升预测准确率 +{(same_correct-opp_correct)*100:.1f}%")

    print(f"{'='*70}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='资金流方向验证')
    parser.add_argument('--days', type=int, default=42, help='验证天数')
    args = parser.parse_args()
    validate_fund(args.days)
