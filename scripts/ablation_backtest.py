#!/usr/bin/env python3
"""
消融实验回测脚本

目的：找出 daily_predict.py 中哪层修正真正有效
方法：模型只跑一次得到原始输出，然后测试 16 种修正组合（4 层修正的 2^4）

修正层:
  L1: signal_correction  (量价+缠论+准确率缩放)
  L2: apply_kouju_correction (短线口诀)
  L3: _apply_trend_bias  (市场趋势偏置)
  L4: sec_bias           (板块偏置)

用法:
  python scripts/ablation_backtest.py                    # 默认回测最近15天
  python scripts/ablation_backtest.py --days 30          # 回测最近30天
  python scripts/ablation_backtest.py --output result.csv
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.indicators import (
    calc_volume_price_features, calc_volume_price_signals,
    calc_chanlun_simple, calc_chanlun_signals
)
from scripts.adaptive_params import (
    load_validation_history, calc_stock_metrics, adaptive_params,
    calc_market_volatility, signal_correction, detect_market_regime
)
from scripts.short_term_rules import (
    detect_trend_state, calc_kouju_rules, apply_kouju_correction,
    get_sideways_conf_factor
)

# ============================================================
# 配置（与 daily_predict.py 保持一致）
# ============================================================
STOCKS = {
    '000001': '平安银行', '000858': '五粮液', '300368': '汇金股份',
    '300430': '诚益通', '300451': '创业惠康', '300678': '中科信息',
    '300782': '卓胜微', '301293': '三博脑科', '600036': '招商银行',
    '600664': '哈药股份',
    '601899': '紫金矿业', '603993': '洛阳钼业', '600011': '华能国际',
    '600900': '长江电力', '601985': '中国核电', '600362': '江西铜业',
    '300624': '万兴科技', '300418': '昆仑万维', '300058': '蓝色光标',
    '600909': '华安证券', '601162': '天风证券', '601211': '国泰海通',
    '002172': '澳洋健康', '300017': '网宿科技',
    '300274': '阳光电源',
    '002714': '牧原股份', '300498': '温氏股份', '000876': '新希望',
    '002157': '正邦科技', '000735': '罗牛山',
    '002261': '拓维信息',
    '600895': '张江高科',
}
STOCK_TAGS = {
    '000001': '银行', '000858': '白酒', '300368': '软件',
    '300430': '软件', '300451': '软件', '300678': '软件',
    '300782': '半导体', '301293': '医药', '600036': '银行',
    '600664': '医药',
    '601899': '铜金', '603993': '铜钴', '600011': '火电',
    '600900': '水电', '601985': '核电', '600362': '铜冶炼',
    '300624': 'AI软件', '300418': 'AI软件', '300058': 'AI营销',
    '600909': '券商', '601162': '券商', '601211': '券商',
    '002172': '医药', '300017': 'CDN',
    '300274': '光伏',
    '002714': '生猪', '300498': '生猪', '000876': '生猪',
    '002157': '生猪', '000735': '生猪',
    '002261': 'AI软件',
    '600895': '园区开发',
}
DATA_DIR = './finetune_csv/data'
LOG_DIR = './prediction_log'
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']

# 预测参数（接近 daily_predict.py 的保守模式）
PRED_PARAMS = {
    'T': 0.6,
    'top_p': 0.9,
    'sample_count': 10,
    'lookback': 120,
}

# 修正层名称
LAYER_NAMES = ['signal', 'kouju', 'trend', 'sector']


def load_df(code, skip_indicators=False):
    """加载股票数据（与 daily_predict.py 一致）"""
    path = os.path.join(DATA_DIR, f'{code}_fundflow.csv')
    df = pd.read_csv(path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0)
    m = df['amount'] == 0
    if m.any():
        df.loc[m, 'amount'] = df.loc[m, 'volume'] * df.loc[m, ['open', 'high', 'low', 'close']].mean(axis=1)
    for fc in ['main_net_inflow', 'huge_net_inflow', 'main_net_inflow_ratio']:
        if fc not in df.columns:
            df[fc] = 0.0
        else:
            df[fc] = df[fc].fillna(0)
    if not skip_indicators:
        vp_feats = calc_volume_price_features(df)
        chanlun_feats = calc_chanlun_simple(df)
        for col in vp_feats.columns:
            df[col] = vp_feats[col]
        for col in chanlun_feats.columns:
            df[col] = chanlun_feats[col]
    return df


def calc_sector_momentum(all_dfs, stock_tags):
    """计算板块动量（与 daily_predict.py 一致）"""
    from collections import defaultdict
    sector_funds = defaultdict(list)
    for code, df in all_dfs.items():
        tag = stock_tags.get(code)
        if tag and 'main_net_inflow_ratio' in df.columns and len(df) > 0:
            sector_funds[tag].append(df['main_net_inflow_ratio'].iloc[-1])
    return {tag: np.mean(funds) for tag, funds in sector_funds.items() if funds}


def apply_trend_bias(chg, regime, short_trend=0.0, short_up_ratio=0.5):
    """趋势偏置（与 daily_predict.py 一致）"""
    if short_trend < -1.5 and short_up_ratio < 0.40:
        if chg > 0:
            chg = chg * 0.4 - 0.3
        else:
            chg = chg * 1.3
        return max(-10, min(10, chg))
    if regime['regime'] == 'uptrend':
        if chg > 0:
            chg = chg * (1 + 0.3 * regime['strength'])
        else:
            chg = chg * (1 - 0.3 * regime['strength']) + 0.3 * regime['strength']
    elif regime['regime'] == 'downtrend':
        if chg < 0:
            chg = chg * (1 + 0.3 * regime['strength'])
        else:
            chg = chg * (1 - 0.3 * regime['strength']) - 0.3 * regime['strength']
    return max(-10, min(10, chg))


def run_model_prediction(predictor, df, last_date):
    """运行模型预测下一天"""
    lb = PRED_PARAMS['lookback']
    x_df = df.iloc[-lb:][FEATURES].reset_index(drop=True)
    x_ts = df.iloc[-lb:]['timestamps'].reset_index(drop=True)

    # 预测目标日
    fd = []
    c = last_date + timedelta(days=1)
    while len(fd) < 1:
        if c.weekday() < 5:
            fd.append(c)
        c += timedelta(days=1)

    pred = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fd),
        pred_len=1, T=PRED_PARAMS['T'], top_p=PRED_PARAMS['top_p'],
        sample_count=PRED_PARAMS['sample_count'], verbose=False
    )
    return pred['close'].iloc[0]


def main():
    parser = argparse.ArgumentParser(description='消融实验回测')
    parser.add_argument('--days', type=int, default=15, help='回测天数')
    parser.add_argument('--output', type=str, default='ablation_results.csv', help='输出文件')
    parser.add_argument('--skip-model', action='store_true', help='跳过模型加载（调试用）')
    args = parser.parse_args()

    print("=" * 70)
    print("  消融实验回测：找出哪层修正真正有效")
    print("=" * 70)

    # 加载模型
    if not args.skip_model:
        print("\n加载模型...")
        tokenizer = KronosTokenizer.from_pretrained('./pretrained/Kronos-Tokenizer-base')
        model = Kronos.from_pretrained('./pretrained/Kronos-base')
        predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5.0)
        print("模型加载完成")
    else:
        predictor = None

    # 获取交易日历
    df0 = load_df('000001', skip_indicators=True)
    all_dates = df0['timestamps'].dt.strftime('%Y-%m-%d').tolist()
    # 取最近 N+1 天，最后一天用于验证
    test_dates = all_dates[-(args.days + 1):-1]
    print(f"\n回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} 天)")

    # 加载验证历史（用于自适应参数）
    val_history = load_validation_history(LOG_DIR, n_days=10)

    # 逐日回测
    all_results = []
    total_start = time.time()

    for day_idx, date_str in enumerate(test_dates):
        day_start = time.time()
        print(f"\n[{day_idx+1}/{len(test_dates)}] {date_str}")

        # 加载所有股票数据
        all_dfs = {}
        for code in STOCKS:
            try:
                all_dfs[code] = load_df(code)
            except Exception as e:
                print(f"  加载 {code} 失败: {e}")

        # 计算市场状态
        market_vol = calc_market_volatility(all_dfs)
        market_regime = detect_market_regime(all_dfs)

        # 短期趋势
        short_trend = 0.0
        short_up_ratio = 0.5
        try:
            rets5 = [(df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
                     for df in all_dfs.values() if len(df) >= 6]
            if rets5:
                short_trend = float(np.mean(rets5))
                short_up_ratio = sum(1 for r in rets5 if r > 0) / len(rets5)
        except:
            pass

        sector_momentum = calc_sector_momentum(all_dfs, STOCK_TAGS)
        top_sector = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)[:3]
        bot_sector = sorted(sector_momentum.items(), key=lambda x: x[1])[:3]
        sector_bias = {}
        for tag, _ in top_sector:
            sector_bias[tag] = 0.3
        for tag, _ in bot_sector:
            sector_bias[tag] = -0.3

        # 逐股预测
        for code, name in STOCKS.items():
            if code not in all_dfs:
                continue
            df = all_dfs[code]

            # 找到当前日期的行
            mask = df['timestamps'].dt.strftime('%Y-%m-%d') == date_str
            if not mask.any():
                continue
            idx = mask.idxmax()
            if idx + 1 >= len(df):
                continue

            # 历史数据
            hist = df.iloc[:idx + 1].copy()
            last_close = hist['close'].iloc[-1]
            last_date = hist['timestamps'].iloc[-1]

            # 实际下一天涨跌
            next_row = df.iloc[idx + 1]
            actual_chg = (next_row['close'] / last_close - 1) * 100

            # 模型预测
            if predictor is not None:
                try:
                    pred_close = run_model_prediction(predictor, hist, last_date)
                    raw_chg = (pred_close / last_close - 1) * 100
                except Exception as e:
                    print(f"  {name} 预测失败: {e}")
                    continue
            else:
                raw_chg = 0.0  # 调试模式

            raw_chg = max(-10, min(10, raw_chg))

            # 计算各层修正
            # L1: signal_correction
            stock_metrics = calc_stock_metrics(val_history, code)
            vp_signals = calc_volume_price_signals(hist)
            chanlun_signals = calc_chanlun_signals(hist)
            l1_chg, l1_detail = signal_correction(raw_chg, vp_signals, chanlun_signals, stock_metrics)

            # L2: kouju
            trend_state = detect_trend_state(hist)
            kouju_rules = calc_kouju_rules(hist, raw_chg, trend_state)
            l2_chg, l2_detail = apply_kouju_correction(l1_chg, kouju_rules)

            # L3: trend_bias
            l3_chg = apply_trend_bias(l2_chg, market_regime, short_trend, short_up_ratio)

            # L4: sector_bias
            tag = STOCK_TAGS.get(code, '')
            sec_bias = sector_bias.get(tag, 0.0)
            l4_chg = l3_chg + sec_bias
            l4_chg = max(-10, min(10, l4_chg))

            all_results.append({
                'date': date_str,
                'code': code,
                'name': name,
                'raw_chg': raw_chg,
                'l1_signal': l1_chg - raw_chg,      # 修正量
                'l2_kouju': l2_chg - l1_chg,
                'l3_trend': l3_chg - l2_chg,
                'l4_sector': l4_chg - l3_chg,
                'actual_chg': actual_chg,
                'market_regime': market_regime['regime'],
                'sector_bias': sec_bias,
            })

        day_elapsed = time.time() - day_start
        print(f"  完成，耗时 {day_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n回测完成，总耗时 {total_elapsed:.1f}s")

    # 保存原始结果
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(args.output, index=False)
    print(f"原始结果已保存: {args.output}")

    # ============================================================
    # 消融分析：测试 16 种组合
    # ============================================================
    print("\n" + "=" * 70)
    print("  消融分析：16 种修正组合对比")
    print("=" * 70)

    # 生成所有组合 (2^4 = 16)
    combinations = list(product([0, 1], repeat=4))

    results = []
    for combo in combinations:
        # combo = (use_signal, use_kouju, use_trend, use_sector)
        apply_signal, apply_kouju, apply_trend, apply_sector = combo

        # 计算每个样本的最终预测
        pred_chgs = []
        for _, row in df_raw.iterrows():
            chg = row['raw_chg']
            if apply_signal:
                chg += row['l1_signal']
            if apply_kouju:
                chg += row['l2_kouju']
            if apply_trend:
                chg += row['l3_trend']
            if apply_sector:
                chg += row['l4_sector']
            pred_chgs.append(chg)

        df_temp = df_raw.copy()
        df_temp['pred_chg'] = pred_chgs

        # 计算准确率
        df_temp['pred_dir'] = df_temp['pred_chg'].apply(lambda x: 1 if x > 0 else -1)
        df_temp['actual_dir'] = df_temp['actual_chg'].apply(lambda x: 1 if x > 0 else -1)
        df_temp['correct'] = df_temp['pred_dir'] == df_temp['actual_dir']

        accuracy = df_temp['correct'].mean()
        avg_error = (df_temp['pred_chg'] - df_temp['actual_chg']).abs().mean()

        # 计算相关系数
        corr = df_temp['pred_chg'].corr(df_temp['actual_chg'])

        combo_name = '+'.join([LAYER_NAMES[i] for i, v in enumerate(combo) if v]) or 'raw'
        results.append({
            'combo': combo_name,
            'signal': apply_signal,
            'kouju': apply_kouju,
            'trend': apply_trend,
            'sector': apply_sector,
            'accuracy': accuracy,
            'avg_error': avg_error,
            'correlation': corr,
            'n_samples': len(df_temp),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    # 打印结果
    print(f"\n{'排名':<4} {'组合':<30} {'准确率':>8} {'平均误差':>10} {'相关系数':>10}")
    print("-" * 70)
    for i, (_, row) in enumerate(df_results.iterrows(), 1):
        marker = " ← 最佳" if i == 1 else ""
        print(f"{i:<4} {row['combo']:<30} {row['accuracy']:>7.2%} {row['avg_error']:>9.2f}% {row['correlation']:>10.4f}{marker}")

    # 保存结果
    result_file = args.output.replace('.csv', '_ablation.csv')
    df_results.to_csv(result_file, index=False)
    print(f"\n消融结果已保存: {result_file}")

    # ============================================================
    # 逐层贡献分析
    # ============================================================
    print("\n" + "=" * 70)
    print("  逐层贡献分析（单独添加每层修正的效果）")
    print("=" * 70)

    baseline = df_results[df_results['combo'] == 'raw'].iloc[0]
    print(f"\n基准（纯模型输出）: 准确率 {baseline['accuracy']:.2%}, 平均误差 {baseline['avg_error']:.2f}%")

    print(f"\n{'修正层':<15} {'准确率':>8} {'变化':>8} {'平均误差':>10} {'误差变化':>10}")
    print("-" * 60)

    for layer_idx, layer_name in enumerate(LAYER_NAMES):
        # 找到只加这一层的组合
        combo = [0, 0, 0, 0]
        combo[layer_idx] = 1
        combo_str = '+'.join([LAYER_NAMES[i] for i, v in enumerate(combo) if v])

        row = df_results[df_results['combo'] == combo_str]
        if not row.empty:
            row = row.iloc[0]
            acc_diff = row['accuracy'] - baseline['accuracy']
            err_diff = row['avg_error'] - baseline['avg_error']
            print(f"{layer_name:<15} {row['accuracy']:>7.2%} {acc_diff:>+7.2%} {row['avg_error']:>9.2f}% {err_diff:>+9.2f}%")

    # 最佳组合
    best = df_results.iloc[0]
    print(f"\n最佳组合: {best['combo']}")
    print(f"  准确率: {best['accuracy']:.2%} (基准 {baseline['accuracy']:.2%}, 提升 {best['accuracy']-baseline['accuracy']:+.2%})")
    print(f"  平均误差: {best['avg_error']:.2f}% (基准 {baseline['avg_error']:.2f}%, 变化 {best['avg_error']-baseline['avg_error']:+.2f}%)")
    print(f"  相关系数: {best['correlation']:.4f} (基准 {baseline['correlation']:.4f})")

    # ============================================================
    # 按市场状态分析
    # ============================================================
    print("\n" + "=" * 70)
    print("  按市场状态分析（最佳组合 vs 基准）")
    print("=" * 70)

    best_combo = best
    df_temp = df_raw.copy()

    # 应用最佳组合
    pred_chgs = []
    for _, row in df_temp.iterrows():
        chg = row['raw_chg']
        if best_combo['signal']:
            chg += row['l1_signal']
        if best_combo['kouju']:
            chg += row['l2_kouju']
        if best_combo['trend']:
            chg += row['l3_trend']
        if best_combo['sector']:
            chg += row['l4_sector']
        pred_chgs.append(chg)

    df_temp['pred_chg_best'] = pred_chgs
    df_temp['pred_dir_best'] = df_temp['pred_chg_best'].apply(lambda x: 1 if x > 0 else -1)
    df_temp['pred_dir_raw'] = df_temp['raw_chg'].apply(lambda x: 1 if x > 0 else -1)
    df_temp['actual_dir'] = df_temp['actual_chg'].apply(lambda x: 1 if x > 0 else -1)
    df_temp['correct_best'] = df_temp['pred_dir_best'] == df_temp['actual_dir']
    df_temp['correct_raw'] = df_temp['pred_dir_raw'] == df_temp['actual_dir']

    for regime in ['uptrend', 'downtrend', 'sideways']:
        sub = df_temp[df_temp['market_regime'] == regime]
        if len(sub) == 0:
            continue
        acc_raw = sub['correct_raw'].mean()
        acc_best = sub['correct_best'].mean()
        print(f"  {regime:<12}: 样本{len(sub):>4} | 基准 {acc_raw:.2%} → 最佳 {acc_best:.2%} ({acc_best-acc_raw:+.2%})")

    print("\n完成！")


if __name__ == '__main__':
    main()
