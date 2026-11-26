"""
自动更新 examples/data 下所有股票数据到最新交易日

从腾讯财经获取前复权日K线，追加新数据到已有CSV。
如果当天是交易日且已收盘，数据会包含当天；否则到最近一个交易日。

用法:
  python scripts/update_today_data.py
  python scripts/update_today_data.py --code 300782 600664  # 只更新指定股票
"""

import os
import sys
import time
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.fetch_stock_data import TencentStrategy, _fill


def update_stock(code: str, data_dir: str = "./examples/data") -> bool:
    """更新单只股票数据到最新交易日"""
    csv_path = os.path.join(data_dir, f"{code}_stock_data.csv")
    if not os.path.exists(csv_path):
        print(f"  {code}: 数据文件不存在，跳过")
        return False

    # 读取现有数据
    df_old = pd.read_csv(csv_path)
    df_old["timestamps"] = pd.to_datetime(df_old["timestamps"])
    last_date = df_old["timestamps"].iloc[-1].strftime("%Y%m%d")
    last_date_str = df_old["timestamps"].iloc[-1].strftime("%Y-%m-%d")

    # 从腾讯获取完整K线
    strategy = TencentStrategy()
    df_new = strategy.fetch_range(code, "20230101", "20991231")

    if df_new.empty:
        print(f"  {code}: 获取数据为空")
        return False

    df_new["timestamps"] = pd.to_datetime(df_new["timestamps"])
    new_last = df_new["timestamps"].iloc[-1].strftime("%Y-%m-%d")

    # 合并策略：K线以新数据为准，资金流保留旧数据中的已有值
    fund_cols = ["main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"]

    # 用 merge 代替 loc 赋值，避免 index 不对齐问题
    df_old["date_key"] = df_old["timestamps"].dt.strftime("%Y-%m-%d")
    df_new["date_key"] = df_new["timestamps"].dt.strftime("%Y-%m-%d")

    # 先删除旧数据中的K线列，用新数据替换
    kline_cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df_old.columns]
    df_old_kline_removed = df_old.drop(columns=kline_cols)

    # 合并：旧数据（保留资金流） + 新K线
    merged = df_old_kline_removed.merge(
        df_new[["date_key"] + kline_cols],
        on="date_key", how="outer",
    )

    # 新日期没有资金流数据，填NaN
    for col in fund_cols:
        if col not in merged.columns:
            merged[col] = float("nan")

    # 恢复 timestamps 列
    merged["timestamps"] = pd.to_datetime(merged["date_key"])
    merged = merged.drop(columns=["date_key"])
    merged = merged.sort_values("timestamps").reset_index(drop=True)
    merged = _fill(merged)

    # 确保列顺序一致
    all_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"] + fund_cols
    merged = merged[[c for c in all_cols if c in merged.columns]]

    # 保存
    merged.to_csv(csv_path, index=False)

    added = len(merged) - len(df_old)
    print(f"  {code}: {last_date_str} -> {new_last} (新增{added}条, 共{len(merged)}条)")
    return True


def main():
    parser = argparse.ArgumentParser(description="自动更新股票数据到最新交易日")
    parser.add_argument("--code", nargs="+", help="指定股票代码，不指定则更新全部")
    parser.add_argument("--data-dir", default="./examples/data", help="数据目录")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.code:
        codes = args.code
    else:
        # 扫描目录下所有 *_stock_data.csv
        codes = []
        for f in sorted(os.listdir(args.data_dir)):
            if f.endswith("_stock_data.csv"):
                codes.append(f.replace("_stock_data.csv", ""))

    if not codes:
        print("未找到需要更新的股票数据文件")
        return

    print(f"开始更新 {len(codes)} 只股票数据...")
    print(f"数据目录: {args.data_dir}\n")

    success = 0
    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}")
        try:
            if update_stock(code, args.data_dir):
                success += 1
        except Exception as e:
            print(f"  {code}: 更新失败 - {e}")
        # 限流控制
        if i < len(codes):
            time.sleep(0.8)

    print(f"\n更新完成: {success}/{len(codes)} 只成功")


if __name__ == "__main__":
    main()
