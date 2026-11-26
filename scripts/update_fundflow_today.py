"""
增量更新 finetune_csv/data/*_fundflow.csv 到最新交易日

策略:
  - 历史K线: baostock (T+1, 有 amount)
  - 今日K线: 腾讯财经 (含当日, amount=0)
  - 今日资金流: 东方财富 datacenter API (最新一天)
  - 历史资金流: 保留 CSV 中已有值, 不覆盖

用法:
  python scripts/update_fundflow_today.py
  python scripts/update_fundflow_today.py --code 300782 600664
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.fetch_stock_data import TencentStrategy
from scripts.fetch_fund_flow import fetch_kline as bs_fetch_kline, fetch_fund_flow_latest


FUND_COLS = ["main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"]
KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def update_one(code: str, data_dir: str) -> str:
    csv_path = os.path.join(data_dir, f"{code}_fundflow.csv")
    if not os.path.exists(csv_path):
        return f"{code}: 文件不存在, 跳过"

    df_old = pd.read_csv(csv_path)
    df_old["timestamps"] = pd.to_datetime(df_old["timestamps"]).dt.strftime("%Y-%m-%d")
    last_date = df_old["timestamps"].iloc[-1]
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    # 1. baostock 拉历史 K线 (last_date+1 ~ today), 有 amount
    start_bs = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    df_bs = bs_fetch_kline(code, start_bs, today.replace("-", ""))
    if not df_bs.empty:
        df_bs["timestamps"] = pd.to_datetime(df_bs["timestamps"]).dt.strftime("%Y-%m-%d")
        df_bs = df_bs[df_bs["timestamps"] > last_date]
    bs_dates = set(df_bs["timestamps"].tolist()) if not df_bs.empty else set()

    # 2. 腾讯拉今日 K线 (兜底, baostock T+1 没出当日), amount=0
    df_tx = TencentStrategy().fetch_range(code, start_bs, "20991231")
    if not df_tx.empty:
        df_tx["timestamps"] = pd.to_datetime(df_tx["timestamps"]).dt.strftime("%Y-%m-%d")
        df_tx = df_tx[df_tx["timestamps"] > last_date]
        # 只保留 baostock 没拿到的日期 (通常是今日)
        df_tx = df_tx[~df_tx["timestamps"].isin(bs_dates)]

    # 合并新 K线
    df_new = pd.concat([df_bs, df_tx], ignore_index=True) if (not df_bs.empty or not df_tx.empty) else pd.DataFrame()
    if df_new.empty:
        return f"{code}: 已是最新 ({last_date})"
    df_new = df_new.sort_values("timestamps").reset_index(drop=True)

    # 3. datacenter 拿最新一天资金流
    latest = fetch_fund_flow_latest(code)
    fund_date = latest.get("date", "") if latest else ""
    fund_main = latest.get("main_net_inflow") if latest else None
    fund_huge = latest.get("huge_net_inflow") if latest else None

    # 构造新行
    new_rows = []
    for _, r in df_new.iterrows():
        row = {"timestamps": r["timestamps"]}
        for c in KLINE_COLS:
            row[c] = r.get(c, np.nan)
        # 仅在 K线日期 == 资金流日期 时填资金流
        if fund_date and r["timestamps"] == fund_date:
            row["main_net_inflow"] = fund_main
            row["huge_net_inflow"] = fund_huge
        else:
            row["main_net_inflow"] = np.nan
            row["huge_net_inflow"] = np.nan
        # ratio 稍后统一算
        new_rows.append(row)
    df_add = pd.DataFrame(new_rows)

    # 合并: 旧数据原样保留, 新行计算 ratio
    df_all = pd.concat([df_old, df_add], ignore_index=True)
    # 对新增行计算 ratio
    add_idx = df_all.index[-len(df_add):]
    mask = (df_all.loc[add_idx, "amount"].notna()) & (df_all.loc[add_idx, "amount"] != 0) & (df_all.loc[add_idx, "main_net_inflow"].notna())
    df_all.loc[add_idx, "main_net_inflow_ratio"] = np.where(
        mask,
        df_all.loc[add_idx, "main_net_inflow"] / df_all.loc[add_idx, "amount"] * 100,
        np.nan,
    )

    # 确保列顺序
    cols = ["timestamps"] + KLINE_COLS + FUND_COLS
    df_all = df_all[[c for c in cols if c in df_all.columns]]
    df_all.to_csv(csv_path, index=False)

    added_dates = df_add["timestamps"].tolist()
    fund_msg = f"资金流={fund_date}" if fund_date else "资金流=无"
    return f"{code}: {last_date} -> {df_all['timestamps'].iloc[-1]} (+{len(df_add)}行, {fund_msg}, 新增{added_dates})"


def main():
    parser = argparse.ArgumentParser(description="增量更新 _fundflow.csv 到最新交易日")
    parser.add_argument("--code", nargs="+", help="股票代码, 不指定则更新全部")
    parser.add_argument("--data-dir", default="./finetune_csv/data", help="数据目录")
    args = parser.parse_args()

    if args.code:
        codes = args.code
    else:
        codes = []
        for f in sorted(os.listdir(args.data_dir)):
            if f.endswith("_fundflow.csv") and not f.startswith("stock_with"):
                codes.append(f.replace("_fundflow.csv", ""))

    if not codes:
        print("未找到需更新的文件")
        return

    print(f"开始增量更新 {len(codes)} 只股票...")
    success = 0
    for i, code in enumerate(codes, 1):
        try:
            msg = update_one(code, args.data_dir)
            print(f"[{i}/{len(codes)}] {msg}")
            if "已" not in msg.split(":")[-1].split("(")[0]:
                success += 1
        except Exception as e:
            print(f"[{i}/{len(codes)}] {code}: 失败 - {e}")
        if i < len(codes):
            time.sleep(0.5)

    # 重新合并 stock_with_fund_flow.csv
    print("\n合并 stock_with_fund_flow.csv ...")
    all_dfs = []
    for code in codes:
        p = os.path.join(args.data_dir, f"{code}_fundflow.csv")
        if os.path.exists(p):
            all_dfs.append(pd.read_csv(p))
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(args.data_dir, "stock_with_fund_flow.csv"), index=False)
        print(f"合并: {len(combined)} 条 -> stock_with_fund_flow.csv")

    print(f"\n完成: {success}/{len(codes)} 只更新成功")


if __name__ == "__main__":
    main()
