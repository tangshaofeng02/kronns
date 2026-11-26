"""
获取 A 股 K线 + 主力资金流数据，合并输出训练用 CSV

数据来源:
  - K线: baostock (稳定, 含 amount)
  - 资金流: 东方财富 datacenter API (datacenter-web.eastmoney.com)

输出列: timestamps, open, high, low, close, volume, amount,
        main_net_inflow, huge_net_inflow, main_net_inflow_ratio

注意: 东方财富 datacenter API 只返回最新一天的资金流数据。
  历史资金流需要每天运行脚本逐步积累缓存。
  如果需要一次性获取大量历史资金流, 需要 push2his.eastmoney.com API
  (该 API 在部分网络环境下可能被封)。
"""

import os
import sys
import time
import argparse
import requests
import baostock as bs
import pandas as pd
import numpy as np

# 东方财富 datacenter API (稳定, 只返回最新一天)
EASTMONEY_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"

# 东方财富 push2his API (可获取历史, 但部分网络被封)
FUND_FLOW_HIS_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"

_DATACENTER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}

_HIS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/",
}


def _secid(code: str) -> str:
    if code.startswith("6") or code.startswith("9"):
        return f"1.{code}"
    return f"0.{code}"


def _bs_prefix(code: str) -> str:
    if code.startswith("6") or code.startswith("9"):
        return f"sh.{code}"
    return f"sz.{code}"


def fetch_kline(code: str, start: str, end: str) -> pd.DataFrame:
    """用 baostock 获取前复权日K线"""
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 else start
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 else end

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        _bs_prefix(code),
        "date,open,high,low,close,volume,amount",
        start_date=start_fmt,
        end_date=end_fmt,
        frequency="d",
        adjustflag="2",
    )
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data, columns=rs.fields)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.rename(columns={"date": "timestamps"}, inplace=True)
    return df


def fetch_fund_flow_latest(code: str) -> dict:
    """从东方财富 datacenter 获取最新一天资金流(只返回最新一天)"""
    try:
        params = {
            "reportName": "RPT_DMSK_TS_STOCKNEW",
            "columns": "ALL",
            "filter": f'(SECURITY_CODE="{code}")',
            "pageSize": "1",
            "pageNumber": "1",
            "sortColumns": "TRADE_DATE",
            "sortTypes": "-1",
        }
        r = requests.get(EASTMONEY_DATACENTER, params=params,
                         headers=_DATACENTER_HEADERS, timeout=15)
        data = r.json()
        if not data.get("success"):
            return {}

        items = data.get("result", {}).get("data", [])
        if not items:
            return {}

        item = items[0]
        super_net = (item.get("SUPERDEAL_INFLOW", 0) or 0) - (item.get("SUPERDEAL_OUTFLOW", 0) or 0)
        return {
            "date": str(item.get("TRADE_DATE", ""))[:10],
            "main_net_inflow": float(item.get("PRIME_INFLOW", 0) or 0),
            "huge_net_inflow": float(super_net),
        }
    except Exception as e:
        print(f"  datacenter资金流获取失败({code}): {e}")
        return {}


def fetch_fund_flow_history(code: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """从东方财富 push2his API 获取历史资金流(部分网络可能被封)"""
    secid = _secid(code)
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "klt": "101",
    }

    for attempt in range(retries):
        try:
            r = requests.get(FUND_FLOW_HIS_URL, params=params,
                             headers=_HIS_HEADERS, timeout=15)
            data = r.json()
            klines = data.get("data", {}).get("klines", [])
            if not klines:
                return pd.DataFrame()

            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                rows.append({
                    "date": parts[0],
                    "main_net_inflow": float(parts[1]),
                    "huge_net_inflow": float(parts[5]),
                })
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            start_fmt = pd.to_datetime(start).strftime("%Y-%m-%d")
            end_fmt = pd.to_datetime(end).strftime("%Y-%m-%d")
            df = df[(df["date"] >= start_fmt) & (df["date"] <= end_fmt)]
            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  历史资金流获取失败({code}): {e}")
                return pd.DataFrame()


def fetch_stock_with_fundflow(
    code: str,
    name: str,
    start: str = "20230101",
    end: str = "20260730",
) -> pd.DataFrame:
    """获取单只股票的 K线 + 资金流合并数据"""
    print(f"  获取 {name}({code}) K线...")
    kline_df = fetch_kline(code, start, end)
    if kline_df.empty:
        print(f"  {name}({code}) K线数据为空，跳过")
        return pd.DataFrame()
    print(f"  K线: {len(kline_df)} 条")

    # 优先尝试 push2his 历史API
    print(f"  获取 {name}({code}) 历史资金流...")
    fund_df = fetch_fund_flow_history(code, start, end)

    # 如果历史API失败, 用 datacenter 最新一天
    if fund_df.empty:
        print(f"  历史API不可用, 获取最新一天资金流...")
        latest = fetch_fund_flow_latest(code)
        if latest:
            print(f"  最新资金流: {latest['date']}, 主力净流入={latest['main_net_inflow']:.0f}")
            fund_df = pd.DataFrame([latest])
        else:
            print(f"  {name}({code}) 资金流数据为空，填充0")

    # 合并
    kline_df["date_key"] = pd.to_datetime(kline_df["timestamps"]).dt.strftime("%Y-%m-%d")
    if not fund_df.empty:
        merged = kline_df.merge(
            fund_df[["date", "main_net_inflow", "huge_net_inflow"]],
            left_on="date_key", right_on="date", how="left",
        )
        merged.drop(columns=["date_key", "date"], inplace=True)
        # 仅在有多天连续资金流数据时 ffill (停牌日等场景)
        # 单日数据不 ffill, 避免错误填充
        fund_matched = merged["main_net_inflow"].notna().sum()
        if fund_matched > 1 and fund_matched < len(merged) * 0.9:
            merged[["main_net_inflow", "huge_net_inflow"]] = (
                merged[["main_net_inflow", "huge_net_inflow"]].ffill().bfill().fillna(0)
            )
    else:
        kline_df.drop(columns=["date_key"], inplace=True)
        merged = kline_df.copy()
        merged["main_net_inflow"] = 0.0
        merged["huge_net_inflow"] = 0.0

    merged["main_net_inflow_ratio"] = np.where(
        merged["amount"] != 0,
        merged["main_net_inflow"] / merged["amount"] * 100,
        0.0,
    )

    print(f"  合并后: {len(merged)} 条, 最后收盘: {merged['close'].iloc[-1]:.2f}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="获取 A 股 K线 + 主力资金流数据")
    parser.add_argument("--stocks", nargs="+", help="股票代码列表，如 300782 600664")
    parser.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260730", help="截止日期 YYYYMMDD")
    parser.add_argument("--output", default="./finetune_csv/data", help="输出目录")
    args = parser.parse_args()

    if not args.stocks:
        args.stocks = [
            "300782", "300368", "600664", "300451", "300430",
            "300678", "301293", "000001", "600036", "000858",
        ]

    stock_names = {
        "300782": "卓胜微", "300368": "汇金股份", "600664": "哈药股份",
        "300451": "创业惠康", "300430": "诚益通", "300678": "中科信息",
        "301293": "三博脑科", "000001": "平安银行", "600036": "招商银行",
        "000858": "五粮液", "601318": "中国平安", "600519": "贵州茅台",
        "000333": "美的集团", "002415": "海康威视", "300750": "宁德时代",
    }

    os.makedirs(args.output, exist_ok=True)
    all_dfs = []

    for code in args.stocks:
        name = stock_names.get(code, code)
        print(f"\n{'='*50}")
        print(f"  处理 {name}({code})")
        print(f"{'='*50}")

        df = fetch_stock_with_fundflow(code, name, args.start, args.end)
        if df.empty:
            continue

        single_path = os.path.join(args.output, f"{code}_fundflow.csv")
        df.to_csv(single_path, index=False)
        print(f"  已保存: {single_path}")

        all_dfs.append(df)
        time.sleep(0.5)

    if all_dfs:
        merged_path = os.path.join(args.output, "stock_with_fund_flow.csv")
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(merged_path, index=False)
        print(f"\n合并数据: {len(combined)} 条 -> {merged_path}")
        print(f"列: {combined.columns.tolist()}")


if __name__ == "__main__":
    main()
