"""
股票数据抓取脚本 - 策略模式

每个数据源是一个 Strategy, 实现 fetch_range 接口。
返回的 DataFrame 统一包含全部字段, 不支持的字段填 NaN。

支持数据源:
  1. baostock    - K线（稳定, T+1, 不含当日）
  2. tencent     - 腾讯财经（K线, 轻量稳定, 含当日数据）
  3. sina        - 新浪财经（K线）
  4. eastmoney   - 东方财富 datacenter API（最新一天 K线+资金流）
  5. composite   - 组合策略: K线从 tencent/sina + 资金流从 eastmoney

用法:
  python3 fetch_stock_data.py --code 300782 --source tencent --start 20260101 --end 20260731
  python3 fetch_stock_data.py --code 300782 --source tencent --date 20260731
  python3 fetch_stock_data.py --code 300782 --source eastmoney --date 20260731
  python3 fetch_stock_data.py --code 300782 --source composite --start 20260701 --end 20260731
"""

import os
import sys
import time
import argparse
import json
import math
import requests
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


# ======================== 常量 ========================

ALL_FIELDS = [
    "timestamps", "open", "high", "low", "close", "volume", "amount",
    "main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio",
]

EASTMONEY_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_FIELDS)


def _fmt(date8: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD"""
    return f"{date8[:4]}-{date8[4:6]}-{date8[6:8]}"


def _prefix(code: str) -> str:
    """sh600664 / sz300782 (无点)"""
    return f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"


def _secid(code: str) -> str:
    """1.600664 / 0.300782"""
    return f"1.{code}" if code.startswith(("6", "9")) else f"0.{code}"


def _fill(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 包含全部字段, 缺失填 NaN"""
    for col in ALL_FIELDS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[ALL_FIELDS]


def _filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按日期范围过滤"""
    s, e = _fmt(start), _fmt(end)
    return df[(df["timestamps"] >= s) & (df["timestamps"] <= e)].reset_index(drop=True)


# ======================== 策略基类 ========================

class DataSourceStrategy(ABC):
    name: str = ""
    supports_fund_flow: bool = False

    @abstractmethod
    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        pass

    def fetch_daily(self, code: str, date: str) -> Optional[pd.Series]:
        df = self.fetch_range(code, date, date)
        if df.empty:
            return None
        return df.iloc[0]


# ======================== BaoStock ========================

class BaoStockStrategy(DataSourceStrategy):
    name = "baostock"
    supports_fund_flow = False

    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        import baostock as bs
        prefix = f"{'sh' if code.startswith(('6','9')) else 'sz'}.{code}"
        bs.login()
        rs = bs.query_history_k_data_plus(
            prefix, "date,open,high,low,close,volume,amount",
            start_date=_fmt(start), end_date=_fmt(end),
            frequency="d", adjustflag="2",
        )
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        bs.logout()

        if not data:
            return empty_df()
        df = pd.DataFrame(data, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.rename(columns={"date": "timestamps"}, inplace=True)
        return _fill(df)


# ======================== 腾讯财经 ========================

class TencentStrategy(DataSourceStrategy):
    name = "tencent"
    supports_fund_flow = False

    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        prefix = _prefix(code)
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,641,qfq"

        r = requests.get(url, headers=headers, timeout=10)
        raw = r.json().get("data", {})
        if isinstance(raw, list):
            raw = raw[0] if raw and isinstance(raw[0], dict) else {}
        stock = raw.get(prefix, {})
        klines = stock.get("qfqday", []) or stock.get("day", [])
        if not klines:
            return empty_df()

        rows = [{
            "timestamps": k[0], "open": float(k[1]), "close": float(k[2]),
            "high": float(k[3]), "low": float(k[4]),
            "volume": float(k[5]) * 100, "amount": 0.0,
        } for k in klines]

        df = pd.DataFrame(rows)
        df = _filter(df, start, end)
        return _fill(df) if not df.empty else empty_df()


# ======================== 新浪财经 ========================

class SinaStrategy(DataSourceStrategy):
    name = "sina"
    supports_fund_flow = False

    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        prefix = _prefix(code)
        url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
               f"/CN_MarketData.getKLineData?symbol={prefix}&scale=240&ma=5&datalen=1000")

        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200 or not r.text.strip().startswith("["):
            return empty_df()

        rows = [{
            "timestamps": d["day"], "open": float(d["open"]),
            "high": float(d["high"]), "low": float(d["low"]),
            "close": float(d["close"]), "volume": float(d["volume"]),
            "amount": 0.0,
        } for d in json.loads(r.text)]

        df = pd.DataFrame(rows)
        df = _filter(df, start, end)
        return _fill(df) if not df.empty else empty_df()


# ======================== 东方财富 datacenter API ========================

class EastMoneyStrategy(DataSourceStrategy):
    """东方财富: 通过 datacenter-web API 获取最新一天 K线+资金流

    注意: 该 API 只返回最新一天的数据, 历史日期无法获取。
    用于组合策略时, 仅能补充最近交易日的资金流字段。
    """
    name = "eastmoney"
    supports_fund_flow = True

    # datacenter 报表只返回最新一天
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://data.eastmoney.com/",
    }

    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        # 1. 从 datacenter 获取最新一天完整数据(含资金流)
        latest = self._fetch_latest(code)
        if latest is None:
            return empty_df()

        # 2. 从腾讯获取K线(补齐 OHLCV + 历史日期)
        kline_df = TencentStrategy().fetch_range(code, start, end)
        if kline_df.empty:
            return empty_df()

        # 3. 将最新一天的资金流合并到K线
        latest_date = latest["timestamps"]
        mask = kline_df["timestamps"] == latest_date
        if mask.any():
            for col in ["main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"]:
                kline_df.loc[mask, col] = latest.get(col, float("nan"))

        return _fill(kline_df)

    def _fetch_latest(self, code: str) -> Optional[dict]:
        """从 datacenter-web 获取最新一天资金流数据"""
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
                             headers=self._HEADERS, timeout=15)
            data = r.json()
            if not data.get("success"):
                return None

            items = data.get("result", {}).get("data", [])
            if not items:
                return None

            item = items[0]
            # PRIME_INFLOW = 主力净流入
            # SUPERDEAL_INFLOW - SUPERDEAL_OUTFLOW = 超大单净流入
            super_net = (item.get("SUPERDEAL_INFLOW", 0) or 0) - (item.get("SUPERDEAL_OUTFLOW", 0) or 0)
            prime_inflow = item.get("PRIME_INFLOW", 0) or 0
            close_price = item.get("CLOSE_PRICE", 0) or 0
            trade_date = str(item.get("TRADE_DATE", ""))[:10]

            # main_net_inflow_ratio 需要成交额才能计算, 此处无法获取, 返回 NaN 由组合策略计算
            ratio = float("nan")

            return {
                "timestamps": trade_date,
                "close": close_price,
                "main_net_inflow": float(prime_inflow),
                "huge_net_inflow": float(super_net),
                "main_net_inflow_ratio": float(ratio),  # 需要从K线amount计算
            }
        except Exception as e:
            print(f"  datacenter请求失败: {e}")
            return None


# ======================== 组合策略 ========================

class CompositeStrategy(DataSourceStrategy):
    """组合策略: K线从轻量API(tencent/sina) + 资金流从东方财富datacenter

    东方财富 datacenter 只返回最新一天资金流, 历史日期的资金流从本地缓存读取。
    每次运行会将新获取的资金流数据追加到缓存文件中。
    """
    name = "composite"
    supports_fund_flow = True

    def __init__(self, kline_source: DataSourceStrategy, fund_source: DataSourceStrategy,
                 cache_dir: str = "./examples/data/fund_cache"):
        self.kline_source = kline_source
        self.fund_source = fund_source
        self.cache_dir = cache_dir

    def fetch_range(self, code: str, start: str, end: str) -> pd.DataFrame:
        # 1. 获取K线
        kline_df = self.kline_source.fetch_range(code, start, end)
        if kline_df.empty:
            return empty_df()
        print(f"  K线({self.kline_source.name}): {len(kline_df)} 条")

        # 2. 获取最新一天资金流
        latest_fund = self._fetch_latest_fund(code)

        # 3. 加载本地缓存
        cache_df = self._load_cache(code)

        # 4. 合并最新资金流到缓存
        if latest_fund is not None:
            cache_df = self._update_cache(code, cache_df, latest_fund)

        # 5. 将缓存中的资金流合并到K线
        if not cache_df.empty and cache_df["main_net_inflow"].notna().any():
            print(f"  资金流(缓存+最新): {cache_df['main_net_inflow'].notna().sum()} 条有数据")
            # 先删除 kline_df 中已有的资金流列, 避免 merge 列名冲突
            for col in ["main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"]:
                if col in kline_df.columns:
                    kline_df.drop(columns=[col], inplace=True)

            kline_df["date_key"] = pd.to_datetime(kline_df["timestamps"]).dt.strftime("%Y-%m-%d")
            cache_df["date_key"] = pd.to_datetime(cache_df["timestamps"]).dt.strftime("%Y-%m-%d")

            merge_cols = ["date_key", "main_net_inflow", "huge_net_inflow"]
            if "main_net_inflow_ratio" in cache_df.columns:
                merge_cols.append("main_net_inflow_ratio")
            merged = kline_df.merge(
                cache_df[merge_cols], on="date_key", how="left",
            )
            merged.drop(columns=["date_key"], inplace=True)

            # 用 amount 计算缺失的 main_net_inflow_ratio
            if "main_net_inflow_ratio" in merged.columns:
                mask = merged["main_net_inflow_ratio"].isna() & merged["main_net_inflow"].notna()
                merged.loc[mask, "main_net_inflow_ratio"] = (
                    merged.loc[mask, "main_net_inflow"] / merged.loc[mask, "amount"] * 100
                )
                mask_zero = mask & (merged["amount"] == 0)
                merged.loc[mask_zero, "main_net_inflow_ratio"] = 0.0
            kline_df = merged
        else:
            print(f"  资金流: 无数据, 填充 NaN")

        return _fill(kline_df)

    def _fetch_latest_fund(self, code: str) -> Optional[dict]:
        """从东方财富 datacenter 获取最新一天资金流"""
        if not isinstance(self.fund_source, EastMoneyStrategy):
            # 非 datacenter 策略, 尝试获取
            fund_df = self.fund_source.fetch_range(code, "20260101", "20261231")
            if not fund_df.empty and fund_df["main_net_inflow"].notna().any():
                last = fund_df[fund_df["main_net_inflow"].notna()].iloc[-1]
                return last.to_dict()
            return None

        return self.fund_source._fetch_latest(code)

    def _load_cache(self, code: str) -> pd.DataFrame:
        """加载本地资金流缓存"""
        cache_path = os.path.join(self.cache_dir, f"{code}_fund_flow_cache.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            return df
        return pd.DataFrame(columns=["timestamps", "main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"])

    def _update_cache(self, code: str, cache_df: pd.DataFrame, latest: dict) -> pd.DataFrame:
        """将最新资金流数据追加到缓存"""
        date = latest.get("timestamps", "")
        if not date:
            return cache_df

        # 如果已有该日期, 更新; 否则追加
        mask = cache_df["timestamps"] == date
        if mask.any():
            for col in ["main_net_inflow", "huge_net_inflow", "main_net_inflow_ratio"]:
                cache_df.loc[mask, col] = latest.get(col, float("nan"))
        else:
            new_row = {
                "timestamps": date,
                "main_net_inflow": latest.get("main_net_inflow", float("nan")),
                "huge_net_inflow": latest.get("huge_net_inflow", float("nan")),
                "main_net_inflow_ratio": latest.get("main_net_inflow_ratio", float("nan")),
            }
            cache_df = pd.concat([cache_df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存缓存
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"{code}_fund_flow_cache.csv")
        cache_df.to_csv(cache_path, index=False)
        return cache_df


# ======================== 策略注册表 ========================

STRATEGIES: dict[str, type[DataSourceStrategy]] = {
    "baostock": BaoStockStrategy,
    "tencent": TencentStrategy,
    "sina": SinaStrategy,
    "eastmoney": EastMoneyStrategy,
}


def get_strategy(name: str) -> DataSourceStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知数据源: {name}, 可选: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]()


def get_composite(kline_name: str, fund_name: str) -> CompositeStrategy:
    return CompositeStrategy(get_strategy(kline_name), get_strategy(fund_name))


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description="股票数据抓取 (策略模式)")
    parser.add_argument("--code", required=True, help="股票代码")
    parser.add_argument("--source", default="tencent",
                        choices=list(STRATEGIES.keys()) + ["composite"], help="数据源")
    parser.add_argument("--kline-source", default="tencent", help="组合策略的K线源")
    parser.add_argument("--fund-source", default="eastmoney", help="组合策略的资金流源")
    parser.add_argument("--date", help="获取单日数据 YYYYMMDD")
    parser.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260731", help="截止日期 YYYYMMDD")
    parser.add_argument("--output", default="./examples/data", help="输出目录")
    args = parser.parse_args()

    if args.source == "composite":
        strategy = get_composite(args.kline_source, args.fund_source)
        print(f"组合策略: K线={args.kline_source}, 资金流={args.fund_source}")
    else:
        strategy = get_strategy(args.source)

    print(f"数据源: {strategy.name}, 股票: {args.code}, 资金流: {'支持' if strategy.supports_fund_flow else '不支持'}")

    if args.date:
        args.start = args.date
        args.end = args.date

    print(f"区间: {args.start}~{args.end}")

    df = strategy.fetch_range(args.code, args.start, args.end)

    if df.empty:
        print("未获取到数据!")
        return

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.code}_stock_data.csv")
    df.to_csv(out_path, index=False)
    print(f"已保存 {len(df)} 条 -> {out_path}")

    fund_ok = df["main_net_inflow"].notna().any()
    fund_count = df["main_net_inflow"].notna().sum()
    print(f"资金流: {fund_count}/{len(df)} 条有数据" if fund_ok else "资金流: 无数据(NaN)")
    print(f"\n最后3条:")
    print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
