#!/usr/bin/env python3
"""
新闻情感打分模块（方案A: 情绪过滤器）

- 个股: 东方财富个股新闻接口, 取最近 N 天标题 -> FinBERT 打分 -> 情绪分/利空占比
- 宏观: 上证/深证指数新闻接口, 取最近 1 天标题 -> 打分 -> 关键词映射到受影响行业
- 模型: 中文金融 FinBERT (yiyanghkust/finbert-tone-chinese), 缓存到本地后离线可用
  首次下载需设置环境变量: HF_ENDPOINT=https://hf-mirror.com

用法(单测):
  python scripts/news_sentiment.py 000001 600036

定时任务(全部股票+宏观新闻抓取存档):
  python scripts/news_sentiment.py --all
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests

NEWS_API = "https://np-listapi.eastmoney.com/comm/web/getListInfo"
MODEL_ID = "yiyanghkust/finbert-tone-chinese"
NEWS_DAYS = 3          # 只看最近3天新闻
PAGE_SIZE = 10         # 每只股票最多取10条
NEG_TITLE_THRESHOLD = 0.6  # 单条新闻 negative 概率超过此值记为利空新闻

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        try:
            # 模型已缓存时离线加载, 避免每次联网校验超时
            _pipeline = pipeline("text-classification", model=MODEL_ID,
                                 truncation=True, max_length=256,
                                 local_files_only=True)
        except Exception:
            _pipeline = pipeline("text-classification", model=MODEL_ID,
                                 truncation=True, max_length=256)
    return _pipeline


def fetch_news(code, days=NEWS_DAYS, page_size=PAGE_SIZE):
    """抓取个股最近N天的新闻标题列表"""
    mtype = "1" if code.startswith(("6", "9")) else "0"
    return _fetch_news_by_type(mtype, code, days, page_size)


def _fetch_news_by_type(mtype, code, days=NEWS_DAYS, page_size=PAGE_SIZE):
    """底层抓取: 指定市场类型(mtype=1 上证/0 深证)和代码, 取最近N天新闻标题"""
    params = {
        "client": "web",
        "mTypeAndCode": f"{mtype}.{code}",
        "type": "1",
        "pageSize": str(page_size),
        "pageIndex": "1",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.eastmoney.com/"}
    try:
        r = requests.get(NEWS_API, params=params, headers=headers, timeout=15)
        data = r.json().get("data", {}).get("list", []) or []
    except Exception:
        return []

    cutoff = datetime.now() - timedelta(days=days)
    titles = []
    for item in data:
        try:
            t = datetime.strptime(item.get("Art_ShowTime", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if t >= cutoff and item.get("Art_Title"):
            # 去掉标题中的HTML标签和空白
            title = re.sub(r"<[^>]+>", "", item["Art_Title"]).strip()
            if title:
                titles.append(title)
    return titles


def score_titles(titles):
    """对新闻标题打分, 返回聚合情绪指标"""
    if not titles:
        return {"count": 0, "score": 0.0, "neg_ratio": 0.0, "pos_ratio": 0.0,
                "negative_titles": [], "positive_titles": []}

    pipe = _get_pipeline()
    preds = pipe(titles, batch_size=8)
    probs = []
    for p in preds:
        label, score = p["label"], float(p["score"])
        if label == "Positive":
            probs.append(score)
        elif label == "Negative":
            probs.append(-score)
        else:
            probs.append(0.0)

    n = len(probs)
    neg_titles = [t for t, s in zip(titles, probs) if s <= -NEG_TITLE_THRESHOLD]
    pos_titles = [t for t, s in zip(titles, probs) if s >= NEG_TITLE_THRESHOLD]
    return {
        "count": n,
        "score": sum(probs) / n,                      # 平均情绪分: [-1, 1]
        "neg_ratio": sum(1 for s in probs if s <= -NEG_TITLE_THRESHOLD) / n,
        "pos_ratio": sum(1 for s in probs if s >= NEG_TITLE_THRESHOLD) / n,
        "negative_titles": neg_titles,
        "positive_titles": pos_titles,
    }


def get_stock_sentiment(code):
    """单只股票: 抓新闻 + 打分"""
    titles = fetch_news(code)
    result = score_titles(titles)
    result["titles"] = titles
    return result


# 情绪过滤规则(在 daily_predict.py 中调用)
# Kronos 看多但强利空新闻 -> 下调; 看空但强利好新闻 -> 上调(减弱看空)
# 同向放大：看多+利好 -> 增强; 看空+利空 -> 增强
STRONG_NEG = -0.15   # 平均情绪分低于此值视为利空（从-0.3放宽）
STRONG_POS = 0.15    # 高于此值视为利好（从0.3放宽）
MIN_NEWS = 1         # 至少1条新闻才启用过滤（从2降低）
CONFLICT_SCALE = 0.3  # 冲突时预测涨跌幅乘以该系数（从0.6提升力度）
SAME_SCALE = 1.3     # 同向放大系数（新增）


def apply_sentiment_filter(pred_chg, sentiment):
    """
    对单只股票的预测涨跌幅应用情绪过滤。
    返回 (调整后chg, 标签orNone)

    新逻辑：
      - 预测涨 + 情绪正面 → 放大（同向共振）
      - 预测跌 + 情绪负面 → 放大（同向共振）
      - 预测涨 + 负面新闻 → 衰减（反向冲突）
      - 预测跌 + 利好新闻 → 衰减（反向冲突）
      - 情绪中性 → 不调整
    """
    if sentiment["count"] < MIN_NEWS:
        return pred_chg, None
    score = sentiment["score"]
    neg_ratio = sentiment.get("neg_ratio", 0)
    pos_ratio = sentiment.get("pos_ratio", 0)

    # 同向放大：预测方向与情绪方向一致
    if pred_chg > 0 and score >= STRONG_POS:
        return pred_chg * SAME_SCALE, "利好共振↑"
    if pred_chg < 0 and score <= STRONG_NEG:
        return pred_chg * SAME_SCALE, "利空共振↓"

    # 反向冲突：预测方向与情绪方向相反
    if pred_chg > 0 and score <= STRONG_NEG and neg_ratio >= 0.3:
        return pred_chg * CONFLICT_SCALE, "负面新闻↓"
    if pred_chg < 0 and score >= STRONG_POS and pos_ratio >= 0.3:
        return pred_chg * CONFLICT_SCALE, "利好新闻↑"

    return pred_chg, None


# ============================================================
# 宏观/行业新闻 (特大国际事件 + 影响行业)
# ============================================================
# 用指数新闻接口拿市场级要闻: 上证/深证成指
MACRO_INDEXES = [("1", "000001"), ("0", "399001")]

# 关键词 -> 行业(STOCK_TAGS 的取值), 顺序无意义, 一条新闻可命中多个行业
INDUSTRY_KEYWORDS = {
    "银行":   ["利率", "LPR", "降准", "降息", "美联储", "加息", "银行", "消费贷", "拨备", "信贷"],
    "券商":   ["券商", "牛市", "成交量", "两融", "北向资金", "沪指", "创业板", "成交额"],
    "白酒":   ["白酒", "茅台", "五粮液", "消费税", "食饮", "食品饮料"],
    "半导体": ["芯片", "半导体", "集成电路", "光刻", "封测", "存储", "国产替代"],
    "AI软件": ["AI", "人工智能", "大模型", "算力", "GPT", "智能体", "AGI"],
    "AI营销": ["营销", "AIGC", "数字营销", "广告"],
    "软件":   ["软件", "信创", "国产软件", "SaaS", "操作系统", "数据库"],
    "医药":   ["医药", "集采", "创新药", "医保", "生物医药", "疫苗", "医疗器械"],
    "铜金":   ["铜价", "智利", "秘鲁", "罢工", "黄金", "金价上涨", "金价下跌", "有色"],
    "铜钴":   ["钴", "锂", "新能源金属", "铜钴"],
    "铜冶炼": ["铜冶炼", "粗铜", "精铜"],
    "火电":   ["火电", "煤价", "动力煤", "电煤"],
    "水电":   ["水电", "发电量", "水电站", "来水"],
    "核电":   ["核电", "反应堆", "核电站", "华龙一号"],
    "光伏":   ["光伏", "硅料", "组件", "储能", "新能源", "太阳能"],
    "CDN":    ["CDN", "边缘计算", "数据中心", "IDC"],
}


def match_industries(title):
    """从新闻标题匹配受影响的行业列表"""
    hits = []
    for ind, kws in INDUSTRY_KEYWORDS.items():
        if any(k in title for k in kws):
            hits.append(ind)
    return hits


def fetch_macro_news(days=1, page_size=30):
    """抓取宏观/行业新闻: 上证+深证指数新闻, 去重, 返回 [(time, title)]"""
    seen = set()
    items = []
    cutoff = datetime.now() - timedelta(days=days)
    for mtype, code in MACRO_INDEXES:
        try:
            params = {
                "client": "web",
                "mTypeAndCode": f"{mtype}.{code}",
                "type": "1",
                "pageSize": str(page_size),
                "pageIndex": "1",
            }
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.eastmoney.com/"}
            r = requests.get(NEWS_API, params=params, headers=headers, timeout=15)
            data = r.json().get("data", {}).get("list", []) or []
        except Exception:
            continue
        for item in data:
            try:
                t = datetime.strptime(item.get("Art_ShowTime", ""), "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if t < cutoff:
                continue
            title = re.sub(r"<[^>]+>", "", item.get("Art_Title", "")).strip()
            if title and title not in seen:
                seen.add(title)
                items.append((t, title))
        time.sleep(0.3)
    # 按时间倒序
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def fetch_all(save_dir="./news_log"):
    """全部股票抓新闻+打分, 存档到 news_log/{date}_{hhmm}.csv (供盘中/盘后情绪跟踪)"""
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.daily_predict import STOCKS

    os.makedirs(save_dir, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d") + "_" + now.strftime("%H%M")

    # ---- 个股情绪 ----
    rows = []
    for code, name in STOCKS.items():
        try:
            s = get_stock_sentiment(code)
        except Exception:
            s = {"count": 0, "score": 0.0, "neg_ratio": 0.0, "pos_ratio": 0.0,
                 "negative_titles": []}
        print(f"  {name}({code}): 情绪分{s['score']:+.2f} 利空占比{s['neg_ratio']:.0%} 共{s['count']}条")
        rows.append({"code": code, "name": name, "news_count": s["count"],
                     "score": round(s["score"], 2), "neg_ratio": round(s["neg_ratio"], 2),
                     "pos_ratio": round(s["pos_ratio"], 2),
                     "negative_titles": "; ".join(s["negative_titles"][:3])})
        time.sleep(0.3)
    stock_out = os.path.join(save_dir, f"{stamp}.csv")
    pd.DataFrame(rows).to_csv(stock_out, index=False)
    print(f"已保存个股: {stock_out}")

    # ---- 宏观/行业新闻 + 情感打分 + 行业映射 ----
    macro_items = fetch_macro_news(days=1, page_size=30)
    titles = [t for _, t in macro_items]
    macro_rows = []
    if titles:
        preds = _get_pipeline()(titles, batch_size=8)
        for (t_time, title), p in zip(macro_items, preds):
            label, score = p["label"], float(p["score"])
            if label == "Positive":
                s = score
            elif label == "Negative":
                s = -score
            else:
                s = 0.0
            industries = match_industries(title)
            macro_rows.append({"time": t_time.strftime("%Y-%m-%d %H:%M"),
                               "title": title, "score": round(s, 2),
                               "industries": ";".join(industries) if industries else ""})
    macro_out = os.path.join(save_dir, f"{stamp}_macro.csv")
    pd.DataFrame(macro_rows).to_csv(macro_out, index=False)
    print(f"已保存宏新闻: {macro_out} ({len(macro_rows)}条)")
    # 打印命中的行业分布(本时段快照)
    industry_hits = {}
    for r in macro_rows:
        if not r["industries"]:
            continue
        for ind in r["industries"].split(";"):
            industry_hits.setdefault(ind, []).append(r)
    for ind, lst in sorted(industry_hits.items(), key=lambda x: -len(x[1])):
        avg = sum(r["score"] for r in lst) / len(lst)
        print(f"  行业[{ind}]: {len(lst)}条 平均情绪分{avg:+.2f}")


if __name__ == "__main__":
    if "--all" in sys.argv:
        fetch_all()
    else:
        codes = sys.argv[1:] or ["600036"]
        for code in codes:
            s = get_stock_sentiment(code)
            print(f"\n{code}: 情绪分={s['score']:+.2f} 利空占比={s['neg_ratio']:.0%} "
                  f"利好占比={s['pos_ratio']:.0%} 共{s['count']}条")
            for t in s["negative_titles"]:
                print(f"  [利空] {t}")
            for t in s["positive_titles"]:
                print(f"  [利好] {t}")
