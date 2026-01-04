from __future__ import annotations

import os
import json
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple

import pandas as pd
import requests
from xgboost import XGBRegressor
import pandas_market_calendars as mcal

from utils.market_calendar import is_market_open
from utils.safe_yfinance import safe_yf_download

warnings.filterwarnings("ignore")

# -----------------------------
# Basic settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
TOP300_CACHE_FILE = os.path.join(CACHE_DIR, "top300_tw.json")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# 你常看的權值股（保留）
FIXED = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]

# -----------------------------
# Helpers
# -----------------------------
def _today_tw() -> str:
    """台北時間今天 YYYY-MM-DD"""
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")


def pre_check() -> bool:
    # 只要不是交易日就不跑（避免無意義的抓資料/訓練）
    if not is_market_open("TW"):
        print("📌 今日非交易日（台股休市）")
        return False
    return True


def calc_pivot(df: pd.DataFrame) -> Tuple[float, float]:
    """近 20 日 Pivot 支撐/壓力（簡易版）"""
    r = df.iloc[-20:]
    h, l, c = float(r["High"].max()), float(r["Low"].min()), float(r["Close"].iloc[-1])
    p = (h + l + c) / 3
    sup = round(2 * p - h, 1)
    res = round(2 * p - l, 1)
    return sup, res


def nth_trading_day_after(start_date: str, n: int, calendar_name: str = "XTAI") -> str:
    """
    回傳 start_date 之後第 n 個交易日（不含 start_date 當天）。
    使用 pandas_market_calendars 的交易所日曆避免週末/假日。
    """
    cal = mcal.get_calendar(calendar_name)
    # 取較寬鬆的區間，避免遇到長假
    schedule = cal.schedule(start_date=start_date, end_date=pd.Timestamp(start_date) + pd.Timedelta(days=60))
    days = schedule.index.strftime("%Y-%m-%d").tolist()
    if start_date in days:
        pos = days.index(start_date)
        target = pos + n
    else:
        # 若 start_date 不是交易日（理論上不會發生，因為 pre_check 已擋）
        # 就找下一個交易日當作起點
        target = n - 1
    if target >= len(days):
        raise RuntimeError("交易日曆不足，請加大 end_date 範圍")
    return days[target]


def _read_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(
            columns=[
                "run_date",
                "ticker",
                "pred",
                "price_at_run",
                "sup",
                "res",
                "settle_date",
                "settle_close",
                "realized_return",
                "hit",
                "status",
                "updated_at",
            ]
        )

    df = pd.read_csv(HISTORY_FILE)

    # 保證欄位完整（避免舊檔案）
    for col in ["settle_close", "realized_return", "hit", "status", "updated_at"]:
        if col not in df.columns:
            df[col] = pd.NA

    if "status" not in df.columns:
        df["status"] = "pending"

    df["status"] = df["status"].fillna("pending")
    return df


def _write_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def _load_top300_cache(today: str) -> List[str] | None:
    try:
        with open(TOP300_CACHE_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("date") == today and isinstance(obj.get("tickers"), list):
            return obj["tickers"]
    except Exception:
        pass
    return None


def _save_top300_cache(today: str, tickers: List[str]) -> None:
    try:
        with open(TOP300_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "tickers": tickers}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_top300_by_volume(today: str) -> List[str]:
    """
    先抓「上市股票清單」→ 用 yfinance 拉 1M 資料計算近 20 日平均量 → 取前 300。
    這一步最容易被 yfinance 429，所以做「當日快取」：同一天內重跑不會再抓一輪。
    """
    cached = _load_top300_cache(today)
    if cached:
        return cached

    # 取得上市清單（四碼）
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    html = requests.get(url, timeout=15).text
    df = pd.read_html(html)[0]
    df.columns = df.iloc[0]
    codes = df.iloc[1:][df.columns[0]].astype(str).str.split(" ").str[0]
    tickers = [f"{c}.TW" for c in codes if c.isdigit() and len(c) == 4]

    # 分批抓取近 1 個月資料，算均量
    data = safe_yf_download(tickers, period="1mo", max_chunk=80)
    avg_vol: Dict[str, float] = {}
    for t, d in data.items():
        if d is None or len(d) < 5:
            continue
        v = float(d["Volume"].tail(20).mean())
        if pd.notna(v) and v > 0:
            avg_vol[t] = v

    top300 = sorted(avg_vol, key=avg_vol.get, reverse=True)[:300]
    _save_top300_cache(today, top300)
    return top300


def settle_history(today: str) -> Tuple[pd.DataFrame, str]:
    """
    對 tw_history.csv 裡「已到期（settle_date <= today）」但尚未結算的項目做結算。
    回傳（更新後的 df, 結算訊息文字）
    """
    hist = _read_history()
    if hist.empty:
        return hist, ""

    # 找待結算且到期的
    pending = hist[(hist["status"] == "pending") & (hist["settle_date"].astype(str) <= today)]
    if pending.empty:
        return hist, ""

    tickers = sorted(pending["ticker"].astype(str).unique().tolist())
    # 為了穩定，直接抓 3mo，足夠涵蓋 settle_date
    data = safe_yf_download(tickers, period="3mo", max_chunk=60)

    settled_rows = []
    for idx, row in pending.iterrows():
        t = str(row["ticker"])
        settle_date = str(row["settle_date"])
        d = data.get(t)
        if d is None or d.empty:
            continue

        # yfinance index 可能是 Timestamp（含時區/不含時區），統一轉字串日期
        d2 = d.copy()
        d2.index = pd.to_datetime(d2.index).strftime("%Y-%m-%d")

        if settle_date not in d2.index:
            # 保守：如果找不到，先略過（不亂補日期）
            continue

        settle_close = float(d2.loc[settle_date, "Close"])
        price_at_run = float(row["price_at_run"])
        rr = (settle_close / price_at_run) - 1.0

        hist.at[idx, "settle_close"] = round(settle_close, 2)
        hist.at[idx, "realized_return"] = rr
        hist.at[idx, "hit"] = int(rr > 0)
        hist.at[idx, "status"] = "settled"
        hist.at[idx, "updated_at"] = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
        settled_rows.append((t, settle_date, rr))

    if not settled_rows:
        return hist, ""

    # 結算摘要（只列最多 8 筆，避免 Discord 太長）
    lines = ["🏁 台股 5 日回測結算（到期項目）"]
    for t, dte, rr in settled_rows[:8]:
        lines.append(f"- {t} @ {dte}: `{rr:+.2%}`")
    if len(settled_rows) > 8:
        lines.append(f"... 另外還有 {len(settled_rows) - 8} 筆已結算")

    return hist, "\n".join(lines) + "\n"


def append_today_predictions(hist: pd.DataFrame, today: str, rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return hist

    now_str = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame(rows)
    df_new["run_date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    # 避免同一天重跑把同一檔重複寫入：用 (run_date, ticker) 去重
    if not hist.empty:
        hist["run_date"] = hist["run_date"].astype(str)
        hist["ticker"] = hist["ticker"].astype(str)
        existing = set(zip(hist["run_date"], hist["ticker"]))
        df_new = df_new[~df_new.apply(lambda r: (today, r["ticker"]) in existing, axis=1)]

    if df_new.empty:
        return hist

    out = pd.concat([hist, df_new], ignore_index=True)
    return out


# -----------------------------
# Main
# -----------------------------
def run() -> None:
    today = _today_tw()

    # 1) 先做歷史結算（到期的就補上）
    hist, settle_msg = settle_history(today)

    # 2) 今日預測（Top300 + FIXED）
    universe = list(dict.fromkeys(FIXED + get_top300_by_volume(today)))
    data = safe_yf_download(universe, period="2y", max_chunk=60)

    feats = ["mom20", "bias", "vol_ratio"]
    results: Dict[str, dict] = {}

    for s, df in data.items():
        if df is None or len(df) < 160:
            continue

        df = df.copy()
        df["mom20"] = df["Close"].pct_change(20)
        ma20 = df["Close"].rolling(20).mean()
        df["bias"] = (df["Close"] - ma20) / ma20
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        df = df.dropna()
        if len(df) < 120:
            continue

        train = df.iloc[:-1]

        model = XGBRegressor(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(train[feats], train["target"])

        pred = float(model.predict(df[feats].iloc[-1:])[0])
        sup, res = calc_pivot(df)

        results[s] = {
            "pred": pred,
            "price": round(float(df["Close"].iloc[-1]), 2),
            "sup": sup,
            "res": res,
        }

    if not results:
        msg = "⚠️ 今日無可用結果（可能資料不足或抓取失敗）"
        _post(msg)
        return

    top = sorted(results.items(), key=lambda kv: kv[1]["pred"], reverse=True)[:5]

    # 3) 寫入歷史（今日 Top5）
    new_rows = []
    for t, r in top:
        settle_date = nth_trading_day_after(today, 5, calendar_name="XTAI")
        new_rows.append(
            {
                "ticker": t,
                "pred": r["pred"],
                "price_at_run": r["price"],
                "sup": r["sup"],
                "res": r["res"],
                "settle_date": settle_date,
                "settle_close": pd.NA,
                "realized_return": pd.NA,
                "hit": pd.NA,
            }
        )
    hist = append_today_predictions(hist, today, new_rows)
    _write_history(hist)

    # 4) 組 Discord 訊息
    msg = f"📈 台股收盤 AI 分析（{today}）\n\n"
    msg += "🔥 預估 5 日報酬 Top 5\n\n"
    for t, r in top:
        msg += f"{t}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']}` (Pivot 支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    if settle_msg:
        msg += "\n\n" + settle_msg

    msg += "\n💡 AI 為機率模型，僅供研究參考"

    _post(msg[:1900])


def _post(content: str) -> None:
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": content}, timeout=15)
        except Exception as e:
            print(f"⚠️ Discord 發送失敗: {e}")
            print(content)
    else:
        print(content)


if __name__ == "__main__":
    if pre_check():
        run()
