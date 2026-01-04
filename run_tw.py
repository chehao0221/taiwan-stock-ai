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

# 權值股（保留你原本那個概念）
FIXED = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]

# -----------------------------
# Helpers
# -----------------------------
def _now_tw() -> datetime:
    return datetime.now(ZoneInfo("Asia/Taipei"))


def _today_tw() -> str:
    return _now_tw().strftime("%Y-%m-%d")


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
    回傳 start_date 之後第 n 個交易日（不含 start_date 當天）
    用交易所日曆避開週末/假日
    """
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(
        start_date=start_date,
        end_date=pd.Timestamp(start_date) + pd.Timedelta(days=60)  # 避免遇到長假不夠用
    )
    days = schedule.index.strftime("%Y-%m-%d").tolist()

    if start_date in days:
        pos = days.index(start_date)
        target = pos + n
    else:
        # 理論上不會發生（因為 pre_check 已擋非交易日）
        target = n - 1

    if target >= len(days):
        raise RuntimeError("交易日曆不足，請加大 end_date 範圍")
    return days[target]


def _read_history() -> pd.DataFrame:
    """
    讀取 tw_history.csv（新格式）
    若不存在：建立空表（包含完整欄位）
    """
    cols = [
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

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(HISTORY_FILE)

    # 補齊缺欄位（就算你之後又換格式也不會炸）
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["status"] = df["status"].fillna("pending")
    df["run_date"] = df["run_date"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["settle_date"] = df["settle_date"].fillna("").astype(str)
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
    先抓「上市股票清單」→ 用 yfinance 拉 1M 資料計算近 20 日平均量 → 取前 300
    這一步最容易被 yfinance 429，所以做「當日快取」：同一天內重跑不會再抓一輪
    """
    cached = _load_top300_cache(today)
    if cached:
        return cached

    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    html = requests.get(url, timeout=15).text
    table = pd.read_html(html)[0]
    table.columns = table.iloc[0]
    codes = table.iloc[1:][table.columns[0]].astype(str).str.split(" ").str[0]
    tickers = [f"{c}.TW" for c in codes if c.isdigit() and len(c) == 4]

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
    結算 tw_history.csv 裡：
    - status == pending
    - settle_date <= today
    並輸出你原本想要的「預估 vs 實際 + ✅/❌」格式
    """
    hist = _read_history()
    if hist.empty:
        return hist, ""

    if "settle_date" not in hist.columns:
        return hist, ""

    # settle_date 全空就不用結算
    if hist["settle_date"].astype(str).str.len().eq(0).all():
        return hist, ""

    pending = hist[
        (hist["status"].astype(str) == "pending")
        & (hist["settle_date"].astype(str) <= today)
        & (hist["settle_date"].astype(str).str.len() > 0)
    ]

    if pending.empty:
        return hist, ""

    tickers = sorted(pending["ticker"].astype(str).unique().tolist())
    data = safe_yf_download(tickers, period="3mo", max_chunk=60)

    settled_lines = []
    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in pending.iterrows():
        t = str(row["ticker"])
        settle_date = str(row["settle_date"])

        d = data.get(t)
        if d is None or d.empty:
            continue

        d2 = d.copy()
        d2.index = pd.to_datetime(d2.index).strftime("%Y-%m-%d")

        if settle_date not in d2.index:
            continue

        settle_close = float(d2.loc[settle_date, "Close"])
        price_at_run = float(row["price_at_run"])
        rr = (settle_close / price_at_run) - 1.0

        pred = row.get("pred", pd.NA)
        try:
            pred_f = float(pred)
        except Exception:
            pred_f = None

        hit = int(rr > 0)
        mark = "✅" if hit == 1 else "❌"

        hist.at[idx, "settle_close"] = round(settle_close, 2)
        hist.at[idx, "realized_return"] = rr
        hist.at[idx, "hit"] = hit
        hist.at[idx, "status"] = "settled"
        hist.at[idx, "updated_at"] = now_str

        if pred_f is None:
            settled_lines.append(f"• {t}: 實際 {rr:+.2%} {mark}")
        else:
            settled_lines.append(f"• {t}: 預估 {pred_f:+.2%} | 實際 {rr:+.2%} {mark}")

    if not settled_lines:
        return hist, ""

    # 保持你原本的顯示方式（標題 + 點列）
    msg = "🏁 5 日回測結算報告\n" + "\n".join(settled_lines[:10])
    if len(settled_lines) > 10:
        msg += f"\n… 另外還有 {len(settled_lines) - 10} 筆已結算"

    return hist, msg


def append_today_predictions(hist: pd.DataFrame, today: str, rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return hist

    now_str = _now_tw().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame(rows)
    df_new["run_date"] = today
    df_new["status"] = "pending"
    df_new["updated_at"] = now_str

    # 避免同一天重跑重複寫入（run_date + ticker）
    if not hist.empty:
        hist["run_date"] = hist["run_date"].astype(str)
        hist["ticker"] = hist["ticker"].astype(str)
        existing = set(zip(hist["run_date"], hist["ticker"]))
        df_new = df_new[~df_new.apply(lambda r: (today, str(r["ticker"])) in existing, axis=1)]

    if df_new.empty:
        return hist

    return pd.concat([hist, df_new], ignore_index=True)


def _post(content: str) -> None:
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": content}, timeout=15)
        except Exception as e:
            print(f"⚠️ Discord 發送失敗: {e}")
            print(content)
    else:
        print(content)


# -----------------------------
# Main
# -----------------------------
def run() -> None:
    today = _today_tw()

    # 1) 先做歷史結算（到期的補上 ✅/❌）
    hist, settle_msg = settle_history(today)

    # 2) 今日預測（Top300 + 權值）
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
        _post("⚠️ 今日無可用結果（可能資料不足或抓取失敗）")
        return

    # 你原本想要「海選 Top5（黑馬）」的樣子
    top = sorted(results.items(), key=lambda kv: kv[1]["pred"], reverse=True)[:5]

    # 3) 寫入歷史（今日 Top5，並計算第 5 個交易日結算日）
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

    # 4) Discord 顯示：維持你 README 那種風格
    msg = f"📊 台股 AI 進階預測報告 ({today})\n\n"
    msg += "🏆 AI 海選 Top 5 (潛力黑馬)\n"
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    for i, (t, r) in enumerate(top):
        medal = medals[i] if i < len(medals) else "•"
        msg += f"{medal} {t}: 預估 {r['pred']:+.2%}\n"
        msg += f"   └ 現價: {r['price']} / 支撐: {r['sup']} / 壓力: {r['res']}\n"

    if settle_msg:
        msg += "\n" + settle_msg + "\n"

    msg += "\n⚠️ 免責聲明：本專案僅供研究參考，不構成任何投資建議。"

    _post(msg[:1900])


if __name__ == "__main__":
    if pre_check():
        run()
