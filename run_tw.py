from utils.market_calendar import is_market_open
from utils.safe_yfinance import safe_yf_download

def pre_check():
    if not is_market_open("TW"):
        print("📌 台股未開盤")
        return False
    return True

import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

FIXED = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]

def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    return round(2*p - h, 1), round(2*p - l, 1)

def get_top300_by_volume():
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    df = pd.read_html(requests.get(url, timeout=10).text)[0]
    df.columns = df.iloc[0]
    codes = df.iloc[1:]["有價證券代號及名稱"].str.split("　").str[0]
    tickers = [f"{c}.TW" for c in codes[codes.str.len() == 4]]

    vol_data = safe_yf_download(tickers, period="1mo", max_chunk=60)
    avg_vol = {
        t: v["Volume"].tail(20).mean()
        for t, v in vol_data.items()
        if "Volume" in v
    }

    return sorted(avg_vol, key=avg_vol.get, reverse=True)[:300]

def run():
    universe = list(dict.fromkeys(FIXED + get_top300_by_volume()))
    data = safe_yf_download(universe, period="2y", max_chunk=60)

    feats = ["mom20", "bias", "vol_ratio"]
    results = {}

    for s, df in data.items():
        if len(df) < 160:
            continue

        df["mom20"] = df["Close"].pct_change(20)
        df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["target"] = df["Close"].shift(-5) / df["Close"] - 1

        train = df.iloc[:-5].dropna()
        if len(train) < 80:
            continue

        model = XGBRegressor(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(train[feats], train["target"])

        pred = float(model.predict(df[feats].iloc[-1:])[0])
        sup, res = calc_pivot(df)

        results[s] = {
            "pred": pred,
            "price": round(df["Close"].iloc[-1], 2),
            "sup": sup,
            "res": res
        }

    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    horses = {k: v for k, v in results.items() if k not in FIXED and v["pred"] > 0}
    top5 = sorted(horses, key=lambda x: horses[x]["pred"], reverse=True)[:5]

    msg += "🏆 **AI 海選 Top 5 (潛力股)**\n"
    for i, s in enumerate(top5):
        r = results[s]
        msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n💎 **指定權值股監控 (固定顯示)**\n"
    for s in FIXED:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n🏁 台股 5 日回測結算報告\n\n💡 AI 為機率模型，僅供研究參考"

    if WEBHOOK_URL:
        import requests
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]})
    else:
        print(msg)

if __name__ == "__main__":
    if pre_check():
        run()
