import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def get_tw_300_pool():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df["code"] = df["有價證券代號及名稱"].str.split("　").str[0]
        stocks = df[df["code"].str.len() == 4]["code"].tolist()
        return [f"{s}.TW" for s in stocks[:300]]
    except:
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "0050.TW"]

def get_market_context():
    try:
        idx = yf.download("^TWII", period="1y", auto_adjust=True, progress=False)
        if idx.empty: return True, 0, 0, None
        idx["ma60"] = idx["Close"].rolling(60).mean()
        curr_p = float(idx["Close"].iloc[-1])
        ma60_p = float(idx["ma60"].iloc[-1])
        return (curr_p > ma60_p), curr_p, ma60_p, idx
    except:
        return True, 0, 0, None

def compute_features(df, market_df=None):
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    
    # ATR 波動率指標
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    df["atr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # 相對強度 (RS Index)
    if market_df is not None:
        df["rs_index"] = df["Close"].pct_change(20) - market_df["Close"].pct_change(20)
    else:
        df["rs_index"] = 0
    
    df["avg_amount"] = (df["Close"] * df["Volume"]).rolling(5).mean()
    df["sup"] = df["Low"].rolling(60).min()
    df["res"] = df["High"].rolling(60).max()
    return df

def audit_and_save(results, top_keys):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])
    
    today = datetime.now().date()
    new_rows = [{"date": today, "symbol": s, "pred_p": results[s]["c"], "pred_ret": results[s]["p"], "settled": False} for s in top_keys]
    hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["date", "symbol"], keep="last")
    hist.to_csv(HISTORY_FILE, index=False)
    return ""

def safe_post(msg: str):
    if not WEBHOOK_URL:
        print(msg); return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=15)
    except: pass

def run():
    is_bull, mkt_p, mkt_ma, mkt_df = get_market_context()
    must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW"]
    watch = list(set(must_watch + get_tw_300_pool()))
    
    print(f"台股分析啟動... (大盤:{'多' if is_bull else '空'})")
    all_data = yf.download(watch, period="5y", group_by="ticker", auto_adjust=True, progress=False)
    
    feats = ["mom20", "rsi", "bias", "vol_ratio", "rs_index"]
    results = {}
    MIN_AMOUNT = 100_000_000 # 1億台幣

    for s in watch:
        try:
            df = all_data[s].dropna()
            if len(df) < 120: continue
            df = compute_features(df, market_df=mkt_df)
            last = df.iloc[-1]
            if last["avg_amount"] < MIN_AMOUNT: continue

            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            pred = float(np.clip(model.predict(train[feats].iloc[-1:])[0], -0.15, 0.15))
            if not is_bull: pred *= 0.5
            if last["atr"] > (df["atr"].mean() * 1.5): pred *= 0.8

            results[s] = {"p": pred, "c": float(last["Close"]), "rs": float(last["rs_index"])}
        except: continue

    horses = {k: v for k, v in results.items() if k not in must_watch}
    top_keys = sorted(horses, key=lambda x: horses[x]['p'], reverse=True)[:5]
    audit_and_save(results, top_keys)

    msg = f"🇹🇼 **台股 AI 進階預報 ({datetime.now():%m/%d})**\n"
    msg += f"{'📈 多頭環境' if is_bull else '⚠️ 空頭警示 (預測已降權)'} | 指數: {mkt_p:.0f}\n"
    msg += "----------------------------------\n"
    for i, s in enumerate(top_keys):
        r = results[s]
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}** 預估 `{r['p']:+.2%}` (RS:{'強' if r['rs']>0 else '弱'})\n"
    
    safe_post(msg[:1900])

if __name__ == "__main__":
    run()
