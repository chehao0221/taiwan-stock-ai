import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# 讀取 GitHub Secret
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 設定區 ======
YEARS = 2           
TOP_PICK = 5        
MIN_VOLUME = 500000 
MUST_WATCH = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"] 

def get_tw_stock_list():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=15)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            if cfi.startswith('ES') or cfi.startswith('CE'):
                code = row['有價證券代號及名稱'].split('\u3000')[0]
                if len(code) <= 5:
                    symbols.append(code + ".TW")
        return list(set(symbols[:100] + MUST_WATCH))
    except:
        return MUST_WATCH

def compute_features(df):
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

def send_to_discord(content):
    if DISCORD_WEBHOOK_URL and content.strip():
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)

def run():
    if not DISCORD_WEBHOOK_URL: return
    symbols = get_tw_stock_list()
    scoring = []
    must_watch_details = [] 
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 100: continue 
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            if full_data.empty: continue

            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            pred = model.predict(df[features].iloc[-1:])[0]
            
            if sym in MUST_WATCH:
                must_watch_details.append({
                    "sym": sym, "pred": pred, "price": df["Close"].iloc[-1],
                    "sup": df.tail(20)['Low'].min(), "res": df.tail(20)['High'].max()
                })
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except: continue

    # 1. 排行榜
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    report = f"🇹🇼 **台股 AI 預測報告** ({today})\n━━━━━━━━━━━━━━━━━━\n"
    for i, (s, p) in enumerate(top_picks):
        report += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    send_to_discord(report)

    # 2. 重點標的分段發送 (純文字版)
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.01 else "💎"
        msg = f"{status} **{item['sym']}** 深度報告\n"
        msg += f"  - 預測報酬: `{item['pred']:+.2%}`\n"
        msg += f"  - 現價: {item['price']:.1f} (支撐: {item['sup']:.1f} / 壓力: {item['res']:.1f})"
        send_to_discord(msg)

if __name__ == "__main__":
    run()
