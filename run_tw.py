import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def compute_features(df):
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    df["rsi"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() / ((-df["Close"].diff().clip(upper=0)).rolling(14).mean() + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    return df

def run():
    if not WEBHOOK_URL: return
    # 這裡放台股海選名單 (範例)
    symbols = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW", "1587.TW", "1727.TW", "1519.TW", "2337.TW", "2349.TW"]
    must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    
    all_results = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5y")
            if len(df) < 60: continue
            sup, res = df['Low'].tail(20).min(), df['High'].tail(20).max()
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            model = XGBRegressor(n_estimators=50, max_depth=3)
            model.fit(train[["mom20", "rsi", "vol_ratio", "bias"]], train["future_return"])
            pred = model.predict(df[["mom20", "rsi", "vol_ratio", "bias"]].iloc[-1:])[0]
            all_results[sym] = {"pred": pred, "price": df["Close"].iloc[-1], "sup": sup, "res": res}
        except: continue

    # 格式化輸出
    report = "━━━━━━━━━━━━━━━━━━\n"
    report += "300 股票前 5 的未來預估\n"
    top_5 = sorted([s for s in all_results if s not in must_watch], key=lambda x: all_results[x]['pred'], reverse=True)[:5]
    for s in top_5:
        item = all_results[s]
        report += f"{s}: 預估 {item['pred']:+.2%}\n   └ 現價: {item['price']:.1f} (支撐: {item['sup']:.1f} / 壓力: {item['res']:.1f})\n"
    
    report += "指定監控標的未來預估\n"
    for s in must_watch:
        if s in all_results:
            item = all_results[s]
            report += f"{s}: 預估 {item['pred']:+.2%}\n   └ 現價: {item['price']:.1f} (支撐: {item['sup']:.1f} / 壓力: {item['res']:.1f})\n"
    
    report += "註：預估值為 AI 對未來 5 個交易日後的走勢判斷。"
    requests.post(WEBHOOK_URL, json={"content": report})

if __name__ == "__main__":
    run()
