import yfinance as yf
import requests
import os
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def run():
    stocks = ["2330.TW", "2317.TW", "0050.TW"]
    for sym in stocks:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2d")
            if df.empty: continue
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            trend = "▲" if change > 0 else "▼" if change < 0 else "—"
            
            payload = {
                "embeds": [{
                    "title": f"🇹🇼 台股快訊: {sym}",
                    "description": f"**價格:** `NT$ {price:.2f}`\n**變動:** `{trend} {change:+.2f}%`",
                    "color": 0x36393f # 中性灰色
                }]
            }
            requests.post(WEBHOOK_URL, json=payload)
        except: pass

if __name__ == "__main__":
    run()
