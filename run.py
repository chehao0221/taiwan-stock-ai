import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings
import os

# 忽略警告訊息
warnings.filterwarnings("ignore")

# 從 GitHub Secrets 讀取 Webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

YEARS = 3
TOP_PICK = 5
MIN_VOLUME = 500000  # 過濾條件：20日平均成交量需大於 500 張

def get_taiwan_list():
    print("🔍 正在從證交所獲取清單...")
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        symbols = []
        for index, row in df.iterrows():
            item = row['有價證券代號及名稱']
            if not isinstance(item, str): continue
            code = item.split('\u3000')[0]
            cfi = str(row['CFICode'])
            if cfi.startswith('ES') or cfi.startswith('CE'):
                symbols.append(code + ".TW")
        return list(set(symbols[:300])) # 掃描前 300 檔標的
    except Exception as e:
        print(f"❌ 抓取失敗: {e}")
        return ["0050.TW", "2330.TW", "2317.TW", "2454.TW", "0056.TW"]

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_features(df):
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    df["rsi"] = compute_rsi(df["Close"])
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

def send_discord(scoring, total_analyzed):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if not DISCORD_WEBHOOK_URL:
        print("❌ 找不到 DISCORD_WEBHOOK_URL 變數，取消發送。")
        return
    
    if not scoring:
        msg = f"⚠️ **台股 AI 選股日報 ({today})**\n今日經 AI 篩選後，無看漲標的。"
    else:
        msg = f"🚀 **台股 AI 選股日報** ({today})\n"
        msg += f"📊 已分析 `{total_analyzed}` 檔高流動性標的\n"
        msg += "━━━━━━━━━━━━━━━\n"
        total_score = sum([max(0, x[1]) for x in scoring])
        for sym, score in scoring:
            weight = (score / total_score) * 100 if total_score > 0 else (100 / len(scoring))
            msg += f"📌 **{sym}**\n"
            msg += f"    ┣ 預期 5 日報酬: `+{score:.2%}`\n"
            msg += f"    ┗ 建議權重: `{weight:.1f}%`\n"
        msg += "━━━━━━━━━━━━━━━\n"
        msg += "⚠️ *註：僅分析成交量 > 500張標的。*"

    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    print("✅ Discord 通知已發送")

def run():
    raw_symbols = get_taiwan_list()
    print(f"📥 下載資料中 (共 {len(raw_symbols)} 檔)...")
    data = yf.download(raw_symbols, period=f"{YEARS}y", progress=False)
    
    scoring = []
    analyzed_count = 0
    features_list = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in raw_symbols:
        try:
            # 處理 yfinance 多股票下載後的欄位結構
            df = data.xs(sym, axis=1, level=1).dropna(how='all') if len(raw_symbols) > 1 else data.dropna(how='all')
            if len(df) < 250: continue
            
            # 流動性檢查
            if df["Volume"].tail(20).mean() < MIN_VOLUME: continue
            
            analyzed_count += 1
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            
            full_data = df.dropna()
            if full_data.empty: continue
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features_list], full_data["future_return"])

            last_features = df[features_list].iloc[-1:].values
            prediction = model.predict(last_features)[0]
            if prediction > 0.005:
                scoring.append((sym, prediction))
        except:
            continue

    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    send_discord(scoring, analyzed_count)

if __name__ == "__main__":
    run()
