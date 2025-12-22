import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings
import os

warnings.filterwarnings("ignore")

# 1. 修正變數名稱以對應您的 GitHub Secret
DISCORD_WEBHOOK_URL = os.getenv("NEWS_WEBHOOK_URL")

# ====== 設定區 ======
YEARS = 3
TOP_PICK = 5
MIN_VOLUME = 500000 
MUST_WATCH = ["2330.TW", "2317.TW", "2454.TW", "0050.TW"] 

# 抓取清單邏輯
def get_combined_list():
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
                symbols.append(code + ".TW")
        # 掃描前 100 檔熱門股確保雲端執行速度，並加入必看清單
        return list(set(symbols[:100] + MUST_WATCH))
    except:
        return MUST_WATCH

# 獲取深度資訊
def get_extra_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # 1. 新聞 (維持簡短格式避免字數爆炸)
        news = ticker.news[:2]
        news_text = "\n".join([f"  - {n.get('title')}" for n in news]) if news else "  (無近期新聞)"
        # 2. 支撐壓力
        hist = ticker.history(period="20d")
        resistance = hist['High'].max()
        support = hist['Low'].min()
        target = ticker.info.get('targetMeanPrice', 'N/A')
        return news_text, support, resistance, target
    except:
        return "  (獲取失敗)", 0, 0, "N/A"

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
    
    symbols = get_combined_list()
    # 逐一抓取數據，對雲端環境較穩定
    scoring = []
    must_watch_details = []
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    print(f"📡 開始 AI 掃描 {len(symbols)} 檔標的...")
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 250: continue
            
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            pred = model.predict(df[features].iloc[-1:])[0]
            
            if sym in MUST_WATCH:
                news, sup, res, target = get_extra_info(sym)
                must_watch_details.append({
                    "sym": sym, "pred": pred, "price": df["Close"].iloc[-1],
                    "news": news, "sup": sup, "res": res, "target": target
                })
            
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except: continue

    # 建立第一段訊息：排行榜
    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    top_msg = f"🇹🇼 **台股 AI 掃描報告** ({today})\n"
    top_msg += "━━━━━━━━━━━━━━━━━━\n"
    top_msg += "🏆 **未來 5 日漲幅預測 Top 5**\n"
    for i, (s, p) in enumerate(scoring):
        top_msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    send_to_discord(top_msg)

    # 建立第二段訊息：重點深度追蹤 (分開傳送避免爆字數)
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.01 else "💎"
        detail_msg = f"{status} **{item['sym']}** 深度追蹤\n"
        detail_msg += f"  - 預測報酬: `{item['pred']:+.2%}`\n"
        detail_msg += f"  - 現價: {item['price']:.1f} (支撐: {item['sup']:.1f} / 壓力: `{item['res']:.1f}`)\n"
        detail_msg += f"  - 法人目標價: `{item['target']}`\n"
        detail_msg += f"  - 最新消息:\n{item['news']}\n"
        send_to_discord(detail_msg)

if __name__ == "__main__":
    run()
