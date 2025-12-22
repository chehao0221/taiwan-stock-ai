import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

# 忽略警告訊息
warnings.filterwarnings("ignore")

# 讀取 GitHub Secret 中的 Discord Webhook URL
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 設定區 ======
YEARS = 2           # 訓練數據年限，設為 2 年可加快雲端執行速度並減少錯誤
TOP_PICK = 5        # 排行榜顯示數量
MIN_VOLUME = 500000 # 篩選成交量大於 500 張的標的
# 必看台股清單
MUST_WATCH = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"] 

def get_tw_stock_list():
    """從證交所抓取台股上市清單"""
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=15)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            # 篩選普通股 (ES) 與 股票型 ETF (CE)
            if cfi.startswith('ES') or cfi.startswith('CE'):
                code = row['有價證券代號及名稱'].split('\u3000')[0]
                if len(code) <= 5: # 過濾權證
                    symbols.append(code + ".TW")
        # 掃描前 100 檔熱門股 + 必看清單，避免 GitHub Actions 執行超時
        return list(set(symbols[:100] + MUST_WATCH))
    except Exception as e:
        print(f"清單抓取失敗: {e}")
        return MUST_WATCH

def compute_features(df):
    """計算 AI 訓練用的技術指標"""
    df = df.copy()
    # 價格動能
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    # RSI 指標
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    # 成交量比與波動率
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

def send_to_discord(content):
    """發送訊息至 Discord，確保不超過 2000 字元限制"""
    if DISCORD_WEBHOOK_URL and content.strip():
        try:
            res = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
            print(f"📡 Discord 狀態碼: {res.status_code}")
        except Exception as e:
            print(f"發送失敗: {e}")

def run():
    if not DISCORD_WEBHOOK_URL:
        print("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL")
        return

    symbols = get_tw_stock_list()
    scoring = []
    must_watch_details = [] 
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    print(f"📡 正在掃描 {len(symbols)} 檔台股標的...")
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            # 抓取歷史數據，若數據不足則跳過以防報錯
            df = ticker.history(period=f"{YEARS}y")
            if len(df) < 100: continue 
            
            df = compute_features(df)
            # 預測目標：未來 5 日報酬率
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            if full_data.empty: continue

            # 初始化並訓練 XGBoost 模型
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            # 進行最新預測
            pred = model.predict(df[features].iloc[-1:])[0]
            
            # 取得當前價格與支撐壓力位
            curr_price = df["Close"].iloc[-1]
            hist_20 = df.tail(20)
            res = hist_20['High'].max()
            sup = hist_20['Low'].min()

            # 儲存必看標的詳細資訊
            if sym in MUST_WATCH:
                must_watch_details.append({
                    "sym": sym, "pred": pred, "price": curr_price, "sup": sup, "res": res
                })
            
            # 儲存高動能標的
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append((sym, pred))
        except Exception as e:
            print(f"⚠️ {sym} 處理出錯: {e}")
            continue

    # --- 分段發送邏輯，徹底解決 400 錯誤 ---

    # 1. 發送第一報：AI 漲幅排行榜
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    report_header = f"🇹🇼 **台股 AI 預測報告** ({today})\n━━━━━━━━━━━━━━━━━━\n"
    report_header += "🏆 **未來 5 日漲幅預測 Top 5**\n"
    for i, (s, p) in enumerate(top_picks):
        report_header += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    send_to_discord(report_header)

    # 2. 發送第二報：重點標的深度分析 (每支股票獨立一則訊息，絕不爆字數)
    for item in must_watch_details:
        status = "🚀" if item['pred'] > 0.01 else "💎"
        detail_msg = f"{status} **{item['sym']}** 深度掃描\n"
        detail_msg += f"  - 預測回報: `{item['pred']:+.2%}`\n"
        detail_msg += f"  - 現價: {item['price']:.1f} (支撐: {item['sup']:.1f} / 壓力: `{item['res']:.1f}`)\n"
        send_to_discord(detail_msg)

if __name__ == "__main__":
    run()
