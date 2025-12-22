import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import os
from xgboost import XGBRegressor
import warnings

# 忽略不必要的警告訊息
warnings.filterwarnings("ignore")

# 讀取 GitHub Secret (請確保在 GitHub Repo 設定中已加入此 Secret)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ====== 參數設定區 ======
YEARS = 5              # 增加到 5 年數據，AI 訓練更準確
TOP_PICK = 5           # 排行榜取前 5 名
MIN_VOLUME_SHARES = 1000000  # 門檻設為 1000 張 (1,000,000 股)
# 核心關注清單
MUST_WATCH = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW", "00991A.TW"] 

def get_tw_stock_list():
    """從證交所抓取台股上市代號"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            # 篩選普通股(ES)與受益憑證(CE)
            if cfi.startswith('ES') or cfi.startswith('CE'):
                code = str(row['有價證券代號及名稱']).split('\u3000')[0]
                if len(code) == 4 or (len(code) == 5 and code.endswith('A')): # 處理如 00991A
                    symbols.append(code + ".TW")
        # 回傳前 150 檔熱門加上必看清單
        return list(set(symbols[:150] + MUST_WATCH))
    except Exception as e:
        print(f"無法取得股票列表: {e}")
        return MUST_WATCH

def compute_features(df):
    """計算技術指標特徵"""
    df = df.copy()
    # 1. 動能指標 (Momentum)
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    
    # 2. 強弱指標 (RSI)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    
    # 3. 量能比例 (Volume Ratio)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    
    # 4. 波動率 (Volatility) - 使用 ATR 簡化版
    df["range"] = df["High"] - df["Low"]
    df["volatility"] = df["range"].rolling(20).mean() / df["Close"]
    
    # 5. 移動平均線偏離度 (Bias)
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / df["ma20"]
    
    return df

def send_to_discord(content):
    if DISCORD_WEBHOOK_URL and content.strip():
        payload = {"content": content}
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
        except Exception as e:
            print(f"Discord 發送失敗: {e}")

def run():
    if not DISCORD_WEBHOOK_URL:
        print("錯誤: 未設定 DISCORD_WEBHOOK_URL")
        return

    symbols = get_tw_stock_list()
    scoring = []
    must_watch_details = [] 
    # 定義特徵清單
    feature_cols = ["mom20", "mom60", "rsi", "vol_ratio", "volatility", "bias"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{YEARS}y")
            
            if len(df) < 100:
                continue 
            
            df = compute_features(df)
            # 預測目標：未來 5 天的報酬率
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            if full_data.empty:
                continue

            # 機器學習：XGBoost 訓練
            model = XGBRegressor(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05, 
                random_state=42,
                n_jobs=-1 # 使用所有 CPU 核心加速
            )
            model.fit(full_data[feature_cols], full_data["future_return"])
            
            # 預測最新一筆資料
            latest_features = df[feature_cols].iloc[-1:].values
            pred = model.predict(latest_features)[0]
            
            # 記錄核心關注股
            if sym in MUST_WATCH:
                must_watch_details.append({
                    "sym": sym, 
                    "pred": pred, 
                    "price": df["Close"].iloc[-1],
                    "sup": df.tail(20)['Low'].min(), 
                    "res": df.tail(20)['High'].max()
                })
            
            # 過濾成交量後加入排行榜候選
            if df["Volume"].tail(10).mean() >= MIN_VOLUME_SHARES:
                scoring.append((sym, pred))
                
        except Exception as e:
            print(f"處理 {sym} 時發生錯誤: {e}")
            continue

    # 1. 整理發送排行榜
    now_tw = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
    top_picks = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    if top_picks:
        report = f"🇹🇼 **台股 AI 預測排行榜** ({now_tw})\n"
        report += "*(預測未來 5 日累計報酬)*\n━━━━━━━━━━━━━━━━━━\n"
        for i, (s, p) in enumerate(top_picks):
            emoji = ['🥇','🥈','🥉','📈','📈'][i]
            report += f"{emoji} **{s}**: `+{p:.2%}`\n"
        send_to_discord(report)

    # 2. 發送核心標的深度報告
    if must_watch_details:
        watch_msg = "🔍 **重點標的監控報告**\n━━━━━━━━━━━━━━━━━━"
        send_to_discord(watch_msg)
        for item in must_watch_details:
            status = "🚀" if item['pred'] > 0.02 else ("⚖️" if item['pred'] < -0.02 else "💎")
            msg = f"{status} **{item['sym']}**\n"
            msg += f"  - 預估報酬: `{item['pred']:+.2%}`\n"
            msg += f"  - 現價: `{item['price']:.2f}`\n"
            msg += f"  - 區間: `(支撐 {item['sup']:.1f} / 壓力 {item['res']:.1f})`"
            send_to_discord(msg)

if __name__ == "__main__":
    run()
