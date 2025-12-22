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
MIN_VOLUME = 500000  # 門檻：20日平均成交量需大於 500 張 (500,000 股)

# ====== 1. 抓取全市場清單 (股票 + ETF) ======
def get_combined_list():
    print("🔍 正在獲取台股全市場 (股票+ETF) 掃描清單...")
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            # ES = 普通股, CE = ETF/受益憑證
            if cfi.startswith('ES') or cfi.startswith('CE'):
                item = row['有價證券代號及名稱']
                code = item.split('\u3000')[0]
                symbols.append(code + ".TW")
        
        # 掃描前 500 檔標的，確保涵蓋所有主要股票與 ETF
        return list(set(symbols[:500]))
    except Exception as e:
        print(f"❌ 抓取失敗: {e}，改用保底清單")
        return ["0050.TW", "2330.TW", "00919.TW", "2317.TW", "0056.TW", "2454.TW"]

# ====== 2. 技術指標計算 ======
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

# ====== 3. 推送 Discord ======
def send_discord(scoring, total_analyzed):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if not DISCORD_WEBHOOK_URL:
        print("❌ 找不到 Webhook 網址")
        return
    
    msg = f"🌟 **台股全市場 AI 強勢股預報** ({today})\n"
    msg += f"📊 已分析 `{total_analyzed}` 檔高流動性標的 (股票+ETF)\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"

    if not scoring:
        msg += "今日經流動性過濾後，無看漲標的。"
    else:
        total_score = sum([max(0, x[1]) for x in scoring])
        for i, (sym, score) in enumerate(scoring):
            medal = ["🥇", "🥈", "🥉", "📈", "📈"][i]
            weight = (score / total_score) * 100 if total_score > 0 else (100 / len(scoring))
            msg += f"{medal} **{sym}**\n"
            msg += f"    ┣ 預估 5 日漲幅: `+{score:.2%}`\n"
            msg += f"    ┗ 建議權重: `{weight:.1f}%`\n"
    
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "💡 *註：結合技術面指標與 XGBoost 預測。分析範圍含個股與 ETF。*"

    payload = {"content": msg}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print("✅ 整合版預測結果已發送")

# ====== 4. 主程式流程 ======
def run():
    symbols = get_combined_list()
    print(f"📥 下載資料中 (共 {len(symbols)} 檔)...")
    data = yf.download(symbols, period=f"{YEARS}y", progress=False)
    
    scoring = []
    analyzed_count = 0
    features_list = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in symbols:
        try:
            df = data.xs(sym, axis=1, level=1).dropna(how='all') if len(symbols) > 1 else data.dropna(how='all')
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
            
            if prediction > 0.003: # 預估漲幅大於 0.3%
                scoring.append((sym, prediction))
        except:
            continue

    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    send_discord(scoring, analyzed_count)

if __name__ == "__main__":
    run()
