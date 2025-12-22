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

# 從環境變數讀取 Webhook (GitHub Secrets)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

YEARS = 3
TOP_PICK = 5
MIN_VOLUME = 500000  # 過濾條件：20日平均成交量需大於 500 張 (500,000 股)

# ====== 自動抓取清單與流動性過濾 ======
def get_taiwan_list():
    print("🔍 正在獲取證交所最新清單...")
    try:
        # 證交所上市證券清單
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
            
            # 抓取普通股 (ES) 與 ETF (CE)
            if cfi.startswith('ES') or cfi.startswith('CE'):
                symbols.append(code + ".TW")
        
        # 先取前 300 檔進行流動性掃描 (涵蓋多數大標的)
        return list(set(symbols[:300]))

    except Exception as e:
        print(f"❌ 抓取失敗: {e}，改用保底清單")
        return ["0050.TW", "0056.TW", "2330.TW", "2317.TW", "2454.TW"]

# ====== 技術指標計算 ======
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

# ====== 推送 Discord ======
def send_discord(scoring, total_analyzed):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not scoring:
        msg = f"⚠️ **台股 AI 選股日報 ({today})**\n今日經流動性過濾與 AI 篩選後，無看漲標的。"
    else:
        msg = f"🚀 **台股 AI 選股日報** ({today})\n"
        msg += f"📊 已過濾流動性並分析 `{total_analyzed}` 檔高質量標的\n"
        msg += "━━━━━━━━━━━━━━━\n"

        total_score = sum([x[1] for x in scoring])
        for sym, score in scoring:
            weight = (score / total_score) * 100 if total_score > 0 else (100 / len(scoring))
            msg += f"📌 **{sym}**\n"
            msg += f"    ┣ 預期報酬: `+{score:.2%}`\n"
            msg += f"    ┗ 權重建議: `{weight:.1f}%`\n"
        
        msg += "━━━━━━━━━━━━━━━\n"
        msg += "⚠️ *註：僅分析日均成交量 > 500張之標的，投資請自負盈虧。*"

    payload = {"content": msg}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print(msg)

# ====== 主流程 ======
def run():
    raw_symbols = get_taiwan_list()
    print(f"📥 下載資料中 (共 {len(raw_symbols)} 檔)...")
    data = yf.download(raw_symbols, period=f"{YEARS}y", group_by='ticker', progress=False)
    
    scoring = []
    analyzed_count = 0
    features_list = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in raw_symbols:
        try:
            df = data[sym].copy().dropna(how='all')
            
            # --- 流動性過濾 ---
            # 檢查最近 20 天平均成交量是否達標
            avg_vol = df["Volume"].tail(20).mean()
            if avg_vol < MIN_VOLUME:
                continue
            
            if len(df) < 250: continue
            
            analyzed_count += 1
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            
            full_data = df.dropna()
            if full_data.empty: continue
            
            X = full_data[features_list]
            y = full_data["future_return"]

            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(X, y)

            last_features = df[features_list].iloc[-1:].values
            prediction = model.predict(last_features)[0]

            # 門檻：預估漲幅需大於 0.5%
            if prediction > 0.005:
                scoring.append((sym, prediction))

        except:
            continue

    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    send_discord(scoring, analyzed_count)

if __name__ == "__main__":
    run()
