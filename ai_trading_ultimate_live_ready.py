import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings

# 忽略警告訊息
warnings.filterwarnings("ignore")

# ====== 你的 Discord Webhook (建議不要公開) ======
import os
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


YEARS = 3
TOP_PICK = 5

# ====== 選股清單 ======
def get_taiwan_list():
    # 包含主要指標 ETF 與 高市值權值股
    etf_list = ["0050.TW", "0056.TW", "006208.TW", "00878.TW", "00940.TW"]
    large_caps = [
        "2330.TW", "2317.TW", "2454.TW", "2603.TW", "2303.TW",
        "2882.TW", "2308.TW", "1301.TW", "1216.TW", "2357.TW",
        "2382.TW", "3231.TW", "2301.TW", "2609.TW", "2615.TW"
    ]
    return list(set(etf_list + large_caps))

# ====== 技術指標計算 ======
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def compute_features(df):
    # 動能因子
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    # 強弱指標
    df["rsi"] = compute_rsi(df["Close"])
    # 量能因子
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    # 波動因子 (新增：標準差)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    
    return df

# ====== 推送 Discord ======
def send_discord(scoring):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not scoring:
        msg = f"⚠️ **台股 AI 選股日報 ({today})**\n今日模型預測無看漲標的，建議觀望。"
    else:
        msg = f"🚀 **台股 AI 選股日報** ({today})\n"
        msg += "根據過去 3 年數據與 XGBoost 模型預測未來 5 日走勢：\n"
        msg += "━━━━━━━━━━━━━━━\n"

        total_score = sum([x[1] for x in scoring])
        for sym, score in scoring:
            # 權知配置邏輯優化
            weight = (score / total_score) * 100 if total_score > 0 else (100 / len(scoring))
            msg += f"📌 **{sym}**\n"
            msg += f"    ┣ AI 預期報酬: `+{score:.2%}`\n"
            msg += f"    ┗ 建議權重: `{weight:.1f}%`\n"
        
        msg += "━━━━━━━━━━━━━━━\n"
        msg += "⚠️ *本報告僅供參考，投資前請自行評估風險。*"

    payload = {"content": msg}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        if response.status_code == 204:
            print(f"[{today}] Discord 推送成功！ ✅")
        else:
            print(f"推送失敗，狀態碼: {response.status_code}")
    except Exception as e:
        print(f"發送請求時出錯: {e}")

# ====== 主流程 ======
def run():
    symbols = get_taiwan_list()
    print(f"📥 正在抓取 {len(symbols)} 檔標的之歷史資料...")
    
    # 批次下載以提升速度
    data = yf.download(symbols, period=f"{YEARS}y", group_by='ticker', progress=False)
    
    scoring = []
    features_list = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in symbols:
        try:
            # 提取單一股票資料並清除缺失值
            df = data[sym].copy().dropna(how='all')
            if len(df) < 250: continue # 數據太少則跳過
            
            df = compute_features(df)
            
            # 目標值：未來 5 天的累積報酬率 (Shift 為負代表看向未來)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            
            # 準備訓練資料
            full_data = df.dropna()
            if full_data.empty: continue
            
            X = full_data[features_list]
            y = full_data["future_return"]

            # 建立並訓練模型
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.07,
                objective='reg:squarederror',
                random_state=42
            )
            model.fit(X, y)

            # 取得最新一天的特徵進行預測
            last_features = df[features_list].iloc[-1:].values
            prediction = model.predict(last_features)[0]

            # 只保留預測報酬為正的標的
            if prediction > 0:
                scoring.append((sym, prediction))

        except Exception as e:
            print(f"❌ 處理 {sym} 時發生錯誤: {e}")
            continue

    # 排序：取預測報酬最高的前 N 名
    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    # 發送結果
    send_discord(scoring)

if __name__ == "__main__":
    run()

