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

# ====== 1. 抓取全台 ETF 清單 ======
def get_etf_list():
    print("🔍 正在獲取全台 ETF 掃描清單...")
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            # 只抓取 CE (ETF/受益憑證)
            if cfi.startswith('CE'):
                item = row['有價證券代號及名稱']
                code = item.split('\u3000')[0]
                symbols.append(code + ".TW")
        
        # 擴大掃描範圍至 1000 檔，確保涵蓋所有 ETF
        return list(set(symbols[:1000]))
    except Exception as e:
        print(f"❌ 抓取失敗: {e}，改用主流 ETF 保底")
        return ["0050.TW", "0056.TW", "00878.TW", "00919.TW", "00929.TW", "00713.TW"]

# ====== 2. 技術指標計算 (技術面) ======
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_features(df):
    # 動能：近 20, 60 日回報
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    # 強弱：RSI
    df["rsi"] = compute_rsi(df["Close"])
    # 量價：成交量比率
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    # 風險：波動率
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

# ====== 3. 推送 Discord ======
def send_discord(scoring, total_analyzed):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if not DISCORD_WEBHOOK_URL:
        print("❌ 找不到 Webhook 網址")
        return
    
    if not scoring:
        msg = f"⚠️ **ETF AI 預測日報 ({today})**\n今日經流動性過濾後，無看漲標的。"
    else:
        msg = f"🏆 **ETF AI 預測日報：未來五日看漲 TOP 5** ({today})\n"
        msg += f"📊 已完成 `{total_analyzed}` 檔高流動性 ETF 深度分析\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"

        total_score = sum([max(0, x[1]) for x in scoring])
        for i, (sym, score) in enumerate(scoring):
            medal = ["🥇", "🥈", "🥉", "📈", "📈"][i]
            weight = (score / total_score) * 100 if total_score > 0 else (100 / len(scoring))
            msg += f"{medal} **{sym}**\n"
            msg += f"    ┣ 預估漲幅: `+{score:.2%}`\n"
            msg += f"    ┗ 權重配置: `{weight:.1f}%`\n"
        
        msg += "━━━━━━━━━━━━━━━━━━\n"
        msg += "💡 *註：本系統僅透過技術面 (量價動能) 進行 XGBoost 預測，不含消息面。投資請自負盈虧。*"

    payload = {"content": msg}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print("✅ ETF 預測結果已發送至 Discord")

# ====== 4. 主程式流程 ======
def run():
    etf_symbols = get_etf_list()
    print(f"📥 下載 ETF 歷史資料中 (共 {len(etf_symbols)} 檔)...")
    data = yf.download(etf_symbols, period="3y", progress=False)
    
    scoring = []
    analyzed_count = 0
    features_list = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in etf_symbols:
        try:
            # 取得該 ETF 數據
            df = data.xs(sym, axis=1, level=1).dropna(how='all') if len(etf_symbols) > 1 else data.dropna(how='all')
            
            if len(df) < 250: continue # 需至少有一年數據
            
            # 流動性檢查 (成交量 > 500 張)
            if df["Volume"].tail(20).mean() < MIN_VOLUME: continue
            
            analyzed_count += 1
            df = compute_features(df)
            # 目標：預測 5 天後的回報率
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            
            full_data = df.dropna()
            if full_data.empty: continue
            
            # 訓練 XGBoost 模型
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features_list], full_data["future_return"])

            # 預測最新數據
            last_features = df[features_list].iloc[-1:].values
            prediction = model.predict(last_features)[0]
            
            # 漲幅門檻 > 0.3% 才入選
            if prediction > 0.003:
                scoring.append((sym, prediction))
        except:
            continue

    # 按預期漲幅排序，取前五名
    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    send_discord(scoring, analyzed_count)

if __name__ == "__main__":
    run()
