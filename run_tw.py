import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定與路徑
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 支撐/壓力位計算 (Pivot Points)
# =========================
def calc_support_resistance(df):
    try:
        # 取最近 20 天的高低點與收盤價
        recent = df.iloc[-20:]
        high = recent['High'].max()
        low = recent['Low'].min()
        close = recent['Close'].iloc[-1]
        
        # 簡單計算 (可視為近期波動區間)
        pivot = (high + low + close) / 3
        resistance = (2 * pivot) - low
        support = (2 * pivot) - high
        return round(support, 1), round(resistance, 1)
    except:
        return 0, 0

# =========================
# 自動抓取台股前 300 檔
# =========================
def get_tw_300_pool():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df["code"] = df["有價證券代號及名稱"].str.split("　").str[0]
        # 過濾四位數代碼 (上市個股)
        stocks = df[df["code"].str.len() == 4]["code"].tolist()
        return [f"{s}.TW" for s in stocks[:300]]
    except:
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "0050.TW"]

# =========================
# 主程序
# =========================
def run():
    # 1. 準備股票池
    fixed_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    pool_stocks = get_tw_300_pool()
    all_watch = list(dict.fromkeys(fixed_watch + pool_stocks))
    
    print(f"🚀 開始海選 {len(all_watch)} 檔標的...")
    
    # 2. 下載數據
    all_data = yf.download(all_watch, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    idx_df = yf.download("^TWII", period="1y", auto_adjust=True, progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]
    
    # 3. 逐股分析
    for s in all_watch:
        try:
            df = all_data[s].dropna()
            if len(df) < 50: continue
            
            # 特徵計算
            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            
            train = df.dropna().iloc[-250:]
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_support_resistance(df)
            
            results[s] = {
                "p": pred,
                "c": float(df["Close"].iloc[-1]),
                "sup": sup,
                "res": res
            }
        except: continue

    # 4. 組合訊息 (比照您截圖的排版)
    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"
    
    # --- 區塊一：海選 Top 5 (排除固定監控的權值股) ---
    msg += "🏆 **AI 海選 Top 5 (潛力黑馬)**\n"
    horses = {k: v for k, v in results.items() if k not in fixed_watch}
    top_5 = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    
    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    for i, s in enumerate(top_5):
        r = results[s]
        msg += f"{medals[i]} **{s}**: 預估 `{r['p']:+.2%}`\n"
        msg += f" └ 現價: `{r['c']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"
        
    msg += "\n🔍 **指定權值股監控 (固定顯示)**\n"
    for s in fixed_watch:
        if s in results:
            r = results[s]
            msg += f"**{s}**: 預估 `{r['p']:+.2%}`\n"
            msg += f" └ 現價: `{r['c']}`\n"

    msg += "\n💡 AI 為機率模型，僅供研究參考"

    # 5. 發送與存檔 (存檔供下週對帳使用)
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else:
        print(msg)
        
    # 儲存預測資料 (結算用)
    new_entries = [{"date": datetime.now().date(), "symbol": s, "pred_p": results[s]['c'], 
                    "pred_ret": results[s]['p'], "settled": "False"} for s in (top_5 + fixed_watch) if s in results]
    pd.DataFrame(new_entries).to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)

if __name__ == "__main__":
    run()
