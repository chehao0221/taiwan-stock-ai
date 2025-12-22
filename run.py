import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from xgboost import XGBRegressor
import warnings
import os

warnings.filterwarnings("ignore")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ====== 設定區 ======
YEARS = 3
TOP_PICK = 5
MIN_VOLUME = 500000 
# 這裡定義您的必看名單
MUST_WATCH = ["2330.TW", "2317.TW", "00919.TW", "0050.TW", "00991A.TW"] 

# ====== 1. 抓取清單 ======
def get_combined_list():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        symbols = []
        for index, row in df.iterrows():
            cfi = str(row['CFICode'])
            if cfi.startswith('ES') or cfi.startswith('CE'):
                code = row['有價證券代號及名稱'].split('\u3000')[0]
                symbols.append(code + ".TW")
        # 結合必看名單與前500檔
        return list(set(symbols[:500] + MUST_WATCH))
    except:
        return MUST_WATCH

# ====== 2. 技術指標 ======
def compute_features(df):
    df["mom20"] = df["Close"].pct_change(20)
    df["mom60"] = df["Close"].pct_change(60)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + up / (down + 1e-9)))
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    return df

# ====== 3. 主流程 ======
def run():
    symbols = get_combined_list()
    data = yf.download(symbols, period=f"{YEARS}y", progress=False)
    
    scoring = [] # 存儲 Top Pick
    must_watch_results = [] # 存儲必看名單結果
    analyzed_count = 0
    features = ["mom20", "mom60", "rsi", "vol_ratio", "volatility"]

    for sym in symbols:
        try:
            df = data.xs(sym, axis=1, level=1).dropna(how='all') if len(symbols) > 1 else data.dropna(how='all')
            if len(df) < 250: continue
            
            analyzed_count += 1
            df = compute_features(df)
            df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            full_data = df.dropna()
            
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(full_data[features], full_data["future_return"])
            
            pred = model.predict(df[features].iloc[-1:])[0]
            
            res_item = (sym, pred)
            # 如果在必看名單中，單獨記錄
            if sym in MUST_WATCH:
                must_watch_results.append(res_item)
            # 如果流動性達標，加入全市場排名
            if df["Volume"].tail(20).mean() >= MIN_VOLUME:
                scoring.append(res_item)
        except: continue

    # 排序
    scoring = sorted(scoring, key=lambda x: x[1], reverse=True)[:TOP_PICK]
    
    # 發送 Discord
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    msg = f"🌟 **AI 全市場掃描報表** ({today})\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 **未來 5 日看漲排行榜**\n"
    for i, (s, p) in enumerate(scoring):
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: `+{p:.2%}`\n"
    
    msg += "\n🔍 **指定標的追蹤**\n"
    for s, p in must_watch_results:
        status = "🔥" if p > 0.01 else "💎" if p > 0 else "☁️"
        msg += f"{status} **{s}**: `+{p:.2%}`\n"
    
    msg += "━━━━━━━━━━━━━━━━━━"
    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    print("✅ 報表已成功發送")

if __name__ == "__main__":
    run()
