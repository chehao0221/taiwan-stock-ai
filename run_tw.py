import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

# =========================
# 基本設定
# =========================
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

def get_tw_300_pool():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df['code'] = df['有價證券代號及名稱'].str.split('　').str[0]
        stocks = df[df['code'].str.len() == 4]['code'].tolist()
        return [f"{s}.TW" for s in stocks[:300]]
    except:
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW"]

def safe_post(msg: str):
    if not WEBHOOK_URL:
        print("\n--- Discord 訊息預覽 ---\n", msg)
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=15)
    except:
        pass

def compute_features(df):
    df = df.copy()
    df["r"] = df["Close"].pct_change()
    df["mom20"] = df["Close"].pct_change(20)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["sup"] = df["Low"].rolling(60).min()
    df["res"] = df["High"].rolling(60).max()
    return df

def audit_and_save(results, top_keys):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])
    
    audit_msg = ""
    today = datetime.now().date()
    deadline = today - timedelta(days=8)
    unsettled = hist[(hist["settled"] == False) & (hist["date"] <= deadline)]
    
    if not unsettled.empty:
        audit_msg = "\n🎯 **5 日預測結算對帳 (TW)**\n"
        for idx, r in unsettled.iterrows():
            try:
                p_df = yf.Ticker(r["symbol"]).history(period="5d")
                if p_df.empty: continue
                curr_p = p_df["Close"].iloc[-1]
                act_ret = (curr_p - r["pred_p"]) / r["pred_p"]
                hit = "✅" if np.sign(act_ret) == np.sign(r["pred_ret"]) else "❌"
                audit_msg += f"`{r['symbol']}` {r['pred_ret']:+.2%} ➜ {act_ret:+.2%} {hit}\n"
                hist.at[idx, "settled"] = True
            except: continue
    
    new_rows = [{"date": today, "symbol": s, "pred_p": results[s]["c"], "pred_ret": results[s]["p"], "settled": False} for s in top_keys]
    hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["date", "symbol"], keep="last")
    hist.to_csv(HISTORY_FILE, index=False)
    return audit_msg

def run():
    must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    pool = get_tw_300_pool()
    watch = list(set(must_watch + pool))
    feats = ["mom20", "rsi", "bias", "vol_ratio"]
    results = {}

    print(f"正在掃描 {len(watch)} 檔台股...")
    all_data = yf.download(watch, period="5y", progress=False, group_by="ticker", auto_adjust=True)

    for s in watch:
        try:
            df = all_data[s].dropna()
            if len(df) < 120: continue
            df = compute_features(df)
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])
            pred = float(np.clip(model.predict(train[feats].iloc[-1:])[0], -0.15, 0.15))
            last = train.iloc[-1]
            results[s] = {"p": pred, "c": float(last["Close"]), "s": float(last["sup"]), "r": float(last["res"])}
        except: continue

    # 邏輯：從海選中排除 must_watch 以選出純黑馬
    potential_horses = {k: v for k, v in results.items() if k not in must_watch}
    top_5_keys = sorted(potential_horses.keys(), key=lambda x: potential_horses[x]['p'], reverse=True)[:5]
    audit_report = audit_and_save(results, top_5_keys)

    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n----------------------------------\n"
    msg += "🏆 **AI 海選 Top 5 (潛力黑馬)**\n"
    ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
    for idx, s in enumerate(top_5_keys):
        i = results[s]
        msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n└ 現價: `{i['c']:.1f}` (支撐: `{i['s']:.1f}` / 壓力: `{i['r']:.1f}`)\n"

    msg += "\n🔍 **指定權值股監控 (固定顯示)**\n"
    for s in must_watch:
        if s in results:
            i = results[s]
            msg += f"**{s}**: `預估 {i['p']:+.2%}`\n└ 現價: `{i['c']:.1f}`\n"

    msg += audit_report + "\n💡 *AI 為機率模型，僅供研究參考*"
    safe_post(msg[:1900])

if __name__ == "__main__":
    run()
