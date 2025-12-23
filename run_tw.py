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

# =========================
# Discord 安全推播
# =========================
def safe_post(msg: str):
    if not WEBHOOK_URL:
        print("\n--- Discord 訊息預覽 ---")
        print(msg)
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print("Discord 發送失敗:", e)

# =========================
# 特徵工程
# =========================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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

# =========================
# 對帳與紀錄
# =========================
def audit_and_save(results: dict, top_keys):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
        hist = hist.dropna(subset=["date"])
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])

    audit_msg = ""
    today_date = datetime.now().date()
    deadline = today_date - timedelta(days=8)

    unsettled = hist[(hist["settled"] == False) & (hist["date"] <= deadline)]

    if not unsettled.empty:
        audit_msg = "\n🎯 **5 日預測結算對帳**\n"
        for idx, r in unsettled.iterrows():
            try:
                if r["pred_p"] <= 0: continue
                price_df = yf.Ticker(r["symbol"]).history(period="5d")
                if price_df.empty: continue
                curr_p = price_df["Close"].iloc[-1]
                actual_ret = (curr_p - r["pred_p"]) / r["pred_p"]
                hit = "✅" if np.sign(actual_ret) == np.sign(r["pred_ret"]) else "❌"
                audit_msg += f"`{r['symbol']}` {r['pred_ret']:+.2%} ➜ {actual_ret:+.2%} {hit}\n"
                hist.at[idx, "settled"] = True
            except: continue

    new_rows = [{"date": today_date, "symbol": s, "pred_p": results[s]["c"], 
                 "pred_ret": results[s]["p"], "settled": False} for s in top_keys]

    hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
    hist = hist.drop_duplicates(subset=["date", "symbol"], keep="last")
    hist.to_csv(HISTORY_FILE, index=False)
    return audit_msg

# =========================
# 主流程
# =========================
def run():
    # 指定監控清單
    must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    watch = must_watch 
    
    feats = ["mom20", "rsi", "bias", "vol_ratio"]
    results = {}

    print(f"[{datetime.now():%H:%M:%S}] 下載市場資料中…")
    all_data = yf.download(watch, period="5y", progress=False, group_by="ticker", auto_adjust=True)

    for s in watch:
        try:
            if s not in all_data: continue
            df = all_data[s].dropna()
            if len(df) < 120: continue

            df = compute_features(df)
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train_df = df.dropna()

            if len(train_df) < 60: continue

            model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42)
            model.fit(train_df[feats], train_df["target"])

            latest_feat = train_df[feats].iloc[-1:]
            pred_ret = float(np.clip(model.predict(latest_feat)[0], -0.15, 0.15))
            last_row = train_df.iloc[-1]

            results[s] = {
                "p": pred_ret,
                "c": float(last_row["Close"]),
                "s": float(last_row["sup"]),
                "r": float(last_row["res"])
            }
        except Exception as e:
            print(f"{s} 發生錯誤: {e}")

    if not results:
        safe_post("⚠️ 今日資料不足，無法產生預測")
        return

    # 篩選預估回報最高的前 5 名
    top_5_keys = sorted(results.keys(), key=lambda x: results[x]['p'], reverse=True)[:5]
    audit_report = audit_and_save(results, top_5_keys)

    # 組合 Discord 訊息 (僅調整顯示順序，邏輯沒變)
    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "----------------------------------\n"
    
    # 1. AI 推薦 Top 5 先顯示
    msg += "🏆 **AI 推薦 Top 5 (最高潛力)**\n"
    ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
    for idx, s in enumerate(top_5_keys):
        i = results[s]
        msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n"
        msg += f"└ 現價: `{i['c']:.1f}` (支撐: `{i['s']:.1f}` / 壓力: `{i['r']:.1f}`)\n"

    # 2. 指定監控標的後顯示
    msg += "\n🔍 **指定監控標的未來預估**\n"
    for s in must_watch:
        if s in results:
            i = results[s]
            msg += f"**{s}**: `預估 {i['p']:+.2%}`\n"
            msg += f"└ 現價: `{i['c']:.1f}` (支撐: `{i['s']:.1f}` / 壓力: `{i['r']:.1f}`)\n"

    # 3. 對帳資訊
    if audit_report:
        msg += audit_report

    msg += "\n💡 *AI 為機率模型，僅供研究參考，非投資建議*"

    if len(msg) > 1900: msg = msg[:1900] + "\n...(訊息過長已截斷)"
    safe_post(msg)
    print(f"[{datetime.now():%H:%M:%S}] 完成")

if __name__ == "__main__":
    run()
