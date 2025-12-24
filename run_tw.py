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
# 大盤趨勢判斷 (季線濾網)
# =========================
def get_market_trend():
    try:
        # 抓取加權指數
        idx = yf.download("^TWII", period="1y", auto_adjust=True, progress=False)
        if idx.empty or len(idx) < 60:
            return True, 0, 0 # 資料不足時預設為多頭

        idx["ma60"] = idx["Close"].rolling(60).mean()
        curr_p = float(idx["Close"].iloc[-1])
        ma60_p = float(idx["ma60"].iloc[-1])
        
        # 判斷是否在季線上
        is_bull = curr_p > ma60_p
        return is_bull, curr_p, ma60_p
    except Exception as e:
        print("Market trend fetch error:", e)
        return True, 0, 0

# =========================
# 台股選股池與特徵工程
# =========================
def get_tw_300_pool():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url, timeout=10)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df["code"] = df["有價證券代號及名稱"].str.split("　").str[0]
        stocks = df[df["code"].str.len() == 4]["code"].tolist()
        return [f"{s}.TW" for s in stocks[:300]]
    except:
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW"]

def compute_features(df):
    df = df.copy()
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
    # 5日平均成交金額 (流動性關鍵)
    df["avg_amount"] = (df["Close"] * df["Volume"]).rolling(5).mean()
    return df

# =========================
# 對帳紀錄 (保留原有 logic)
# =========================
def audit_and_save(results, top_keys):
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
    else:
        hist = pd.DataFrame(columns=["date", "symbol", "pred_p", "pred_ret", "settled"])

    today = datetime.now().date()
    # 自動清理重複並儲存新預測
    new_rows = [{"date": today, "symbol": s, "pred_p": results[s]["c"], 
                 "pred_ret": results[s]["p"], "settled": False} for s in top_keys]
    hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
    hist = hist.drop_duplicates(subset=["date", "symbol"], keep="last")
    hist.to_csv(HISTORY_FILE, index=False)
    return "" # 此處可擴充對帳訊息

def safe_post(msg: str):
    if not WEBHOOK_URL:
        print(f"\n--- Discord 預覽 ---\n{msg}")
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=15)
    except:
        pass

# =========================
# 主流程
# =========================
def run():
    # 1. 取得大盤資訊 (不論多空都繼續)
    is_bull, tw_p, ma60 = get_market_trend()
    
    must_watch = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    watch = list(set(must_watch + get_tw_300_pool()))

    feats = ["mom20", "rsi", "bias", "vol_ratio"]
    results = {}
    MIN_AMOUNT = 100_000_000 # 1億台幣門檻

    print(f"掃描中... 目前大盤: {'多頭' if is_bull else '空頭 (將標示風險)'}")

    all_data = yf.download(watch, period="5y", group_by="ticker", auto_adjust=True, progress=False)

    for s in watch:
        try:
            if s not in all_data or all_data[s].empty: continue
            df = compute_features(all_data[s].dropna())
            
            # 流動性檢查
            last_row = df.iloc[-1]
            if last_row["avg_amount"] < MIN_AMOUNT:
                continue

            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna()
            if len(train) < 60: continue

            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train[feats], train["target"])

            latest_feat = train[feats].iloc[-1:]
            pred = float(np.clip(model.predict(latest_feat)[0], -0.15, 0.15))

            results[s] = {
                "p": pred, "c": float(last_row["Close"]),
                "amt": float(last_row["avg_amount"])
            }
        except:
            continue

    # 選出黑馬
    horses = {k: v for k, v in results.items() if k not in must_watch}
    top_keys = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    audit_and_save(results, top_keys)

    # 4. 訊息封裝
    msg = f"🏛 **台股 AI 預測報告 ({datetime.now():%m/%d})**\n"
    
    if is_bull:
        msg += f"📈 **市場環境：多頭** (加權指數 > 季線)\n"
    else:
        msg += f"⚠️ **風險預警：空頭環境** (加權指數 < 季線)\n"
        msg += f"└ *目前大盤收 `{tw_p:.0f}`，低於季線 `{ma60:.0f}`，選股勝率可能下降。*\n"
    
    msg += "----------------------------------\n"
    msg += "🏆 **AI 潛力黑馬 Top 5** (5日均量 > 1億)\n"

    for i, s in enumerate(top_keys):
        r = results[s]
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}** 預估 `{r['p']:+.2%}` | 現價 `{r['c']:.1f}`\n"

    msg += "\n🔍 **權值股與指數監測**\n"
    for s in must_watch:
        if s in results:
            msg += f"`{s}` 預估 `{results[s]['p']:+.2%}`\n"

    safe_post(msg[:1900])

if __name__ == "__main__":
    run()
