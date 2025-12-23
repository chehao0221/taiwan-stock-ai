import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
HISTORY_FILE = "tw_history.csv"

# (compute_features 函式維持不變...)
def compute_features(df):
    df = df.copy()
    df["mom20"] = df["Close"].pct_change(20)
    df["rsi"] = 100 - (100 / (1 + df["Close"].diff().clip(lower=0).rolling(14).mean() / ((-df["Close"].diff().clip(upper=0)).rolling(14).mean() + 1e-9)))
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bias"] = (df["Close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["sup"] = df["Low"].rolling(60).min()
    df["res"] = df["High"].rolling(60).max()
    return df

def audit_and_save(current_results, top_5_keys):
    audit_msg = ""
    if os.path.exists(HISTORY_FILE):
        hist_df = pd.read_csv(HISTORY_FILE)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        deadline = datetime.now() - timedelta(days=7)
        to_settle = hist_df[(hist_df['date'] <= deadline) & (hist_df['settled'] == False)]
        if not to_settle.empty:
            audit_msg = "\n🎯 **5日預測結算對帳單**\n"
            for idx, row in to_settle.iterrows():
                try:
                    curr_p = yf.Ticker(row['symbol']).history(period="1d")['Close'].iloc[-1]
                    actual_ret = (curr_p - row['pred_p']) / row['pred_p']
                    is_hit = "✅ 命中" if (actual_ret > 0 and row['pred_ret'] > 0) or (actual_ret < 0 and row['pred_ret'] < 0) else "❌ 錯誤"
                    audit_msg += f" `{row['symbol']}`: 預估 `{row['pred_ret']:+.2%}` ➔ 實際 `{actual_ret:+.2%}` ({is_hit})\n"
                    hist_df.at[idx, 'settled'] = True
                except: continue
        hist_df.to_csv(HISTORY_FILE, index=False)
    else:
        hist_df = pd.DataFrame(columns=['date', 'symbol', 'pred_p', 'pred_ret', 'settled'])

    new_recs = [{'date': datetime.now().strftime("%Y-%m-%d"), 'symbol': s, 'pred_p': current_results[s]['c'], 'pred_ret': current_results[s]['p'], 'settled': False} for s in top_5_keys]
    hist_df = pd.concat([hist_df, pd.DataFrame(new_recs)], ignore_index=True)
    hist_df.to_csv(HISTORY_FILE, index=False)
    return audit_msg

def run():
    if not WEBHOOK_URL: return
    # (海選 300 檔與分析邏輯維持不變...)
    # 這裡僅列出發送訊息的 Formatting 部分
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"🇹🇼 **台股 AI 預測報告 ({today})**\n"
    msg += "----------------------------------\n"
    msg += "🏆 **300 股票前 5 的未來預估**\n"
    
    # ... (假設 top_5 已經計算完成)
    ranks = ["🥇", "🥈", "🥉", "📈", "📈"]
    for idx, s in enumerate(top_5):
        i = results[s]
        msg += f"{ranks[idx]} **{s}**: `預估 {i['p']:+.2%}`\n"
        msg += f"└ 現價: `{i['c']:.1f}` (支撐: {i['s']:.1f} / 壓力: {i['r']:.1f})\n"

    msg += "\n💎 **指定監控標的未來預估**\n"
    for s in must_watch:
        if s in results:
            i = results[s]
            msg += f"⭐ **{s}**: `預估 {i['p']:+.2%}`\n"
            msg += f"└ 現價: `{i['c']:.1f}` (支撐: {i['s']:.1f} / 壓力: {i['r']:.1f})\n"
    
    # ... (加上對帳單 audit_report)
    msg += audit_report
    msg += "\n💡 *註：預估值為 AI 對未來 5 個交易日後的走勢判斷。*"
    
    requests.post(WEBHOOK_URL, json={"content": msg})
