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
HISTORY_FILE_TW = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# =========================
# 工具函數：支撐壓力計算 & 股票池
# =========================
def calc_sup_res(df):
    try:
        recent = df.iloc[-20:]
        h, l, c = recent['High'].max(), recent['Low'].min(), recent['Close'].iloc[-1]
        p = (h + l + c) / 3
        return round(2*p - h, 1), round(2*p - l, 1) # 支撐, 壓力
    except: return 0, 0

def get_tw_300():
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        df = pd.read_html(requests.get(url, timeout=10).text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        codes = df[df["有價證券代號及名稱"].str.split("　").str[0].str.len() == 4]["有價證券代號及名稱"].str.split("　").str[0].tolist()
        return [f"{s}.TW" for s in codes[:300]]
    except: return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW"]

# =========================
# 5 日回測結算邏輯
# =========================
def get_settle_report():
    if not os.path.exists(HISTORY_FILE_TW): return ""
    try:
        df = pd.read_csv(HISTORY_FILE_TW)
        df['date'] = pd.to_datetime(df['date'])
        # 找出 5 天前且未結算的
        mask = (df['settled'].astype(str).str.upper() == 'FALSE') & (df['date'] <= datetime.now() - timedelta(days=5))
        to_settle = df[mask].copy()
        if to_settle.empty: return "\n📊 **5日回測**: 尚無待結算數據 (需累積 5 天資料)。"

        report = "\n🏁 **5 日回測結算報告 (對帳單)**\n"
        syms = to_settle['symbol'].unique().tolist()
        prices = yf.download(syms, period="5d", auto_adjust=True, progress=False)['Close']
        
        for idx, row in to_settle.iterrows():
            s = row['symbol']
            try:
                curr_p = float(prices[s].dropna().iloc[-1]) if isinstance(prices, pd.DataFrame) else float(prices.iloc[-1])
                ret = (curr_p - row['pred_p']) / row['pred_p']
                win = (ret > 0 and row['pred_ret'] > 0) or (ret < 0 and row['pred_ret'] < 0)
                df.at[idx, 'settled'] = 'True'
                report += f"• `{s}`: 預估 {row['pred_ret']:+.2%} | 實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            except: continue
        df.to_csv(HISTORY_FILE_TW, index=False)
        return report
    except: return ""

# =========================
# 主程序
# =========================
def run():
    fixed = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]
    pool = get_tw_300()
    watch = list(dict.fromkeys(fixed + pool))
    
    print(f"🚀 開始海選 {len(watch)} 檔標的...")
    data = yf.download(watch, period="2y", auto_adjust=True, group_by="ticker", progress=False)
    
    results = {}
    feats = ["mom20", "bias", "vol_ratio"]
    
    for s in watch:
        try:
            df = data[s].dropna()
            if len(df) < 50: continue
            df["mom20"], df["bias"], df["vol_ratio"] = df["Close"].pct_change(20), (df["Close"]-df["Close"].rolling(20).mean())/df["Close"].rolling(20).mean(), df["Volume"]/df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1
            train = df.dropna().iloc[-250:]
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05).fit(train[feats], train["target"])
            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_sup_res(df)
            results[s] = {"p": pred, "c": float(df["Close"].iloc[-1]), "sup": sup, "res": res}
        except: continue

    # 組合訊息
    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"
    msg += "🏆 **AI 海選 Top 5 (潛力黑馬)**\n"
    
    horses = {k: v for k, v in results.items() if k not in fixed}
    top_5 = sorted(horses, key=lambda x: horses[x]["p"], reverse=True)[:5]
    
    for i, s in enumerate(top_5):
        r = results[s]
        msg += f"{['🥇','🥈','🥉','📈','📈'][i]} **{s}**: 預估 `{r['p']:+.2%}`\n"
        msg += f" └ 現價: `{r['c']}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n🔍 **指定權值股監控 (固定顯示)**\n"
    for s in fixed:
        if s in results:
            r = results[s]
            msg += f"**{s}**: 預估 `{r['p']:+.2%}`\n └ 現價: `{r['c']}`\n"

    # 加上回測報告
    msg += get_settle_report()
    msg += "\n💡 AI 為機率模型，僅供研究參考"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else: print(msg)

    # 存檔供未來結算
    new_hist = [{"date": datetime.now().date(), "symbol": s, "pred_p": results[s]['c'], "pred_ret": results[s]['p'], "settled": "False"} for s in (top_5 + fixed) if s in results]
    pd.DataFrame(new_hist).to_csv(HISTORY_FILE_TW, mode='a', header=not os.path.exists(HISTORY_FILE_TW), index=False)

if __name__ == "__main__":
    run()
