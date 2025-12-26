from utils.market_calendar import is_market_open

def pre_check():
    if not is_market_open("TW"):
        print("📌 因假日或節日，股市未開盤，停止動作")
        return False
    return True

import yfinance as yf
import pandas as pd
import requests
import os
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# =========================
# 基本設定
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "tw_history.csv")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# 固定權值股（不動）
FIXED = ["2330.TW", "2317.TW", "2454.TW", "0050.TW", "2308.TW", "2382.TW"]

# =========================
# 技術工具
# =========================
def calc_pivot(df):
    r = df.iloc[-20:]
    h, l, c = r["High"].max(), r["Low"].min(), r["Close"].iloc[-1]
    p = (h + l + c) / 3
    return round(2*p - h, 1), round(2*p - l, 1)

def get_top300_by_volume():
    """
    取得近 20 個交易日『平均成交量』前 300 檔台股
    """
    try:
        # 從 TWSE 官方名單抓全部上市股票代碼
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        df = pd.read_html(requests.get(url, timeout=10).text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        codes = df["有價證券代號及名稱"].str.split("　").str[0]
        codes = codes[codes.str.len() == 4]
        tickers = [f"{c}.TW" for c in codes]

        # 抓最近 1 個月成交量
        vol_data = yf.download(
            tickers,
            period="1mo",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True
        )

        avg_vol = {}
        for t in tickers:
            try:
                v = vol_data[t]["Volume"].dropna().tail(20).mean()
                if v > 0:
                    avg_vol[t] = v
            except:
                continue

        top300 = sorted(avg_vol, key=avg_vol.get, reverse=True)[:300]
        return top300

    except:
        # 保底
        return FIXED

# =========================
# 5 日回測結算
# =========================
def get_settle_report():
    if not os.path.exists(HISTORY_FILE):
        return "\n📊 **5 日回測**：尚無可結算資料\n"

    df = pd.read_csv(HISTORY_FILE)
    unsettled = df[df["settled"] == False]

    if unsettled.empty:
        return "\n📊 **5 日回測**：尚無可結算資料\n"

    report = "\n🏁 **5 日回測結算報告**\n"
    for idx, row in unsettled.iterrows():
        try:
            p = yf.download(row["symbol"], period="7d", auto_adjust=True, progress=False)
            if p.empty:
                continue
            exit_price = p["Close"].iloc[-1]
            ret = (exit_price - row["entry_price"]) / row["entry_price"]
            win = (ret > 0 and row["pred_ret"] > 0) or (ret < 0 and row["pred_ret"] < 0)

            report += (
                f"• `{row['symbol']}` 預估 {row['pred_ret']:+.2%} | "
                f"實際 `{ret:+.2%}` {'✅' if win else '❌'}\n"
            )
            df.at[idx, "settled"] = True
        except:
            continue

    # 只留 180 天
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= datetime.now() - timedelta(days=180)]
    df.to_csv(HISTORY_FILE, index=False)

    return report

# =========================
# 主程式
# =========================
def run():
    universe = list(dict.fromkeys(FIXED + get_top300_by_volume()))
    data = yf.download(universe, period="2y", auto_adjust=True, group_by="ticker", progress=False)

    feats = ["mom20", "bias", "vol_ratio"]
    results = {}

    for s in universe:
        try:
            df = data[s].dropna()
            if len(df) < 160:
                continue

            df["mom20"] = df["Close"].pct_change(20)
            df["bias"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["target"] = df["Close"].shift(-5) / df["Close"] - 1

            train = df.iloc[:-5].dropna()
            if len(train) < 80:
                continue

            model = XGBRegressor(
                n_estimators=90,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(train[feats], train["target"])

            pred = float(model.predict(df[feats].iloc[-1:])[0])
            sup, res = calc_pivot(df)

            results[s] = {
                "pred": pred,
                "price": round(df["Close"].iloc[-1], 2),
                "sup": sup,
                "res": res
            }
        except:
            continue

    msg = f"📊 **台股 AI 進階預測報告 ({datetime.now():%Y-%m-%d})**\n"
    msg += "------------------------------------------\n\n"

    medals = ["🥇", "🥈", "🥉", "📈", "📈"]
    horses = {k: v for k, v in results.items() if k not in FIXED and v["pred"] > 0}
    top_5 = sorted(horses, key=lambda x: horses[x]["pred"], reverse=True)[:5]

    msg += "🏆 **AI 海選 Top 5 (潛力黑馬)**\n"
    for i, s in enumerate(top_5):
        r = results[s]
        msg += f"{medals[i]} {s}: 預估 `{r['pred']:+.2%}`\n"
        msg += f" └ 現價: `{r['price']:.2f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += "\n🔍 **指定權值股監控 (固定顯示)**\n"
    for s in FIXED:
        if s in results:
            r = results[s]
            msg += f"{s}: 預估 `{r['pred']:+.2%}`\n"
            msg += f" └ 現價: `{r['price']:.2f}` (支撐: `{r['sup']}` / 壓力: `{r['res']}`)\n"

    msg += get_settle_report()
    msg += "\n💡 AI 為機率模型，僅供研究參考"

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=15)
    else:
        print(msg)

    hist = [{
        "date": datetime.now().date(),
        "symbol": s,
        "entry_price": results[s]["price"],
        "pred_ret": results[s]["pred"],
        "settled": False
    } for s in (top_5 + FIXED) if s in results]

    if hist:
        pd.DataFrame(hist).to_csv(
            HISTORY_FILE,
            mode="a",
            header=not os.path.exists(HISTORY_FILE),
            index=False
        )

if __name__ == "__main__":
    if pre_check():
        run()
