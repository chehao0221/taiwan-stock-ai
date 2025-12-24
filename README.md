🚀 Taiwan Stock AI Predictor (台股 AI 自動化預測)
這是一個基於機器學習的自動化台股分析工具。透過 GitHub Actions 每日盤後自動抓取資料、訓練 XGBoost 模型，並將預測結果與支撐壓力位推播至 Discord 頻道，同時具備「五日後自動對帳」功能來驗證 AI 準確度。

🌟 核心功能
自動化運行：每日台北時間 8:30 (盤前) 自動啟動分析。

機器學習預測：使用 XGBoost 迴歸模型預測未來 5 日回報率。

技術指標分析：自動計算 RSI、MA20 乖離率、成交量比率及 60 日支撐/壓力位。

回測對帳系統：自動追蹤 5 天前的預測，並比對實際股價進行「勝率對帳」。

Discord 整合：預測結果與對帳報表第一時間推播至手機。

🛠️ 技術棧
語言: Python 3.10

模型: XGBoost (Regressor)

資料來源: yfinance (Yahoo Finance API)

自動化: GitHub Actions

資料庫: CSV 檔案 (透過 Git 自動更新紀錄)

📊 預測邏輯與特徵
模型主要觀察以下特徵 (features)：

mom20: 20 日動量指標。

rsi: 強弱指標，判斷是否超買或超賣。

bias: 股價與 20 日均線的乖離率。

vol_ratio: 當前成交量與 20 日均量的比率。

🚀 快速開始
1. 複製專案
Bash

git clone https://github.com/你的用戶名/taiwan-stock-ai-main.git
cd taiwan-stock-ai-main
2. 設定 Discord Webhook
在 Discord 頻道中：編輯頻道 -> 整合 -> Webhook -> 建立 Webhook。

複製 Webhook URL。

3. GitHub Secrets 設定 (重要)
為了讓自動化腳本能發送訊息，請在 GitHub Repo 設定：

前往 Settings -> Secrets and variables -> Actions。

點擊 New repository secret。

名稱輸入：DISCORD_WEBHOOK_URL。

數值輸入：你的 Discord Webhook 網址。

4. 權限設定
為了讓 GitHub Actions 能自動更新 tw_history.csv，請確保：

Settings -> Actions -> General -> Workflow permissions 設為 Read and write permissions。

📅 運行時間
自動觸發: 每週一至週五 14:00 (台北時間)。

手動觸發: 可在 GitHub Actions 頁面點擊 Run workflow 立即執行。

📝 報表示例
📊 台股 AI 進階預測報告 (2025-12-23)
⭐ 2330.TW 預估 5 日：+2.45% └ 現價 1030.00｜支撐 1010.00｜壓力 1080.00

🎯 5 日預測結算對帳 2454.TW +1.50% ➜ +2.10% ✅ 2317.TW -0.50% ➜ +0.20% ❌

⚠️ 免責聲明
本專案僅供機器學習研究參考，不構成任何投資建議。股市投資有風險，AI 預測可能存在誤差，請投資人審慎評估。
