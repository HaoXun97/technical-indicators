# 股票技術分析系統 - 模組化版本

[![Python application](https://github.com/HaoXun97/technical-indicators/actions/workflows/python-app.yml/badge.svg)](https://github.com/HaoXun97/technical-indicators/actions/workflows/python-app.yml)

## 📋 專案概述

這是一個重新設計的股票技術分析系統，採用模組化架構，專注於：

1. **智能數據更新**：自動檢查資料庫與外部數據的差異，只更新需要的部分
2. **分離式技術指標計算及型態辨識**：先更新 OHLCV 數據，再計算技術指標及型態
3. **模組化設計**：清晰的分層架構，易於維護和擴展
4. **智能訊號分析**：全新的訊號分析模組，提供多種技術指標訊號和交易訊號生成

## 🏗️ 系統架構

```
├── config/                         # 配置模組
├── providers/                      # 數據提供者
├── calculators/                    # 計算器
├── repositories/                   # 資料庫操作
├── services/                       # 業務服務層
├── utils/                          # 工具模組
├── pattern_detection/              # K 線型態辨識
├── signals/                        # 訊號分析模組
├── output/                         # 輸出目錄
├── main.py                         # 主程式入口
├── Indicators.py                   # 技術指標計算
├── signals.py                      # 訊號分析模組
├── config.ini                      # 配置檔案
├── .env                            # 環境變數配置
└── requirements.txt                # 依賴套件清單
```

## ⚡ 核心功能

### 1. 數據更新流程

```python
# 系統會自動執行以下流程：
1. 檢查資料庫中現有數據
2. 從 yfinance 獲取最新數據
3. 比對最近 30 天的數據差異
4. 只更新有差異的 OHLCV 數據
5. 重新計算並更新技術指標
6. 辨識 K 線型態
7. 執行訊號分析和交易訊號生成
8. 將結果存入資料庫
```

### 2. 數據比對機制

- 預設檢查最近 30 天的數據
- 使用 0.001 的價格容差進行比較
- 自動偵測新增、修正或缺失的數據
- 避免不必要的全量更新

### 3. 技術指標計算

支援以下技術指標：

- **RSI**: 5, 7, 10, 14, 21 期間
- **MACD**: DIF, MACD, 柱狀圖
- **KDJ**: RSV, K, D, J 值
- **移動平均**: MA5, MA10, MA20, MA60, EMA12, EMA26
- **布林通道**: 上軌, 中軌, 下軌
- **其他**: ATR, CCI, Williams %R, 動量指標

### 4. K 線型態

使用 ta-lib 辨識基本 K 線型態

### 5. 訊號分析系統

全新的訊號分析模組，提供：

- **多種技術指標訊號**：MA 交叉、布林通道、MACD、RSI、KD、CCI、威廉指標等
- **趨勢判斷**：自動識別趨勢方向和強度
- **背離偵測**：MACD 背離訊號識別
- **異常偵測**：價格和成交量異常識別
- **支撐阻力位**：動態計算支撐和阻力位
- **交易訊號生成**：綜合多個指標生成買賣訊號
- **訊號權重系統**：不同訊號具有不同權重，提高準確性

## 🚀 使用方式

### 基本使用

```bash

# 基本用法
python main.py [市場選項] [時間間隔選項] [功能選項] [股票代號...]

# 指定特定股票
python main.py 2330 2454      # 預設台股可不用輸入市場選項
python main.py --us AAPL NVDA

# 特定數據間隔 (小時線 K 線)
python main.py --1h 2330

# 顯示幫助資訊
python main.py --help

# 重新計算技術指標（不更新 OHLCV 數據）
python main.py --indicators-only 2330

# 擴展歷史數據模式（獲取比資料庫更早的數據）
python main.py --expand-history 2330

# 辨識歷史 K 線型態
python main.py --pattern 2330

# 執行訊號分析（預設自動執行）
python main.py --signals 2330

# 指定訊號分析輸出路徑
python main.py --signals --signals-output ./output/ 2330

# 顯示所有資料表的統計資訊
python main.py --show-all-stats
```

### 市場選項

```
--tw 台股 (預設)
--us 美股
--etf ETF
--index 指數
--forex 外匯
--crypto 加密貨幣
--futures 期貨
```

## ⚙️ 配置說明

### .env 檔案

可另外新增 .env.local 檔案，程式會優先讀取 .env.local 的設定：

```env
# 資料庫配置
use_windows_auth = true    # 若使用 Windows 認證則設為 true

MSSQL_SERVER = localhost   # 資料庫伺服器位址
MSSQL_DATABASE = master    # 資料庫名稱
MSSQL_TABLE = stock_data_1d # 預設資料表
MSSQL_USER = username      # 資料庫使用者名稱
MSSQL_PASSWORD = password  # 資料庫密碼

# 輸出配置
OUTPUT_CSV = ./output/     # CSV 輸出目錄
```

## 📊 處理結果

系統會顯示詳細的處理結果：

```
📊 [1/3] 處理 2330
   ✅ 成功 | 新增: 0 筆 | 更新: 5 筆 | 指標: 250 筆
   📅 時間範圍: 2024-06-15 ~ 2024-06-20
   ⏱️  處理時間: 3.45 秒

處理結果摘要:
✅ 成功處理: 3/3 個股票
📊 新增記錄: 0 筆
🔄 更新記錄: 15 筆
📈 技術指標更新: 750 筆

訊號分析結果:
📊 開始執行訊號分析
[1/1] 分析 2330...
✅ 2330 訊號分析完成
📁 輸出檔案: ./output/signals_2330_stock_data_1d.csv

總執行時間: 12.34 秒
- 資料讀取: 2.10秒 (17.0%)
- 指標計算: 8.45秒 (68.5%)
- 訊號生成: 1.23秒 (10.0%)
- 資料儲存: 0.56秒 (4.5%)
```

## 📝 日誌記錄

系統會在 `stock_analyzer.log` 中記錄詳細的執行資訊：

- 數據獲取過程
- 比對結果詳情
- 更新操作記錄
- 錯誤和警告資訊

## 🐛 故障排除

### 常見問題

1. **資料庫連接失敗**

   - 檢查 config.ini 中的資料庫設定
   - 確認 ODBC 驅動程式已安裝
   - 驗證網絡連接和認證資訊

2. **技術指標計算失敗**

   - 確保有足夠的歷史數據（至少 60 筆）
   - 檢查 talib 套件是否正確安裝

3. **數據獲取失敗**
   - 檢查網絡連接
   - 驗證股票代號格式
   - 確認 yfinance 套件版本

## 📚 依賴套件

詳見 [requirements.txt](./requirements.txt)

## ⚠️ 舊版 (CSV 儲存模式)

舊版使用方式: [前往查看](./OLD_VERSION.md)
