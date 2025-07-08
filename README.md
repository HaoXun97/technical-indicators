# 股票技術分析系統 - 模組化版本

[![Python application](https://github.com/HaoXun97/technical-indicators/actions/workflows/python-app.yml/badge.svg)](https://github.com/HaoXun97/technical-indicators/actions/workflows/python-app.yml)

## 📋 專案概述

這是一個重新設計的股票技術分析系統，採用模組化架構，專注於：

1. **智能數據更新**：自動檢查資料庫與外部數據的差異，只更新需要的部分
2. **分離式技術指標計算及型態辨識**：先更新 OHLCV 數據，再計算技術指標及型態
3. **模組化設計**：清晰的分層架構，易於維護和擴展

## 🏗️ 系統架構

```
├── config/                 # 配置模組
│   └── database_config.py  # 資料庫配置與連接管理
├── providers/              # 數據提供者
│   └── stock_data_provider.py  # 外部API數據獲取
├── calculators/            # 計算器
│   └── technical_indicators.py  # 技術指標計算
├── repositories/           # 資料庫操作
│   └── stock_data_repository.py  # 數據存儲與比對
├── services/               # 業務服務層
│   └── stock_data_service.py  # 主要業務邏輯
├── utils/                  # 工具模組
│   └── display_utils.py    # 統計顯示工具
├── pattern_detection/      # K 線型態辨識
│   └── pattern_detection.py  # 基本型態辨識
├── main.py                 # 主程式入口
└── config.ini              # 配置檔案
```

## ⚡ 核心功能

### 1. 數據更新流程

```python
# 系統會自動執行以下流程：
1. 檢查資料庫中現有數據
2. 從yfinance獲取最新數據
3. 比對最近30天的數據差異
4. 只更新有差異的OHLCV數據
5. 重新計算並更新技術指標
6. 辨識 K 線型態
7. 將結果存入資料庫
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

### config.ini 範例

```ini
[database]
server = your_server_name
database = your_database_name
driver = ODBC Driver 17 for SQL Server
# SQL Server 登入 (若未填寫則使用 Windows 認證登入)
# username = your_username
# password = your_password

[import_settings]
log_level = INFO
```

### .env 範例

```env
db_username=your_username
db_password=your_password
use_windows_auth=true    # 若使用 Windows 認證則設為 true
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

```
yfinance
pandas
numpy
talib
sqlalchemy
pyodbc
configparser
```

## 🔮 未來擴展

- 支援更多技術指標
- 增加數據驗證機制
- 實作數據回測功能
- 添加圖表視覺化
- 支援多資料來源
- 辨識複雜型態

## ⚠️ 舊版 (CSV 儲存模式)

舊版使用方式: [前往查看](./OLD_VERSION.md)
