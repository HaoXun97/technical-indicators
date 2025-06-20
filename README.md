# 股票技術分析系統 - 模組化版本

## 📋 專案概述

這是一個重新設計的股票技術分析系統，採用模組化架構，專注於：

1. **智能數據更新**：自動檢查資料庫與外部數據的差異，只更新需要的部分
2. **分離式技術指標計算**：先更新 OHLCV 數據，再計算技術指標
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
├── main.py                 # 主程式入口
└── config.ini              # 配置檔案
```

## ⚡ 核心功能

### 1. 智能數據更新流程

```python
# 系統會自動執行以下流程：
1. 檢查資料庫中現有數據
2. 從yfinance獲取最新數據
3. 比對最近30天的數據差異
4. 只更新有差異的OHLCV數據
5. 重新計算並更新技術指標
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

## 🚀 使用方式

### 基本使用

```bash
# 使用預設股票列表 (2330, AAPL) (預設一年日 K 資料)
python main.py

# 指定特定股票
python main.py 2330 2454 AAPL TSLA

# 特定數據間隔 (60 分 K)
python main.py --1h 2330

# 顯示幫助資訊
python main.py --help

# 只更新技術指標（不檢查OHLCV數據）
python main.py --indicators-only

# 只更新特定股票 60 分 K 的技術指標
python main.py --indicators-only --1h 2330 AAPL

# 擴展歷史數據模式（獲取比資料庫更早的數據）
python main.py --expand-history 2330

# 顯示所有資料表的統計資訊
python main.py --show-all-stats
```

## ⚙️ 配置說明

### config.ini 範例

```ini
[database]
server = your_server_name
database = your_database_name
driver = ODBC Driver 17 for SQL Server
# 可選的認證資訊
username = your_username
password = your_password

[import_settings]
log_level = INFO
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
