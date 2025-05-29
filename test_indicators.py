import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

from Indicators import (
    DataProvider,
    IndicatorCalculator,
    TechnicalAnalyzer,
    ResultExporter,
    AnalysisReporter,
    StockPrice,
    Period,
    TimeInterval
)


class TestDataProvider:
    """測試數據提供者"""

    def setup_method(self):
        """每個測試前的設置"""
        self.provider = DataProvider()

    def test_format_symbol_taiwan_stock(self):
        """測試台股代號格式化"""
        assert self.provider._format_symbol("2330") == "2330.TW"
        assert self.provider._format_symbol("2317") == "2317.TW"

    def test_format_symbol_already_formatted(self):
        """測試已格式化的股票代號"""
        assert self.provider._format_symbol("2330.TW") == "2330.TW"
        assert self.provider._format_symbol("1234.TWO") == "1234.TWO"

    def test_adjust_period_for_minute_interval(self):
        """測試分鐘級間隔的期間調整"""
        result = self.provider._adjust_period_for_interval("max", "1m")
        assert result == "60d"

        result = self.provider._adjust_period_for_interval("5y", "5m")
        assert result == "60d"

    def test_adjust_period_for_hour_interval(self):
        """測試小時級間隔的期間調整"""
        result = self.provider._adjust_period_for_interval("max", "1h")
        assert result == "730d"

    def test_adjust_period_no_change(self):
        """測試不需要調整的期間"""
        result = self.provider._adjust_period_for_interval("1y", "1d")
        assert result == "1y"

    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """測試成功獲取股票數據"""
        # 模擬返回的數據
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        result = self.provider.get_stock_data(
            "2330", Period.MONTH_1, TimeInterval.DAY_1)

        assert result is not None
        assert len(result) == 3
        assert "Open" in result.columns

    @patch('yfinance.Ticker')
    def test_get_stock_data_empty_result(self, mock_ticker):
        """測試獲取空數據"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        result = self.provider.get_stock_data(
            "INVALID", Period.MONTH_1, TimeInterval.DAY_1)

        assert result is None


class TestIndicatorCalculator:
    """測試指標計算器"""

    def setup_method(self):
        """每個測試前的設置"""
        self.calculator = IndicatorCalculator()

        # 建立測試數據
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 固定隨機種子

        self.test_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_calculate_all_indicators_success(self):
        """測試成功計算所有指標"""
        indicators = self.calculator.calculate_all_indicators(self.test_data)

        # 檢查是否返回字典
        assert isinstance(indicators, dict)
        assert len(indicators) > 0

        # 檢查 RSI 指標
        assert "RSI(14)" in indicators
        assert "RSI(5)" in indicators

        # 檢查 MACD 指標
        assert "DIF" in indicators
        assert "MACD" in indicators
        assert "MACD_Histogram" in indicators

        # 檢查 KDJ 指標
        assert "K" in indicators
        assert "D" in indicators
        assert "J" in indicators
        assert "RSV" in indicators

        # 檢查移動平均線
        assert "MA5" in indicators
        assert "MA20" in indicators
        assert "EMA12" in indicators

        # 檢查布林通道
        assert "BB_Upper" in indicators
        assert "BB_Middle" in indicators
        assert "BB_Lower" in indicators

    def test_calculate_rsi(self):
        """測試 RSI 計算"""
        close = self.test_data["Close"].values
        indicators = self.calculator._calculate_rsi(
            close, self.test_data.index)

        assert "RSI(14)" in indicators
        assert "RSI(5)" in indicators
        assert len(indicators["RSI(14)"]) == len(self.test_data)

        # RSI 值應該在 0-100 之間
        rsi_values = indicators["RSI(14)"].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_macd(self):
        """測試 MACD 計算"""
        close = self.test_data["Close"].values
        indicators = self.calculator._calculate_macd(
            close, self.test_data.index)

        assert "DIF" in indicators
        assert "MACD" in indicators
        assert "MACD_Histogram" in indicators
        assert len(indicators["DIF"]) == len(self.test_data)

    def test_calculate_stochastic(self):
        """測試 KDJ 指標計算"""
        high = self.test_data["High"].values
        low = self.test_data["Low"].values
        close = self.test_data["Close"].values

        indicators = self.calculator._calculate_stochastic(
            high, low, close, self.test_data.index
        )

        assert "RSV" in indicators
        assert "K" in indicators
        assert "D" in indicators
        assert "J" in indicators

        # 檢查 RSV 值範圍
        rsv_values = indicators["RSV"].dropna()
        assert len(rsv_values) > 0, "RSV 應該有有效值"

        # RSV 值應該在 0-100 範圍內
        assert all(0 <= val <= 100 for val in rsv_values), \
            f"RSV 值超出範圍: min={rsv_values.min():.2f}, max={rsv_values.max():.2f}"

        # 檢查 K 和 D 值的有效性
        k_values = indicators["K"].dropna()
        d_values = indicators["D"].dropna()

        assert len(k_values) > 0, "K 值應該有有效值"
        assert len(d_values) > 0, "D 值應該有有效值"

        # J 值可能超出 0-100 範圍，這是正常的
        j_values = indicators["J"].dropna()
        assert len(j_values) > 0, "J 值應該有有效值"

    def test_calculate_stochastic_edge_cases(self):
        """測試 KDJ 指標的邊界情況"""
        # 創建一個所有價格相同的測試案例（會導致除零情況）
        same_price_data = pd.DataFrame({
            'High': [100] * 20,
            'Low': [100] * 20,
            'Close': [100] * 20,
        }, index=pd.date_range('2023-01-01', periods=20))

        high = same_price_data["High"].values
        low = same_price_data["Low"].values
        close = same_price_data["Close"].values

        indicators = self.calculator._calculate_stochastic(
            high, low, close, same_price_data.index
        )

        # 當所有價格相同時，RSV 應該是 50
        rsv_values = indicators["RSV"].dropna()
        assert all(val == 50 for val in rsv_values), "當價格不變時，RSV 應該為 50"

    def test_calculate_all_indicators_empty_data(self):
        """測試空數據時的錯誤處理"""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="數據必須是非空的 DataFrame"):
            self.calculator.calculate_all_indicators(empty_data)

    def test_calculate_all_indicators_insufficient_data(self):
        """測試數據不足時的警告"""
        small_data = self.test_data.head(30)  # 只有30筆數據

        indicators = self.calculator.calculate_all_indicators(small_data)

        # 即使數據不足，也應該返回一些指標
        assert isinstance(indicators, dict)


class TestResultExporter:
    """測試結果匯出器"""

    def setup_method(self):
        """每個測試前的設置"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ResultExporter(self.temp_dir)

    def test_init_creates_output_directory(self):
        """測試初始化時創建輸出目錄"""
        assert self.exporter.output_dir.exists()

    def test_save_to_json_success(self):
        """測試成功保存 JSON"""
        test_results = {
            "2330": {
                "symbol": "2330",
                "price": {"close": 500.0},
                "indicators": {"RSI(14)": 55.0}
            }
        }

        filepath = self.exporter.save_to_json(test_results, "test.json")

        assert filepath != ""
        assert Path(filepath).exists()

        # 檢查檔案內容
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert "2330" in loaded_data
        assert loaded_data["2330"]["symbol"] == "2330"

    def test_save_to_csv_success(self):
        """測試成功保存 CSV"""
        # 建立測試數據
        test_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        })

        test_indicators = {
            "RSI(14)": pd.Series([50.0, 55.0]),
            "MA20": pd.Series([100.5, 101.5])
        }

        filepath = self.exporter.save_to_csv(
            "2330", test_data, test_indicators)

        assert filepath != ""
        assert Path(filepath).exists()

        # 檢查 CSV 內容
        loaded_data = pd.read_csv(filepath, index_col=0)
        assert "RSI(14)" in loaded_data.columns
        assert "MA20" in loaded_data.columns

    def test_clean_results_for_json(self):
        """測試清理結果以便 JSON 序列化"""
        test_results = {
            "2330": {
                "symbol": "2330",
                "indicators": {"RSI": 55.0},
                "_data": "should_be_removed",
                "_indicators": "should_be_removed"
            }
        }

        cleaned = self.exporter._clean_results_for_json(test_results)

        assert "_data" not in cleaned["2330"]
        assert "_indicators" not in cleaned["2330"]
        assert "symbol" in cleaned["2330"]
        assert "indicators" in cleaned["2330"]


class TestTechnicalAnalyzer:
    """測試技術分析器"""

    def setup_method(self):
        """每個測試前的設置"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = TechnicalAnalyzer(self.temp_dir)

    def test_init(self):
        """測試分析器初始化"""
        assert self.analyzer.data_provider is not None
        assert self.analyzer.indicator_calculator is not None
        assert self.analyzer.exporter is not None
        assert self.analyzer.logger is not None

    @patch.object(DataProvider, 'get_stock_data')
    def test_analyze_stock_no_data(self, mock_get_data):
        """測試無法獲取數據的情況"""
        mock_get_data.return_value = None

        result = self.analyzer.analyze_stock("INVALID")

        assert "error" in result
        assert "無法獲取" in result["error"]

    @patch.object(DataProvider, 'get_stock_data')
    def test_analyze_stock_insufficient_data(self, mock_get_data):
        """測試數據不足的情況"""
        small_data = pd.DataFrame({
            'Open': [100] * 30,
            'High': [105] * 30,
            'Low': [95] * 30,
            'Close': [103] * 30,
            'Volume': [1000] * 30
        })
        mock_get_data.return_value = small_data

        result = self.analyzer.analyze_stock("2330")

        assert "error" in result
        assert "數據不足" in result["error"]

    def test_get_latest_indicator_values(self):
        """測試獲取最新指標值"""
        test_indicators = {
            "RSI(14)": pd.Series([50.0, 55.0, 60.0]),
            "MA20": pd.Series([100.0, 101.0, 102.0]),
            "Empty": pd.Series([])
        }

        latest_values = self.analyzer._get_latest_indicator_values(
            test_indicators)

        assert latest_values["RSI(14)"] == 60.0
        assert latest_values["MA20"] == 102.0
        assert "Empty" not in latest_values or latest_values["Empty"] is None


class TestStockPrice:
    """測試股價數據結構"""

    def test_stock_price_creation(self):
        """測試股價對象創建"""
        price = StockPrice(
            open=100.0,
            high=105.0,
            low=95.0,
            close=103.0,
            volume=1000
        )

        assert price.open == 100.0
        assert price.high == 105.0
        assert price.low == 95.0
        assert price.close == 103.0
        assert price.volume == 1000


class TestAnalysisReporter:
    """測試分析報告生成器"""

    def test_print_analysis_summary(self, capsys):
        """測試打印分析摘要"""
        test_results = {
            "2330": {
                "symbol": "2330",
                "date": "2023-12-01 00:00:00",
                "price": {
                    "open": 500.0,
                    "high": 510.0,
                    "low": 495.0,
                    "close": 505.0,
                    "volume": 10000
                },
                "indicators": {
                    "RSI(14)": 75.0,  # 超買
                    "DIF": 2.5,
                    "MACD": 2.0,
                    "K": 80.0,
                    "D": 75.0,
                    "J": 90.0
                },
                "total_records": 100,
                "interval": "1d"
            },
            "ERROR_STOCK": {
                "error": "無法獲取數據"
            }
        }

        AnalysisReporter.print_analysis_summary(test_results)

        captured = capsys.readouterr()
        assert "技術分析結果摘要" in captured.out
        assert "2330" in captured.out
        assert "超買" in captured.out  # RSI > 70
        assert "多頭" in captured.out  # DIF > MACD
        assert "ERROR_STOCK" in captured.out
        assert "無法獲取數據" in captured.out


class TestEnums:
    """測試枚舉類別"""

    def test_time_interval_enum(self):
        """測試時間間隔枚舉"""
        assert TimeInterval.MINUTE_1.value == "1m"
        assert TimeInterval.DAY_1.value == "1d"
        assert TimeInterval.WEEK_1.value == "1wk"

    def test_period_enum(self):
        """測試時間週期枚舉"""
        assert Period.DAY_1.value == "1d"
        assert Period.YEAR_1.value == "1y"
        assert Period.MAX.value == "max"


# 整合測試
class TestIntegration:
    """整合測試"""

    def setup_method(self):
        """設置整合測試環境"""
        self.temp_dir = tempfile.mkdtemp()

    @patch.object(DataProvider, 'get_stock_data')
    def test_full_analysis_workflow(self, mock_get_data):
        """測試完整分析流程"""
        # 模擬股票數據
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

        mock_get_data.return_value = mock_data

        analyzer = TechnicalAnalyzer(self.temp_dir)

        # 執行分析
        result = analyzer.analyze_stock(
            "2330", Period.YEAR_1, TimeInterval.DAY_1)

        # 檢查結果
        assert "error" not in result
        assert result["symbol"] == "2330"
        assert "price" in result
        assert "indicators" in result
        assert result["total_records"] == 100

        # 檢查指標是否存在
        indicators = result["indicators"]
        assert "RSI(14)" in indicators
        assert "DIF" in indicators
        assert "K" in indicators


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
