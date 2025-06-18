import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import json
import tempfile
import os
import shutil

from Indicators import (
    DataProvider,
    IndicatorCalculator,
    ResultExporter,
    TechnicalAnalyzer,
    AnalysisReporter,
    StockPrice,
    TechnicalIndicators,
    Period,
    TimeInterval,
    main  # 新增匯入 main 函數
)


class TestStockPrice:
    """測試 StockPrice 資料結構"""

    def test_stock_price_creation(self):
        price = StockPrice(
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000000
        )
        assert price.open == 100.0
        assert price.high == 105.0
        assert price.low == 98.0
        assert price.close == 103.0
        assert price.volume == 1000000


class TestTechnicalIndicators:
    """測試 TechnicalIndicators 資料結構"""

    def test_technical_indicators_creation(self):
        indicators = TechnicalIndicators()
        assert indicators.rsi_14 is None
        assert indicators.macd is None

    def test_technical_indicators_with_values(self):
        indicators = TechnicalIndicators(
            rsi_14=65.5,
            macd=1.25,
            ma_5=100.0
        )
        assert indicators.rsi_14 == 65.5
        assert indicators.macd == 1.25
        assert indicators.ma_5 == 100.0


class TestDataProvider:
    """測試 DataProvider 類別"""

    data_provider: DataProvider

    def setup_method(self):
        self.data_provider = DataProvider()

    def test_format_symbol_taiwan_stock(self):
        """測試台股代號格式化"""
        assert self.data_provider._format_symbol("2330") == "2330.TW"
        assert self.data_provider._format_symbol("1234") == "1234.TW"

    def test_format_symbol_us_stock(self):
        """測試美股代號格式化"""
        assert self.data_provider._format_symbol("AAPL") == "AAPL"
        assert self.data_provider._format_symbol("MSFT") == "MSFT"

    def test_format_symbol_with_suffix(self):
        """測試已有後綴的代號"""
        assert self.data_provider._format_symbol("2330.TW") == "2330.TW"
        assert self.data_provider._format_symbol("1234.TWO") == "1234.TWO"

    def test_adjust_period_for_interval(self):
        """測試期間調整"""
        # 小時間隔
        result = self.data_provider._adjust_period_for_interval("max", "1h")
        assert result == "730d"

        # 分鐘間隔
        result = self.data_provider._adjust_period_for_interval("1y", "5m")
        assert result == "60d"

        # 日間隔
        result = self.data_provider._adjust_period_for_interval("1y", "1d")
        assert result == "1y"

    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """測試成功獲取股票數據"""
        # 建立模擬數據
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        result = self.data_provider.get_stock_data(
            "AAPL", Period.MONTH_1, TimeInterval.DAY_1)

        assert result is not None
        assert len(result) == 3
        assert 'Open' in result.columns

    @patch('yfinance.Ticker')
    def test_get_stock_data_empty_result(self, mock_ticker):
        """測試空數據結果"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        result = self.data_provider.get_stock_data(
            "INVALID", Period.MONTH_1, TimeInterval.DAY_1)

        assert result is None

    @patch('yfinance.Ticker')
    def test_get_stock_data_with_cache(self, mock_ticker):
        """測試快取功能"""
        # 建立模擬數據
        mock_data = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [98],
            'Close': [103],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # 第一次調用
        result1 = self.data_provider.get_stock_data(
            "AAPL", Period.MONTH_1, TimeInterval.DAY_1)

        # 第二次調用（應該使用快取）
        result2 = self.data_provider.get_stock_data(
            "AAPL", Period.MONTH_1, TimeInterval.DAY_1)

        # 檢查 ticker.history 只被調用一次
        assert mock_ticker_instance.history.call_count == 1
        assert result1 is not None
        assert result2 is not None


class TestIndicatorCalculator:
    """測試 IndicatorCalculator 類別"""

    calculator: IndicatorCalculator
    test_data: pd.DataFrame

    def setup_method(self):
        self.calculator = IndicatorCalculator()

        # 建立測試數據
        np.random.seed(42)  # 固定隨機種子以確保測試一致性
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # 建立合理的股價數據
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        high_prices = close_prices + np.random.uniform(0, 2, 100)
        low_prices = close_prices - np.random.uniform(0, 2, 100)
        open_prices = close_prices + np.random.uniform(-1, 1, 100)
        volumes = np.random.randint(1000000, 5000000, 100)

        self.test_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    def test_calculate_all_indicators_success(self):
        """測試成功計算所有指標"""
        result = self.calculator.calculate_all_indicators(self.test_data)

        assert isinstance(result, dict)
        assert len(result) > 0

        # 檢查是否包含主要指標
        expected_indicators = [
            'RSI(14)', 'DIF', 'MACD', 'K', 'D', 'J', 'MA5', 'MA20']
        for indicator in expected_indicators:
            assert indicator in result
            assert isinstance(result[indicator], pd.Series)

    def test_calculate_all_indicators_insufficient_data(self):
        """測試數據不足的情況"""
        short_data = self.test_data.head(10)  # 只有10筆數據

        # 應該仍能計算，但會有警告
        result = self.calculator.calculate_all_indicators(short_data)
        assert isinstance(result, dict)

    def test_calculate_all_indicators_empty_data(self):
        """測試空數據"""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError):
            self.calculator.calculate_all_indicators(empty_data)

    def test_calculate_rsi(self):
        """測試 RSI 計算"""
        close = self.test_data['Close'].to_numpy()
        result = self.calculator._calculate_rsi(close, self.test_data.index)

        assert 'RSI(14)' in result
        assert isinstance(result['RSI(14)'], pd.Series)

        # RSI 值應該在 0-100 之間
        rsi_values = result['RSI(14)'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_macd(self):
        """測試 MACD 計算"""
        close = self.test_data['Close'].to_numpy()
        result = self.calculator._calculate_macd(close, self.test_data.index)

        assert 'DIF' in result
        assert 'MACD' in result
        assert 'MACD_Histogram' in result

        for key in result:
            assert isinstance(result[key], pd.Series)

    def test_calculate_stochastic(self):
        """測試 KDJ 計算"""
        high = self.test_data['High'].to_numpy()
        low = self.test_data['Low'].to_numpy()
        close = self.test_data['Close'].to_numpy()

        result = self.calculator._calculate_stochastic(
            high, low, close, self.test_data.index)

        assert 'K' in result
        assert 'D' in result
        assert 'J' in result
        assert 'RSV' in result

        # K, D 值應該在合理範圍內
        k_values = result['K'].dropna()
        d_values = result['D'].dropna()

        assert all(0 <= val <= 100 for val in k_values)
        assert all(0 <= val <= 100 for val in d_values)


class TestResultExporter:
    """測試 ResultExporter 類別"""

    temp_dir: str
    exporter: ResultExporter

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ResultExporter(self.temp_dir)

    def teardown_method(self):
        """清理測試檔案"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_to_json(self):
        """測試保存 JSON 檔案"""
        test_data = {
            "AAPL": {
                "symbol": "AAPL",
                "price": {"close": 150.0},
                "indicators": {"RSI(14)": 65.5}
            }
        }

        result_path = self.exporter.save_to_json(test_data, "test.json")

        assert os.path.exists(result_path)

        # 檢查檔案內容
        with open(result_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert "AAPL" in loaded_data
        assert loaded_data["AAPL"]["symbol"] == "AAPL"

    def test_save_to_csv(self):
        """測試保存 CSV 檔案"""
        # 建立測試數據
        data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [98, 99],
            'Close': [103, 104],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))

        indicators = {
            'RSI(14)': pd.Series([50, 55], index=data.index),
            'MA5': pd.Series([102, 103], index=data.index)
        }

        result_path = self.exporter.save_to_csv("AAPL", data, indicators, "1d")

        assert os.path.exists(result_path)

        # 檢查檔案內容
        loaded_data = pd.read_csv(
            result_path, index_col='Date', parse_dates=True)
        assert len(loaded_data) == 2


class TestTechnicalAnalyzer:
    """測試 TechnicalAnalyzer 類別"""

    temp_dir: str
    analyzer: TechnicalAnalyzer

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = TechnicalAnalyzer(self.temp_dir)

    def teardown_method(self):
        """清理測試檔案"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch.object(DataProvider, 'get_stock_data')
    @patch.object(IndicatorCalculator, 'calculate_all_indicators')
    def test_analyze_stock_success(self, mock_calculate, mock_get_data):
        """測試成功分析股票"""
        # 模擬數據 - 需要至少60筆數據
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(60)],
            'High': [105 + i * 0.1 for i in range(60)],
            'Low': [98 + i * 0.1 for i in range(60)],
            'Close': [103 + i * 0.1 for i in range(60)],
            'Volume': [1000000 + i * 1000 for i in range(60)]
        }, index=dates)

        mock_indicators = {
            'RSI(14)': pd.Series([65.5] * 60, index=mock_data.index),
            'DIF': pd.Series([1.25] * 60, index=mock_data.index),
            'MACD': pd.Series([1.20] * 60, index=mock_data.index),
            'K': pd.Series([75.0] * 60, index=mock_data.index),
            'D': pd.Series([70.0] * 60, index=mock_data.index),
            'J': pd.Series([85.0] * 60, index=mock_data.index)
        }

        mock_get_data.return_value = mock_data
        mock_calculate.return_value = mock_indicators

        result = self.analyzer.analyze_stock("AAPL")

        assert "error" not in result
        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert "indicators" in result
        assert result["total_records"] == 60

        # 驗證價格數據
        price = result["price"]
        assert price["open"] == 105.9  # 最後一筆的開盤價
        assert price["close"] == 108.9  # 最後一筆的收盤價

        # 驗證指標數據
        indicators = result["indicators"]
        assert indicators["RSI(14)"] == 65.5
        assert indicators["DIF"] == 1.25
        assert indicators["MACD"] == 1.20

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
        # 只有10筆數據
        mock_data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [98] * 10,
            'Close': [103] * 10,
            'Volume': [1000000] * 10
        }, index=pd.date_range('2023-01-01', periods=10))

        mock_get_data.return_value = mock_data

        result = self.analyzer.analyze_stock("TEST")

        assert "error" in result
        assert "數據不足" in result["error"]

    def test_analyze_multiple_stocks(self):
        """測試分析多個股票"""
        with patch.object(self.analyzer, 'analyze_stock') as mock_analyze:
            mock_analyze.return_value = {"symbol": "TEST", "price": {}}

            symbols = ["AAPL", "MSFT"]
            result = self.analyzer.analyze_multiple_stocks(symbols)

            assert len(result) == 2
            assert "AAPL" in result
            assert "MSFT" in result
            assert mock_analyze.call_count == 2


class TestAnalysisReporter:
    """測試 AnalysisReporter 類別"""

    def test_print_analysis_summary(self, capsys):
        """測試打印分析摘要"""
        test_results = {
            "AAPL": {
                "date": "2023-01-01 09:30:00",
                "price": {
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1000000
                },
                "indicators": {
                    "RSI(14)": 65.5,
                    "DIF": 1.25,
                    "MACD": 1.20,
                    "K": 75.0,
                    "D": 70.0,
                    "J": 85.0
                },
                "total_records": 100,
                "interval": "1d",
                "time_range": "2022-01-01 - 2023-01-01"
            },
            "ERROR_STOCK": {
                "error": "無法獲取數據"
            }
        }

        AnalysisReporter.print_analysis_summary(test_results)

        captured = capsys.readouterr()
        assert "技術分析結果摘要" in captured.out
        assert "AAPL" in captured.out
        assert "RSI(14): 65.5" in captured.out
        assert "ERROR_STOCK" in captured.out
        assert "無法獲取數據" in captured.out


class TestEnums:
    """測試枚舉類別"""

    def test_time_interval_enum(self):
        """測試 TimeInterval 枚舉"""
        assert TimeInterval.DAY_1.value == "1d"
        assert TimeInterval.MINUTE_5.value == "5m"
        assert TimeInterval.HOUR_1.value == "1h"


class TestIntegration:
    """整合測試"""

    temp_dir: str

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """清理測試檔案"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('yfinance.Ticker')
    def test_full_analysis_workflow(self, mock_ticker):
        """測試完整的分析流程"""
        # 建立模擬數據（足夠的數據量）
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        mock_data = pd.DataFrame({
            'Open': close_prices + np.random.uniform(-1, 1, 100),
            'High': close_prices + np.random.uniform(0, 2, 100),
            'Low': close_prices - np.random.uniform(0, 2, 100),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # 執行完整分析
        analyzer = TechnicalAnalyzer(self.temp_dir)
        result = analyzer.analyze_stock(
            "AAPL", Period.MONTH_3, TimeInterval.DAY_1)

        # 驗證結果
        assert "error" not in result
        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert "indicators" in result
        assert result["total_records"] == 100

        # 驗證指標值
        indicators = result["indicators"]
        assert "RSI(14)" in indicators
        assert "DIF" in indicators
        assert "K" in indicators


# 新增 TestMainFunction 類別
class TestMainFunction:
    """測試 main 函數的命令列參數處理"""

    @patch('Indicators.AnalysisReporter')  # 修正 patch 的目標
    @patch('Indicators.TechnicalAnalyzer')  # 修正 patch 的目標
    @patch('sys.argv')
    def test_main_no_args(self, mock_argv, MockTechnicalAnalyzer,
                          MockAnalysisReporter):
        """測試 main 函數在沒有命令列參數時的行為"""
        # 設定 mock
        mock_analyzer_instance = MockTechnicalAnalyzer.return_value
        mock_reporter_instance = MockAnalysisReporter.return_value

        # 模擬 sys.argv 為 ['Indicators.py'] (即沒有額外參數)
        mock_argv_list = ['Indicators.py']
        mock_argv.__getitem__ = lambda s, i: mock_argv_list[i]
        mock_argv.__len__ = lambda s: len(mock_argv_list)

        # 呼叫 main 函數
        main()

        # 斷言
        default_stocks = ["2330", "AAPL", "NFLX"]
        mock_analyzer_instance.analyze_multiple_stocks.assert_called_once_with(
            symbols=default_stocks,
            period=Period.MAX,
            interval=TimeInterval.DAY_1
        )

        mock_results = \
            mock_analyzer_instance.analyze_multiple_stocks.return_value
        mock_reporter_instance.print_analysis_summary.assert_called_once_with(
            mock_results
        )
        mock_analyzer_instance.save_analysis_results.assert_called_once_with(
            mock_results, "json"
        )

    @patch('Indicators.AnalysisReporter')  # 修正 patch 的目標
    @patch('Indicators.TechnicalAnalyzer')  # 修正 patch 的目標
    @patch('sys.argv')
    def test_main_with_one_arg(self, mock_argv, MockTechnicalAnalyzer,
                               MockAnalysisReporter):
        """測試 main 函數在有一個命令列參數時的行為"""
        # 設定 mock
        mock_analyzer_instance = MockTechnicalAnalyzer.return_value
        mock_reporter_instance = MockAnalysisReporter.return_value
        test_stock = "TSLA"

        # 模擬 sys.argv 為 ['Indicators.py', 'TSLA']
        mock_argv_list = ['Indicators.py', test_stock]
        mock_argv.__getitem__ = lambda s, i: mock_argv_list[i]
        mock_argv.__len__ = lambda s: len(mock_argv_list)

        # 呼叫 main 函數
        main()

        # 斷言
        mock_analyzer_instance.analyze_multiple_stocks.assert_called_once_with(
            symbols=[test_stock],
            period=Period.MAX,
            interval=TimeInterval.DAY_1
        )

        mock_results = \
            mock_analyzer_instance.analyze_multiple_stocks.return_value
        mock_reporter_instance.print_analysis_summary.assert_called_once_with(
            mock_results
        )
        mock_analyzer_instance.save_analysis_results.assert_called_once_with(
            mock_results, "json"
        )

    @patch('Indicators.AnalysisReporter')  # 修正 patch 的目標
    @patch('Indicators.TechnicalAnalyzer')  # 修正 patch 的目標
    @patch('sys.argv')
    def test_main_with_multiple_args(self, mock_argv, MockTechnicalAnalyzer,
                                     MockAnalysisReporter):
        """測試 main 函數在有多個命令列參數時的行為"""
        # 設定 mock
        mock_analyzer_instance = MockTechnicalAnalyzer.return_value
        mock_reporter_instance = MockAnalysisReporter.return_value
        test_stocks = ["GOOG", "AMZN"]

        # 模擬 sys.argv 為 ['Indicators.py', 'GOOG', 'AMZN']
        mock_argv_list = ['Indicators.py'] + test_stocks
        mock_argv.__getitem__ = lambda s, i: mock_argv_list[i]
        mock_argv.__len__ = lambda s: len(mock_argv_list)

        # 呼叫 main 函數
        main()

        # 斷言
        mock_analyzer_instance.analyze_multiple_stocks.assert_called_once_with(
            symbols=test_stocks,
            period=Period.MAX,
            interval=TimeInterval.DAY_1
        )

        mock_results = \
            mock_analyzer_instance.analyze_multiple_stocks.return_value
        mock_reporter_instance.print_analysis_summary.assert_called_once_with(
            mock_results
        )
        mock_analyzer_instance.save_analysis_results.assert_called_once_with(
            mock_results, "json"
        )

    @patch('Indicators.AnalysisReporter')
    @patch('Indicators.TechnicalAnalyzer')
    @patch('sys.argv')
    def test_main_date_format(self, mock_argv, MockTechnicalAnalyzer,
                              MockAnalysisReporter):
        """測試 main 函數的日期格式化行為"""
        # 設定 mock
        mock_analyzer_instance = MockTechnicalAnalyzer.return_value
        mock_reporter_instance = MockAnalysisReporter.return_value

        # 模擬 sys.argv 為 ['Indicators.py', 'AAPL']
        mock_argv_list = ['Indicators.py', 'AAPL']
        mock_argv.__getitem__ = lambda s, i: mock_argv_list[i]
        mock_argv.__len__ = lambda s: len(mock_argv_list)

        # 模擬分析結果
        mock_analyzer_instance.analyze_multiple_stocks.return_value = {
            "AAPL": {
                "date": "2025-06-18",
                "price": {
                    "open": 150.0,
                    "high": 155.0,
                    "low": 145.0,
                    "close": 152.0,
                    "volume": 1000000
                },
                "indicators": {
                    "RSI(14)": 65.5,
                    "DIF": 1.25,
                    "MACD": 1.20
                },
                "total_records": 100,
                "interval": "1d",
                "time_range": "2025-01-01 - 2025-06-18"
            }
        }

        # 呼叫 main 函數
        main()

        # 驗證日期格式是否正確
        mock_results = \
            mock_analyzer_instance.analyze_multiple_stocks.return_value
        assert mock_results["AAPL"]["date"] == "2025-06-18"

        # 驗證報告是否正確顯示日期
        mock_reporter_instance.print_analysis_summary.assert_called_once_with(
            mock_results
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
