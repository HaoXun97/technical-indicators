"""
股票技術分析與資料庫整合系統
結合股票數據抓取、技術指標計算和 SQL Server 資料庫儲存功能
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from talib._ta_lib import MA_Type
import configparser
import logging
import time
import sys
from datetime import datetime
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import warnings

# 抑制警告
warnings.filterwarnings("ignore")


class TimeInterval(Enum):
    """時間間隔枚舉"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"


class Period(Enum):
    """時間週期枚舉"""
    DAY_1 = "1d"
    DAY_5 = "5d"
    MONTH_1 = "1mo"
    MONTH_3 = "3mo"
    MONTH_6 = "6mo"
    YEAR_1 = "1y"
    YEAR_2 = "2y"
    YEAR_5 = "5y"
    YEAR_10 = "10y"
    MAX = "max"


@dataclass
class ProcessResult:
    """處理結果統計"""
    symbol: str
    success: bool
    new_records: int = 0
    updated_records: int = 0
    total_records: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0
    date_range: Optional[str] = None


class DatabaseConfig:
    """資料庫配置類"""

    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        self.server = self.config.get('database', 'server')
        self.database = self.config.get('database', 'database')
        self.driver = self.config.get('database', 'driver')

        try:
            self.username = self.config.get('database', 'username')
            self.password = self.config.get('database', 'password')
        except (configparser.NoOptionError, configparser.NoSectionError):
            self.username = None
            self.password = None

    def get_sqlalchemy_url(self) -> str:
        """生成 SQLAlchemy 連接字串"""
        if self.username and self.password:
            return (f"mssql+pyodbc://{self.username}:{self.password}@"
                    f"{self.server}/{self.database}?driver={self.driver}")
        else:
            return (f"mssql+pyodbc://@{self.server}/{self.database}"
                    f"?driver={self.driver}&trusted_connection=yes")


class ProgressReporter:
    """進度報告器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str):
        self.logger.info(message)
        print(f"ℹ️  {message}")

    def success(self, message: str):
        self.logger.info(f"SUCCESS: {message}")
        print(f"✅ {message}")

    def warning(self, message: str):
        self.logger.warning(message)
        print(f"⚠️  {message}")

    def error(self, message: str):
        self.logger.error(message)
        print(f"❌ {message}")

    def progress(self, message: str):
        self.logger.debug(message)
        print(f"🔄 {message}")


class StockDataProvider:
    """股票數據提供者"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}

    def get_stock_data(self, symbol: str, period: Union[Period, str, int],
                       interval: Union[TimeInterval, str]
                       ) -> Optional[pd.DataFrame]:
        """獲取股票數據 - 支援動態期間參數"""
        try:
            formatted_symbol = self._format_symbol(symbol)

            # 處理不同類型的 period 參數
            if isinstance(period, Period):
                period_str = period.value
            elif isinstance(period, int):
                period_str = f"{period}d"  # 整數天數轉換為天數字串
            else:
                period_str = period

            interval_str = interval.value if isinstance(
                interval, TimeInterval) else interval

            # 調整期間以避免API限制
            adjusted_period = self._adjust_period_for_interval(
                period_str, interval_str)

            # 嘗試獲取數據
            symbols_to_try = [formatted_symbol]
            if (formatted_symbol.endswith('.TW') and symbol.isdigit()
                    and len(symbol) == 4):
                symbols_to_try.append(f"{symbol}.TWO")

            for attempt_symbol in symbols_to_try:
                # 抑制 yfinance 輸出
                import io
                from contextlib import redirect_stderr, redirect_stdout

                loggers_to_silence = ['yfinance', 'urllib3', 'requests']
                original_levels = {}
                for logger_name in loggers_to_silence:
                    logger = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger.level
                    logger.setLevel(logging.CRITICAL)

                devnull = io.StringIO()

                try:
                    with redirect_stderr(devnull), redirect_stdout(devnull):
                        ticker = yf.Ticker(attempt_symbol)
                        data = ticker.history(
                            period=adjusted_period,
                            interval=interval_str,
                            auto_adjust=False,
                            actions=False,
                            timeout=10
                        )

                        if not data.empty:
                            # 移除 Adj Close 欄位
                            if 'Adj Close' in data.columns:
                                data = data.drop(columns=['Adj Close'])

                            self.logger.info(
                                f"成功獲取 {attempt_symbol} 數據：{len(data)} 筆")
                            return data

                finally:
                    # 恢復日誌級別
                    for logger_name, level in original_levels.items():
                        logging.getLogger(logger_name).setLevel(level)

            self.logger.error(f"無法獲取 {symbol} 的數據")
            return None

        except Exception as e:
            self.logger.error(f"獲取 {symbol} 數據錯誤: {e}")
            return None

    def _format_symbol(self, symbol: str) -> str:
        """格式化股票代號"""
        if symbol.isdigit() and len(symbol) == 4:
            return f"{symbol}.TW"

        if symbol.endswith((".TW", ".TWO")):
            return symbol

        if (any(c.isalpha() for c in symbol) and
                len(symbol) <= 5 and not symbol.isdigit()):
            return symbol

        return f"{symbol}.TW"

    def _adjust_period_for_interval(self, period: str, interval: str) -> str:
        """根據間隔調整期間"""
        if interval in ["1h", "2h", "4h", "6h", "12h"]:
            if period in ["max", "5y", "10y"]:
                return "730d"
        elif interval in ["1m", "2m", "5m", "15m", "30m", "90m"]:
            if period in ["max", "5y", "10y", "2y", "1y"]:
                return "60d"
        return period


class TechnicalIndicatorCalculator:
    """技術指標計算器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算所有技術指標並返回完整的 DataFrame"""
        if data.empty or len(data) < 60:
            raise ValueError("數據不足，無法計算技術指標")

        # 創建結果 DataFrame
        result_df = data.copy()

        # 轉換為 numpy arrays
        high = data["High"].values.astype("float64")
        low = data["Low"].values.astype("float64")
        close = data["Close"].values.astype("float64")
        # volume = data["Volume"].values.astype("float64")

        try:
            # RSI 指標
            result_df['RSI(5)'] = talib.RSI(close, timeperiod=5)
            result_df['RSI(7)'] = talib.RSI(close, timeperiod=7)
            result_df['RSI(10)'] = talib.RSI(close, timeperiod=10)
            result_df['RSI(14)'] = talib.RSI(close, timeperiod=14)
            result_df['RSI(21)'] = talib.RSI(close, timeperiod=21)

            # MACD 指標
            macd_line, signal_line, macd_histogram = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9)
            result_df['DIF'] = macd_line
            result_df['MACD'] = signal_line
            result_df['MACD_Histogram'] = macd_histogram

            # KDJ 指標
            kdj_data = self._calculate_kdj(high, low, close)
            result_df['RSV'] = kdj_data['RSV']
            result_df['K'] = kdj_data['K']
            result_df['D'] = kdj_data['D']
            result_df['J'] = kdj_data['J']

            # 移動平均線
            result_df['MA5'] = talib.SMA(close, timeperiod=5)
            result_df['MA10'] = talib.SMA(close, timeperiod=10)
            result_df['MA20'] = talib.SMA(close, timeperiod=20)
            result_df['MA60'] = talib.SMA(close, timeperiod=60)
            result_df['EMA12'] = talib.EMA(close, timeperiod=12)
            result_df['EMA26'] = talib.EMA(close, timeperiod=26)

            # 布林通道
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=MA_Type.SMA)
            result_df['BB_Upper'] = bb_upper
            result_df['BB_Middle'] = bb_middle
            result_df['BB_Lower'] = bb_lower

            # 其他指標
            result_df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            result_df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
            result_df['WILLR'] = talib.WILLR(high, low, close, timeperiod=20)
            result_df['MOM'] = talib.MOM(close, timeperiod=10)

            self.logger.info("技術指標計算完成")
            return result_df

        except Exception as e:
            self.logger.error(f"計算技術指標錯誤: {e}")
            raise

    def _calculate_kdj(self, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> Dict[str, np.ndarray]:
        """計算 KDJ 指標"""
        n = 9
        rsv = np.full(len(close), np.nan)
        k_values = np.full(len(close), np.nan)
        d_values = np.full(len(close), np.nan)

        # 計算 RSV
        for i in range(n - 1, len(close)):
            period_high = np.max(high[i - n + 1: i + 1])
            period_low = np.min(low[i - n + 1: i + 1])

            if period_high != period_low:
                rsv_value = ((close[i] - period_low) /
                             (period_high - period_low)) * 100
                rsv[i] = max(0, min(100, rsv_value))
            else:
                rsv[i] = 50

        # 計算 K 和 D
        first_valid_idx = n - 1
        k_values[first_valid_idx] = rsv[first_valid_idx] if not np.isnan(
            rsv[first_valid_idx]) else 50
        d_values[first_valid_idx] = k_values[first_valid_idx]

        for i in range(first_valid_idx + 1, len(close)):
            if not np.isnan(rsv[i]):
                k_values[i] = (2 / 3) * k_values[i - 1] + (1 / 3) * rsv[i]
                d_values[i] = (2 / 3) * d_values[i - 1] + (1 / 3) * k_values[i]

        # 計算 J
        j_values = 3 * k_values - 2 * d_values

        return {
            'RSV': rsv,
            'K': k_values,
            'D': d_values,
            'J': j_values
        }


class StockAnalyzerDB:
    """股票分析資料庫整合系統"""

    def __init__(self, config_file: str = "config.ini"):
        # 初始化配置
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        self.db_config = DatabaseConfig(config_file)
        self.data_provider = StockDataProvider()
        self.indicator_calculator = TechnicalIndicatorCalculator()

        # 設置日誌
        log_level = self.config.get(
            'import_settings', 'log_level', fallback='INFO')
        self.logger = self._setup_logger(log_level)
        self.reporter = ProgressReporter(self.logger)

        # 初始化資料庫連接
        self.engine = create_engine(
            self.db_config.get_sqlalchemy_url(),
            pool_pre_ping=True,
            pool_recycle=3600
        )

        # 確保資料表存在
        self._ensure_table_exists()

    def _setup_logger(self, log_level: str = 'INFO') -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger('StockAnalyzerDB')
        logger.handlers = []

        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)

        try:
            file_handler = logging.FileHandler(
                'stock_analyzer.log', encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

        return logger

    @contextmanager
    def get_connection(self):
        """資料庫連接上下文管理器"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def test_connection(self) -> bool:
        """測試資料庫連接"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            self.reporter.success("資料庫連接測試成功")
            return True
        except Exception as e:
            self.reporter.error(f"資料庫連接測試失敗: {e}")
            return False

    def _ensure_table_exists(self):
        """確保資料表存在"""
        create_table_sql = text("""
        IF NOT EXISTS (SELECT * FROM sysobjects
                                WHERE name='stock_data' AND xtype='U')
        BEGIN
            CREATE TABLE stock_data (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol NVARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open_price DECIMAL(18,6),
                high_price DECIMAL(18,6),
                low_price DECIMAL(18,6),
                close_price DECIMAL(18,6),
                volume BIGINT,
                rsi_5 DECIMAL(8,4),
                rsi_7 DECIMAL(8,4),
                rsi_10 DECIMAL(8,4),
                rsi_14 DECIMAL(8,4),
                rsi_21 DECIMAL(8,4),
                dif DECIMAL(10,6),
                macd DECIMAL(10,6),
                macd_histogram DECIMAL(10,6),
                rsv DECIMAL(8,4),
                k_value DECIMAL(8,4),
                d_value DECIMAL(8,4),
                j_value DECIMAL(8,4),
                ma5 DECIMAL(18,6),
                ma10 DECIMAL(18,6),
                ma20 DECIMAL(18,6),
                ma60 DECIMAL(18,6),
                ema12 DECIMAL(18,6),
                ema26 DECIMAL(18,6),
                bb_upper DECIMAL(18,6),
                bb_middle DECIMAL(18,6),
                bb_lower DECIMAL(18,6),
                atr DECIMAL(10,6),
                cci DECIMAL(10,4),
                willr DECIMAL(8,4),
                mom DECIMAL(10,6),
                created_at DATETIME2 DEFAULT GETDATE(),
                updated_at DATETIME2 DEFAULT GETDATE(),
                CONSTRAINT UK_stock_symbol_date UNIQUE (symbol, date)
            );

            -- 創建索引
            CREATE NONCLUSTERED INDEX IX_stock_data_symbol_date
                ON stock_data (symbol, date DESC);
            CREATE NONCLUSTERED INDEX IX_stock_data_date
                ON stock_data (date DESC);
            CREATE NONCLUSTERED INDEX IX_stock_data_symbol
                ON stock_data (symbol);
        END
        """)

        try:
            with self.get_connection() as conn:
                conn.execute(create_table_sql)
                conn.commit()
            self.logger.info("資料表檢查/創建完成")
        except Exception as e:
            self.reporter.error(f"創建資料表失敗: {e}")
            raise

    def get_existing_data_info(self, symbol: str) -> Dict[str, Any]:
        """獲取已存在的數據資訊"""
        try:
            with self.get_connection() as conn:
                query = text("""
                SELECT
                    COUNT(*) as record_count,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    MAX(updated_at) as last_updated
                FROM stock_data
                WHERE symbol = :symbol
                """)

                result = conn.execute(query, {"symbol": symbol}).fetchone()

                if result and result[0] > 0:
                    return {
                        'exists': True,
                        'record_count': result[0],
                        'earliest_date': result[1],
                        'latest_date': result[2],
                        'last_updated': result[3]
                    }
                else:
                    return {'exists': False}

        except Exception as e:
            self.logger.warning(f"獲取數據資訊失敗: {e}")
            return {'exists': False}

    def process_stock(self, symbol: str,
                      period: Union[Period, str] = Period.MAX,
                      interval: Union[TimeInterval, str] = TimeInterval.DAY_1,
                      force_update: bool = False
                      ) -> ProcessResult:
        """處理單一股票：獲取數據、計算指標、儲存到資料庫

        Args:
            symbol: 股票代號
            period: 時間週期
            interval: 時間間隔
            force_update: 是否強制更新所有數據（包括已存在的數據）
        """
        start_time = time.time()
        result = ProcessResult(symbol=symbol, success=False)
        new_data = pd.DataFrame()  # 初始化 new_data 變數

        try:
            self.reporter.progress(f"開始處理股票 {symbol}")

            # 1. 首先檢查資料庫中的現有數據
            existing_info = self.get_existing_data_info(symbol)

            # 2. 獲取完整的股票數據（根據指定的期間）
            fetch_period = period.value if isinstance(
                period, Period) else period

            data = self.data_provider.get_stock_data(
                symbol, fetch_period, interval)
            if data is None or data.empty:
                result.error_message = "無法獲取股票數據"
                return result

            self.reporter.info(f"{symbol}: 獲取到 {len(data)} 筆原始數據")

            # 3. 計算技術指標
            try:
                complete_data = (
                    self.indicator_calculator.calculate_all_indicators(data)
                )
                self.reporter.info(f"{symbol}: 技術指標計算完成")
            except Exception as e:
                result.error_message = f"技術指標計算失敗: {e}"
                return result

            # 4. 判斷需要儲存的數據範圍
            if force_update:
                # 強制更新模式：重新處理所有數據
                self.reporter.info(f"{symbol}: 強制更新模式 - 將重新處理所有數據")
                new_data = complete_data
                saved_count = self._save_data_to_db(
                    complete_data, symbol, mode='replace')
                result.new_records = saved_count
                self.reporter.success(
                    f"{symbol}: 強制更新完成 - 處理了 {saved_count} 筆數據")

            elif existing_info['exists']:
                earliest_db_date = existing_info['earliest_date']
                latest_db_date = existing_info['latest_date']

                # 統一處理日期格式
                if hasattr(earliest_db_date, 'date'):
                    earliest_db_date = earliest_db_date.date()
                elif isinstance(earliest_db_date, str):
                    from datetime import datetime
                    earliest_db_date = datetime.strptime(
                        earliest_db_date, '%Y-%m-%d').date()

                if hasattr(latest_db_date, 'date'):
                    latest_db_date = latest_db_date.date()
                elif isinstance(latest_db_date, str):
                    from datetime import datetime
                    latest_db_date = datetime.strptime(
                        latest_db_date, '%Y-%m-%d').date()

                # 分析新獲取的數據範圍
                data_dates = complete_data.index.to_series().dt.date
                earliest_data_date = data_dates.min()
                latest_data_date = data_dates.max()

                # 找出需要新增的數據（包括更早和更新的數據）
                historical_data = complete_data[data_dates < earliest_db_date]
                future_data = complete_data[data_dates > latest_db_date]

                # 檢查重疊期間的數據是否需要更新
                overlap_data = complete_data[
                    (data_dates >= earliest_db_date) &
                    (data_dates <= latest_db_date)
                ]

                # 檢測需要更新的重疊數據
                outdated_data = self._detect_outdated_data(
                    symbol, overlap_data)
                if not outdated_data.empty:
                    self.reporter.info(
                        f"{symbol}: 檢測到 {len(outdated_data)} 筆數據需要更新")

                # 合併所有需要處理的數據
                new_data = pd.concat(
                    [historical_data, future_data, outdated_data])

                if not new_data.empty:
                    saved_count = self._save_data_to_db(
                        new_data, symbol, mode='append')
                    result.new_records = saved_count

                    # 提供詳細的更新資訊
                    historical_count = len(historical_data)
                    future_count = len(future_data)
                    updated_count = len(outdated_data)

                    update_msg = f"{symbol}: 處理 {saved_count} 筆數據"
                    details = []
                    if historical_count > 0:
                        details.append(f"歷史: {historical_count} 筆")
                    if future_count > 0:
                        details.append(f"最新: {future_count} 筆")
                    if updated_count > 0:
                        details.append(f"更新: {updated_count} 筆")

                    if details:
                        update_msg += f" ({', '.join(details)})"

                    self.reporter.success(update_msg)

                    # 顯示數據覆蓋範圍
                    if historical_count > 0:
                        hist_start = historical_data.index.min().strftime(
                            '%Y-%m-%d')
                        hist_end = historical_data.index.max().strftime(
                            '%Y-%m-%d')
                        self.reporter.info(
                            f"{symbol}: 新增歷史數據範圍: {hist_start} ~ {hist_end}")

                    if future_count > 0:
                        fut_start = future_data.index.min().strftime(
                            '%Y-%m-%d')
                        fut_end = future_data.index.max().strftime(
                            '%Y-%m-%d')
                        self.reporter.info(
                            f"{symbol}: 新增最新數據範圍: {fut_start} ~ {fut_end}")

                    if updated_count > 0:
                        upd_start = outdated_data.index.min().strftime(
                            '%Y-%m-%d')
                        upd_end = outdated_data.index.max().strftime(
                            '%Y-%m-%d')
                        self.reporter.info(
                            f"{symbol}: 更新數據範圍: {upd_start} ~ {upd_end}")

                else:
                    # 檢查是否需要提示使用強制更新
                    try:
                        # 安全地比較日期，避免 Series 布林值問題
                        earliest_data_condition = (
                            earliest_data_date <= latest_db_date)
                        latest_data_condition = (
                            latest_data_date >= earliest_db_date)

                        # 檢查條件是否為 Series，如果是則使用 any() 方法
                        if (hasattr(earliest_data_condition, 'any') and
                                not isinstance(earliest_data_condition, bool)):
                            earliest_data_check = bool(
                                earliest_data_condition.any())
                        else:
                            earliest_data_check = bool(earliest_data_condition)

                        if (hasattr(latest_data_condition, 'any') and
                                not isinstance(latest_data_condition, bool)):
                            latest_data_check = bool(
                                latest_data_condition.any())
                        else:
                            latest_data_check = bool(latest_data_condition)

                        if earliest_data_check and latest_data_check:
                            self.reporter.info(
                                f"{symbol}: 數據庫已包含所有可用數據，無需更新")
                            self.reporter.info(
                                f"{symbol}: 如需更新已存在的數據，"
                                f"請使用 force_update=True 參數")
                        else:
                            self.reporter.info(
                                f"{symbol}: 數據庫已包含所有可用數據，無需更新")
                    except Exception as date_compare_error:
                        self.logger.warning(
                            f"{symbol}: 日期比較時發生錯誤: {date_compare_error}")
                        self.reporter.info(
                            f"{symbol}: 數據庫已包含所有可用數據，無需更新")

                    result.success = True
                    result.total_records = existing_info['record_count']
                    result.processing_time = time.time() - start_time
                    return result
            else:
                # 全新股票，儲存所有數據
                new_data = complete_data  # 確保 new_data 被定義
                saved_count = self._save_data_to_db(
                    complete_data, symbol, mode='replace')
                result.new_records = saved_count
                self.reporter.success(f"{symbol}: 儲存 {saved_count} 筆完整數據")

            # 5. 更新結果統計
            result.success = True
            result.total_records = len(complete_data)
            result.processing_time = time.time() - start_time

            # 設置日期範圍
            if not new_data.empty:
                start_date = new_data.index.min().strftime('%Y-%m-%d')
                end_date = new_data.index.max().strftime('%Y-%m-%d')
                result.date_range = f"{start_date} ~ {end_date}"
            else:
                start_date = complete_data.index.min().strftime('%Y-%m-%d')
                end_date = complete_data.index.max().strftime('%Y-%m-%d')
                result.date_range = f"{start_date} ~ {end_date}"

        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            self.reporter.error(f"{symbol}: 處理失敗 - {e}")

        return result

    def _detect_outdated_data(self, symbol: str,
                              overlap_data: pd.DataFrame) -> pd.DataFrame:
        """檢測需要更新的重疊數據

        Args:
            symbol: 股票代號
            overlap_data: 與資料庫日期範圍重疊的數據

        Returns:
            需要更新的數據 DataFrame
        """
        if overlap_data.empty:
            return pd.DataFrame()

        try:
            # 只檢查最近30天的數據以提升效率
            from datetime import datetime, timedelta

            # 計算30天前的日期
            thirty_days_ago = datetime.now().date() - timedelta(days=30)

            # 過濾出最近30天的數據
            overlap_dates = overlap_data.index.to_series().dt.date
            recent_overlap_data = overlap_data[overlap_dates >=
                                               thirty_days_ago]

            if recent_overlap_data.empty:
                self.logger.info(f"{symbol}: 最近30天內沒有重疊數據需要檢查")
                return pd.DataFrame()

            self.logger.info(
                f"{symbol}: 檢查最近30天內的 {len(recent_overlap_data)} 筆重疊數據")

            # 獲取資料庫中對應日期的數據
            dates_to_check = (recent_overlap_data.index.to_series()
                              .dt.date.tolist())

            # 如果日期數量過多，分批處理以避免 SQL Server 參數限制
            batch_size = 1000  # SQL Server 建議的批次大小
            all_db_data = []

            for i in range(0, len(dates_to_check), batch_size):
                batch_dates = dates_to_check[i:i + batch_size]

                # 構建查詢條件
                date_placeholders = ','.join([
                    ':date' + str(j) for j in range(len(batch_dates))
                ])

                query = text(f"""
                SELECT date, open_price, high_price,
                             low_price, close_price, volume
                FROM stock_data
                WHERE symbol = :symbol
                AND date IN ({date_placeholders})
                ORDER BY date
                """)

                # 準備參數
                params = {'symbol': symbol}
                for j, date in enumerate(batch_dates):
                    params[f'date{j}'] = date.strftime('%Y-%m-%d')

                with self.get_connection() as conn:
                    batch_results = conn.execute(query, params).fetchall()
                    all_db_data.extend(batch_results)

            if not all_db_data:
                return recent_overlap_data  # 如果資料庫中沒有數據，返回所有重疊數據

            # 轉換為 DataFrame 以便比較
            db_df = pd.DataFrame(
                all_db_data,
                columns=['date', 'open_price', 'high_price',
                         'low_price', 'close_price', 'volume']
            )
            db_df['date'] = pd.to_datetime(db_df['date']).dt.date
            db_df.set_index('date', inplace=True)

            outdated_rows = []

            for date_val in recent_overlap_data.index.to_series().dt.date:
                if date_val in db_df.index:
                    # 比較 OHLCV 數據
                    date_mask = (recent_overlap_data.index
                                 .to_series().dt.date == date_val)
                    new_row = recent_overlap_data.loc[date_mask].iloc[0]
                    db_row = db_df.loc[date_val]

                    # 檢查是否有重要差異（調整容差以檢測手動修改）
                    tolerance = 1e-3  # 增加容差到 0.001，能檢測到手動修改的差異
                    price_fields = ['Open', 'High', 'Low', 'Close']
                    db_price_fields = [
                        'open_price', 'high_price', 'low_price', 'close_price'
                    ]

                    needs_update = False

                    # 檢查價格數據
                    for new_field, db_field in zip(price_fields,
                                                   db_price_fields):
                        # 安全地處理新數據值
                        new_val_raw = new_row[new_field]
                        if pd.notna(new_val_raw):
                            # 如果是 Series，取第一個值
                            if hasattr(new_val_raw, 'iloc'):
                                new_val = float(new_val_raw.iloc[0])
                            else:
                                new_val = float(new_val_raw)
                        else:
                            new_val = None

                        # 安全地處理資料庫值
                        try:
                            db_val_raw = db_row[db_field]
                            # 如果是 Series，取第一個值
                            if hasattr(db_val_raw, 'iloc'):
                                db_val_raw = db_val_raw.iloc[0]

                            # 使用 pandas 的 isna 函數並檢查結果
                            if hasattr(pd.isna(db_val_raw), 'any'):
                                # 如果是 Series，使用 any() 方法
                                is_na = pd.isna(db_val_raw).any()
                            else:
                                # 如果是單一值，直接使用布林值
                                is_na = bool(pd.isna(db_val_raw))

                            if is_na:
                                db_val = None
                            else:
                                # 安全地轉換為 float
                                if hasattr(db_val_raw, 'iloc'):
                                    db_val = float(db_val_raw.iloc[0])
                                else:
                                    db_val = float(db_val_raw)
                        except (ValueError, TypeError):
                            db_val = None

                        if new_val is None and db_val is None:
                            continue
                        elif new_val is None or db_val is None:
                            needs_update = True
                            self.logger.info(
                                f"{symbol}: 檢測到 {date_val} {new_field} 值不一致 - "
                                f"新值: {new_val}, 資料庫值: {db_val}")
                            break
                        elif abs(new_val - db_val) > tolerance:
                            needs_update = True
                            self.logger.info(
                                f"{symbol}: 檢測到 {date_val} {new_field} "
                                f"差異超過容差 - "
                                f"新值: {new_val:.6f}, 資料庫值: {db_val:.6f}, "
                                f"差異: {abs(float(new_val) - db_val):.6f}")
                            break

                    # 檢查成交量
                    if not needs_update:
                        new_vol_raw = new_row['Volume']
                        if pd.notna(new_vol_raw):
                            # 如果是 Series，取第一個值
                            if hasattr(new_vol_raw, 'iloc'):
                                new_vol = int(new_vol_raw.iloc[0])
                            else:
                                new_vol = int(new_vol_raw)
                        else:
                            new_vol = None

                        # 安全地處理資料庫成交量值
                        try:
                            db_vol_raw = db_row['volume']
                            # 如果是 Series，取第一個值
                            if hasattr(db_vol_raw, 'iloc'):
                                db_vol_raw = db_vol_raw.iloc[0]

                            # 使用 pandas 的 isna 函數並檢查結果
                            if hasattr(pd.isna(db_vol_raw), 'any'):
                                # 如果是 Series，使用 any() 方法
                                is_na = pd.isna(db_vol_raw).any()
                            else:
                                # 如果是單一值，直接使用布林值
                                is_na = bool(pd.isna(db_vol_raw))

                            if is_na:
                                db_vol = None
                            else:
                                # 安全地轉換為 int
                                if hasattr(db_vol_raw, 'iloc'):
                                    db_vol = int(db_vol_raw.iloc[0])
                                else:
                                    db_vol = int(db_vol_raw)
                        except (ValueError, TypeError):
                            db_vol = None

                        if new_vol != db_vol:
                            needs_update = True
                            self.logger.info(
                                f"{symbol}: 檢測到 {date_val} Volume 值不一致 - "
                                f"新值: {new_vol}, 資料庫值: {db_vol}")
                            # break  # 這裡不需要立即中斷，繼續檢查其他欄位

                    # 檢查是否所有價格都為零（可能被手動修改）
                    if not needs_update:
                        try:
                            open_price_val = db_row['open_price']
                            high_price_val = db_row['high_price']
                            low_price_val = db_row['low_price']
                            close_price_val = db_row['close_price']

                            # 如果是 Series，取第一個值
                            if hasattr(open_price_val, 'iloc'):
                                open_price_val = open_price_val.iloc[0]
                            if hasattr(high_price_val, 'iloc'):
                                high_price_val = high_price_val.iloc[0]
                            if hasattr(low_price_val, 'iloc'):
                                low_price_val = low_price_val.iloc[0]
                            if hasattr(close_price_val, 'iloc'):
                                close_price_val = close_price_val.iloc[0]

                            # 安全地轉換為 float 並檢查
                            try:
                                # 安全地處理每個價格值
                                if hasattr(open_price_val, 'iloc'):
                                    open_val = (float(open_price_val.iloc[0])
                                                if open_price_val is not None
                                                else 0)
                                else:
                                    open_val = (float(open_price_val)
                                                if open_price_val is not None
                                                else 0)

                                if hasattr(high_price_val, 'iloc'):
                                    high_val = (float(high_price_val.iloc[0])
                                                if high_price_val is not None
                                                else 0)
                                else:
                                    high_val = (float(high_price_val)
                                                if high_price_val is not None
                                                else 0)

                                if hasattr(low_price_val, 'iloc'):
                                    low_val = (float(low_price_val.iloc[0])
                                               if low_price_val is not None
                                               else 0)
                                else:
                                    low_val = (float(low_price_val)
                                               if low_price_val is not None
                                               else 0)

                                if hasattr(close_price_val, 'iloc'):
                                    close_val = (float(close_price_val.iloc[0])
                                                 if close_price_val is not None
                                                 else 0)
                                else:
                                    close_val = (float(close_price_val)
                                                 if close_price_val is not None
                                                 else 0)

                                if (open_val == 0 and high_val == 0 and
                                        low_val == 0 and close_val == 0):
                                    needs_update = True
                                    self.logger.info(
                                        f"{symbol}: 檢測到異常數據 {date_val} - "
                                        f"所有價格為0"
                                    )
                            except (ValueError, TypeError):
                                # 如果轉換失敗，也認為需要更新
                                needs_update = True
                        except Exception:
                            # 發生其他錯誤時，也認為需要更新
                            needs_update = True

                    if needs_update:
                        outdated_rows.append(new_row.name)
                        self.logger.debug(f"{symbol}: 數據需要更新 {date_val}")
                else:
                    # 資料庫中沒有這個日期的數據，需要新增
                    date_series = recent_overlap_data.index.to_series()
                    matching_indices = recent_overlap_data.loc[
                        date_series.dt.date == date_val
                    ].index
                    if len(matching_indices) > 0:
                        outdated_rows.append(matching_indices[0])

            if outdated_rows:
                result = recent_overlap_data.loc[outdated_rows]
                self.logger.info(
                    f"{symbol}: 找到 {len(result)} 筆需要更新的數據（最近30天內）")
                return result
            else:
                self.logger.info(f"{symbol}: 最近30天內沒有需要更新的數據")
                return pd.DataFrame()

        except Exception as e:
            self.logger.warning(f"{symbol}: 檢測過時數據時發生錯誤: {e}")
            # 發生錯誤時，為了安全起見，不更新任何重疊數據
            return pd.DataFrame()

    def _save_data_to_db(self, data: pd.DataFrame,
                         symbol: str, mode: str = 'append') -> int:
        """儲存數據到資料庫"""
        if data.empty:
            self.logger.warning(f"{symbol}: 要儲存的數據為空")
            return 0

        self.logger.info(f"{symbol}: 準備儲存 {len(data)} 筆數據，模式: {mode}")

        # 準備資料庫格式的數據
        db_data = self._prepare_data_for_db(data, symbol)

        if db_data.empty:
            self.logger.error(f"{symbol}: 準備資料庫格式的數據失敗，結果為空")
            return 0

        self.logger.info(f"{symbol}: 準備了 {len(db_data)} 筆資料庫格式數據")

        try:
            if mode == 'replace':
                # 先刪除現有數據
                with self.get_connection() as conn:
                    delete_sql = text(
                        "DELETE FROM stock_data WHERE symbol = :symbol")
                    delete_result = conn.execute(
                        delete_sql, {"symbol": symbol})
                    deleted_rows = delete_result.rowcount
                    conn.commit()
                    self.logger.info(f"{symbol}: 刪除了 {deleted_rows} 筆舊數據")

            # 使用 UPSERT 方式儲存
            saved_count = self._upsert_data(db_data)
            self.logger.info(f"{symbol}: 成功儲存 {saved_count} 筆數據")
            return saved_count

        except Exception as e:
            self.logger.error(f"{symbol}: 儲存數據失敗: {e}")
            # 打印詳細的錯誤資訊
            import traceback
            self.logger.error(f"{symbol}: 詳細錯誤: {traceback.format_exc()}")
            raise

    def _prepare_data_for_db(self, data: pd.DataFrame,
                             symbol: str) -> pd.DataFrame:
        """準備資料庫格式的數據"""
        try:
            self.logger.info(f"{symbol}: 開始準備資料庫格式數據，原始數據形狀: {data.shape}")
            self.logger.info(f"{symbol}: 原始數據欄位: {list(data.columns)}")

            # 先獲取行數，然後創建包含所有必要欄位的 DataFrame
            num_rows = len(data)

            # 安全地處理日期索引 - 完全避免 .date 屬性
            try:
                # 確保索引是 datetime 類型
                if isinstance(data.index, pd.DatetimeIndex):
                    datetime_index = data.index
                else:
                    datetime_index = pd.to_datetime(data.index)

                # 使用 normalize() 方法去除時間部分，只保留日期
                dates = datetime_index.normalize().date

            except (AttributeError, TypeError):
                # 如果還是失敗，使用更安全的方法
                try:
                    datetime_index = pd.to_datetime(data.index)
                    # 轉換為日期字符串，然後再轉回日期對象
                    date_strings = datetime_index.strftime('%Y-%m-%d')
                    dates = [datetime.strptime(ds, '%Y-%m-%d').date()
                             for ds in date_strings]
                except Exception as e:
                    self.logger.error(f"{symbol}: 日期轉換失敗: {e}")
                    # 最後備用方案：使用字符串格式
                    dates = pd.to_datetime(data.index).strftime('%Y-%m-%d')

            # 基本資料 - 正確創建 DataFrame
            db_data = pd.DataFrame({
                'symbol': [symbol] * num_rows,  # 創建與數據行數相同的 symbol 欄位
                'date': dates,
                'open_price': data['Open'].where(
                    pd.notnull(data['Open']), None),
                'high_price': data['High'].where(
                    pd.notnull(data['High']), None),
                'low_price': data['Low'].where(
                    pd.notnull(data['Low']), None),
                'close_price': data['Close'].where(
                    pd.notnull(data['Close']), None),
                'volume': data['Volume'].where(
                    pd.notnull(data['Volume']), None).astype('Int64')
            })

            # 技術指標映射
            indicator_mapping = {
                'RSI(5)': 'rsi_5',
                'RSI(7)': 'rsi_7',
                'RSI(10)': 'rsi_10',
                'RSI(14)': 'rsi_14',
                'RSI(21)': 'rsi_21',
                'DIF': 'dif',
                'MACD': 'macd',
                'MACD_Histogram': 'macd_histogram',
                'RSV': 'rsv',
                'K': 'k_value',
                'D': 'd_value',
                'J': 'j_value',
                'MA5': 'ma5',
                'MA10': 'ma10',
                'MA20': 'ma20',
                'MA60': 'ma60',
                'EMA12': 'ema12',
                'EMA26': 'ema26',
                'BB_Upper': 'bb_upper',
                'BB_Middle': 'bb_middle',
                'BB_Lower': 'bb_lower',
                'ATR': 'atr',
                'CCI': 'cci',
                'WILLR': 'willr',
                'MOM': 'mom'
            }

            # 檢查並添加技術指標，正確處理 NaN 值
            indicators_found = []
            for file_col, db_col in indicator_mapping.items():
                if file_col in data.columns:
                    indicators_found.append(file_col)
                    # 使用 where 方法保持 NaN 為 None，並應用適當的四捨五入
                    if db_col in ['rsi_5', 'rsi_7', 'rsi_10', 'rsi_14',
                                  'rsi_21', 'rsv', 'k_value', 'd_value',
                                  'j_value', 'willr']:
                        # 對於百分比指標，NaN 保持為 None
                        series_data = data[file_col].round(4)
                        db_data[db_col] = series_data.where(
                            pd.notnull(series_data), None)
                    elif db_col in ['dif', 'macd', 'macd_histogram',
                                    'atr', 'mom']:
                        series_data = data[file_col].round(6)
                        db_data[db_col] = series_data.where(
                            pd.notnull(series_data), None)
                    elif db_col in ['cci']:
                        series_data = data[file_col].round(4)
                        db_data[db_col] = series_data.where(
                            pd.notnull(series_data), None)
                    else:  # MA 和 BB 系列
                        series_data = data[file_col].round(6)
                        db_data[db_col] = series_data.where(
                            pd.notnull(series_data), None)
                else:
                    # 如果技術指標欄位不存在，設為 None
                    db_data[db_col] = None

            self.logger.info(f"{symbol}: 找到技術指標: {indicators_found}")
            self.logger.info(f"{symbol}: 準備後的數據形狀: {db_data.shape}")

            # 檢查數據完整性
            valid_dates = db_data['date'].notna().sum()
            valid_symbols = db_data['symbol'].notna().sum()
            self.logger.info(f"{symbol}: 有效日期數量: {valid_dates}")
            self.logger.info(f"{symbol}: 有效 symbol 數量: {valid_symbols}")

            # 檢查 NULL 值的情況（只顯示有 NULL 的欄位）
            nan_counts = db_data.isnull().sum()
            nan_fields = {col: count for col, count in nan_counts.items(
            ) if count > 0 and count < len(db_data)}
            if nan_fields:
                self.logger.info(f"{symbol}: 部分 NULL 值欄位: {nan_fields}")

            # 驗證 symbol 欄位是否正確
            if valid_symbols != num_rows:
                self.logger.error(
                    f"{symbol}: symbol 欄位設置失敗！預期: "
                    f"{num_rows}, 實際: {valid_symbols}")
                # 嘗試修復
                db_data['symbol'] = symbol
                valid_symbols_after_fix = db_data['symbol'].notna().sum()
                self.logger.info(
                    f"{symbol}: 修復後 symbol 數量: {valid_symbols_after_fix}")

            return db_data

        except Exception as e:
            self.logger.error(f"{symbol}: 準備資料庫格式數據失敗: {e}")
            import traceback
            self.logger.error(f"{symbol}: 詳細錯誤: {traceback.format_exc()}")
            return pd.DataFrame()

    def _upsert_data(self, data: pd.DataFrame) -> int:
        """使用 UPSERT 方式儲存數據"""
        if data.empty:
            self.logger.warning("UPSERT: 數據為空")
            return 0

        symbol = data['symbol'].iloc[0] if not data.empty else "UNKNOWN"
        self.logger.info(f"{symbol}: 開始 UPSERT 操作，數據筆數: {len(data)}")

        merge_sql = text("""
        MERGE stock_data AS target
        USING (SELECT :symbol as symbol, :date as date) AS source
        ON (target.symbol = source.symbol AND target.date = source.date)
        WHEN MATCHED THEN
            UPDATE SET
                open_price = :open_price, high_price = :high_price,
                low_price = :low_price, close_price = :close_price,
                volume = :volume, rsi_5 = :rsi_5, rsi_7 = :rsi_7,
                rsi_10 = :rsi_10, rsi_14 = :rsi_14, rsi_21 = :rsi_21,
                dif = :dif, macd = :macd, macd_histogram = :macd_histogram,
                rsv = :rsv, k_value = :k_value, d_value = :d_value,
                j_value = :j_value, ma5 = :ma5, ma10 = :ma10,
                ma20 = :ma20, ma60 = :ma60, ema12 = :ema12, ema26 = :ema26,
                bb_upper = :bb_upper, bb_middle = :bb_middle,
                bb_lower = :bb_lower, atr = :atr, cci = :cci,
                willr = :willr, mom = :mom, updated_at = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (symbol, date, open_price, high_price,
                         low_price, close_price, volume,
                         rsi_5, rsi_7, rsi_10, rsi_14, rsi_21, dif, macd,
                   macd_histogram, rsv, k_value, d_value, j_value, ma5, ma10,
                   ma20, ma60, ema12, ema26, bb_upper, bb_middle, bb_lower,
                   atr, cci, willr, mom)
            VALUES (:symbol, :date, :open_price, :high_price, :low_price,
                   :close_price, :volume, :rsi_5, :rsi_7, :rsi_10, :rsi_14,
                   :rsi_21, :dif, :macd, :macd_histogram, :rsv, :k_value,
                   :d_value, :j_value, :ma5, :ma10, :ma20, :ma60, :ema12,
                   :ema26, :bb_upper, :bb_middle, :bb_lower, :atr, :cci,
                   :willr, :mom)
        OUTPUT $action;
        """)

        saved_count = 0
        error_count = 0

        try:
            with self.get_connection() as conn:
                for i, (_, row) in enumerate(data.iterrows()):
                    try:
                        params = row.to_dict()

                        # 確保所有參數都是有效的類型
                        for key, value in params.items():
                            if pd.isna(value):
                                params[key] = None
                            elif key == 'volume' and value is not None:
                                params[key] = int(value)

                        # 記錄前幾筆數據的詳細資訊
                        if i < 3:
                            self.logger.debug(
                                f"{symbol}: 第 {i+1} 筆數據參數: {params}")

                        result = conn.execute(merge_sql, params)
                        action = result.fetchone()

                        if action and action[0] in ['INSERT', 'UPDATE']:
                            saved_count += 1
                            if i < 3:
                                self.logger.debug(
                                    f"{symbol}: 第 {i+1} 筆執行動作: {action[0]}")
                        else:
                            if i < 3:
                                self.logger.warning(
                                    f"{symbol}: 第 {i+1} 筆無動作回傳: {action}")

                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # 只記錄前5個錯誤
                            self.logger.error(
                                f"{symbol}: 第 {i+1} 筆數據儲存失敗: {e}")
                            if i < 3:  # 只在前3筆時記錄詳細數據
                                self.logger.debug(
                                    f"{symbol}: 失敗的數據: {row.to_dict()}")

                conn.commit()
                self.logger.info(
                    f"{symbol}: UPSERT 完成 - 成功: "
                    f"{saved_count}, 錯誤: {error_count}")

        except Exception as e:
            self.logger.error(f"{symbol}: UPSERT 操作失敗: {e}")
            import traceback
            self.logger.error(
                f"{symbol}: UPSERT 詳細錯誤: {traceback.format_exc()}")
            raise

        return saved_count

    def process_multiple_stocks(
            self, symbols: List[str],
            period: Union[Period, str] = Period.MAX,
            interval: Union[TimeInterval, str] = TimeInterval.DAY_1,
            force_update: bool = False
    ) -> Dict[str, ProcessResult]:
        """批量處理多個股票

        Args:
            symbols: 股票代號列表
            period: 時間週期
            interval: 時間間隔
            force_update: 是否強制更新所有數據（包括已存在的數據）
        """
        self.reporter.info(f"開始批量處理 {len(symbols)} 個股票")
        if force_update:
            self.reporter.info("🔄 強制更新模式已啟用 - 將重新處理所有數據")

        results = {}
        success_count = 0
        total_new_records = 0

        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 [{i}/{len(symbols)}] 處理 {symbol}")

            result = self.process_stock(symbol, period, interval, force_update)
            results[symbol] = result

            if result.success:
                success_count += 1
                total_new_records += result.new_records

                if force_update and result.new_records > 0:
                    print(
                        f"   ✅ 強制更新成功 | 處理: {result.new_records} 筆 | "
                        f"總計: {result.total_records} 筆")
                else:
                    print(
                        f"   ✅ 成功 | 新增: {result.new_records} 筆 | "
                        f"總計: {result.total_records} 筆")
                print(f"   📅 時間範圍: {result.date_range}")
                print(f"   ⏱️  處理時間: {result.processing_time:.2f} 秒")
            else:
                print(f"   ❌ 失敗: {result.error_message}")

        # 顯示總結
        print(f"\n{'='*60}")
        if force_update:
            print("📈 強制更新批量處理完成")
        else:
            print("📈 批量處理完成")
        print(f"✅ 成功: {success_count}/{len(symbols)} 個股票")
        print(f"📊 總處理記錄: {total_new_records:,} 筆")
        print(f"{'='*60}")

        return results

    def get_database_statistics(self) -> Dict[str, Any]:
        """獲取資料庫統計資訊"""
        try:
            with self.get_connection() as conn:
                # 整體統計
                overall_query = text("""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM stock_data
                """)

                overall_result = conn.execute(overall_query).fetchone()

                # 各股票統計
                symbol_query = text("""
                SELECT
                    symbol,
                    COUNT(*) as record_count,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    MAX(updated_at) as last_updated
                FROM stock_data
                GROUP BY symbol
                ORDER BY record_count DESC
                """)

                symbol_results = conn.execute(symbol_query).fetchall()

                return {
                    'total_records': (overall_result[0] if overall_result
                                      else 0),
                    'unique_symbols': (overall_result[1] if overall_result
                                       else 0),
                    'date_range': {
                        'earliest': (overall_result[2] if overall_result
                                     else None),
                        'latest': (overall_result[3] if overall_result
                                   else None)
                    },
                    'symbols': [
                        {
                            'symbol': row[0],
                            'records': row[1],
                            'start_date': row[2],
                            'end_date': row[3],
                            'last_updated': row[4]
                        } for row in symbol_results
                    ]
                }

        except Exception as e:
            self.logger.error(f"獲取統計資訊失敗: {e}")
            return {}


def print_statistics(stats: Dict[str, Any]):
    """列印資料庫統計資訊"""
    if not stats:
        print("無法獲取統計資訊")
        return

    print("\n📊 資料庫統計資訊")
    print(f"{'='*50}")
    print(f"總記錄數: {stats['total_records']:,}")
    print(f"股票數量: {stats['unique_symbols']}")

    if stats['date_range']['earliest'] and stats['date_range']['latest']:
        print(
            f"日期範圍: {stats['date_range']['earliest']} ~ "
            f"{stats['date_range']['latest']}")

    if stats['symbols']:
        print("\n📋 各股票詳情:")
        for symbol_info in stats['symbols'][:10]:  # 只顯示前10個
            print(f"  {symbol_info['symbol']}: {symbol_info['records']:,} 筆 "
                  f"({symbol_info['start_date']} ~ {symbol_info['end_date']})")

        if len(stats['symbols']) > 10:
            print(f"  ... 還有 {len(stats['symbols']) - 10} 個股票")


def main():
    """主程式"""
    try:
        # 創建分析器
        analyzer = StockAnalyzerDB()

        # 測試資料庫連接
        if not analyzer.test_connection():
            print("❌ 資料庫連接失敗，程式結束")
            return

        # 預設股票列表
        default_stocks = ["2330", "AAPL"]

        # 從命令行參數獲取股票代號
        if len(sys.argv) > 1:
            target_stocks = sys.argv[1:]
            print(f"ℹ️  使用命令行參數: {', '.join(target_stocks)}")
        else:
            target_stocks = default_stocks
            print(f"ℹ️  使用預設股票: {', '.join(target_stocks)}")

        print("🚀 股票技術分析與資料庫整合系統")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 處理股票
        results = analyzer.process_multiple_stocks(
            symbols=target_stocks,
            period=Period.YEAR_1,
            interval=TimeInterval.DAY_1,
            force_update=False  # 改回 False 使用正常的增量更新模式
        )

        # 顯示統計資訊
        stats = analyzer.get_database_statistics()
        print_statistics(stats)

        print("\n✅ 程式執行完成！")
        print("📝 詳細日誌請查看: stock_analyzer.log")

        # 返回處理結果供後續使用
        return results

    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
        logging.error(f"主程式錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()
