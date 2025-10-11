import yfinance as yf
import pandas as pd
from pandas import DataFrame, Series, Index
import numpy as np
import talib
from talib._ta_lib import MA_Type
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Union, Literal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import sys

# 設定警告和日誌
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
class StockPrice:
    """股價數據結構"""

    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TechnicalIndicators:
    """技術指標數據結構"""

    rsi_14: Optional[float] = None
    rsi_5: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    rsv: Optional[float] = None
    k_percent: Optional[float] = None
    d_percent: Optional[float] = None
    j_percent: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ma_5: Optional[float] = None
    ma_10: Optional[float] = None
    ma_20: Optional[float] = None
    ma_60: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    atr: Optional[float] = None
    cci: Optional[float] = None
    willr: Optional[float] = None


class DataProvider:
    """數據提供者類別"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._cache = {}

    def get_stock_data(
        self,
        symbol: str,
        period: Union[Period, str],
        interval: Union[TimeInterval, str],
        use_cache: bool = True,
    ) -> Optional[DataFrame]:
        """獲取股票數據"""
        try:
            formatted_symbol: str = self._format_symbol(symbol)

            # 提取枚舉值
            period_str: str | Any = (
                period.value if isinstance(period, Period) else period
            )
            interval_str: str | Any = (
                interval.value if isinstance(
                    interval, TimeInterval) else interval
            )

            cache_key: str = f"{formatted_symbol}_{period_str}_{interval_str}"

            if use_cache and cache_key in self._cache:
                self.logger.info(f"從快取獲取 {formatted_symbol} 數據")
                return self._cache[cache_key]

            # 調整期間以避免API限制
            adjusted_period: str = self._adjust_period_for_interval(
                period_str, interval_str
            )

            # 嘗試獲取數據，如果是台股且失敗，則嘗試上櫃市場
            data: DataFrame = DataFrame()
            symbols_to_try = [formatted_symbol]

            # 如果是台股 .TW 後綴，也嘗試 .TWO 後綴
            if (formatted_symbol.endswith('.TW') and
                    symbol.isdigit() and len(symbol) == 4):
                symbols_to_try.append(f"{symbol}.TWO")

            success_symbol = None

            for attempt_symbol in symbols_to_try:

                # 完全禁用所有相關的日誌輸出
                import io
                from contextlib import redirect_stderr, redirect_stdout

                # 保存原始日誌級別
                loggers_to_silence = [
                    'yfinance',
                    'urllib3',
                    'requests',
                    'urllib3.connectionpool',
                    'requests.packages.urllib3.connectionpool'
                ]
                original_levels = {}
                for logger_name in loggers_to_silence:
                    logger = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger.level
                    logger.setLevel(logging.CRITICAL)

                # 創建空的輸出流來捕獲所有輸出
                devnull = io.StringIO()

                try:
                    with redirect_stderr(devnull), redirect_stdout(devnull):
                        ticker: yf.Ticker = yf.Ticker(attempt_symbol)

                        # 添加額外的錯誤處理和重試機制
                        import time
                        max_retries = 2
                        for attempt in range(max_retries):
                            try:
                                data = ticker.history(
                                    period=adjusted_period,
                                    interval=interval_str,
                                    auto_adjust=False,
                                    actions=False,
                                    timeout=10
                                )

                                # 如果成功獲取數據，跳出重試循環
                                if not data.empty:
                                    success_symbol = attempt_symbol
                                    break

                            except Exception:
                                if attempt < max_retries - 1:
                                    time.sleep(0.5)
                                    continue
                                else:
                                    # 只在調試模式下記錄
                                    pass

                finally:
                    # 恢復原始日誌級別
                    for logger_name, level in original_levels.items():
                        logging.getLogger(logger_name).setLevel(level)

                # 如果成功獲取數據，跳出循環
                if not data.empty:
                    break

            if data.empty:
                self.logger.error(
                    f"❌ 無法獲取 {symbol} 的數據（已嘗試所有可能的市場："
                    f"{', '.join(symbols_to_try)}）")
                return None

            # 移除 Adj Close 欄位（如果存在）
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])

            self.logger.info(f"✅ 成功從 {success_symbol} 獲取 {len(data)} 筆數據")

            if use_cache:
                # 使用成功的symbol作為cache key
                cache_key = f"{success_symbol}_{period_str}_{interval_str}"
                self._cache[cache_key] = data

            return data

        except Exception as e:
            self.logger.error(f"❌ 獲取 {symbol} 數據錯誤: {e}")
            return None

    def _format_symbol(self, symbol: str) -> str:
        """格式化股票代號"""
        # 對於台股，添加 .TW 後綴
        if symbol.isdigit() and len(symbol) == 4:
            return f"{symbol}.TW"

        # 如果已經有後綴，直接返回
        if symbol.endswith((".TW", ".TWO")):
            return symbol

        # 美股代號特徵：
        # 1. 包含英文字母
        # 2. 長度通常在1-5個字符
        # 3. 可能包含數字但不是純數字
        if (any(c.isalpha() for c in symbol) and
            len(symbol) <= 5 and
                not symbol.isdigit()):
            return symbol

        # 默認情況，假設是台股
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


class IndicatorCalculator:
    """技術指標計算器"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """計算所有技術指標"""
        if not isinstance(data, DataFrame) or data.empty:
            raise ValueError("數據必須是非空的 DataFrame")

        if len(data) < 60:
            self.logger.warning("數據長度不足，某些指標可能無法計算")

        # 轉換為 numpy arrays
        high: np.ndarray = data["High"].values.astype("float64")
        low: np.ndarray = data["Low"].values.astype("float64")
        close: np.ndarray = data["Close"].values.astype("float64")
        volume: np.ndarray = data["Volume"].values.astype("float64")

        indicators: dict[str, Series] = {}

        try:
            # RSI 指標
            indicators.update(self._calculate_rsi(close, data.index))

            # MACD 指標
            indicators.update(self._calculate_macd(close, data.index))

            # KD 指標
            indicators.update(self._calculate_stochastic(
                high, low, close, data.index))

            # 移動平均線
            indicators.update(
                self._calculate_moving_averages(close, data.index))

            # 布林通道
            indicators.update(
                self._calculate_bollinger_bands(close, data.index))

            # 其他指標
            indicators.update(
                self._calculate_other_indicators(
                    high, low, close, volume, data.index)
            )

        except Exception as e:
            self.logger.error(f"計算指標錯誤: {e}")
            return {}

        return indicators

    def _calculate_rsi(
        self, close: np.ndarray, index: Index
    ) -> Dict[str, Series]:
        """計算 RSI 指標"""
        return {
            "RSI(5)": Series(talib.RSI(close, timeperiod=5), index=index),
            "RSI(7)": Series(talib.RSI(close, timeperiod=7), index=index),
            "RSI(10)": Series(talib.RSI(close, timeperiod=10), index=index),
            "RSI(14)": Series(talib.RSI(close, timeperiod=14), index=index),
            "RSI(21)": Series(talib.RSI(close, timeperiod=21), index=index),
        }

    def _calculate_macd(
        self, close: np.ndarray, index: Index
    ) -> Dict[str, Series]:
        """計算 MACD 指標"""
        macd_line, signal_line, macd_histogram = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        return {
            "DIF": Series(macd_line, index=index),
            "MACD": Series(signal_line, index=index),
            "MACD_Histogram": Series(macd_histogram, index=index),
        }

    def _calculate_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        index: Index
    ) -> Dict[str, Series]:
        """計算 KDJ 指標"""

        # 計算 RSV (Raw Stochastic Value)
        # RSV = (收盤價 - 最近 n 日最低價) / (最近 n 日最高價 - 最近 n 日最低價) * 100
        n = 9  # KD 指標的週期參數

        rsv: np.ndarray = np.full(len(close), np.nan)
        k_values: np.ndarray = np.full(len(close), np.nan)
        d_values: np.ndarray = np.full(len(close), np.nan)

        for i in range(n - 1, len(close)):
            # 計算最近 n 日的最高價和最低價
            period_high: float = np.max(high[i - n + 1: i + 1])
            period_low: float = np.min(low[i - n + 1: i + 1])

            # 計算 RSV
            if period_high != period_low:
                rsv_value = ((close[i] - period_low) /
                             (period_high - period_low)) * 100
                # 確保 RSV 值在 0-100 範圍內
                rsv[i] = max(0, min(100, rsv_value))
            else:
                rsv[i] = 50  # 避免除零錯誤

        # 初始化 K 和 D 值
        # 第一個 K 值通常設為 50，或使用 RSV 值
        first_valid_idx: int = n - 1
        k_values[first_valid_idx] = (
            rsv[first_valid_idx] if not np.isnan(rsv[first_valid_idx]) else 50
        )
        d_values[first_valid_idx] = k_values[first_valid_idx]

        # 計算 K 和 D 值
        for i in range(first_valid_idx + 1, len(close)):
            if not np.isnan(rsv[i]):
                # K = 2/3 × K 前一日 + 1/3 × RSV
                k_values[i] = (2 / 3) * k_values[i - 1] + (1 / 3) * rsv[i]
                # D = 2/3 × D 前一日 + 1/3 × K
                d_values[i] = (2 / 3) * d_values[i - 1] + (1 / 3) * k_values[i]

        # 計算 J 指標：J = 3 * K - 2 * D
        j_values = 3 * k_values - 2 * d_values

        return {
            "RSV": Series(rsv, index=index),
            "K": Series(k_values, index=index),
            "D": Series(d_values, index=index),
            "J": Series(j_values, index=index),
        }

    def _calculate_moving_averages(
        self, close: np.ndarray, index: Index
    ) -> Dict[str, Series]:
        """計算移動平均線"""
        indicators: dict[str, Series] = {}

        # 簡單移動平均線
        for period in [5, 10, 20, 60]:
            indicators[f"MA{period}"] = Series(
                talib.SMA(close, timeperiod=period), index=index
            )

        # 指數移動平均線
        indicators["EMA12"] = Series(
            talib.EMA(close, timeperiod=12), index=index)
        indicators["EMA26"] = Series(
            talib.EMA(close, timeperiod=26), index=index)

        return indicators

    def _calculate_bollinger_bands(
        self, close: np.ndarray, index: Index
    ) -> Dict[str, Series]:
        """計算布林通道"""
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=MA_Type.SMA
        )
        return {
            "BB_Upper": Series(bb_upper, index=index),
            "BB_Middle": Series(bb_middle, index=index),
            "BB_Lower": Series(bb_lower, index=index),
        }

    def _calculate_other_indicators(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        index: Index,
    ) -> Dict[str, Series]:
        """計算其他技術指標"""
        return {
            "ATR": Series(
                talib.ATR(high, low, close, timeperiod=14), index=index
            ),
            "CCI": Series(
                talib.CCI(high, low, close, timeperiod=14), index=index
            ),
            "WILLR": Series(
                talib.WILLR(high, low, close, timeperiod=20), index=index
            ),
            "MOM": Series(
                talib.MOM(close, timeperiod=10), index=index
            ),
        }


class DecimalPrecisionHelper:
    """小數位數處理工具類"""

    @staticmethod
    def get_indicator_decimal_places(indicator_name: str) -> int:
        """根據指標類型返回建議的小數位數"""

        # 價格相關指標 - 保持原始格式（不強制小數位）
        price_indicators = [
            'MA', 'EMA', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR'
        ]

        # 震盪指標 - 2位
        oscillators = [
            'RSI', 'K', 'D', 'J', 'CCI', 'WILLR', 'MOM', 'RSV'
        ]

        # MACD系列 - 4位（數值較小需要更高精度）
        macd_indicators = ['DIF', 'MACD', 'MACD_Histogram']

        if any(ind in indicator_name for ind in price_indicators):
            return 2  # 價格相關指標保持2位
        elif any(ind in indicator_name for ind in oscillators):
            return 2  # 震盪指標2位
        elif any(ind in indicator_name for ind in macd_indicators):
            return 4  # MACD系列4位
        else:
            return 2  # 預設值

    @staticmethod
    def round_value_by_type(value: Any, name: str = "") -> Any:
        """根據類型和名稱智能四捨五入數值"""
        if pd.isna(value) or value is None:
            return None

        if isinstance(value, (int, float)):
            if name:
                decimal_places = (
                    DecimalPrecisionHelper.get_indicator_decimal_places(name)
                )
                return round(float(value), decimal_places)
            else:
                return round(float(value), 2)  # 預設2位

        return value

    @staticmethod
    def round_series_by_name(series: Series, name: str) -> Series:
        """根據指標名稱四捨五入Series"""
        decimal_places = (
            DecimalPrecisionHelper.get_indicator_decimal_places(name)
        )
        return series.round(decimal_places)


class ResultExporter:
    """結果匯出器"""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_to_json(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """保存結果為 JSON，使用智能小數位數處理"""
        if filename is None:
            filename = "analysis.json"

        filepath: Path = self.output_dir / filename

        try:
            # 清理結果，移除無法序列化的物件
            clean_results: Dict[str, Any] = self._clean_results_for_json(
                results
            )

            # 使用智能小數位數處理
            def round_values_intelligently(
                data: Any, parent_key: str = ""
            ) -> Any:
                if isinstance(data, dict):
                    return {
                        k: round_values_intelligently(v, k)
                        for k, v in data.items()
                    }
                elif isinstance(data, list):
                    return [
                        round_values_intelligently(v, parent_key)
                        for v in data
                    ]
                elif isinstance(data, (int, float)) and not pd.isna(data):
                    # 對特定欄位進行特殊處理
                    if parent_key in ['open', 'high', 'low', 'close']:
                        return round(float(data), 2)  # 價格數據2位小數
                    elif parent_key in ['total_records', 'volume']:
                        return int(data)  # 記錄數和成交量保持整數
                    else:
                        return DecimalPrecisionHelper.round_value_by_type(
                            data, parent_key
                        )
                return data

            clean_results = round_values_intelligently(clean_results)

            existing_data: Dict[str, Any] = {}
            # 檢查檔案是否存在且非空
            if filepath.exists() and filepath.stat().st_size > 0:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, dict):
                        self.logger.warning(
                            f"現有的 JSON 檔案 {filepath} 格式不正確，將會覆寫。")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.logger.warning(f"無法解析現有的 JSON 檔案 {filepath}。將會覆寫。")
                    existing_data = {}
                except Exception as e:
                    self.logger.error(f"讀取現有 JSON 檔案 {filepath} 錯誤: {e}。將會覆寫。")
                    existing_data = {}

            # 合併數據：新的結果會覆寫或添加條目
            if not isinstance(existing_data, dict):
                existing_data = {}
            existing_data.update(clean_results)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    existing_data, f, ensure_ascii=False, indent=2,
                    default=str)

            return str(filepath)

        except Exception as e:
            self.logger.error(f"保存 JSON 錯誤: {e}")
            return ""

    def save_to_csv(
        self,
        symbol: str,
        data: DataFrame,
        indicators: Dict[str, Series],
        interval: str
    ) -> str:
        """保存歷史數據為 CSV，使用智能小數位數處理"""
        try:
            new_combined_data: DataFrame = data.copy()

            # 對價格數據進行四捨五入到2位小數
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in new_combined_data.columns:
                    new_combined_data[col] = new_combined_data[col].round(2)

            # 對指標數據使用智能小數位數處理
            for name, series in indicators.items():
                new_combined_data[name] = (
                    DecimalPrecisionHelper.round_series_by_name(series, name)
                )

            date_format = '%Y-%m-%d %H:%M:%S'
            if (interval.endswith('d') or
                interval.endswith('wk') or
                    interval.endswith('mo')):
                date_format = '%Y-%m-%d'

            if isinstance(new_combined_data.index, pd.DatetimeIndex):
                if new_combined_data.index.tz is not None:
                    import pytz
                    taiwan_tz = pytz.timezone('Asia/Taipei')
                    new_combined_data.index = (
                        new_combined_data.index.tz_convert(taiwan_tz)
                        .tz_localize(None)
                    )
                new_combined_data.index = pd.Index([
                    dt.strftime(date_format)
                    for dt in new_combined_data.index
                ], name='Date')
            elif new_combined_data.index.name != 'Date':
                new_combined_data.index.name = 'Date'

            filename: str = f"{symbol}.csv"
            filepath: Path = self.output_dir / filename
            final_df: DataFrame

            if filepath.exists() and filepath.stat().st_size > 0:
                try:
                    existing_df = pd.read_csv(
                        filepath, index_col='Date', encoding='utf-8-sig'
                    )
                    if isinstance(existing_df.index, pd.DatetimeIndex):
                        existing_df.index = existing_df.index.strftime(
                            date_format
                        )
                    else:
                        existing_df.index = existing_df.index.astype(str)
                    existing_df.index.name = 'Date'

                    merged_df = pd.concat([existing_df, new_combined_data])
                    final_df = merged_df[
                        ~merged_df.index.duplicated(keep='last')
                    ]
                    final_df = final_df.sort_index()
                    self.logger.info(f"已更新 CSV 檔案: {filepath}")

                except pd.errors.EmptyDataError:
                    self.logger.warning(
                        f"現有的 CSV 檔案 {filepath} 為空。將創建新檔案。"
                    )
                    final_df = new_combined_data.sort_index()
                except Exception as e_read:
                    self.logger.warning(
                        f"讀取或合併現有 CSV 檔案 {filepath} 錯誤: {e_read}。"
                        "將覆寫。"
                    )
                    final_df = new_combined_data.sort_index()
            else:
                final_df = new_combined_data.sort_index()
                if filepath.exists():
                    self.logger.info(
                        f"現有的 CSV 檔案 {filepath} 為空。將寫入新數據。"
                    )
                else:
                    self.logger.info(
                        f"CSV 檔案不存在。將創建新檔案: {filepath}"
                    )

            final_df.to_csv(filepath, encoding="utf-8-sig", index=True)
            return str(filepath)

        except Exception as e:
            self.logger.error(f"保存 CSV 錯誤 ({symbol}): {e}")
            return ""

    def _clean_results_for_json(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """清理結果以便 JSON 序列化，使用智能小數位數處理"""
        clean_results: Dict[str, Dict[str, Any]] = {}
        for symbol, data in results.items():
            if isinstance(data, dict):
                clean_data: Dict[str, Any] = {}
                for k, v in data.items():
                    if not k.startswith("_"):
                        if k == "indicators" and isinstance(v, dict):
                            # 對指標數據使用智能小數位數處理
                            clean_data[k] = {
                                ind_name: (
                                    DecimalPrecisionHelper.round_value_by_type(
                                        ind_value, ind_name
                                    )
                                )
                                for ind_name, ind_value in v.items()
                            }
                        else:
                            clean_data[k] = v
                clean_results[symbol] = clean_data
            else:
                clean_results[symbol] = data
        return clean_results


class TechnicalAnalyzer:
    """技術分析器主類別"""

    def __init__(self, output_dir: str = "output") -> None:
        self.data_provider = DataProvider()
        self.indicator_calculator = IndicatorCalculator()
        self.exporter = ResultExporter(output_dir)
        self.logger = logging.getLogger(__name__)

    def analyze_stock(
        self,
        symbol: str,
        period: Union[Period, str] = Period.YEAR_2,
        interval: Union[TimeInterval, str] = TimeInterval.DAY_1,
    ) -> Dict[str, Any]:
        """分析單一股票"""
        try:
            # 獲取數據
            data: DataFrame | None = self.data_provider.get_stock_data(
                symbol, period, interval
            )
            if data is None:
                return {"error": f"無法獲取 {symbol} 的數據"}

            if len(data) < 60:
                return {"error": f"{symbol} 數據不足（需要至少60筆數據）"}

            # 計算指標
            indicators: dict[str, Series] = (
                self.indicator_calculator.calculate_all_indicators(data)
            )
            if not indicators:
                return {"error": f"無法計算 {symbol} 的技術指標"}

            # 組裝結果
            latest: Series = data.iloc[-1]
            interval_str: str | Any = (
                interval.value if isinstance(
                    interval, TimeInterval) else interval
            )

            # 根據 interval 決定日期格式
            if (interval_str.endswith('d') or
                interval_str.endswith('wk') or
                    interval_str.endswith('mo')):
                date_format_str = '%Y-%m-%d'
            else:
                date_format_str = '%Y-%m-%d %H:%M:%S'

            result: dict[str, Any] = {
                "symbol": symbol,
                "date": pd.to_datetime(data.index[-1]).strftime(
                    date_format_str  # 使用條件日期格式
                ),
                "price": StockPrice(
                    open=round(float(latest["Open"]), 2),    # 四捨五入到2位小數
                    high=round(float(latest["High"]), 2),    # 四捨五入到2位小數
                    low=round(float(latest["Low"]), 2),      # 四捨五入到2位小數
                    close=round(float(latest["Close"]), 2),  # 四捨五入到2位小數
                    volume=int(latest["Volume"]),
                ).__dict__,
                "indicators": self._get_latest_indicator_values(indicators),
                "total_records": len(data),
                "interval": interval_str,
                "period": (
                    period.value if isinstance(period, Period) else period
                ),
                "time_range": (
                    f"{pd.to_datetime(data.index[0]).strftime('%Y-%m-%d')} - "
                    f"{pd.to_datetime(data.index[-1]).strftime('%Y-%m-%d')}"
                ),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "_data": data,
                "_indicators": indicators,
            }

            return result

        except Exception as e:
            self.logger.error(f"分析 {symbol} 錯誤: {e}")
            return {"error": f"分析 {symbol} 時發生錯誤: {str(e)}"}

    def analyze_multiple_stocks(
        self,
        symbols: List[str],
        period: Union[Period, str] = Period.YEAR_2,
        interval: Union[TimeInterval, str] = TimeInterval.DAY_1,
    ) -> Dict[str, Any]:
        """分析多個股票"""
        results: dict[str, dict[str, Any]] = {}

        for symbol in symbols:
            self.logger.info(f"正在分析 {symbol}...")
            results[symbol] = self.analyze_stock(symbol, period, interval)

        return results

    def save_analysis_results(
        self, results: Dict[str, Any], format_type: str = "json"
    ) -> List[str]:
        """保存分析結果"""
        saved_files: list[str] = []

        if format_type.lower() == "json":
            json_file: str = self.exporter.save_to_json(results)
            if json_file:
                saved_files.append(json_file)

        # 保存 CSV
        for symbol, result in results.items():
            if ("error" not in result and "_data" in result and
                    "_indicators" in result and "interval" in result):
                # 確保 interval 存在
                csv_file: str = self.exporter.save_to_csv(
                    # 傳遞 interval
                    symbol,
                    result["_data"],
                    result["_indicators"],
                    result["interval"]
                )
                if csv_file:
                    saved_files.append(csv_file)

        return saved_files

    def _get_latest_indicator_values(
        self, indicators: Dict[str, Series]
    ) -> Dict[str, Optional[float]]:
        """獲取指標的最新值，使用智能小數位數處理"""
        latest_values: dict[str, float | None] = {}

        for name, series in indicators.items():
            if isinstance(series, Series) and not series.empty:
                latest_value: Any = series.iloc[-1]
                if not pd.isna(latest_value):
                    # 使用智能小數位數處理
                    processed_value = (
                        DecimalPrecisionHelper.round_value_by_type(
                            latest_value, name
                        )
                    )
                    latest_values[name] = processed_value
                else:
                    latest_values[name] = None

        return latest_values


class AnalysisReporter:
    """分析報告生成器"""

    @staticmethod
    def print_analysis_summary(results: Dict[str, Any]) -> None:
        """打印分析摘要"""
        print("\n=== 技術分析結果摘要 ===", flush=True)

        for symbol, data in results.items():
            if "error" in data:
                print(f"\n❌ {symbol}: {data['error']}", flush=True)
                continue

            print(f"\n📊 {symbol} ({data['date']}):", flush=True)
            price: Any = data["price"]
            print(
                f"   價格: 開 {price['open']:.2f} | "
                f"高 {price['high']:.2f} | "
                f"低 {price['low']:.2f} | "
                f"收 {price['close']:.2f}", flush=True
            )
            print(f"   成交量: {price['volume']:,}", flush=True)

            indicators: Any = data["indicators"]

            # 顯示關鍵指標
            if indicators.get("RSI(14)"):
                rsi: float = indicators["RSI(14)"]
                rsi_status: Literal["超買", "超賣", "正常"] = (
                    "超買" if rsi > 70 else "超賣" if rsi < 30 else "正常"
                )
                print(f"   RSI(14): {rsi:.2f} ({rsi_status})", flush=True)
            else:
                print("   RSI(14): N/A", flush=True)

            if indicators.get("DIF") and indicators.get("MACD"):
                macd_trend: Literal["多頭", "空頭"] = (
                    "多頭" if indicators["DIF"] > indicators["MACD"] else "空頭"
                )
                print(
                    f"   MACD: {indicators['MACD']:.4f} | "
                    f"DIF: {indicators['DIF']:.4f} ({macd_trend})", flush=True
                )
            else:
                print("   MACD: N/A", flush=True)

            if indicators.get("K") and indicators.get("D"):
                kd_trend: Literal["多頭", "空頭"] = (
                    "多頭" if indicators["K"] > indicators["D"] else "空頭"
                )
                j_value: Optional[float] = indicators.get("J")
                if j_value is not None:
                    j_signal: Literal["強多", "強空", "正常"] = (
                        "強多" if j_value > 100 else
                        "強空" if j_value < 0 else "正常"
                    )
                    print(
                        f"   KDJ: K={indicators['K']:.2f}, "
                        f"D={indicators['D']:.2f}, "
                        f"J={j_value:.2f} ({kd_trend}, J:{j_signal})",
                        flush=True
                    )
                else:
                    print(
                        f"   KDJ: K={indicators['K']:.2f}, "
                        f"D={indicators['D']:.2f}, "
                        f"J=N/A ({kd_trend})", flush=True
                    )

            if (indicators.get("BB_Upper") and
                indicators.get("BB_Middle") and
                    indicators.get("BB_Lower")):
                BB_Trend: Literal["上升", "下降", "平穩"] = (
                    "上升" if indicators["BB_Upper"] > indicators["BB_Middle"]
                    else "下降"
                    if indicators["BB_Upper"] < indicators["BB_Middle"]
                    else "平穩"
                )
                print(
                    f"   布林通道: 上軌: {indicators['BB_Upper']:.2f}, "
                    f"中軌: {indicators['BB_Middle']:.2f}, "
                    f"下軌: {indicators['BB_Lower']:.2f} | 趨勢: {BB_Trend}",
                    flush=True
                )
            else:
                print("   布林通道: N/A", flush=True)

            if (indicators.get("MA5") and indicators.get("MA10") and
                    indicators.get("MA20") and indicators.get("MA60")):
                ma_trend: Literal["上升", "下降", "平穩"] = (
                    "上升" if indicators["MA5"] > indicators["MA10"] else
                    "下降" if indicators["MA5"] < indicators["MA10"] else
                    "平穩"
                )
                print(
                    f"   MA5: {indicators['MA5']:.2f} | "
                    f"MA10: {indicators['MA10']:.2f} | "
                    f"MA20: {indicators['MA20']:.2f} | "
                    f"MA60: {indicators['MA60']:.2f} | 趨勢: {ma_trend}",
                    flush=True
                )
            else:
                print("   移動平均線: N/A", flush=True)

            # 修復：檢查 time_range 是否存在，如果不存在則顯示基本信息
            time_range_info = data.get('time_range', 'N/A')
            print(
                f"   數據筆數: {data['total_records']} ({time_range_info}) | "
                f"間隔: {data['interval']}", flush=True)


def main() -> None:
    """主程序"""
    try:
        # 創建分析器
        analyzer: TechnicalAnalyzer = TechnicalAnalyzer()
        reporter: AnalysisReporter = AnalysisReporter()

        # 測試股票
        default_stocks: list[str] = ["2330", "AAPL", "NFLX"]

        # 從命令行參數獲取股票代號
        if len(sys.argv) > 1:
            target_stocks = sys.argv[1:]
            print(f"ℹ️ 使用命令行傳入的股票代號: {', '.join(target_stocks)}", flush=True)
        else:
            target_stocks = default_stocks
            print(
                f"ℹ️ 未提供命令行參數，使用預設股票代號: {', '.join(target_stocks)}",
                flush=True)

        print("🚀 開始技術分析", flush=True)

        # 執行分析
        results: dict[str, Any] = analyzer.analyze_multiple_stocks(
            symbols=target_stocks,  # 使用 target_stocks
            period=Period.MAX,
            interval=TimeInterval.DAY_1
        )

        # 顯示結果
        reporter.print_analysis_summary(results)

        # 保存結果
        saved_files: list[str] = analyzer.save_analysis_results(
            results, "json")

        print(f"\n💾 已保存 {len(saved_files)} 個檔案:", flush=True)
        for file in saved_files:
            print(f"   📄 {file}", flush=True)

    except Exception as e:
        logging.error(f"執行錯誤: {e}")


if __name__ == "__main__":
    main()
