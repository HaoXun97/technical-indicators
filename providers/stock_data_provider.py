"""
股票數據提供者模組
負責從外部API獲取股票數據
"""

import yfinance as yf
import pandas as pd
import logging
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional, Union
from enum import Enum


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


class StockDataProvider:
    """股票數據提供者"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_stock_data(self, symbol: str, period: Union[Period, str, int],
                       interval: Union[TimeInterval, str]
                       ) -> Optional[pd.DataFrame]:
        """獲取股票數據"""
        try:
            formatted_symbol = self._format_symbol(symbol)

            # 處理不同類型的 period 參數
            if isinstance(period, Period):
                period_str = period.value
            elif isinstance(period, int):
                period_str = f"{period}d"
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
        # 特殊符號開頭或包含 =、- 的金融商品，直接回傳原始 symbol
        if symbol.startswith('^') or '=' in symbol or '-' in symbol:
            return symbol

        # 常見 Yahoo Finance 市場後綴
        market_suffixes = (
            ".TW", ".TWO", ".SS", ".SZ", ".HK", ".KS", ".T", ".AX", ".L",
            ".TO", ".V", ".SI", ".NZ", ".SA", ".MX", ".ST", ".HE", ".PA",
            ".MI", ".VI", ".MC", ".SW", ".OL", ".CO", ".IR", ".IS", ".TA",
            ".BK", ".JK", ".KL", ".SG", ".B", ".PR", ".IL", ".F", ".DE",
            ".AT", ".BR", ".PL", ".IC", ".NE", ".NS", ".SSX", ".J", ".JO"
        )
        if symbol.endswith(market_suffixes):
            return symbol

        if symbol.isdigit() and len(symbol) == 4:
            return f"{symbol}.TW"

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
