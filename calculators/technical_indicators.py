"""
技術指標計算模組
負責計算各種股票技術指標
"""

import pandas as pd
import numpy as np
import talib
from talib._ta_lib import MA_Type
import logging
from typing import Dict


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

    def calculate_indicators_from_ohlcv(self, ohlcv_data: pd.DataFrame
                                        ) -> Dict[str, pd.Series]:
        """從OHLCV數據計算技術指標，返回指標字典"""
        if ohlcv_data.empty or len(ohlcv_data) < 60:
            raise ValueError("數據不足，無法計算技術指標")

        indicators = {}

        # 檢查欄位名稱並統一處理
        if 'High' in ohlcv_data.columns:
            # 使用大寫欄位名稱（來自外部API）
            high = ohlcv_data["High"].values.astype("float64")
            low = ohlcv_data["Low"].values.astype("float64")
            close = ohlcv_data["Close"].values.astype("float64")
        elif 'high_price' in ohlcv_data.columns:
            # 使用資料庫欄位名稱
            high = ohlcv_data["high_price"].values.astype("float64")
            low = ohlcv_data["low_price"].values.astype("float64")
            close = ohlcv_data["close_price"].values.astype("float64")
        else:
            raise ValueError(f"無法識別的欄位格式，可用欄位: {list(ohlcv_data.columns)}")

        try:
            # RSI 指標
            indicators['rsi_5'] = pd.Series(
                talib.RSI(close, timeperiod=5), index=ohlcv_data.index)
            indicators['rsi_7'] = pd.Series(
                talib.RSI(close, timeperiod=7), index=ohlcv_data.index)
            indicators['rsi_10'] = pd.Series(
                talib.RSI(close, timeperiod=10), index=ohlcv_data.index)
            indicators['rsi_14'] = pd.Series(
                talib.RSI(close, timeperiod=14), index=ohlcv_data.index)
            indicators['rsi_21'] = pd.Series(
                talib.RSI(close, timeperiod=21), index=ohlcv_data.index)

            # MACD 指標
            macd_line, signal_line, macd_histogram = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['dif'] = pd.Series(macd_line, index=ohlcv_data.index)
            indicators['macd'] = pd.Series(signal_line, index=ohlcv_data.index)
            indicators['macd_histogram'] = pd.Series(
                macd_histogram, index=ohlcv_data.index)

            # KDJ 指標
            kdj_data = self._calculate_kdj(high, low, close)
            indicators['rsv'] = pd.Series(
                kdj_data['RSV'], index=ohlcv_data.index)
            indicators['k_value'] = pd.Series(
                kdj_data['K'], index=ohlcv_data.index)
            indicators['d_value'] = pd.Series(
                kdj_data['D'], index=ohlcv_data.index)
            indicators['j_value'] = pd.Series(
                kdj_data['J'], index=ohlcv_data.index)

            # 移動平均線
            indicators['ma5'] = pd.Series(
                talib.SMA(close, timeperiod=5), index=ohlcv_data.index)
            indicators['ma10'] = pd.Series(
                talib.SMA(close, timeperiod=10), index=ohlcv_data.index)
            indicators['ma20'] = pd.Series(
                talib.SMA(close, timeperiod=20), index=ohlcv_data.index)
            indicators['ma60'] = pd.Series(
                talib.SMA(close, timeperiod=60), index=ohlcv_data.index)
            indicators['ema12'] = pd.Series(
                talib.EMA(close, timeperiod=12), index=ohlcv_data.index)
            indicators['ema26'] = pd.Series(
                talib.EMA(close, timeperiod=26), index=ohlcv_data.index)

            # 布林通道
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=MA_Type.SMA)
            indicators['bb_upper'] = pd.Series(
                bb_upper, index=ohlcv_data.index)
            indicators['bb_middle'] = pd.Series(
                bb_middle, index=ohlcv_data.index)
            indicators['bb_lower'] = pd.Series(
                bb_lower, index=ohlcv_data.index)

            # 其他指標
            indicators['atr'] = pd.Series(
                talib.ATR(high, low, close, timeperiod=14
                          ), index=ohlcv_data.index)
            indicators['cci'] = pd.Series(
                talib.CCI(high, low, close, timeperiod=14
                          ), index=ohlcv_data.index)
            indicators['willr'] = pd.Series(talib.WILLR(
                high, low, close, timeperiod=20), index=ohlcv_data.index)
            indicators['mom'] = pd.Series(
                talib.MOM(close, timeperiod=10), index=ohlcv_data.index)

            self.logger.info("技術指標計算完成")
            return indicators

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
