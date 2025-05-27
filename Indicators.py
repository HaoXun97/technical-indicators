import yfinance as yf
import pandas as pd
import numpy as np
import talib
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 設定警告和日誌
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # date: str

@dataclass
class TechnicalIndicators:
    """技術指標數據結構"""
    rsi_14: Optional[float] = None
    rsi_5: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    k_percent: Optional[float] = None
    d_percent: Optional[float] = None
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
    obv: Optional[float] = None

class DataProvider:
    """數據提供者類別"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
    
    def get_stock_data(self, symbol: str, period: Union[Period, str], 
                      interval: Union[TimeInterval, str], use_cache: bool = True) -> Optional[pd.DataFrame]:
        """獲取股票數據"""
        try:
            formatted_symbol = self._format_symbol(symbol)
            
            # 提取枚舉值
            period_str = period.value if isinstance(period, Period) else period
            interval_str = interval.value if isinstance(interval, TimeInterval) else interval
            
            cache_key = f"{formatted_symbol}_{period_str}_{interval_str}"
            
            if use_cache and cache_key in self._cache:
                self.logger.info(f"從快取獲取 {formatted_symbol} 數據")
                return self._cache[cache_key]
            
            # 調整期間以避免API限制
            adjusted_period = self._adjust_period_for_interval(period_str, interval_str)
            
            ticker = yf.Ticker(formatted_symbol)
            data = ticker.history(period=adjusted_period, interval=interval_str)
            
            if data.empty:
                self.logger.warning(f"無法獲取 {formatted_symbol} 的數據")
                return None
            
            self.logger.info(f"成功獲取 {formatted_symbol} 數據: {len(data)} 筆")
            
            if use_cache:
                self._cache[cache_key] = data
                
            return data
            
        except Exception as e:
            self.logger.error(f"獲取 {symbol} 數據錯誤: {e}")
            return None
    
    def _format_symbol(self, symbol: str) -> str:
        """格式化股票代號"""
        if not symbol.endswith(('.TW', '.TWO')):
            return f"{symbol}.TW"
        return symbol
    
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
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """計算所有技術指標"""
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("數據必須是非空的 DataFrame")
        
        if len(data) < 60:
            self.logger.warning("數據長度不足，某些指標可能無法計算")
        
        # 轉換為 numpy arrays
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        
        indicators = {}
        
        try:
            # RSI 指標
            indicators.update(self._calculate_rsi(close, data.index))
            
            # MACD 指標
            indicators.update(self._calculate_macd(close, data.index))
            
            # KD 指標
            indicators.update(self._calculate_stochastic(high, low, close, data.index))
            
            # 移動平均線
            indicators.update(self._calculate_moving_averages(close, data.index))
            
            # 布林通道
            indicators.update(self._calculate_bollinger_bands(close, data.index))
            
            # 其他指標
            indicators.update(self._calculate_other_indicators(high, low, close, volume, data.index))
            
        except Exception as e:
            self.logger.error(f"計算指標錯誤: {e}")
            return {}
            
        return indicators
    
    def _calculate_rsi(self, close: np.ndarray, index: pd.Index) -> Dict[str, pd.Series]:
        """計算 RSI 指標"""
        return {
            'RSI(5)': pd.Series(talib.RSI(close, timeperiod=5), index=index),
            'RSI(7)': pd.Series(talib.RSI(close, timeperiod=7), index=index),
            'RSI(10)': pd.Series(talib.RSI(close, timeperiod=10), index=index),
            'RSI(14)': pd.Series(talib.RSI(close, timeperiod=14), index=index),
            'RSI(21)': pd.Series(talib.RSI(close, timeperiod=21), index=index),
        }
    
    def _calculate_macd(self, close: np.ndarray, index: pd.Index) -> Dict[str, pd.Series]:
        """計算 MACD 指標"""
        macd_line, signal_line, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return {
            'DIF': pd.Series(macd_line, index=index),
            'MACD': pd.Series(signal_line, index=index),
            'MACD_Histogram': pd.Series(macd_histogram, index=index),
        }
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, index: pd.Index) -> Dict[str, pd.Series]:
        """計算 KD 指標"""
        k_percent, d_percent = talib.STOCH(
            high, low, close, 
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        return {
            'K': pd.Series(k_percent, index=index),
            'D': pd.Series(d_percent, index=index),
        }
    
    def _calculate_moving_averages(self, close: np.ndarray, index: pd.Index) -> Dict[str, pd.Series]:
        """計算移動平均線"""
        indicators = {}
        
        # 簡單移動平均線
        for period in [5, 10, 20, 60]:
            indicators[f'MA{period}'] = pd.Series(talib.SMA(close, timeperiod=period), index=index)
        
        # 指數移動平均線
        indicators['EMA12'] = pd.Series(talib.EMA(close, timeperiod=12), index=index)
        indicators['EMA26'] = pd.Series(talib.EMA(close, timeperiod=26), index=index)
        
        return indicators
    
    def _calculate_bollinger_bands(self, close: np.ndarray, index: pd.Index) -> Dict[str, pd.Series]:
        """計算布林通道"""
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        return {
            'BB_Upper': pd.Series(bb_upper, index=index),
            'BB_Middle': pd.Series(bb_middle, index=index),
            'BB_Lower': pd.Series(bb_lower, index=index),
        }
    
    def _calculate_other_indicators(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, volume: np.ndarray, 
                                  index: pd.Index) -> Dict[str, pd.Series]:
        """計算其他技術指標"""
        return {
            'ATR': pd.Series(talib.ATR(high, low, close, timeperiod=14), index=index),
            'CCI': pd.Series(talib.CCI(high, low, close, timeperiod=14), index=index),
            'WILLR': pd.Series(talib.WILLR(high, low, close, timeperiod=9), index=index),
            'MOM': pd.Series(talib.MOM(close, timeperiod=10), index=index),
            'OBV': pd.Series(talib.OBV(close, volume), index=index),
            'Volume_MA': pd.Series(talib.SMA(volume, timeperiod=5), index=index),
        }

class ResultExporter:
    """結果匯出器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_to_json(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """保存結果為 JSON"""
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        try:
            # 清理結果，移除無法序列化的物件
            clean_results = self._clean_results_for_json(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"結果已保存至: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存 JSON 錯誤: {e}")
            return ""
    
    def save_to_csv(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> str:
        """保存歷史數據為 CSV"""
        try:
            # 合併數據和指標
            combined_data = data.copy()
            for name, series in indicators.items():
                combined_data[name] = series
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            combined_data.to_csv(filepath, encoding='utf-8-sig')
            self.logger.info(f"CSV 已保存至: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存 CSV 錯誤: {e}")
            return ""
    
    def _clean_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """清理結果以便 JSON 序列化"""
        clean_results = {}
        for symbol, data in results.items():
            if isinstance(data, dict):
                clean_data = {k: v for k, v in data.items() if not k.startswith('_')}
                clean_results[symbol] = clean_data
            else:
                clean_results[symbol] = data
        return clean_results

class TechnicalAnalyzer:
    """技術分析器主類別"""
    
    def __init__(self, output_dir: str = "output"):
        self.data_provider = DataProvider()
        self.indicator_calculator = IndicatorCalculator()
        self.exporter = ResultExporter(output_dir)
        self.logger = logging.getLogger(__name__)
    
    def analyze_stock(self, symbol: str, 
                     period: Union[Period, str] = Period.YEAR_2, 
                     interval: Union[TimeInterval, str] = TimeInterval.DAY_1) -> Dict[str, Any]:
        """分析單一股票"""
        try:
            # 獲取數據
            data = self.data_provider.get_stock_data(symbol, period, interval)
            if data is None:
                return {'error': f'無法獲取 {symbol} 的數據'}
            
            if len(data) < 60:
                return {'error': f'{symbol} 數據不足（需要至少60筆數據）'}
            
            # 計算指標
            indicators = self.indicator_calculator.calculate_all_indicators(data)
            if not indicators:
                return {'error': f'無法計算 {symbol} 的技術指標'}
            
            # 組裝結果
            latest = data.iloc[-1]
            interval_str = interval.value if isinstance(interval, TimeInterval) else interval
            
            result = {
                'symbol': symbol,
                'date': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'price': StockPrice(
                    open=float(latest['Open']),
                    high=float(latest['High']),
                    low=float(latest['Low']),
                    close=float(latest['Close']),
                    volume=int(latest['Volume']),
                ).__dict__,
                'indicators': self._get_latest_indicator_values(indicators),
                'total_records': len(data),
                'interval': interval_str,
                'period': period.value if isinstance(period, Period) else period,
                'time_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '_data': data,
                '_indicators': indicators
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析 {symbol} 錯誤: {e}")
            return {'error': f'分析 {symbol} 時發生錯誤: {str(e)}'}
    
    def analyze_multiple_stocks(self, symbols: List[str], 
                              period: Union[Period, str] = Period.YEAR_2,
                              interval: Union[TimeInterval, str] = TimeInterval.DAY_1) -> Dict[str, Any]:
        """分析多個股票"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"正在分析 {symbol}...")
            results[symbol] = self.analyze_stock(symbol, period, interval)
        
        return results
    
    def save_analysis_results(self, results: Dict[str, Any], format_type: str = "json") -> List[str]:
        """保存分析結果"""
        saved_files = []
        
        if format_type.lower() == "json":
            json_file = self.exporter.save_to_json(results)
            if json_file:
                saved_files.append(json_file)
        
        # 保存 CSV
        for symbol, result in results.items():
            if 'error' not in result and '_data' in result and '_indicators' in result:
                csv_file = self.exporter.save_to_csv(symbol, result['_data'], result['_indicators'])
                if csv_file:
                    saved_files.append(csv_file)
        
        return saved_files
    
    def _get_latest_indicator_values(self, indicators: Dict[str, pd.Series]) -> Dict[str, Optional[float]]:
        """獲取指標的最新值"""
        latest_values = {}
        
        for name, series in indicators.items():
            if isinstance(series, pd.Series) and not series.empty:
                latest_value = series.iloc[-1]
                latest_values[name] = float(latest_value) if not pd.isna(latest_value) else None
        
        return latest_values

class AnalysisReporter:
    """分析報告生成器"""
    
    @staticmethod
    def print_analysis_summary(results: Dict[str, Any]) -> None:
        """打印分析摘要"""
        print("\n=== 技術分析結果摘要 ===")
        
        for symbol, data in results.items():
            if 'error' in data:
                print(f"\n❌ {symbol}: {data['error']}")
                continue
            
            print(f"\n📊 {symbol} ({data['date']}):")
            price = data['price']
            print(f"   價格: 開 {price['open']:.2f} | 高 {price['high']:.2f} | 低 {price['low']:.2f} | 收 {price['close']:.2f}")
            print(f"   成交量: {price['volume']:,}")
            
            indicators = data['indicators']
            
            # 顯示關鍵指標
            if indicators.get('RSI(14)'):
                rsi = indicators['RSI(14)']
                rsi_status = "超買" if rsi > 70 else "超賣" if rsi < 30 else "正常"
                print(f"   RSI(14): {rsi:.2f} ({rsi_status})")
            
            if indicators.get('MACD') and indicators.get('MACD_Signal'):
                macd_trend = "多頭" if indicators['MACD'] > indicators['MACD_Signal'] else "空頭"
                print(f"   MACD: {indicators['MACD']:.4f} ({macd_trend})")
            
            if indicators.get('K') and indicators.get('D'):
                kd_trend = "多頭" if indicators['K'] > indicators['D'] else "空頭"
                print(f"   KD: K={indicators['K']:.2f}, D={indicators['D']:.2f} ({kd_trend})")
            
            print(f"   數據筆數: {data['total_records']} | 間隔: {data['interval']}")

def main() -> None:
    """主程序"""
    try:
        # 創建分析器
        analyzer = TechnicalAnalyzer()
        reporter = AnalysisReporter()
        
        # 測試股票
        test_stocks = ['2330', '2317', '2454']  # 台積電、鴻海、聯發科
        
        print("🚀 開始技術分析")
        
        # 執行分析
        results = analyzer.analyze_multiple_stocks(
            symbols=test_stocks,
            period=Period.MAX,
            interval=TimeInterval.DAY_1
        )
        
        # 顯示結果
        reporter.print_analysis_summary(results)
        
        # 保存結果
        saved_files = analyzer.save_analysis_results(results, "json")
        
        print(f"\n💾 已保存 {len(saved_files)} 個檔案:")
        for file in saved_files:
            print(f"   📄 {file}")
            
    except Exception as e:
        logging.error(f"執行錯誤: {e}")

if __name__ == "__main__":
    main()