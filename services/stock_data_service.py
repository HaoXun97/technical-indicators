"""
股票數據服務模組
主要業務邏輯處理，協調各個組件
"""

import logging
import time
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd

from providers.stock_data_provider import (
    StockDataProvider, Period, TimeInterval
)
from calculators.technical_indicators import TechnicalIndicatorCalculator
from repositories.stock_data_repository import StockDataRepository


@dataclass
class ProcessResult:
    """處理結果統計"""
    symbol: str
    success: bool
    new_records: int = 0
    updated_records: int = 0
    indicator_updates: int = 0
    total_records: int = 0
    error_message: str = None
    processing_time: float = 0.0
    date_range: str = None


class ProgressReporter:
    """進度報告器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str):
        self.logger.info(message)
        print(f"ℹ️  {message}", flush=True)

    def success(self, message: str):
        self.logger.info(f"SUCCESS: {message}")
        print(f"✅ {message}", flush=True)

    def warning(self, message: str):
        self.logger.warning(message)
        print(f"⚠️  {message}", flush=True)

    def error(self, message: str):
        self.logger.error(message)
        print(f"❌ {message}", flush=True)

    def progress(self, message: str):
        self.logger.debug(message)
        print(f"🔄 {message}", flush=True)


class StockDataService:
    """股票數據服務"""

    def __init__(self, config_file: str = "config.ini"):
        self.data_provider = StockDataProvider()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.repository = StockDataRepository(config_file)

        # 設置日誌
        self.logger = self._setup_logger()
        self.reporter = ProgressReporter(self.logger)

    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger('StockDataService')
        if logger.handlers:
            return logger  # 避免重複設置

        logger.setLevel(logging.INFO)

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

    def test_connection(self, market_type: str = "tw") -> bool:
        """測試指定市場的資料庫連接"""
        try:
            db_manager = self.repository._get_db_manager(market_type)
            success = db_manager.test_connection()
            if success:
                self.reporter.success(f"{market_type} 資料庫連接測試成功")
            else:
                self.reporter.error(f"{market_type} 資料庫連接測試失敗")
            return success
        except Exception as e:
            self.reporter.error(f"{market_type} 資料庫連接測試失敗: {e}")
            return False

    def process_stock(self, symbol: str,
                      period: Union[Period, str] = Period.YEAR_1,
                      interval: Union[TimeInterval, str] = TimeInterval.DAY_1,
                      check_days: int = 30,
                      expand_history: bool = False,
                      market_type: str = "tw") -> ProcessResult:
        """
        處理單一股票數據

        Args:
            symbol: 股票代號
            period: 抓取數據的時間週期
            interval: 數據間隔
            check_days: 檢查最近N天的數據差異
            expand_history: 是否擴展歷史數據（當設為True時，會嘗試獲取比資料庫更早的歷史數據）
        """
        start_time = time.time()
        result = ProcessResult(symbol=symbol, success=False)

        try:
            interval_str = interval.value if isinstance(
                interval, TimeInterval) else str(interval)
            self.reporter.progress(f"開始處理股票 {symbol} (間隔: {interval_str})")

            # 步驟1: 檢查資料庫中的現有數據
            db_info = self.repository.get_stock_data_info(
                symbol, interval_str, market_type=market_type)

            # 步驟2: 從外部API獲取股票數據
            external_data = self.data_provider.get_stock_data(
                symbol, period, interval)
            if external_data is None or external_data.empty:
                result.error_message = "無法獲取股票數據"
                return result

            self.reporter.info(
                f"{symbol} ({interval_str}): 獲取到 {len(external_data)} 筆外部數據")

            # 步驟3: 比對數據並找出需要更新的部分
            if db_info['exists']:
                # 獲取資料庫中的日期範圍並標準化時區
                db_earliest = pd.to_datetime(db_info['earliest_date'])
                db_latest = pd.to_datetime(db_info['latest_date'])

                # 移除時區資訊以便比較
                if db_earliest.tz is not None:
                    db_earliest = db_earliest.tz_localize(None)
                if db_latest.tz is not None:
                    db_latest = db_latest.tz_localize(None)

                # 獲取外部數據的日期範圍並標準化時區
                ext_earliest = external_data.index.min()
                ext_latest = external_data.index.max()

                # 移除時區資訊以便比較
                if hasattr(ext_earliest, 'tz') and ext_earliest.tz is not None:
                    ext_earliest = ext_earliest.tz_localize(None)
                if hasattr(ext_latest, 'tz') and ext_latest.tz is not None:
                    ext_latest = ext_latest.tz_localize(None)

                self.reporter.info(
                    f"{symbol} ({interval_str}): 資料庫範圍 {db_earliest.date()} ~ "
                    f"{db_latest.date()}")
                self.reporter.info(
                    f"{symbol} ({interval_str}): 外部數據範圍 {ext_earliest.date()} "
                    f"~ {ext_latest.date()}")

                # 找出需要新增的數據
                new_data_parts = []

                # 1. 檢查歷史數據擴展（只在 expand_history=True 時執行）
                if expand_history and ext_earliest < db_earliest:
                    # 確保外部數據索引也沒有時區資訊
                    external_data_no_tz = external_data.copy()
                    if (hasattr(external_data_no_tz.index, 'tz') and
                            external_data_no_tz.index.tz is not None):
                        external_data_no_tz.index = (
                            external_data_no_tz.index.tz_localize(None))

                    historical_data = external_data_no_tz[
                        external_data_no_tz.index < db_earliest]
                    if not historical_data.empty:
                        new_data_parts.append(('historical', historical_data))
                        self.reporter.info(
                            f"{symbol} ({interval_str}): 發現 "
                            f"{len(historical_data)} 筆更早的歷史數據")

                # 2. 檢查未來數據（新的數據）
                if ext_latest > db_latest:
                    # 確保外部數據索引也沒有時區資訊
                    external_data_no_tz = external_data.copy()
                    if (hasattr(external_data_no_tz.index, 'tz') and
                            external_data_no_tz.index.tz is not None):
                        external_data_no_tz.index = (
                            external_data_no_tz.index.tz_localize(None))

                    future_data = external_data_no_tz[
                        external_data_no_tz.index > db_latest]
                    if not future_data.empty:
                        new_data_parts.append(('future', future_data))
                        self.reporter.info(
                            f"{symbol} ({interval_str}): 發現 "
                            f"{len(future_data)} 筆新數據")

                # 3. 檢查最近數據的差異（用於日常更新）
                if not expand_history:  # 只在非擴展模式下檢查最近數據差異
                    recent_external_data = external_data.tail(check_days)
                    outdated_dates = (
                        self.repository.compare_with_external_data(
                            symbol, recent_external_data, interval_str))

                    if outdated_dates:
                        update_data = external_data.loc[outdated_dates]
                        new_data_parts.append(('updated', update_data))
                        self.reporter.info(
                            f"{symbol} ({interval_str}): 發現 "
                            f"{len(update_data)} 筆需要更新的數據")

                # 處理所有需要更新的數據
                if new_data_parts:
                    total_updated = 0
                    date_ranges = []
                    for data_type, data_part in new_data_parts:
                        if not data_part.empty:
                            updated_count = self.repository.upsert_ohlcv_data(
                                symbol,
                                data_part,
                                interval_str, market_type=market_type
                            )
                            total_updated += updated_count
                            start_date = (
                                data_part.index.min().strftime('%Y-%m-%d'))
                            end_date = (
                                data_part.index.max().strftime('%Y-%m-%d'))
                            date_ranges.append(f"{start_date}~{end_date}")
                            type_name = {'historical': '歷史', 'future': '最新',
                                         'updated': '更新'}[
                                data_type]
                            self.reporter.success(
                                f"{symbol} ({interval_str}): 處理 "
                                f"{updated_count} 筆{type_name}數據")
                    result.updated_records = total_updated
                    result.date_range = ', '.join(date_ranges)
                    # 重新計算技術指標和型態訊號
                    indicator_result = self.update_indicators_and_patterns(
                        symbol, interval_str, market_type=market_type)
                    result.indicator_updates = indicator_result
                else:
                    if expand_history:
                        self.reporter.info(
                            f"{symbol} ({interval_str}): 歷史數據已完整，無需擴展")
                    else:
                        self.reporter.info(
                            f"{symbol} ({interval_str}): 最近 "
                            f"{check_days} 天數據無需更新")
            else:
                # 全新股票，儲存所有數據
                saved_count = self.repository.upsert_ohlcv_data(
                    symbol,
                    external_data,
                    interval_str,
                    market_type=market_type
                )
                result.new_records = saved_count
                self.reporter.success(
                    f"{symbol} ({interval_str}): 新增 {saved_count} 筆數據")
                # 新增後也要計算技術指標和型態訊號
                indicator_result = self.update_indicators_and_patterns(
                    symbol, interval_str, market_type=market_type)
                result.indicator_updates = indicator_result

            # 步驟5: 獲取最終統計
            final_info = self.repository.get_stock_data_info(
                symbol, interval_str)
            result.total_records = final_info.get('record_count', 0)
            result.success = True
            result.processing_time = time.time() - start_time

        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            self.reporter.error(f"{symbol} ({interval_str}): 處理失敗 - {e}")

        return result

    def update_technical_indicators(self, symbol: str, interval: str = '1d',
                                    full_history: bool = False,
                                    market_type: str = "tw") -> int:
        """重新計算並更新技術指標

        Args:
            symbol: 股票代號
            interval: 時間間隔
            full_history: 是否更新完整歷史數據的技術指標
            market_type: 市場類型
        """
        try:
            self.reporter.progress(f"{symbol} ({interval}): 開始更新技術指標")

            if full_history:
                # 獲取所有OHLCV數據用於計算技術指標
                self.reporter.info(f"{symbol} ({interval}): 使用完整歷史模式，獲取所有數據")
                ohlcv_data = self.repository.get_all_ohlcv_data(
                    symbol, interval, market_type=market_type)
            else:
                # 獲取足夠的OHLCV數據用於計算技術指標（至少200天）
                ohlcv_data = self.repository.get_latest_ohlcv_data(
                    symbol, interval, days=500, market_type=market_type)

            if ohlcv_data.empty:
                self.reporter.warning(f"{symbol} ({interval}): 無數據可用於計算技術指標")
                return 0

            if len(ohlcv_data) < 60:
                self.reporter.warning(
                    f"{symbol} ({interval}): 數據不足（"
                    f"{len(ohlcv_data)} 筆），無法計算技術指標")
                return 0

            self.reporter.info(
                f"{symbol} ({interval}): 使用 {len(ohlcv_data)} 筆數據計算技術指標")

            # 計算技術指標
            indicators = (
                self.indicator_calculator.calculate_indicators_from_ohlcv(
                    ohlcv_data))

            # 更新到資料庫
            updated_count = self.repository.update_technical_indicators(
                symbol, indicators, interval, market_type=market_type)

            self.reporter.success(
                f"{symbol} ({interval}): 更新了 {updated_count} 筆技術指標")
            return updated_count

        except Exception as e:
            self.reporter.error(f"{symbol} ({interval}): 技術指標更新失敗 - {e}")
            return 0

    def process_multiple_stocks(self, symbols: List[str],
                                period: Union[Period, str] = Period.YEAR_1,
                                interval: Union[TimeInterval,
                                                str] = TimeInterval.DAY_1,
                                check_days: int = 30,
                                expand_history: bool = False,
                                market_type: str = "tw"
                                ) -> Dict[str, ProcessResult]:
        """批量處理多個股票"""
        interval_str = interval.value if isinstance(
            interval, TimeInterval) else str(interval)

        if expand_history:
            self.reporter.info(
                f"🔄 歷史數據擴展模式 - 開始批量處理 {len(symbols)} 個股票 (間隔: {interval_str})")
            self.reporter.info("將嘗試獲取比資料庫更早的歷史數據")
        else:
            self.reporter.info(
                f"開始批量處理 {len(symbols)} 個股票 (間隔: {interval_str})")

        results = {}
        success_count = 0
        total_updates = 0
        total_new_records = 0

        for i, symbol in enumerate(symbols, 1):
            if expand_history:
                print(
                    f"\n📊 [{i}/{len(symbols)}] 擴展歷史數據 "
                    f"{symbol} ({interval_str})", flush=True)
            else:
                print(
                    f"\n📊 [{i}/{len(symbols)}] 處理 {symbol} ({interval_str})",
                    flush=True)

            result = self.process_stock(
                symbol,
                period,
                interval,
                check_days,
                expand_history,
                market_type=market_type
            )
            results[symbol] = result

            if result.success:
                success_count += 1
                total_updates += result.updated_records
                total_new_records += result.new_records

                if result.updated_records > 0 or result.new_records > 0:
                    print(f"   ✅ 成功 | 新增: {result.new_records} 筆 | "
                          f"更新: {result.updated_records} 筆 | "
                          f"指標: {result.indicator_updates} 筆", flush=True)
                    print(f"   📅 時間範圍: {result.date_range}", flush=True)
                else:
                    print(
                        f"   ✅ 成功 | 無需更新 | 總計: {result.total_records} 筆",
                        flush=True)

                print(
                    f"   ⏱️  處理時間: {result.processing_time:.2f} 秒", flush=True)
            else:
                print(f"   ❌ 失敗: {result.error_message}", flush=True)

        # 顯示總結
        print(f"\n{'='*60}", flush=True)
        if expand_history:
            print(f"📈 歷史數據擴展完成 ({interval_str})", flush=True)
        else:
            print(f"📈 批量處理完成 ({interval_str})", flush=True)
        print(f"✅ 成功: {success_count}/{len(symbols)} 個股票", flush=True)
        print(f"📊 新增記錄: {total_new_records:,} 筆", flush=True)
        print(f"📊 更新記錄: {total_updates:,} 筆", flush=True)
        print(f"{'='*60}", flush=True)

        return results

    def force_update_all_indicators(self, symbols: List[str] = None,
                                    interval: str = '1d',
                                    full_history: bool = True,
                                    market_type: str = "tw"
                                    ) -> Dict[str, int]:
        """強制更新所有股票的技術指標

        Args:
            symbols: 股票代號列表，如果為None則獲取所有股票
            interval: 時間間隔
            full_history: 是否更新完整歷史數據的技術指標
            market_type: 市場類型
        """
        if symbols is None:
            symbols = self.repository.get_symbols_list(
                interval, market_type=market_type)
        mode_text = "完整歷史" if full_history else "最近數據"
        self.reporter.info(
            f"開始強制更新 {len(symbols)} 個股票的技術指標 ({interval}) - {mode_text}模式")

        results = {}
        for i, symbol in enumerate(symbols, 1):
            print(
                f"\n🔄 [{i}/{len(symbols)}] 更新 {symbol} ({interval}) "
                f"技術指標 ({mode_text})", flush=True)
            updated_count = self.update_technical_indicators(
                symbol, interval, full_history, market_type=market_type)
            results[symbol] = updated_count

            if updated_count > 0:
                print(f"   ✅ 更新了 {updated_count} 筆技術指標", flush=True)
            else:
                print("   ⚠️  無更新或數據不足", flush=True)

        return results

    def get_database_statistics(
            self,
            interval: str = '1d',
            market_type: str = "tw"
    ) -> Dict[str, Any]:
        """獲取指定間隔的資料庫統計資訊"""
        return self.repository.get_database_statistics(
            interval,
            market_type=market_type
        )

    def get_all_database_statistics(
            self,
            market_type: str = "tw"
    ) -> Dict[str, Dict[str, Any]]:
        """獲取所有間隔表的資料庫統計資訊"""
        return self.repository.get_all_tables_statistics(
            market_type=market_type
        )

    def expand_historical_data(self, symbols: List[str] = None,
                               interval: str = '1d'
                               ) -> Dict[str, ProcessResult]:
        """專用的歷史數據擴展功能"""
        if symbols is None:
            symbols = self.repository.get_symbols_list(interval)

        self.reporter.info(f"🔄 開始擴展 {len(symbols)} 個股票的歷史數據 ({interval})")

        return self.process_multiple_stocks(
            symbols=symbols,
            period=Period.MAX,  # 使用最大期間以獲取所有可用的歷史數據
            interval=TimeInterval.DAY_1 if interval == '1d' else interval,
            expand_history=True
        )

    def update_pattern_signals_for_stocks(
            self,
            symbols: List[str],
            interval: str = '1d',
            market_type: str = "tw",
            recent_only: bool = False
    ) -> Dict[str, int]:
        """為多個股票更新K線型態訊號

        Args:
            symbols: 股票代號列表
            interval: 時間間隔
            market_type: 市場類型
            recent_only: 是否僅更新最近數據（預設False=所有歷史數據）

        Returns:
            Dict[str, int]: 每個股票的更新筆數
        """
        from pattern_detection.pattern_detection import (
            detect_patterns, combine_patterns
        )

        results = {}

        for symbol in symbols:
            try:
                self.reporter.progress(f"開始處理 {symbol} 的K線型態偵測")

                # 獲取OHLCV數據
                if recent_only:
                    # 僅獲取最近30筆數據
                    ohlcv_data = self.repository.get_latest_ohlcv_data(
                        symbol, interval, days=30, market_type=market_type)
                    self.reporter.info(f"{symbol}: 獲取最近30筆數據進行型態偵測")
                else:
                    # 獲取所有歷史數據
                    ohlcv_data = self.repository.get_all_ohlcv_data(
                        symbol, interval, market_type=market_type)
                    self.reporter.info(f"{symbol}: 獲取所有歷史數據進行型態偵測")

                if ohlcv_data.empty:
                    self.reporter.warning(f"{symbol}: 無數據可用於型態偵測")
                    results[symbol] = 0
                    continue

                if len(ohlcv_data) < 10:
                    self.reporter.warning(
                        f"{symbol}: 數據不足（{len(ohlcv_data)} 筆），"
                        f"無法進行型態偵測")
                    results[symbol] = 0
                    continue                # 重新命名欄位以符合 pattern_detection 的需求
                pattern_df = ohlcv_data.copy()
                pattern_df = pattern_df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close'
                })

                # 確保必要欄位存在
                required_cols = ['open', 'high', 'low', 'close']
                missing_cols = [col for col in required_cols
                                if col not in pattern_df.columns]
                if missing_cols:
                    self.reporter.warning(
                        f"{symbol}: 缺少必要欄位: {missing_cols}")
                    results[symbol] = 0
                    continue

                # 確保數據類型為 float64 (double)
                for col in required_cols:
                    pattern_df[col] = pd.to_numeric(
                        pattern_df[col], errors='coerce').astype('float64')

                # 進行型態偵測
                pattern_df = detect_patterns(pattern_df)

                # 組合型態訊號
                pattern_df['PatternSignals'] = pattern_df.apply(
                    combine_patterns, axis=1)

                # 篩選出有型態訊號的數據
                signals_series = pattern_df['PatternSignals']

                # 更新到資料庫
                updated_count = self.repository.update_pattern_signals(
                    symbol, signals_series, interval, market_type=market_type)

                results[symbol] = updated_count
                self.reporter.success(
                    f"{symbol}: 成功更新 {updated_count} 筆型態訊號")

            except Exception as e:
                self.reporter.error(f"{symbol}: 型態偵測失敗 - {e}")
                results[symbol] = 0

        return results

    def update_indicators_and_patterns(self, symbol: str, interval: str = '1d',
                                       full_history: bool = False,
                                       market_type: str = "tw") -> int:
        """重新計算並更新技術指標和K線型態訊號

        Args:
            symbol: 股票代號
            interval: 時間間隔
            full_history: 是否更新完整歷史數據
            market_type: 市場類型

        Returns:
            int: 更新的筆數
        """
        try:
            # 更新技術指標
            indicator_count = self.update_technical_indicators(
                symbol, interval, full_history, market_type)

            # 更新型態訊號（正常模式只更新最近30筆）
            pattern_results = self.update_pattern_signals_for_stocks(
                [symbol], interval, market_type, recent_only=True)
            pattern_count = pattern_results.get(symbol, 0)

            self.reporter.info(
                f"{symbol} ({interval}): 技術指標更新 {indicator_count} 筆，"
                f"型態訊號更新 {pattern_count} 筆")

            return indicator_count

        except Exception as e:
            self.reporter.error(
                f"{symbol} ({interval}): 指標和型態更新失敗 - {e}")
            return 0
