"""
資料庫操作模組
負責股票數據的資料庫操作，包括數據比對、更新和技術指標儲存
支援根據時間間隔動態創建不同的資料表
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import text
from config.database_config import DatabaseManager
from providers.stock_data_provider import TimeInterval


class StockDataRepository:
    """股票數據存儲庫"""

    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    def _get_database_name(self, market_type: str) -> str:
        """根據市場類型回傳資料庫名稱"""
        mapping = {
            "tw": "market_stock_tw",
            "us": "market_stock_us",
            "etf": "market_etf",
            "index": "market_index",
            "forex": "market_forex",
            "crypto": "market_crypto",
            "futures": "market_futures"
        }
        return mapping.get(market_type.lower(), "market_stock_tw")  # 預設台股

    def _get_db_manager(self, market_type: str):
        """根據市場類型取得對應的 DatabaseManager 實例"""
        db_name = self._get_database_name(market_type)
        return DatabaseManager(self.config_file, database=db_name)

    def _get_table_name(self, interval: str) -> str:
        """根據時間間隔獲取對應的表名"""
        # 時間間隔到表名的映射
        interval_mapping = {
            '1m': 'stock_data_1m',
            '5m': 'stock_data_5m',
            '15m': 'stock_data_15m',
            '30m': 'stock_data_30m',
            '1h': 'stock_data_1h',
            '1d': 'stock_data_1d',
            '1wk': 'stock_data_1wk',
            '1mo': 'stock_data_1mo'
        }

        # 標準化間隔字符串
        if isinstance(interval, TimeInterval):
            interval_str = interval.value
        else:
            interval_str = str(interval).lower()

        table_name = interval_mapping.get(interval_str, 'stock_data_1d')
        self.logger.info(f"時間間隔 {interval_str} 對應表名: {table_name}")
        return table_name

    def _ensure_table_exists(self, interval: str, market_type: str = "tw"):
        """確保指定間隔的資料表存在"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            # 先檢查資料表是否存在
            check_table_sql = text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = :table_name
            """)
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    check_table_sql, {"table_name": table_name}).fetchone()
                table_exists = result[0] > 0
                if not table_exists:
                    # 創建新資料表
                    create_table_sql = text(f"""
                    CREATE TABLE {table_name} (
                        id BIGINT IDENTITY(1,1) PRIMARY KEY,
                        symbol NVARCHAR(10) NOT NULL,
                        datetime DATETIME2 NOT NULL,
                        open_price DECIMAL(16,4),
                        high_price DECIMAL(16,4),
                        low_price DECIMAL(16,4),
                        close_price DECIMAL(16,4),
                        volume BIGINT,
                        rsi_5 DECIMAL(8,4),
                        rsi_7 DECIMAL(8,4),
                        rsi_10 DECIMAL(8,4),
                        rsi_14 DECIMAL(8,4),
                        rsi_21 DECIMAL(8,4),
                        dif DECIMAL(12,4),
                        macd DECIMAL(12,4),
                        macd_histogram DECIMAL(12,4),
                        rsv DECIMAL(8,4),
                        k_value DECIMAL(8,4),
                        d_value DECIMAL(8,4),
                        j_value DECIMAL(8,4),
                        ma5 DECIMAL(16,4),
                        ma10 DECIMAL(16,4),
                        ma20 DECIMAL(16,4),
                        ma60 DECIMAL(16,4),
                        ema12 DECIMAL(16,4),
                        ema26 DECIMAL(16,4),
                        bb_upper DECIMAL(16,4),
                        bb_middle DECIMAL(16,4),
                        bb_lower DECIMAL(16,4),
                        atr DECIMAL(12,4),
                        cci DECIMAL(12,4),
                        willr DECIMAL(8,4),
                        mom DECIMAL(12,4),
                        pattern_signals NVARCHAR(1000),
                        created_at DATETIME2 DEFAULT GETDATE(),
                        updated_at DATETIME2 DEFAULT GETDATE()
                    )
                    """)
                    with db_manager.get_connection() as conn:
                        conn.execute(create_table_sql)
                        conn.commit()
                        self.logger.info(f"創建新的 {table_name} 資料表")

                        # 創建唯一約束
                        try:
                            constraint_name = (
                                f"UK_{table_name}_symbol_datetime"
                            )
                            constraint_sql = text(f"""
                            ALTER TABLE {table_name}
                            ADD CONSTRAINT {constraint_name}
                            UNIQUE (symbol, datetime)
                            """)
                            conn.execute(constraint_sql)
                            conn.commit()
                            self.logger.info(f"創建 {table_name} 唯一約束成功")
                        except Exception as e:
                            if "already an object named" not in str(e):
                                self.logger.warning(
                                    f"創建 {table_name} 約束失敗，但可以繼續: {e}")

                        # 創建索引
                        try:
                            index_sqls = [
                                text(
                                    f"CREATE NONCLUSTERED INDEX "
                                    f"IX_{table_name}_symbol_datetime "
                                    f"ON {table_name} "
                                    f"(symbol, datetime DESC)"),
                                text(
                                    f"CREATE NONCLUSTERED INDEX "
                                    f"IX_{table_name}_datetime "
                                    f"ON {table_name} (datetime DESC)"),
                                text(
                                    f"CREATE NONCLUSTERED INDEX "
                                    f"IX_{table_name}_symbol "
                                    f"ON {table_name} (symbol)")
                            ]

                            for idx_sql in index_sqls:
                                conn.execute(idx_sql)
                            conn.commit()
                            self.logger.info(f"創建 {table_name} 索引成功")
                        except Exception as e:
                            self.logger.warning(
                                f"創建 {table_name} 索引失敗，但可以繼續: {e}")

                else:
                    self.logger.info(f"{table_name} 資料表已存在")
        except Exception as e:
            self.logger.error(f"檢查/創建 {table_name} 資料表失敗: {e}")
            # 如果是約束相關錯誤，可以繼續執行
            if ("already an object named" in str(e) or
                    "Invalid object name" in str(e)):
                self.logger.info(f"{table_name} 資料表創建可能有問題，但嘗試繼續執行")
            else:
                raise

    def _ensure_pattern_signals_column(self, interval: str,
                                       market_type: str = "tw"):
        """確保資料表有 pattern_signals 欄位"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            # 檢查 pattern_signals 欄位是否存在
            check_column_sql = text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = :table_name AND COLUMN_NAME = 'pattern_signals'
            """)
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    check_column_sql, {"table_name": table_name}).fetchone()
                column_exists = result[0] > 0

                if not column_exists:
                    # 新增 pattern_signals 欄位
                    add_column_sql = text(f"""
                    ALTER TABLE {table_name}
                    ADD pattern_signals NVARCHAR(1000)
                    """)
                    conn.execute(add_column_sql)
                    conn.commit()
                    self.logger.info(f"為 {table_name} 新增 pattern_signals 欄位")
                else:
                    self.logger.info(f"{table_name} 的 pattern_signals 欄位已存在")
        except Exception as e:
            self.logger.error(f"檢查/新增 {table_name} pattern_signals 欄位失敗: {e}")

    def get_stock_data_info(
        self,
        symbol: str,
        interval: str = '1d',
        market_type: str = "tw"
    ) -> Dict[str, Any]:
        """獲取股票在指定間隔資料表中的數據資訊"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            with db_manager.get_connection() as conn:
                query = text(f"""
                SELECT
                    COUNT(*) as record_count,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date,
                    MAX(updated_at) as last_updated
                FROM {table_name}
                WHERE symbol = :symbol
                """)
                result = conn.execute(query, {"symbol": symbol}).fetchone()
                if result and result[0] > 0:
                    return {
                        'exists': True,
                        'record_count': result[0],
                        'earliest_date': result[1],
                        'latest_date': result[2],
                        'last_updated': result[3],
                        'table_name': table_name
                    }
                else:
                    return {'exists': False, 'table_name': table_name}
        except Exception as e:
            self.logger.warning(f"獲取 {table_name} 數據資訊失敗: {e}")
            return {'exists': False, 'table_name': table_name}

    def get_latest_ohlcv_data(
        self,
        symbol: str,
        interval: str = '1d',
        days: int = 30,
        market_type: str = "tw"
    ) -> pd.DataFrame:
        """獲取指定間隔表中最近N天的OHLCV數據"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            with db_manager.get_connection() as conn:
                query = text(f"""
                SELECT TOP (:days) datetime, open_price, high_price,
                       low_price, close_price, volume
                FROM {table_name}
                WHERE symbol = :symbol
                ORDER BY datetime DESC
                """)
                result = conn.execute(query, {"symbol": symbol, "days": days})
                df = pd.DataFrame(result.fetchall(),
                                  columns=['datetime', 'open_price',
                                           'high_price', 'low_price',
                                           'close_price', 'volume'])
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df
        except Exception as e:
            self.logger.error(f"獲取 {table_name} OHLCV數據失敗: {e}")
            return pd.DataFrame()

    def get_all_ohlcv_data(
        self,
        symbol: str,
        interval: str = '1d',
        market_type: str = "tw"
    ) -> pd.DataFrame:
        """獲取指定間隔表中該股票的所有OHLCV數據"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            with db_manager.get_connection() as conn:
                query = text(f"""
                SELECT datetime, open_price, high_price,
                       low_price, close_price, volume
                FROM {table_name}
                WHERE symbol = :symbol
                ORDER BY datetime ASC
                """)
                result = conn.execute(query, {"symbol": symbol})
                df = pd.DataFrame(result.fetchall(),
                                  columns=['datetime', 'open_price',
                                           'high_price', 'low_price',
                                           'close_price', 'volume'])
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    # 標準化欄位名稱，與 get_latest_ohlcv_data 保持一致
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df
        except Exception as e:
            self.logger.error(f"獲取 {table_name} 所有數據失敗: {e}")
            return pd.DataFrame()

    def compare_with_external_data(
            self,
            symbol: str,
            external_data: pd.DataFrame,
            interval: str = '1d',
            tolerance: float = 0.001,
            market_type: str = "tw"
    ) -> List[datetime]:
        """比對指定間隔表的數據與外部數據，返回需要更新的日期列表"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            if external_data.empty:
                return []
            start_date = external_data.index.min()
            end_date = external_data.index.max()
            with db_manager.get_connection() as conn:
                query = text(f"""
                SELECT datetime, open_price, high_price, low_price,
                       close_price, volume
                FROM {table_name}
                WHERE symbol = :symbol
                AND datetime BETWEEN :start_date AND :end_date
                ORDER BY datetime
                """)
                result = conn.execute(query, {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                })
                db_data = pd.DataFrame(result.fetchall(),
                                       columns=['datetime', 'open_price',
                                                'high_price', 'low_price',
                                                'close_price', 'volume'])
            if db_data.empty:
                return external_data.index.tolist()
            db_data['datetime'] = pd.to_datetime(db_data['datetime'])
            db_data.set_index('datetime', inplace=True)
            outdated_dates = []
            for dt in external_data.index:
                if dt not in db_data.index:
                    outdated_dates.append(dt)
                else:
                    ext_row = external_data.loc[dt]
                    db_row = db_data.loc[dt]
                    price_fields = [('Open', 'open_price'),
                                    ('High', 'high_price'),
                                    ('Low', 'low_price'),
                                    ('Close', 'close_price')]
                    needs_update = False
                    for ext_field, db_field in price_fields:
                        ext_val = float(ext_row[ext_field])
                        db_val = float(db_row[db_field]) if pd.notna(
                            db_row[db_field]) else None
                        if db_val is None or abs(ext_val - db_val) > tolerance:
                            needs_update = True
                            break
                    if not needs_update:
                        ext_vol = int(ext_row['Volume'])
                        db_vol = int(db_row['volume']) if pd.notna(
                            db_row['volume']) else None
                        if db_vol is None or ext_vol != db_vol:
                            needs_update = True
                    if needs_update:
                        outdated_dates.append(dt)
            self.logger.info(
                f"{symbol}: 在 {table_name} 中找到 {len(outdated_dates)} 筆需要更新的數據")
            return outdated_dates
        except Exception as e:
            self.logger.error(f"數據比對失敗: {e}")
            return []

    def upsert_ohlcv_data(
            self,
            symbol: str,
            data: pd.DataFrame,
            interval: str = '1d',
            market_type: str = "tw"
    ) -> int:
        """插入或更新指定間隔表的OHLCV數據"""
        if data.empty:
            return 0
        self._ensure_table_exists(interval, market_type)
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        upsert_sql = text(f"""
        MERGE {table_name} AS target
        USING (SELECT :symbol as symbol, :datetime as datetime) AS source
        ON (target.symbol = source.symbol
        AND target.datetime = source.datetime)
        WHEN MATCHED THEN
            UPDATE SET
                open_price = :open_price, high_price = :high_price,
                low_price = :low_price, close_price = :close_price,
                volume = :volume, updated_at = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (symbol, datetime, open_price, high_price,
                   low_price, close_price, volume)
            VALUES (:symbol, :datetime, :open_price, :high_price,
                   :low_price, :close_price, :volume);
        """)
        saved_count = 0
        try:
            with db_manager.get_connection() as conn:
                for dt, row in data.iterrows():
                    params = {
                        'symbol': symbol,
                        'datetime': (dt.to_pydatetime()
                                     if isinstance(dt, pd.Timestamp) else dt),
                        'open_price': (float(row['Open'])
                                       if pd.notna(row['Open']) else None),
                        'high_price': (float(row['High'])
                                       if pd.notna(row['High']) else None),
                        'low_price': (float(row['Low'])
                                      if pd.notna(row['Low']) else None),
                        'close_price': (float(row['Close'])
                                        if pd.notna(row['Close']) else None),
                        'volume': (int(row['Volume'])
                                   if pd.notna(row['Volume']) else None)
                    }
                    conn.execute(upsert_sql, params)
                    saved_count += 1
                conn.commit()
                self.logger.info(
                    f"{symbol}: 成功更新 {table_name} 中 {saved_count} 筆OHLCV數據")
        except Exception as e:
            self.logger.error(f"更新 {table_name} OHLCV數據失敗: {e}")
            raise
        return saved_count

    def update_technical_indicators(
            self,
            symbol: str,
            indicators: Dict[str, pd.Series],
            interval: str = '1d',
            market_type: str = "tw"
    ) -> int:
        """更新指定間隔表的技術指標"""
        if not indicators:
            return 0
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        indicator_fields = ', '.join(
            [f"{field} = :{field}" for field in indicators.keys()])
        update_sql = text(f"""
        UPDATE {table_name}
        SET {indicator_fields}, updated_at = GETDATE()
        WHERE symbol = :symbol AND datetime = :datetime
        """)
        updated_count = 0
        try:
            with db_manager.get_connection() as conn:
                common_index = next(iter(indicators.values())).index
                for dt in common_index:
                    params = {'symbol': symbol, 'datetime': dt.to_pydatetime(
                    ) if isinstance(dt, pd.Timestamp) else dt}
                    for field, series in indicators.items():
                        value = series.loc[dt] if dt in series.index else None
                        if pd.notna(value):
                            if field in ['rsi_5', 'rsi_7', 'rsi_10', 'rsi_14',
                                         'rsi_21', 'rsv', 'k_value', 'd_value',
                                         'j_value', 'willr']:
                                params[field] = round(float(value), 4)
                            elif field in ['dif', 'macd', 'macd_histogram',
                                           'atr', 'mom']:
                                params[field] = round(float(value), 6)
                            elif field == 'cci':
                                params[field] = round(float(value), 4)
                            else:
                                params[field] = round(float(value), 6)
                        else:
                            params[field] = None
                    result = conn.execute(update_sql, params)
                    if result.rowcount > 0:
                        updated_count += 1
                conn.commit()
                self.logger.info(
                    f"{symbol}: 成功更新 {table_name} 中 {updated_count} 筆技術指標")
        except Exception as e:
            self.logger.error(f"更新 {table_name} 技術指標失敗: {e}")
            raise
        return updated_count

    def update_pattern_signals(
            self,
            symbol: str,
            pattern_signals: pd.Series,
            interval: str = '1d',
            market_type: str = "tw"
    ) -> int:
        """更新指定間隔表的型態訊號"""
        if pattern_signals.empty:
            return 0

        self._ensure_pattern_signals_column(interval, market_type)
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)

        update_sql = text(f"""
        UPDATE {table_name}
        SET pattern_signals = :pattern_signals, updated_at = GETDATE()
        WHERE symbol = :symbol AND datetime = :datetime
        """)

        updated_count = 0
        try:
            with db_manager.get_connection() as conn:
                for dt, signals in pattern_signals.items():
                    params = {
                        'symbol': symbol,
                        'datetime': (dt.to_pydatetime()
                                     if isinstance(dt, pd.Timestamp) else dt),
                        'pattern_signals': signals if signals else None
                    }
                    result = conn.execute(update_sql, params)
                    if result.rowcount > 0:
                        updated_count += 1
                conn.commit()
                self.logger.info(
                    f"{symbol}: 成功更新 {table_name} 中 {updated_count} 筆型態訊號")
        except Exception as e:
            self.logger.error(f"更新 {table_name} 型態訊號失敗: {e}")
            raise
        return updated_count

    def get_symbols_list(
            self,
            interval: str = '1d',
            market_type: str = "tw"
    ) -> List[str]:
        """獲取指定間隔表中所有股票代號"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            with db_manager.get_connection() as conn:
                query = text(
                    f"SELECT DISTINCT symbol "
                    f"FROM {table_name} ORDER BY symbol")
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            self.logger.error(f"獲取 {table_name} 股票代號列表失敗: {e}")
            return []

    def get_database_statistics(
            self,
            interval: str = '1d',
            market_type: str = "tw"
    ) -> Dict[str, Any]:
        """獲取指定間隔表的資料庫統計資訊"""
        table_name = self._get_table_name(interval)
        db_manager = self._get_db_manager(market_type)
        try:
            with db_manager.get_connection() as conn:
                # 整體統計
                overall_query = text(f"""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date
                FROM {table_name}
                """)
                overall_result = conn.execute(overall_query).fetchone()

                # 各股票統計
                symbol_query = text(f"""
                SELECT
                    symbol,
                    COUNT(*) as record_count,
                    MIN(datetime) as start_date,
                    MAX(datetime) as end_date,
                    MAX(updated_at) as last_updated
                FROM {table_name}
                GROUP BY symbol
                ORDER BY record_count DESC
                """)
                symbol_results = conn.execute(symbol_query).fetchall()

                return {
                    'table_name': table_name,
                    'interval': interval,
                    'total_records': (overall_result[0]
                                      if overall_result else 0),
                    'unique_symbols': (overall_result[1]
                                       if overall_result else 0),
                    'date_range': {
                        'earliest': (overall_result[2]
                                     if overall_result else None),
                        'latest': (overall_result[3]
                                   if overall_result else None)
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
            self.logger.error(f"獲取 {table_name} 統計資訊失敗: {e}")
            return {'table_name': table_name, 'interval': interval}

    def get_all_tables_statistics(
            self,
            market_type: str = "tw"
    ) -> Dict[str, Dict[str, Any]]:
        """獲取所有間隔表的統計資訊"""
        intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
        all_stats = {}
        for interval in intervals:
            try:
                table_name = self._get_table_name(interval)
                db_manager = self._get_db_manager(market_type)
                with db_manager.get_connection() as conn:
                    check_query = text("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = :table_name
                    """)
                    result = conn.execute(
                        check_query, {"table_name": table_name}).fetchone()
                    if result[0] > 0:
                        stats = self.get_database_statistics(
                            interval, market_type)
                        if stats.get('total_records', 0) > 0:
                            all_stats[interval] = stats
            except Exception as e:
                self.logger.warning(f"獲取 {interval} 統計資訊失敗: {e}")
        return all_stats
