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
        self.db_manager = DatabaseManager(config_file)
        self.logger = logging.getLogger(__name__)

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

    def _ensure_table_exists(self, interval: str):
        """確保指定間隔的資料表存在"""
        table_name = self._get_table_name(interval)

        try:
            # 先檢查資料表是否存在
            check_table_sql = text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = :table_name
            """)

            with self.db_manager.get_connection() as conn:
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
                        updated_at DATETIME2 DEFAULT GETDATE()
                    )
                    """)

                    conn.execute(create_table_sql)
                    conn.commit()
                    self.logger.info(f"創建新的 {table_name} 資料表")

                    # 創建唯一約束
                    try:
                        constraint_name = f"UK_{table_name}_symbol_datetime"
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
                                f"ON {table_name} (symbol, datetime DESC)"),
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
                        self.logger.warning(f"創建 {table_name} 索引失敗，但可以繼續: {e}")

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

    def get_stock_data_info(self, symbol: str, interval: str = '1d'
                            ) -> Dict[str, Any]:
        """獲取股票在指定間隔資料表中的數據資訊"""
        table_name = self._get_table_name(interval)

        try:
            with self.db_manager.get_connection() as conn:
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

    def get_latest_ohlcv_data(self, symbol: str, interval: str = '1d',
                              days: int = 30) -> pd.DataFrame:
        """獲取指定間隔表中最近N天的OHLCV數據"""
        table_name = self._get_table_name(interval)

        try:
            with self.db_manager.get_connection() as conn:
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
                    # 重命名欄位以匹配技術指標計算器的期望格式
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                return df

        except Exception as e:
            self.logger.error(f"獲取 {table_name} OHLCV數據失敗: {e}")
            return pd.DataFrame()

    def get_all_ohlcv_data(self, symbol: str, interval: str = '1d'
                           ) -> pd.DataFrame:
        """獲取指定間隔表中該股票的所有OHLCV數據"""
        table_name = self._get_table_name(interval)

        try:
            with self.db_manager.get_connection() as conn:
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
                    # 保持資料庫欄位名稱不變，技術指標計算器現在能處理這種格式

                return df

        except Exception as e:
            self.logger.error(f"獲取 {table_name} 所有數據失敗: {e}")
            return pd.DataFrame()

    def compare_with_external_data(self, symbol: str,
                                   external_data: pd.DataFrame,
                                   interval: str = '1d',
                                   tolerance: float = 0.001) -> List[datetime]:
        """比對指定間隔表的數據與外部數據，返回需要更新的日期列表"""
        table_name = self._get_table_name(interval)

        try:
            # 獲取外部數據的日期範圍
            if external_data.empty:
                return []

            start_date = external_data.index.min()
            end_date = external_data.index.max()

            # 獲取資料庫中對應日期範圍的數據
            with self.db_manager.get_connection() as conn:
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
                # 如果資料庫沒有數據，返回所有外部數據的日期
                return external_data.index.tolist()

            db_data['datetime'] = pd.to_datetime(db_data['datetime'])
            db_data.set_index('datetime', inplace=True)

            outdated_dates = []

            for dt in external_data.index:
                if dt not in db_data.index:
                    outdated_dates.append(dt)
                else:
                    # 比較OHLCV數據
                    ext_row = external_data.loc[dt]
                    db_row = db_data.loc[dt]

                    # 檢查價格數據
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

                    # 檢查成交量
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

    def upsert_ohlcv_data(self, symbol: str, data: pd.DataFrame,
                          interval: str = '1d') -> int:
        """插入或更新指定間隔表的OHLCV數據"""
        if data.empty:
            return 0

        # 確保表存在
        self._ensure_table_exists(interval)
        table_name = self._get_table_name(interval)

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
            with self.db_manager.get_connection() as conn:
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

    def update_technical_indicators(self, symbol: str,
                                    indicators: Dict[str, pd.Series],
                                    interval: str = '1d') -> int:
        """更新指定間隔表的技術指標"""
        if not indicators:
            return 0

        table_name = self._get_table_name(interval)

        # 構建更新SQL
        indicator_fields = ', '.join(
            [f"{field} = :{field}" for field in indicators.keys()])

        update_sql = text(f"""
        UPDATE {table_name}
        SET {indicator_fields}, updated_at = GETDATE()
        WHERE symbol = :symbol AND datetime = :datetime
        """)

        updated_count = 0
        try:
            with self.db_manager.get_connection() as conn:
                # 獲取所有指標的索引（應該都相同）
                common_index = next(iter(indicators.values())).index

                for dt in common_index:
                    params = {'symbol': symbol, 'datetime': dt.to_pydatetime(
                    ) if isinstance(dt, pd.Timestamp) else dt}

                    # 添加所有指標值
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
                            else:  # MA 和 BB 系列
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

    def get_symbols_list(self, interval: str = '1d') -> List[str]:
        """獲取指定間隔表中所有股票代號"""
        table_name = self._get_table_name(interval)

        try:
            with self.db_manager.get_connection() as conn:
                query = text(
                    f"SELECT DISTINCT symbol "
                    f"FROM {table_name} ORDER BY symbol")
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            self.logger.error(f"獲取 {table_name} 股票代號列表失敗: {e}")
            return []

    def get_database_statistics(self, interval: str = '1d') -> Dict[str, Any]:
        """獲取指定間隔表的資料庫統計資訊"""
        table_name = self._get_table_name(interval)

        try:
            with self.db_manager.get_connection() as conn:
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

    def get_all_tables_statistics(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有間隔表的統計資訊"""
        intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
        all_stats = {}

        for interval in intervals:
            try:
                table_name = self._get_table_name(interval)
                # 檢查表是否存在
                with self.db_manager.get_connection() as conn:
                    check_query = text("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = :table_name
                    """)
                    result = conn.execute(
                        check_query, {"table_name": table_name}).fetchone()

                    if result[0] > 0:
                        stats = self.get_database_statistics(interval)
                        if stats.get('total_records', 0) > 0:
                            all_stats[interval] = stats
            except Exception as e:
                self.logger.warning(f"獲取 {interval} 統計資訊失敗: {e}")

        return all_stats
