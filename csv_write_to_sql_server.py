import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob
import logging
import configparser
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time
import sys
from dataclasses import dataclass
from functools import lru_cache
import hashlib


@dataclass
class ImportStats:
    """匯入統計資料結構"""
    imported_rows: int = 0
    updated_rows: int = 0
    skipped_rows: int = 0
    total_rows: int = 0
    elapsed_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    mode_used: str = "unknown"
    detection_reason: Optional[str] = None
    quality_report: Optional[Dict] = None


class ProgressReporter:
    """進度報告器 - 分離日誌和終端輸出"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.console_output = True

    def info(self, message: str, log_only: bool = False):
        """資訊訊息"""
        self.logger.info(message)
        if not log_only and self.console_output:
            print(f"ℹ️  {message}")

    def success(self, message: str, log_only: bool = False):
        """成功訊息"""
        self.logger.info(f"SUCCESS: {message}")
        if not log_only and self.console_output:
            print(f"✅ {message}")

    def warning(self, message: str, log_only: bool = False):
        """警告訊息"""
        self.logger.warning(message)
        if not log_only and self.console_output:
            print(f"⚠️  {message}")

    def error(self, message: str, log_only: bool = False):
        """錯誤訊息"""
        self.logger.error(message)
        if not log_only and self.console_output:
            print(f"❌ {message}")

    def progress(self, message: str, log_only: bool = False):
        """進度訊息"""
        self.logger.debug(message)
        if not log_only and self.console_output:
            print(f"🔄 {message}")

    def header(self, title: str, width: int = 80):
        """標題"""
        self.logger.info(f"=== {title} ===")
        if self.console_output:
            print(f"\n{'='*width}")
            print(f"{title:^{width}}")
            print(f"{'='*width}")

    def separator(self, width: int = 80):
        """分隔符"""
        if self.console_output:
            print(f"{'-'*width}")


class DatabaseConfig:
    """資料庫配置類"""

    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        # 讀取配置
        self.server = self.config.get('database', 'server')
        self.database = self.config.get('database', 'database')
        self.driver = self.config.get('database', 'driver')

        # 可選的認證資訊
        try:
            self.username = self.config.get('database', 'username')
            self.password = self.config.get('database', 'password')
        except (configparser.NoOptionError, configparser.NoSectionError):
            self.username = None
            self.password = None

    @lru_cache(maxsize=1)
    def get_sqlalchemy_url(self) -> str:
        """生成 SQLAlchemy 連接字串（快取結果）"""
        if self.username and self.password:
            return (f"mssql+pyodbc://{self.username}:{self.password}@"
                    f"{self.server}/{self.database}?driver={self.driver}")
        else:
            return (f"mssql+pyodbc://@{self.server}/{self.database}"
                    f"?driver={self.driver}&trusted_connection=yes")


class DataCleaner:
    """數據清理工具類"""

    # 欄位對應表 - 使用常數避免重複定義
    COLUMN_MAPPING = {
        'Date': 'date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'volume',
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

    # 數值欄位列表
    NUMERIC_COLUMNS = [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'rsi_5', 'rsi_7', 'rsi_10', 'rsi_14', 'rsi_21',
        'dif', 'macd', 'macd_histogram',
        'rsv', 'k_value', 'd_value', 'j_value',
        'ma5', 'ma10', 'ma20', 'ma60', 'ema12', 'ema26',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'cci', 'willr', 'mom'
    ]

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """清理和準備 DataFrame"""
        # 創建副本避免修改原始數據
        df_clean = df.copy()

        # 重新命名欄位
        df_clean = df_clean.rename(columns=cls.COLUMN_MAPPING)

        # 新增股票代碼欄位
        df_clean['symbol'] = symbol

        # 轉換日期格式
        df_clean['date'] = pd.to_datetime(df_clean['date'])

        # 優化數值轉換 - 只處理實際存在的數值欄位
        existing_numeric_cols = [
            col for col in cls.NUMERIC_COLUMNS if col in df_clean.columns]
        if existing_numeric_cols:
            df_clean[existing_numeric_cols] = (
                df_clean[existing_numeric_cols].apply(
                    pd.to_numeric, errors='coerce'))

        # 移除完全空白的行
        df_clean = df_clean.dropna(subset=['date', 'symbol'])

        return df_clean

    @classmethod
    def validate_data_quality(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """驗證資料品質"""
        if df.empty:
            return {
                'total_rows': 0,
                'null_counts': {},
                'duplicate_dates': 0,
                'date_range': {'min': None, 'max': None}
            }

        quality_report = {
            'total_rows': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_dates': (df['date'].duplicated().sum()
                                if 'date' in df.columns else 0),
            'date_range': {
                'min': df['date'].min() if 'date' in df.columns else None,
                'max': df['date'].max() if 'date' in df.columns else None
            }
        }
        return quality_report

    @staticmethod
    def calculate_data_hash(df: pd.DataFrame) -> str:
        """計算數據框的雜湊值，用於快速比較"""
        if df.empty:
            return ""

        # 選擇關鍵欄位計算雜湊
        key_columns = ['date', 'close_price', 'volume']
        hash_data = df[key_columns].to_string()
        return hashlib.md5(hash_data.encode()).hexdigest()


class DataComparator:
    """數據比較工具類"""

    # 比較閾值配置
    COMPARISON_THRESHOLDS = {
        'price_columns': ['open_price', 'high_price',
                          'low_price', 'close_price'],
        'price_threshold': 0.0001,  # 0.01%
        'volume_threshold': 0,  # 完全精確比較
        'rsi_threshold': 0.01,
        'macd_threshold': 0.001,
        'kdj_threshold': 0.01,
        'ma_threshold': 0.0001,
        'bb_threshold': 0.0001,
        'atr_threshold': 0.001,
        'cci_willr_threshold': 0.1,
        'mom_threshold': 0.001,
        'default_threshold': 0.01
    }

    @classmethod
    def get_comparison_threshold(cls, column: str) -> float:
        """根據欄位類型獲取比較閾值"""
        if column in cls.COMPARISON_THRESHOLDS['price_columns']:
            return cls.COMPARISON_THRESHOLDS['price_threshold']
        elif column == 'volume':
            return cls.COMPARISON_THRESHOLDS['volume_threshold']
        elif column.startswith('rsi_'):
            return cls.COMPARISON_THRESHOLDS['rsi_threshold']
        elif column in ['dif', 'macd', 'macd_histogram']:
            return cls.COMPARISON_THRESHOLDS['macd_threshold']
        elif column in ['rsv', 'k_value', 'd_value', 'j_value']:
            return cls.COMPARISON_THRESHOLDS['kdj_threshold']
        elif column.startswith('ma') or column.startswith('ema'):
            return cls.COMPARISON_THRESHOLDS['ma_threshold']
        elif column.startswith('bb_'):
            return cls.COMPARISON_THRESHOLDS['bb_threshold']
        elif column == 'atr':
            return cls.COMPARISON_THRESHOLDS['atr_threshold']
        elif column in ['cci', 'willr']:
            return cls.COMPARISON_THRESHOLDS['cci_willr_threshold']
        elif column == 'mom':
            return cls.COMPARISON_THRESHOLDS['mom_threshold']
        else:
            return cls.COMPARISON_THRESHOLDS['default_threshold']

    @classmethod
    def compare_values(cls, file_val: Any, db_val: Any, column: str) -> bool:
        """比較兩個值是否有顯著差異"""
        # 處理空值
        file_val = file_val if pd.notna(file_val) else 0
        db_val = db_val if pd.notna(db_val) else 0

        # 特殊處理成交量（整數比較）
        if column == 'volume':
            file_val = int(file_val) if file_val != 0 else 0
            db_val = int(db_val) if db_val != 0 else 0

        threshold = cls.get_comparison_threshold(column)

        # 對於價格相關欄位，使用相對閾值
        if column in cls.COMPARISON_THRESHOLDS['price_columns']:
            threshold = max(
                abs(db_val * cls.COMPARISON_THRESHOLDS['price_threshold']),
                0.001)

        diff = abs(file_val - db_val)
        return diff > threshold


class StockDataImporter:
    """股票數據匯入器 - 支援智能更新模式"""

    def __init__(self, config_file: str = "config.ini"):
        # 讀取配置
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file, encoding='utf-8')

        # 初始化資料庫配置
        self.db_config = DatabaseConfig(config_file)

        # 讀取匯入設定
        self.batch_size = self.config_parser.getint(
            'import_settings', 'batch_size', fallback=30)
        log_level = self.config_parser.get(
            'import_settings', 'log_level', fallback='INFO')

        # 初始化
        self.engine = create_engine(
            self.db_config.get_sqlalchemy_url(),
            pool_pre_ping=True,
            pool_recycle=3600  # 1小時回收連線
        )
        self.logger = self._setup_logger(log_level)
        self.reporter = ProgressReporter(self.logger)

        # 讀取路徑配置
        self.output_dir = self.config_parser.get(
            'paths', 'output_directory', fallback='./output')

    def _setup_logger(self, log_level: str = 'INFO') -> logging.Logger:
        """設置日誌記錄器 - 僅輸出到檔案"""
        logger = logging.getLogger('StockDataImporter')

        # 清除現有的處理器避免重複
        logger.handlers = []

        # 設置日誌級別
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # 只設定檔案處理器，移除控制台處理器
        try:
            file_handler = logging.FileHandler(
                'stock_import.log', encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # 如果無法創建日誌檔案，使用標準錯誤輸出
            print(f"警告：無法創建日誌檔案，將使用控制台輸出: {e}", file=sys.stderr)

        return logger

    @contextmanager
    def get_connection(self):
        """獲取資料庫連接的上下文管理器"""
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

    def create_table_if_not_exists(self):
        """如果表不存在則創建表"""
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
            self.reporter.info("資料表和索引檢查/創建完成", log_only=True)
        except Exception as e:
            self.reporter.error(f"創建資料表失敗: {e}")
            raise

    def get_existing_data_info(self, symbol: str) -> Dict[str, Any]:
        """獲取已存在數據的資訊（優化版本）"""
        try:
            with self.get_connection() as conn:
                # 基本資訊查詢 + 快速雜湊檢查
                query = text("""
                SELECT
                    COUNT(*) as record_count,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    MAX(updated_at) as last_updated,
                    CHECKSUM_AGG(CHECKSUM(CONCAT(CAST(date AS VARCHAR),
                                                 CAST(close_price AS VARCHAR),
                                                 CAST(volume AS VARCHAR))))
                        as data_checksum
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
                        'last_updated': result[3],
                        'data_checksum': result[4] if result[4] else 0
                    }
                else:
                    return {'exists': False}

        except Exception as e:
            self.reporter.warning(f"獲取已存在數據資訊失敗: {e}", log_only=True)
            return {'exists': False}

    def quick_file_check(self, file_path: str, symbol: str) -> Dict[str, Any]:
        """快速檢查檔案是否需要處理（平衡效能與準確性）"""
        try:
            # 獲取資料庫資訊
            existing_info = self.get_existing_data_info(symbol)

            if not existing_info.get('exists', False):
                return {'needs_processing': True, 'reason': '新股票需要完整匯入'}

            # 讀取檔案的完整日期範圍
            try:
                df_dates = pd.read_csv(file_path, usecols=['Date'])
                if df_dates.empty:
                    return {'needs_processing': False, 'reason': '檔案為空'}

                df_dates['Date'] = pd.to_datetime(df_dates['Date'])
                file_earliest_date = df_dates['Date'].min().date()
                file_latest_date = df_dates['Date'].max().date()
                file_record_count = len(df_dates)

            except Exception as e:
                return {'needs_processing': True, 'reason': f'無法讀取檔案日期範圍: {e}'}

            # 獲取資料庫日期範圍
            db_earliest_date = existing_info['earliest_date']
            db_latest_date = existing_info['latest_date']
            db_record_count = existing_info.get('record_count', 0)

            if hasattr(db_earliest_date, 'date'):
                db_earliest_date = db_earliest_date.date()
            if hasattr(db_latest_date, 'date'):
                db_latest_date = db_latest_date.date()

            # 檢查是否有歷史數據需要匯入
            has_historical_data = file_earliest_date < db_earliest_date
            has_future_data = file_latest_date > db_latest_date

            # 如果有歷史數據或新數據，需要處理
            if has_historical_data or has_future_data:
                reasons = []
                if has_historical_data:
                    historical_count = len(
                        df_dates[df_dates['Date'].dt.date < db_earliest_date])
                    reasons.append(
                        f"{historical_count} 筆歷史數據 (檔案最早: "
                        f"{file_earliest_date}, 資料庫最早: {db_earliest_date})")

                if has_future_data:
                    future_count = len(
                        df_dates[df_dates['Date'].dt.date > db_latest_date])
                    reasons.append(
                        f"{future_count} 筆新數據 (檔案最新: {file_latest_date}, "
                        f"資料庫最新: {db_latest_date})")

                return {
                    'needs_processing': True,
                    'reason': '發現 ' + ' 和 '.join(reasons),
                    'existing_info': existing_info
                }

            # 檢查記錄數是否顯著不同
            if abs(file_record_count - db_record_count) > 10:  # 允許小幅差異
                return {
                    'needs_processing': True,
                    'reason': (f'記錄數差異過大 (檔案: {file_record_count}, '
                               f'資料庫: {db_record_count})'),
                    'existing_info': existing_info
                }

            # 檢查檔案修改時間（調整為更寬鬆的檢查）
            file_stat = os.stat(file_path)
            file_mtime = file_stat.st_mtime
            last_updated = existing_info.get('last_updated')

            if last_updated:
                import datetime
                if hasattr(last_updated, 'timestamp'):
                    db_timestamp = last_updated.timestamp()
                else:
                    db_timestamp = last_updated

                # 降低時間差閾值，讓檔案更容易被認為需要更新
                time_diff_hours = (file_mtime - db_timestamp) / 3600

                # 如果檔案比資料庫新10分鐘以上，就進行進一步檢查
                if time_diff_hours > 0.17:  # 10分鐘 = 0.17小時
                    return {
                        'needs_processing': True,
                        'reason': f'檔案修改時間較新 ({time_diff_hours:.1f}小時)，需要詳細檢查',
                        'existing_info': existing_info
                    }

            # 最後的保險檢查：如果日期範圍完全相同但今天是工作日，也檢查一下
            today = datetime.date.today()
            is_weekday = today.weekday() < 5  # 0-4 是週一到週五

            # 如果是工作日且檔案最新日期是今天或昨天，進行進一步檢查
            if is_weekday and (file_latest_date >= today
                               - datetime.timedelta(days=1)):
                return {
                    'needs_processing': True,
                    'reason': '工作日且檔案包含最新數據，進行詳細檢查',
                    'existing_info': existing_info
                }

            # 所有快速檢查都通過，可能不需要更新
            return {
                'needs_processing': False,
                'reason': (f'快速檢查：檔案可能無重大變化 (檔案: {file_earliest_date}'
                           f"~{file_latest_date}, 資料庫: "
                           f"{db_earliest_date}~{db_latest_date})"),
                'existing_info': existing_info
            }

        except Exception as e:
            # 如果快速檢查失敗，回退到完整處理以確保不遺漏數據
            return {'needs_processing': True, 'reason': f'快速檢查失敗，將進行完整處理: {e}'}

    def detect_data_changes_optimized(self, df_clean: pd.DataFrame,
                                      existing_info: Dict) -> Dict[str, Any]:
        """優化版數據變化檢測（減少資料庫查詢）"""
        if not existing_info.get('exists', False):
            return {
                'mode': 'insert_all',
                'data_to_process': df_clean,
                'reason': '股票不存在於資料庫'
            }

        # 獲取檔案和資料庫的日期範圍
        file_earliest_date = df_clean['date'].min().date()
        file_latest_date = df_clean['date'].max().date()
        db_earliest_date = existing_info['earliest_date']
        db_latest_date = existing_info['latest_date']

        if hasattr(db_earliest_date, 'date'):
            db_earliest_date = db_earliest_date.date()
        if hasattr(db_latest_date, 'date'):
            db_latest_date = db_latest_date.date()

        # 快速檢查：如果檔案日期範圍完全在資料庫範圍內，且記錄數相同
        if (file_earliest_date >= db_earliest_date and
            file_latest_date <= db_latest_date and
                len(df_clean) == existing_info.get('record_count', 0)):

            # 使用快速雜湊比較（如果支援）
            if 'data_checksum' in existing_info:
                file_key_data = df_clean[[
                    'date', 'close_price', 'volume']].copy()
                file_key_data['date'] = file_key_data['date'].dt.strftime(
                    '%Y-%m-%d')
                file_hash = hash(str(file_key_data.values.tolist()))

                # 簡單的雜湊比較（不是完全準確，但可以快速排除大部分不變的情況）
                file_hash_mod = abs(file_hash) % 1000000
                db_hash_mod = abs(existing_info['data_checksum']) % 1000000
                if file_hash_mod == db_hash_mod:
                    return {
                        'mode': 'skip',
                        'data_to_process': pd.DataFrame(),
                        'reason': '快速雜湊檢查：數據無變化'
                    }

        # 檢查是否有新的歷史數據或未來數據
        historical_data = df_clean[df_clean['date'].dt.date < db_earliest_date]
        future_data = df_clean[df_clean['date'].dt.date > db_latest_date]

        # 如果有歷史數據或未來數據需要插入
        if not historical_data.empty or not future_data.empty:
            new_data_frames = []
            reasons = []

            if not historical_data.empty:
                new_data_frames.append(historical_data)
                reasons.append(f"{len(historical_data)} 筆歷史數據")

            if not future_data.empty:
                new_data_frames.append(future_data)
                reasons.append(f"{len(future_data)} 筆新數據")

            if new_data_frames:
                combined_new_data = pd.concat(
                    new_data_frames, ignore_index=True)
                return {
                    'mode': 'incremental',
                    'data_to_process': combined_new_data,
                    'reason': "發現 " + " 和 ".join(reasons)
                }

        # 只對可能變化的期間進行詳細比較（限制範圍）
        overlap_start = max(file_earliest_date, db_earliest_date)
        overlap_end = min(file_latest_date, db_latest_date)

        if overlap_start <= overlap_end:
            # 只檢查最近30天的數據變化（減少比較範圍）
            import datetime
            recent_date = max(
                overlap_end - datetime.timedelta(days=30), overlap_start)

            overlap_file_data = df_clean[
                (df_clean['date'].dt.date >= recent_date) &
                (df_clean['date'].dt.date <= overlap_end)
            ]

            # 限制比較的行數
            if not overlap_file_data.empty and len(overlap_file_data) <= 100:
                changed_rows = self._compare_with_database(
                    overlap_file_data, df_clean['symbol'].iloc[0])

                if not changed_rows.empty:
                    return {
                        'mode': 'update_changed',
                        'data_to_process': changed_rows,
                        'reason': f'最近30天內檢測到 {len(changed_rows)} 筆數據有變化'
                    }

        return {
            'mode': 'skip',
            'data_to_process': pd.DataFrame(),
            'reason': '有限範圍檢查：數據無重大變化'
        }

    def _compare_with_database(self, df: pd.DataFrame,
                               stock_symbol: str) -> pd.DataFrame:
        """與資料庫數據進行詳細比較"""
        try:
            if df.empty:
                return pd.DataFrame()

            with self.get_connection() as conn:
                # 獲取資料庫中對應日期的數據
                dates = df['date'].dt.date.tolist()
                date_str = ','.join([f"'{date}'" for date in dates])

                query = text(f"""
                SELECT date, open_price, high_price, low_price, close_price,
                       volume, rsi_5, rsi_7, rsi_10, rsi_14, rsi_21,
                       dif, macd, macd_histogram,
                             rsv, k_value, d_value, j_value,
                       ma5, ma10, ma20, ma60, ema12, ema26,
                       bb_upper, bb_middle, bb_lower, atr, cci, willr, mom
                FROM stock_data
                WHERE symbol = :symbol AND date IN ({date_str})
                """)

                db_result = conn.execute(
                    query, {"symbol": stock_symbol}).fetchall()

                if not db_result:
                    return df  # 如果資料庫沒有數據，返回所有檔案數據

                # 轉換資料庫結果為字典
                db_data = {}
                for row in db_result:
                    date_key = row[0] if hasattr(row[0], 'date') else row[0]
                    if hasattr(date_key, 'date'):
                        date_key = date_key.date()

                    db_data[date_key] = {
                        col: (float(val) if val is not None else 0)
                        for col, val in zip(DataCleaner.NUMERIC_COLUMNS,
                                            row[1:])
                    }

                # 比較數據並找出變化
                changed_rows = []
                for _, row in df.iterrows():
                    date_key = row['date'].date()
                    if date_key in db_data:
                        db_row = db_data[date_key]

                        # 檢查是否有顯著差異
                        has_changes = any(
                            DataComparator.compare_values(
                                row.get(col, 0), db_row.get(col, 0), col)
                            for col in DataCleaner.NUMERIC_COLUMNS
                            if col in row.index
                        )

                        if has_changes:
                            changed_rows.append(row)

                return (pd.DataFrame(changed_rows) if changed_rows
                        else pd.DataFrame())

        except Exception as e:
            self.logger.warning(f"數據比較失敗: {e}")
            return df  # 比較失敗時返回所有數據以確保更新

    def import_single_file(self, file_path: str, symbol: Optional[str] = None,
                           update_mode: str = 'smart') -> ImportStats:
        """匯入單一 CSV 檔案（優化版本）"""
        start_time = time.time()
        result = ImportStats()

        try:
            # 提取股票代碼
            if symbol is None:
                symbol = os.path.basename(file_path).replace('.csv', '')

            filename = os.path.basename(file_path)
            self.reporter.progress(f"處理檔案 {filename} ({symbol})")

            # 智能模式下先進行快速檢查
            if update_mode == 'smart':
                quick_check = self.quick_file_check(file_path, symbol)

                # 如果快速檢查發現不需要處理，提早返回
                if not quick_check['needs_processing']:
                    result.success = True
                    result.mode_used = 'skip'
                    result.detection_reason = quick_check['reason']
                    result.elapsed_time = time.time() - start_time

                    # 仍需要讀取檔案來計算跳過的行數
                    try:
                        df_sample = pd.read_csv(file_path, nrows=1)
                        if not df_sample.empty:
                            # 快速計算總行數
                            with open(file_path, 'r', encoding='utf-8') as f:
                                result.total_rows = sum(
                                    1 for _ in f) - 1  # 減去標題行
                            result.skipped_rows = result.total_rows
                    except (pd.errors.EmptyDataError, pd.errors.ParserError,
                            FileNotFoundError, UnicodeDecodeError, OSError):
                        result.total_rows = 0
                        result.skipped_rows = 0

                    print("   📋 模式: 跳過處理 | 待處理: 0 筆")
                    print(f"   💡 原因: {result.detection_reason}")
                    self.reporter.success(f"{filename}: 快速檢查 - 數據無變化，跳過處理")
                    return result

            # 讀取 CSV（只有在需要時才讀取完整檔案）
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                result.error_message = f"讀取檔案失敗: {e}"
                self.reporter.error(f"讀取檔案 {filename} 失敗: {e}")
                return result

            result.total_rows = len(df)

            if df.empty:
                result.error_message = "檔案沒有數據"
                self.reporter.warning(f"檔案 {filename} 沒有數據")
                return result

            # 清理資料
            df_clean = DataCleaner.clean_dataframe(df, symbol)

            # 資料品質檢查
            result.quality_report = DataCleaner.validate_data_quality(df_clean)

            # 根據模式決定處理方式
            if update_mode == 'smart':
                # 如果快速檢查已經提供了資料庫資訊，使用它
                existing_info = quick_check.get('existing_info')
                if not existing_info:
                    existing_info = self.get_existing_data_info(symbol)

                # 使用優化的檢測方法
                change_detection = self.detect_data_changes_optimized(
                    df_clean, existing_info)
                actual_mode = change_detection['mode']
                data_to_process = change_detection['data_to_process']
                result.detection_reason = change_detection['reason']
            else:
                # 非智能模式，使用原有方法
                existing_info = self.get_existing_data_info(symbol)
                change_detection = self.detect_data_changes_optimized(
                    df_clean, existing_info)
                actual_mode = change_detection['mode']
                data_to_process = change_detection['data_to_process']
                result.detection_reason = f"使用指定模式: {update_mode}"

            result.mode_used = actual_mode

            # 顯示處理模式和原因
            mode_descriptions = {
                'insert_all': '新增插入',
                'incremental': '增量更新',
                'update_changed': '差異更新',
                'update_all': '完整更新',
                'skip': '跳過處理'
            }
            mode_desc = mode_descriptions.get(actual_mode, actual_mode)

            print(f"   📋 模式: {mode_desc} | 待處理: {len(data_to_process):,} 筆")
            print(f"   💡 原因: {result.detection_reason}")

            # 如果是跳過模式，直接返回
            if actual_mode == 'skip':
                result.success = True
                result.skipped_rows = len(df_clean)
                result.elapsed_time = time.time() - start_time
                self.reporter.success(f"{filename}: 數據無變化，跳過處理")
                return result

            # 執行匯入
            import_result = self._execute_import(
                data_to_process, symbol, actual_mode)

            result.imported_rows = import_result.get('imported_rows', 0)
            result.updated_rows = import_result.get('updated_rows', 0)
            result.skipped_rows = import_result.get('skipped_rows', 0)

            # 計算跳過的行數
            if actual_mode in ['incremental', 'update_changed']:
                result.skipped_rows += len(df_clean) - len(data_to_process)

            result.success = (result.imported_rows >
                              0 or result.updated_rows > 0)
            result.elapsed_time = time.time() - start_time

            # 顯示結果
            if result.success:
                summary = (f"新增 {result.imported_rows:,} | "
                           f"更新 {result.updated_rows:,} | "
                           f"跳過 {result.skipped_rows:,} | "
                           f"{result.elapsed_time:.2f}秒")
                self.reporter.success(f"{filename}: {summary}")
            else:
                self.reporter.warning(f"{filename}: 處理失敗")

        except Exception as e:
            result.error_message = str(e)
            result.elapsed_time = time.time() - start_time
            self.reporter.error(f"處理檔案 {filename} 失敗: {e}")

        return result

    def _execute_import(self, df: pd.DataFrame, symbol: str,
                        mode: str) -> Dict[str, int]:
        """執行實際的匯入操作"""
        if df.empty:
            return {'imported_rows': 0, 'updated_rows': 0, 'skipped_rows': 0}

        if mode == 'replace':
            return self._replace_all_data(df, symbol)
        elif mode in ['incremental', 'insert_all']:
            return self._incremental_update(df, symbol)
        elif mode in ['update_changed', 'update_all', 'update']:
            return self._update_all_data(df, symbol)
        else:  # insert mode
            return self._insert_new_data(df, symbol)

    def _replace_all_data(self, df: pd.DataFrame,
                          symbol: str) -> Dict[str, int]:
        """替換模式：刪除舊數據，插入新數據"""
        result = {'imported_rows': 0, 'updated_rows': 0, 'skipped_rows': 0}

        try:
            with self.get_connection() as conn:
                # 刪除舊數據
                delete_query = text(
                    "DELETE FROM stock_data WHERE symbol = :symbol")
                conn.execute(delete_query, {"symbol": symbol})

                # 插入新數據
                df_clean = df.where(pd.notnull(df), None)
                df_clean.to_sql(
                    name='stock_data',
                    con=self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

                conn.commit()
                result['imported_rows'] = len(df)
                self.logger.info(f"替換模式：刪除舊數據並插入 {len(df)} 筆新數據")

        except Exception as e:
            self.logger.error(f"替換模式失敗: {e}")
            raise

        return result

    def _incremental_update(self, df: pd.DataFrame,
                            symbol: str) -> Dict[str, int]:
        """增量更新模式：插入新數據"""
        result = {'imported_rows': 0, 'updated_rows': 0, 'skipped_rows': 0}

        try:
            df_clean = df.where(pd.notnull(df), None)
            df_clean.to_sql(
                name='stock_data',
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            result['imported_rows'] = len(df)

        except Exception as e:
            self.logger.error(f"增量更新失敗: {e}")
            # 如果批次插入失敗，改用 UPSERT
            result = self._upsert_batch_by_batch(df, symbol)

        return result

    def _update_all_data(self, df: pd.DataFrame,
                         symbol: str) -> Dict[str, int]:
        """更新模式：使用 MERGE 語句進行 UPSERT"""
        return self._upsert_batch_by_batch(df, symbol)

    def _insert_new_data(self, df: pd.DataFrame,
                         symbol: str) -> Dict[str, int]:
        """插入模式：直接插入新數據"""
        result = {'imported_rows': 0, 'updated_rows': 0, 'skipped_rows': 0}

        try:
            df_clean = df.where(pd.notnull(df), None)
            df_clean.to_sql(
                name='stock_data',
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            result['imported_rows'] = len(df)

        except Exception as e:
            self.logger.warning(f"批次插入失敗，改用 UPSERT 模式: {e}")
            result = self._upsert_batch_by_batch(df, symbol)

        return result

    def _upsert_batch_by_batch(self, df: pd.DataFrame) -> Dict[str, int]:
        """逐筆進行 UPSERT 操作"""
        result = {'imported_rows': 0, 'updated_rows': 0, 'skipped_rows': 0}

        # 準備 MERGE SQL
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
            INSERT (symbol, date, open_price, high_price, low_price,
                         close_price, volume,
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

        with self.get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    # 準備參數，將 NaN 轉換為 None
                    params = row.to_dict()
                    for key, value in params.items():
                        if pd.isna(value):
                            params[key] = None

                    merge_result = conn.execute(merge_sql, params)
                    action = merge_result.fetchone()

                    if action and action[0] == 'INSERT':
                        result['imported_rows'] += 1
                    elif action and action[0] == 'UPDATE':
                        result['updated_rows'] += 1

                except Exception as row_error:
                    self.logger.debug(
                        f"處理行失敗 {row['symbol']} {row['date']}: {row_error}")
                    result['skipped_rows'] += 1

            conn.commit()

        return result

    def import_directory(self, directory_path: Optional[str] = None,
                         update_mode: str = 'smart') -> Dict[str, ImportStats]:
        """批量匯入目錄中的所有 CSV 檔案"""
        if directory_path is None:
            directory_path = self.output_dir

        if not os.path.exists(directory_path):
            self.reporter.error(f"目錄不存在: {directory_path}")
            return {}

        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

        if not csv_files:
            self.reporter.warning("目錄中沒有找到 CSV 檔案")
            return {}

        self.reporter.header("股票數據匯入程序")
        self.reporter.info(f"目錄: {directory_path}")
        self.reporter.info(f"找到 {len(csv_files)} 個檔案，使用 {update_mode} 模式")

        # 確保資料表存在
        self.create_table_if_not_exists()

        self.reporter.separator()

        results = {}
        success_count = 0
        total_imported = 0
        total_updated = 0

        for i, file_path in enumerate(csv_files, 1):
            file_name = os.path.basename(file_path)
            print(f"\n📁 [{i}/{len(csv_files)}] {file_name}")

            result = self.import_single_file(
                file_path, update_mode=update_mode)
            results[file_name] = result

            if result.success:
                success_count += 1
                total_imported += result.imported_rows
                total_updated += result.updated_rows

        # 顯示總結
        self.reporter.separator()
        self.reporter.header("匯入結果摘要")

        print("\n📊 結果統計:")
        print(f"   ✅ 成功: {success_count} 個檔案")
        print(f"   ❌ 失敗: {len(csv_files) - success_count} 個檔案")
        print(f"   📈 總新增: {total_imported:,} 筆")
        print(f"   🔄 總更新: {total_updated:,} 筆")

        return results

    def get_import_statistics(self) -> Dict[str, Any]:
        """獲取匯入統計資訊"""
        try:
            with self.get_connection() as conn:
                stats_query = text("""
                SELECT COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    MAX(created_at) as last_import_time
                FROM stock_data
                """)

                result = conn.execute(stats_query).fetchone()

                symbols_query = text("""
                SELECT symbol, COUNT(*) as record_count,
                       MIN(date) as start_date, MAX(date) as end_date
                FROM stock_data
                GROUP BY symbol
                ORDER BY record_count DESC
                """)

                symbols_result = conn.execute(symbols_query).fetchall()

                return {
                    'total_records': result[0] if result else 0,
                    'unique_symbols': result[1] if result else 0,
                    'date_range': {
                        'earliest': result[2] if result else None,
                        'latest': result[3] if result else None
                    },
                    'last_import': result[4] if result else None,
                    'symbols': [
                        {
                            'symbol': row[0],
                            'records': row[1],
                            'start_date': row[2],
                            'end_date': row[3]
                        } for row in symbols_result
                    ]
                }

        except Exception as e:
            self.logger.error(f"獲取統計資訊失敗: {e}")
            return {}


def print_import_summary(results: Dict[str, ImportStats],
                         reporter: ProgressReporter):
    """打印匯入摘要"""
    reporter.separator()
    reporter.header("匯入結果摘要")

    successful_files = [f for f, r in results.items() if r.success]
    failed_files = [f for f, r in results.items() if not r.success]

    print("\n📊 結果統計:")
    print(f"   ✅ 成功: {len(successful_files)} 個檔案")
    print(f"   ❌ 失敗: {len(failed_files)} 個檔案")

    if successful_files:
        print("\n📈 成功處理的檔案:")
        total_new = sum(r.imported_rows for r in results.values())
        total_updated = sum(r.updated_rows for r in results.values())
        total_time = sum(r.elapsed_time for r in results.values())

        for filename in successful_files:
            result = results[filename]
            print(f"   • {filename}")
            print(
                f"     新增: {result.imported_rows:,} | 更新: "
                f"{result.updated_rows:,} | 跳過: {result.skipped_rows:,}")
            print(
                f"     模式: {result.mode_used} | 耗時: "
                f"{result.elapsed_time:.2f}秒")

        print("\n🎯 總計:")
        print(f"   新增資料: {total_new:,} 筆")
        print(f"   更新資料: {total_updated:,} 筆")
        print(f"   總耗時: {total_time:.2f} 秒")

    if failed_files:
        print("\n❌ 失敗的檔案:")
        for filename in failed_files:
            result = results[filename]
            print(f"   • {filename}: {result.error_message or '未知錯誤'}")


def print_database_statistics(stats: Dict[str, Any],
                              reporter: ProgressReporter):
    """打印資料庫統計資訊"""
    if not stats:
        return

    reporter.separator()
    reporter.header("資料庫統計資訊")

    print("\n📈 整體統計:")
    print(f"   總記錄數: {stats['total_records']:,}")
    print(f"   股票數量: {stats['unique_symbols']}")
    print(
        f"   日期範圍: {stats['date_range']['earliest']} ~ "
        f"{stats['date_range']['latest']}")
    print(f"   最後更新: {stats['last_import']}")

    if stats['symbols']:
        print("\n📋 各股票統計:")
        for symbol_info in stats['symbols'][:10]:
            print(f"   • {symbol_info['symbol']}: "
                  f"{symbol_info['records']:,} 筆 "
                  f"({symbol_info['start_date']} ~ {symbol_info['end_date']})")

        if len(stats['symbols']) > 10:
            print(f"   ... 還有 {len(stats['symbols']) - 10} 個股票")


def main():
    """主程式"""
    try:
        # 創建匯入器
        importer = StockDataImporter()

        # 測試連接
        if not importer.test_connection():
            importer.reporter.error("資料庫連接失敗，程式結束")
            return

        # 匯入數據
        results = importer.import_directory(update_mode='smart')

        # 顯示結果摘要
        print_import_summary(results, importer.reporter)

        # 顯示資料庫統計
        stats = importer.get_import_statistics()
        print_database_statistics(stats, importer.reporter)

        # 結束訊息
        importer.reporter.separator()
        importer.reporter.success("程序執行完成！")
        print("\n📝 詳細日誌請查看: stock_import.log")

    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
        logging.error(f"主程式錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()
