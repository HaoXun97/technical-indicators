"""
資料庫配置模組
處理資料庫連線配置和連線管理
"""

import configparser
import os
from sqlalchemy import create_engine
from contextlib import contextmanager
import logging
import urllib.parse
from dotenv import load_dotenv


class DatabaseConfig:
    """資料庫配置類"""

    def __init__(self, config_file: str = "config.ini", database: str = None):
        # 載入 .env 檔案
        load_dotenv()

        # 讀取配置檔案
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        # 從配置檔案讀取基本設定
        self.server = self.config.get('database', 'server')
        self.database = self.config.get('database', 'database')
        self.driver = self.config.get('database', 'driver')

        if database:
            self.database = database

        # 優先從環境變數讀取帳號密碼，如果沒有則從配置檔案讀取
        env_username = os.getenv('db_username')
        env_password = os.getenv('db_password')
        env_use_windows_auth = os.getenv('use_windows_auth', 'false').lower()

        config_username = self.config.get('database', 'username',
                                          fallback=None)
        config_password = self.config.get('database', 'password',
                                          fallback=None)
        config_use_windows_auth = self.config.get('database',
                                                  'use_windows_auth',
                                                  fallback='false').lower()

        self.username = env_username or config_username
        self.password = env_password or config_password
        self.use_windows_auth = (env_use_windows_auth == 'true' or
                                 config_use_windows_auth == 'true')

        # 如果明確設定使用 Windows 驗證，則忽略帳號密碼
        if self.use_windows_auth:
            self.username = None
            self.password = None
        # 確保 username 和 password 不是空字串
        elif self.username == '':
            self.username = None
        elif self.password == '':
            self.password = None

    def get_sqlalchemy_url(self) -> str:
        """生成 SQLAlchemy 連接字串"""
        if self.use_windows_auth:
            # 明確設定使用 Windows 驗證
            return (f"mssql+pyodbc://@{self.server}/{self.database}"
                    f"?driver={self.driver}&trusted_connection=yes")
        elif self.username and self.password:
            # 有提供完整的帳號密碼，使用 SQL Server 驗證
            username = urllib.parse.quote_plus(self.username)
            password = urllib.parse.quote_plus(self.password)
            return (f"mssql+pyodbc://{username}:{password}@"
                    f"{self.server}/{self.database}?driver={self.driver}")
        else:
            # 沒有完整的帳號密碼，回退到 Windows 驗證
            return (f"mssql+pyodbc://@{self.server}/{self.database}"
                    f"?driver={self.driver}&trusted_connection=yes")

    def debug_config(self):
        """除錯：顯示目前的配置值"""
        print(f"Debug - Server: {self.server}")
        print(f"Debug - Database: {self.database}")
        print(f"Debug - Driver: {self.driver}")
        print(f"Debug - Username: {repr(self.username)}")
        print(f"Debug - Password: {repr(self.password)}")
        print(f"Debug - Use Windows Auth: {self.use_windows_auth}")
        print(f"Debug - Connection URL: {self.get_sqlalchemy_url()}")


class DatabaseManager:
    """資料庫管理器"""

    def __init__(self, config_file: str = "config.ini", database: str = None):
        self.db_config = DatabaseConfig(config_file, database=database)
        self.logger = logging.getLogger(__name__)

        # 初始化資料庫連接
        self.engine = create_engine(
            self.db_config.get_sqlalchemy_url(),
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

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
            from sqlalchemy import text
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("資料庫連接測試成功")
            return True
        except Exception as e:
            self.logger.error(f"資料庫連接測試失敗: {e}")
            return False
