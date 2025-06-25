"""
資料庫配置模組
處理資料庫連線配置和連線管理
"""

import configparser
from sqlalchemy import create_engine
from contextlib import contextmanager
import logging


class DatabaseConfig:
    """資料庫配置類"""

    def __init__(self, config_file: str = "config.ini", database: str = None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        self.server = self.config.get('database', 'server')
        self.database = self.config.get('database', 'database')
        self.driver = self.config.get('database', 'driver')

        if database:
            self.database = database

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
