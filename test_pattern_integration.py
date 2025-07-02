"""
測試 pattern_detection 整合功能
"""

import sys
import os
from services.stock_data_service import StockDataService

# 添加項目根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_pattern_integration():
    """測試 pattern detection 整合"""
    print("🧪 測試 K線型態偵測整合功能")

    # 創建服務實例
    service = StockDataService()

    # 測試資料庫連接
    if not service.test_connection():
        print("❌ 資料庫連接失敗")
        return False

    print("✅ 資料庫連接成功")

    # 測試更新方法
    try:
        results = service.update_pattern_signals_for_stocks(
            ["2330"], "1d", "tw"
        )
        print(f"✅ 型態訊號更新測試完成: {results}")
    except Exception as e:
        print(f"❌ 型態訊號更新測試失敗: {e}")
        return False

    print("🎉 所有測試通過！")
    return True


if __name__ == "__main__":
    test_pattern_integration()
