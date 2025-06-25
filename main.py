"""
股票技術分析系統 - 主程式
重新設計的模組化架構，支援數據更新檢查和技術指標計算
支援根據時間間隔動態創建不同的資料表
"""

import sys
import warnings
from datetime import datetime

from services.stock_data_service import StockDataService
from providers.stock_data_provider import Period, TimeInterval
from utils.display_utils import (
    print_statistics, print_all_statistics, format_processing_summary)

# 抑制警告
warnings.filterwarnings("ignore")


def parse_arguments():
    """解析命令行參數"""
    args = sys.argv[1:]

    # 預設值
    interval = TimeInterval.DAY_1
    stocks = ["2330", "2317"]
    indicators_only = False
    show_all_stats = False
    expand_history = False
    market_type = "tw"  # 預設台股

    # 支援的市場參數
    market_type_map = {
        "--tw": "tw",
        "--us": "us",
        "--etf": "etf",
        "--index": "index",
        "--forex": "forex",
        "--crypto": "crypto",
        "--futures": "futures"
    }

    if not args:
        return (stocks, interval, indicators_only,
                show_all_stats, expand_history, market_type)

    # 檢查特殊模式
    if "--indicators-only" in args:
        indicators_only = True
        args.remove("--indicators-only")

    if "--show-all-stats" in args:
        show_all_stats = True
        args.remove("--show-all-stats")

    if "--expand-history" in args:
        expand_history = True
        args.remove("--expand-history")

    # 檢查市場參數
    for k in list(market_type_map.keys()):
        if k in args:
            market_type = market_type_map[k]
            args.remove(k)
            break

    # 檢查間隔參數
    interval_map = {
        "--1m": TimeInterval.MINUTE_1,
        "--5m": TimeInterval.MINUTE_5,
        "--15m": TimeInterval.MINUTE_15,
        "--30m": TimeInterval.MINUTE_30,
        "--1h": TimeInterval.HOUR_1,
        "--1d": TimeInterval.DAY_1,
        "--1wk": TimeInterval.WEEK_1,
        "--1mo": TimeInterval.MONTH_1,
    }

    for arg in list(args):
        if arg in interval_map:
            interval = interval_map[arg]
            args.remove(arg)
            break

    # 剩餘的參數視為股票代號
    if args:
        stocks = args

    return (
        stocks,
        interval,
        indicators_only,
        show_all_stats,
        expand_history,
        market_type
    )


def main():
    """主程式"""
    try:
        # 解析命令行參數
        (stocks, interval, indicators_only,
         show_all_stats, expand_history, market_type) = parse_arguments()

        # 創建服務實例
        service = StockDataService()

        # 測試資料庫連接
        if not service.test_connection(market_type=market_type):
            print("❌ 資料庫連接失敗，程式結束")
            return

        interval_str = interval.value

        print("🚀 股票技術分析系統 - 模組化版本")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ 時間間隔: {interval_str}")
        print(f"🌏 市場類型: {market_type}")

        # 顯示所有資料表統計資訊模式
        if show_all_stats:
            print("📊 顯示所有資料表統計資訊模式")
            all_stats = service.get_all_database_statistics(
                market_type=market_type)
            print_all_statistics(all_stats)
            return

        # 僅更新技術指標模式
        if indicators_only:
            print("🔄 技術指標更新模式 - 完整歷史數據")
            print(f"🎯 目標股票: {', '.join(stocks)}")
            print("📋 處理流程:")
            print("   1️⃣  獲取每個股票的所有歷史OHLCV數據")
            print("   2️⃣  重新計算所有技術指標")
            print("   3️⃣  更新資料庫中的技術指標欄位")

            results = service.force_update_all_indicators(
                stocks,
                interval_str,
                full_history=True,
                market_type=market_type
            )

            total_updated = sum(results.values())
            success_count = sum(1 for count in results.values() if count > 0)

            print(f"\n📊 技術指標更新完成 ({interval_str})")
            print(f"✅ 成功更新: {success_count}/{len(results)} 個股票")
            print(f"📈 總更新筆數: {total_updated:,} 筆")
            print("📝 已重新計算所有歷史數據的技術指標")
            return

        # 歷史數據擴展模式
        if expand_history:
            print("🔄 歷史數據擴展模式")
            print(f"🎯 目標股票: {', '.join(stocks)}")
            print("📋 處理流程:")
            print("   1️⃣  檢查資料庫現有數據範圍")
            print("   2️⃣  獲取完整歷史數據")
            print("   3️⃣  比對並新增更早的歷史數據")
            print("   4️⃣  重新計算技術指標")

            results = service.process_multiple_stocks(
                symbols=stocks,
                period=Period.MAX,
                interval=interval,
                expand_history=True,
                market_type=market_type
            )

            print(format_processing_summary(results))
            stats = service.get_database_statistics(
                interval_str, market_type=market_type)
            print_statistics(stats)
            print(f"\n✅ 歷史數據擴展完成！(間隔: {interval_str})")
            print("📝 詳細日誌請查看: stock_analyzer.log")
            return

        # 正常處理模式
        print(f"🎯 目標股票: {', '.join(stocks)}")
        print("📋 處理流程:")
        print("   1️⃣  檢查對應間隔表的現有數據")
        print("   2️⃣  比對外部數據差異")
        print("   3️⃣  更新OHLCV數據")
        print("   4️⃣  重新計算技術指標")

        results = service.process_multiple_stocks(
            symbols=stocks,
            period=Period.YEAR_1,
            interval=interval,
            check_days=30,
            expand_history=False,
            market_type=market_type
        )

        print(format_processing_summary(results))
        stats = service.get_database_statistics(
            interval_str, market_type=market_type)
        print_statistics(stats)
        print(f"\n✅ 程式執行完成！(間隔: {interval_str})")
        print("📝 詳細日誌請查看: stock_analyzer.log")
        return results

    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
        import logging
        logging.error(f"主程式錯誤: {e}", exc_info=True)


def show_help():
    """顯示幫助資訊"""
    help_text = """
🚀 股票技術分析系統 - 使用說明

基本用法:
  python main.py [市場選項] [時間間隔選項] [功能選項] [股票代號...]

市場選項:
  --tw        台股 (預設)
  --us        美股
  --etf       ETF
  --index     指數
  --forex     外匯
  --crypto    加密貨幣
  --futures   期貨

時間間隔選項:
  --1m    1分鐘數據 (存入 stock_data_1m)
  --5m    5分鐘數據 (存入 stock_data_5m)
  --15m   15分鐘數據 (存入 stock_data_15m)
  --30m   30分鐘數據 (存入 stock_data_30m)
  --1h    1小時數據 (存入 stock_data_1h)
  --1d    日線數據 (存入 stock_data_1d) [預設]
  --1wk   週線數據 (存入 stock_data_1wk)
  --1mo   月線數據 (存入 stock_data_1mo)

功能選項:
  --indicators-only     僅更新技術指標，不檢查OHLCV數據
  --show-all-stats      顯示所有資料表的統計資訊
  --expand-history      擴展歷史數據模式（獲取比資料庫更早的數據）
  --help                顯示此幫助資訊

使用範例:
  python main.py                    # 使用預設股票和日線數據 (台股)
  python main.py --us AAPL          # 查詢美股AAPL
  python main.py --tw 2330 2317    # 查詢台股2330、2317
  python main.py --etf 0050         # 查詢台灣ETF 0050
  python main.py --1h --us TSLA     # 查詢美股TSLA 1小時線
  python main.py --indicators-only --us AAPL  # 僅更新美股AAPL技術指標
  python main.py --show-all-stats --us   # 顯示美股所有資料表統計
  python main.py --expand-history --us AAPL  # 擴展美股AAPL歷史數據

📊 歷史數據擴展功能:
  --expand-history 選項會：
  1. 檢查資料庫中現有的數據範圍
  2. 獲取所有可用的歷史數據 (使用max期間)
  3. 自動新增比資料庫更早的歷史數據
  4. 重新計算所有技術指標

  適用情況：
  - 之前只存了最近一年的數據，現在想要完整歷史數據
  - 新增股票需要完整歷史數據
  - 資料庫數據不完整需要補充

📊 資料表對應:
  每個時間間隔會自動創建對應的資料表，如：
  - 1分鐘數據 → stock_data_1m
  - 5分鐘數據 → stock_data_5m
  - 1小時數據 → stock_data_1h
  - 日線數據 → stock_data_1d
"""
    print(help_text)


if __name__ == "__main__":
    # 檢查是否為幫助模式
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
    else:
        main()
