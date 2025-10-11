"""
股票技術分析系統 - 主程式
重新設計的模組化架構，支援數據更新檢查和技術指標計算
支援根據時間間隔動態創建不同的資料表
"""

import sys
import warnings
from datetime import datetime
import os
import re as _re

from services.stock_data_service import StockDataService
from providers.stock_data_provider import Period, TimeInterval
from utils.display_utils import (
    print_statistics, print_all_statistics, format_processing_summary)
from signals.analyzer import analyze_signals_from_db_with_symbol
from dotenv import load_dotenv

# 載入環境變數
env_local = '.env.local'
if os.path.exists(env_local):
    load_dotenv(env_local, override=True)
else:
    load_dotenv()

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
    pattern_only = False
    market_type = "tw"  # 預設台股
    run_signals = True  # 是否執行訊號分析
    signals_output = None  # 訊號分析輸出路徑

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
                show_all_stats, expand_history, pattern_only, market_type,
                run_signals, signals_output)

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

    if "--pattern" in args:
        pattern_only = True
        args.remove("--pattern")

    # 檢查訊號分析參數
    if "--signals" in args:
        run_signals = True
        args.remove("--signals")

    # 檢查訊號輸出路徑
    if "--signals-output" in args:
        idx = args.index("--signals-output")
        if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            signals_output = args[idx + 1]
            args.remove(args[idx + 1])
        args.remove("--signals-output")

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
        pattern_only,
        market_type,
        run_signals,
        signals_output
    )


def run_signal_analysis(
        stocks, interval_str, market_type, signals_output=None):
    """執行訊號分析

    Args:
        stocks: 股票代號列表
        interval_str: 時間間隔字串 (例如: '1d', '1h')
        market_type: 市場類型 (例如: 'tw', 'us')
        signals_output: 輸出路徑（可選）
    """
    print("\n" + "="*60, flush=True)
    print("📊 開始執行訊號分析", flush=True)
    print("="*60, flush=True)

    # 從環境變數讀取資料庫連線資訊
    server = os.getenv('MSSQL_SERVER')
    user = os.getenv('MSSQL_USER')
    password = os.getenv('MSSQL_PASSWORD')

    # 根據市場類型決定資料庫名稱
    market_db_map = {
        'tw': 'market_stock_tw',
        'us': 'market_stock_us',
        'etf': 'market_etf',
        'index': 'market_index',
        'forex': 'market_forex',
        'crypto': 'market_crypto',
        'futures': 'market_futures'
    }
    default_db = os.getenv('MSSQL_DATABASE', 'market_stock_tw')
    database = market_db_map.get(market_type, default_db)

    # 根據時間間隔決定資料表名稱
    table = f"stock_data_{interval_str}"

    print(f"🎯 分析股票: {', '.join(stocks)}", flush=True)
    print(f"📊 資料庫: {database}", flush=True)
    print(f"📋 資料表: {table}", flush=True)

    # 對每個股票執行訊號分析
    multiple = len(stocks) > 1
    for i, symbol in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] 分析 {symbol}...", flush=True)

        # 決定輸出路徑
        out_path = None
        if signals_output:
            is_default = signals_output == '__DEFAULT_OUTPUT__'
            is_dir = os.path.isdir(signals_output)
            if is_default or is_dir:
                # 預設輸出到 output 目錄
                if is_default:
                    out_dir = os.path.join(os.getcwd(), 'output')
                else:
                    out_dir = signals_output
                os.makedirs(out_dir, exist_ok=True)
                safe_symbol = _re.sub(r'[^0-9A-Za-z]+', '_', symbol)
                safe_table = _re.sub(r'[^0-9A-Za-z]+', '_', table)
                filename = f"signals_{safe_symbol}_{safe_table}.csv"
                out_path = os.path.join(out_dir, filename)
            elif multiple:
                # 多個股票且指定了檔案路徑，加入股票代號
                base, ext = os.path.splitext(signals_output)
                ext = ext if ext else '.csv'
                out_path = f"{base}_{symbol}{ext}"
            else:
                # 單一股票直接使用指定路徑
                out_path = signals_output

        try:
            analyze_signals_from_db_with_symbol(
                server=server,
                database=database,
                table=table,
                user=user,
                password=password,
                output_path=out_path,
                symbol=symbol
            )
            print(f"✅ {symbol} 訊號分析完成", flush=True)
            if out_path:
                print(f"📁 輸出檔案: {out_path}", flush=True)
        except Exception as e:
            print(f"❌ {symbol} 訊號分析失敗: {e}", flush=True)

    print("\n" + "="*60, flush=True)
    print("✅ 所有訊號分析完成", flush=True)
    print("="*60 + "\n", flush=True)


def main():
    """主程式"""
    try:
        # 解析命令行參數
        (stocks, interval, indicators_only, show_all_stats,
         expand_history, pattern_only, market_type,
         run_signals, signals_output) = parse_arguments()

        # 創建服務實例
        service = StockDataService()

        # 測試資料庫連接
        if not service.test_connection(market_type=market_type):
            print("❌ 資料庫連接失敗，程式結束", flush=True)
            return

        interval_str = interval.value

        print("🚀 股票技術分析系統 - 模組化版本", flush=True)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"⏰ 時間間隔: {interval_str}", flush=True)
        print(f"🌏 市場類型: {market_type}", flush=True)

        # 顯示所有資料表統計資訊模式
        if show_all_stats:
            print("📊 顯示所有資料表統計資訊模式", flush=True)
            all_stats = service.get_all_database_statistics(
                market_type=market_type)
            print_all_statistics(all_stats)
            return

        # 僅更新技術指標模式
        if indicators_only:
            print("🔄 技術指標更新模式 - 完整歷史數據", flush=True)
            print(f"🎯 目標股票: {', '.join(stocks)}", flush=True)
            print("📋 處理流程:", flush=True)
            print("   1️⃣  獲取每個股票的所有歷史OHLCV數據", flush=True)
            print("   2️⃣  重新計算所有技術指標", flush=True)
            print("   3️⃣  更新資料庫中的技術指標欄位", flush=True)

            results = service.force_update_all_indicators(
                stocks,
                interval_str,
                full_history=True,
                market_type=market_type
            )

            total_updated = sum(results.values())
            success_count = sum(1 for count in results.values() if count > 0)

            print(f"\n📊 技術指標更新完成 ({interval_str})", flush=True)
            print(f"✅ 成功更新: {success_count}/{len(results)} 個股票", flush=True)
            print(f"📈 總更新筆數: {total_updated:,} 筆", flush=True)
            print("📝 已重新計算所有歷史數據的技術指標", flush=True)

            # 如果啟用訊號分析，執行訊號分析
            if run_signals:
                run_signal_analysis(stocks, interval_str,
                                    market_type, signals_output)
            return

        # 歷史數據擴展模式
        if expand_history:
            print("🔄 歷史數據擴展模式", flush=True)
            print(f"🎯 目標股票: {', '.join(stocks)}", flush=True)
            print("📋 處理流程:", flush=True)
            print("   1️⃣  檢查資料庫現有數據範圍", flush=True)
            print("   2️⃣  獲取完整歷史數據", flush=True)
            print("   3️⃣  比對並新增更早的歷史數據", flush=True)
            print("   4️⃣  重新計算技術指標", flush=True)

            results = service.process_multiple_stocks(
                symbols=stocks,
                period=Period.MAX,
                interval=interval,
                expand_history=True,
                market_type=market_type
            )

            print(format_processing_summary(results), flush=True)
            stats = service.get_database_statistics(
                interval_str, market_type=market_type)
            print_statistics(stats)
            print(f"\n✅ 歷史數據擴展完成！(間隔: {interval_str})", flush=True)
            print("📝 詳細日誌請查看: stock_analyzer.log", flush=True)

            # 如果啟用訊號分析，執行訊號分析
            if run_signals:
                run_signal_analysis(stocks, interval_str,
                                    market_type, signals_output)
            return

        # 型態偵測模式
        if pattern_only:
            print("🔍 K 線型態偵測模式 - 完整歷史數據", flush=True)
            print(f"🎯 目標股票: {', '.join(stocks)}", flush=True)
            print("📋 處理流程:", flush=True)
            print("   1️⃣  獲取所有歷史 OHLCV 數據", flush=True)
            print("   2️⃣  進行完整 K 線型態偵測", flush=True)
            print("   3️⃣  更新資料庫中的型態訊號欄位", flush=True)

            results = service.update_pattern_signals_for_stocks(
                stocks,
                interval_str,
                market_type=market_type,
                recent_only=False
            )

            total_updated = sum(results.values())
            success_count = sum(1 for count in results.values() if count > 0)

            print(f"\n📊 K 線型態偵測完成 ({interval_str})", flush=True)
            print(f"✅ 成功更新: {success_count}/{len(results)} 個股票", flush=True)
            print(f"📈 總更新筆數: {total_updated:,} 筆", flush=True)
            print("📝 已完成所有歷史數據的 K 線型態分析", flush=True)

            # 如果啟用訊號分析，執行訊號分析
            if run_signals:
                run_signal_analysis(stocks, interval_str,
                                    market_type, signals_output)
            return

        # 正常處理模式
        print(f"🎯 目標股票: {', '.join(stocks)}", flush=True)
        print("📋 處理流程:", flush=True)
        print("   1️⃣  檢查對應間隔表的現有數據", flush=True)
        print("   2️⃣  比對外部數據差異", flush=True)
        print("   3️⃣  更新 OHLCV 數據", flush=True)
        print("   4️⃣  計算技術指標", flush=True)
        print("   5️⃣  進行 K 線型態偵測", flush=True)
        print("   6️⃣  進行訊號分析", flush=True)

        results = service.process_multiple_stocks(
            symbols=stocks,
            period=Period.YEAR_1,
            interval=interval,
            check_days=30,
            expand_history=False,
            market_type=market_type
        )

        print(format_processing_summary(results), flush=True)
        stats = service.get_database_statistics(
            interval_str, market_type=market_type)
        print_statistics(stats)
        print(f"\n✅ 程式執行完成！(間隔: {interval_str})", flush=True)
        print("📝 詳細日誌請查看: stock_analyzer.log", flush=True)

        # 如果啟用訊號分析，執行訊號分析
        if run_signals:
            run_signal_analysis(stocks, interval_str,
                                market_type, signals_output)

        return results

    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}", flush=True)
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
  --pattern             僅更新K線型態訊號（完整歷史數據）
  --show-all-stats      顯示所有資料表的統計資訊
  --expand-history      擴展歷史數據模式（獲取比資料庫更早的數據）
  --signals             執行訊號分析（在主要工作完成後）
  --signals-output PATH 指定訊號分析輸出路徑（可選）
  --help                顯示此幫助資訊

使用範例:
  python main.py                    # 使用預設股票和日線數據 (台股)
  python main.py --us AAPL          # 查詢美股AAPL
  python main.py --tw 2330 2317    # 查詢台股2330、2317
  python main.py --etf 0050         # 查詢台灣ETF 0050
  python main.py --1h --us TSLA     # 查詢美股TSLA 1小時線
  python main.py --indicators-only --us AAPL  # 僅更新美股AAPL技術指標
  python main.py --pattern --tw 2330  # 更新台股2330完整歷史型態訊號
  python main.py --show-all-stats --us   # 顯示美股所有資料表統計
  python main.py --expand-history --us AAPL  # 擴展美股AAPL歷史數據
  python main.py --signals --tw 2330 2317  # 更新數據後執行訊號分析
  python main.py --signals --signals-output ./output/ --us AAPL  # 指定輸出路徑

📊 歷史數據擴展功能:
  --expand-history 選項會：
  1. 檢查資料庫中現有的數據範圍
  2. 獲取所有可用的歷史數據 (使用 max 期間)
  3. 自動新增比資料庫更早的歷史數據
  4. 重新計算所有技術指標

  適用情況：
  - 之前只存了最近一年的數據，現在想要完整歷史數據
  - 新增股票需要完整歷史數據
  - 資料庫數據不完整需要補充

🔍 K 線型態偵測功能:
  --pattern 選項：
  1. 獲取所有歷史 OHLCV 數據
  2. 使用TA-Lib進行完整 K 線型態偵測
  3. 將型態訊號存入 pattern_signals 欄位

📊 正常模式（不指定特殊選項）會同時：
  1. 更新 OHLCV 數據
  2. 計算技術指標
  3. 進行 K 線型態偵測
  4. 將所有結果存入資料庫

🔔 訊號分析功能:
  --signals 選項會在主要工作完成後：
  1. 從資料庫讀取更新後的 OHLCV 數據
  2. 計算各種技術指標訊號
  3. 生成交易訊號
  4. 將訊號儲存到 trade_signals_* 資料表
  5. 可選擇性輸出到 CSV 檔案

  可搭配所有模式使用：
  - 正常模式 + --signals：更新數據並分析訊號
  - --indicators-only --signals：更新指標後分析訊號
  - --pattern --signals：更新型態後分析訊號
"""
    print(help_text, flush=True)


if __name__ == "__main__":
    # 檢查是否為幫助模式
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
    else:
        main()
