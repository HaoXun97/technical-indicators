# -*- coding: utf-8 -*-
"""主分析流程：讀取資料、計算指標、產生訊號、儲存與報告"""

import os
import sys
import time
import pandas as pd

# 當以腳本直接執行時，確保專案根目錄在 sys.path，讓子資料夾 `signals` 可被絕對匯入
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from signals import db as dbmod
    from signals import indicators as ind
    from signals import trades as tradesmod
except Exception:
    # 最後備援：嘗試相對匯入（若此模組被作為 package 匯入）
    from .signals import db as dbmod
    from .signals import indicators as ind
    from .signals import trades as tradesmod


def analyze_signals_from_db(
    server, database, table, user, password, output_path=None
):
    total_start_time = time.time()

    print("開始從資料庫讀取資料...", flush=True)
    read_start = time.time()
    df = dbmod.read_ohlcv_from_mssql(server, database, table, user, password)
    read_time = time.time() - read_start
    print(f"讀取完成，共 {len(df)} 筆資料，耗時 {read_time:.2f} 秒", flush=True)

    print("開始計算技術指標...", flush=True)
    calc_start = time.time()

    df = df.sort_values('datetime').reset_index(drop=True)

    df = ind.ma_cross_signal(df)
    df = ind.bollinger_signal(df)
    df = ind.macd_signal(df)
    df = ind.trend_signal(df)
    df = ind.macd_divergence(df)
    df = ind.anomaly_detection(df)
    df = ind.rsi_signal(df)
    df = ind.kd_signal(df)
    df = ind.support_resistance_signal(df)
    df = ind.volume_anomaly_signal(df)
    df = ind.ema_cross_signal(df)
    df = ind.cci_signal(df)
    df = ind.willr_signal(df)
    df = ind.momentum_signal(df)

    calc_time = time.time() - calc_start
    print(f"指標計算完成，耗時 {calc_time:.2f} 秒", flush=True)

    signal_start = time.time()
    df = tradesmod.generate_trade_signals(df)
    signal_time = time.time() - signal_start
    print(f"訊號生成完成，耗時 {signal_time:.2f} 秒", flush=True)

    # 根據輸入的資料表名稱決定要儲存到哪個 trade_signals 表
    def _signals_table_for_data_table(data_table: str) -> str:
        # 支援 schema.table 或 [schema].[table] 等格式
        if not data_table:
            return 'trade_signals'
        base = data_table.split('.')[-1]
        base = base.replace('[', '').replace(']', '')
        if '_' in base:
            suffix = base.split('_')[-1]
            return f'trade_signals_{suffix}'
        return 'trade_signals'

    signals_table = _signals_table_for_data_table(table)
    print(f"開始儲存結果到資料庫（目標表：{signals_table}）...", flush=True)
    save_start = time.time()
    dbmod.save_signals_to_mssql(
        df, server, database, user, password, table_name=signals_table
    )
    save_time = time.time() - save_start
    print(f"資料庫儲存完成，耗時 {save_time:.2f} 秒", flush=True)

    if output_path:
        csv_start = time.time()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        csv_time = time.time() - csv_start
        print(f"CSV檔案儲存完成，耗時 {csv_time:.2f} 秒，路徑: {output_path}", flush=True)

    total_time = time.time() - total_start_time
    print(f"\n總執行時間: {total_time:.2f} 秒", flush=True)
    print(
        f"- 資料讀取: {read_time:.2f}秒 ({read_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 指標計算: {calc_time:.2f}秒 ({calc_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 訊號生成: {signal_time:.2f}秒 ({signal_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 資料儲存: {save_time:.2f}秒 ({save_time/total_time*100:.1f}%)",
        flush=True)

    tradesmod.print_analysis_summary(df)


def _print_db_connection_help(err, server, database, user):
    """輔助函式：顯示資料庫連線錯誤訊息"""
    print("\n[錯誤] 無法連線到 MSSQL 資料庫", flush=True)
    print("\n--- 原始錯誤 ---", flush=True)
    try:
        # 若 err 為 pyodbc 的錯誤物件，通常 .args 含有詳細資訊
        print(repr(err), flush=True)
    except Exception:
        print(str(err), flush=True)
    print("----------------------------------------\n", flush=True)


def _print_table_error_help(err, table, server, database):
    """輔助函式：顯示資料表相關錯誤訊息"""
    print("\n[錯誤] 找不到指定的資料表或物件。", flush=True)
    print("\n--- 原始錯誤（供技術人員參考） ---", flush=True)
    try:
        print(repr(err), flush=True)
    except Exception:
        print(str(err), flush=True)
    print("----------------------------------------\n", flush=True)


def analyze_signals_from_db_with_symbol(
    server,
    database,
    table,
    user,
    password,
    output_path=None,
    symbol=None,
):
    total_start_time = time.time()

    print(f"開始分析 symbol={symbol}" if symbol else "開始分析全部資料", flush=True)

    if symbol and symbol != 'Unknown':
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={user};PWD={password}"
        )
        import pyodbc
        try:
            conn = pyodbc.connect(conn_str)
        except Exception as e:
            _print_db_connection_help(e, server, database, user)
            return
        with conn:
            check_query = f"SELECT COUNT(*) FROM {table} WHERE symbol = ?"
            cursor = conn.cursor()
            try:
                count = cursor.execute(check_query, symbol).fetchval()
            except Exception as e:
                # 可能是資料表不存在或 SQL 語法/物件錯誤
                _print_table_error_help(e, table, server, database)
                return

            if count == 0:
                print(f"找不到 symbol={symbol} 的資料，程式結束。", flush=True)
                return

            print(f"找到 {count} 筆 {symbol} 的資料，開始讀取...", flush=True)
            query = f"SELECT * FROM {table} WHERE symbol = ? "
            query += "ORDER BY datetime"
            read_start = time.time()
            try:
                df = pd.read_sql(query, conn, params=[symbol])
            except Exception as e:
                _print_table_error_help(e, table, server, database)
                return
            read_time = time.time() - read_start
            print(f"讀取完成，耗時 {read_time:.2f} 秒", flush=True)
    else:
        read_start = time.time()
        df = dbmod.read_ohlcv_from_mssql(
            server, database, table, user, password
        )
        read_time = time.time() - read_start

    if df.empty:
        print("沒有資料可分析，程式結束。", flush=True)
        return

    print("開始計算技術指標...", flush=True)
    calc_start = time.time()

    signals = []

    signals = []

    df = ind.ma_cross_signal(df)
    signals.append("MA交叉")

    df = ind.bollinger_signal(df)
    signals.append("布林通道")

    df = ind.macd_signal(df)
    signals.append("MACD交叉")

    df = ind.trend_signal(df)
    signals.append("趨勢判斷")

    df = ind.macd_divergence(df)
    signals.append("MACD背離")

    df = ind.anomaly_detection(df)
    signals.append("異常偵測")

    df = ind.rsi_signal(df)
    signals.append("RSI訊號")

    df = ind.kd_signal(df)
    signals.append("KD訊號")

    df = ind.support_resistance_signal(df)
    signals.append("壓力支撐位")

    df = ind.volume_anomaly_signal(df)
    signals.append("成交量異常")

    df = ind.ema_cross_signal(df)
    signals.append("EMA交叉")

    df = ind.cci_signal(df)
    signals.append("CCI訊號")

    df = ind.willr_signal(df)
    signals.append("威廉指標")

    df = ind.momentum_signal(df)
    signals.append("動量指標")

    calc_time = time.time() - calc_start
    print(f"指標計算完成，耗時 {calc_time:.2f} 秒，共計算 {len(signals)} 個指標", flush=True)

    signal_start = time.time()
    df = tradesmod.generate_trade_signals(df)
    signal_time = time.time() - signal_start
    print(f"訊號生成完成，耗時 {signal_time:.2f} 秒", flush=True)

    # 決定 trade_signals 表名，並儲存
    def _signals_table_for_data_table(data_table: str) -> str:
        if not data_table:
            return 'trade_signals'
        base = data_table.split('.')[-1]
        base = base.replace('[', '').replace(']', '')
        if '_' in base:
            suffix = base.split('_')[-1]
            return f'trade_signals_{suffix}'
        return 'trade_signals'

    signals_table = _signals_table_for_data_table(table)
    save_start = time.time()
    print(f"開始儲存結果到資料庫（目標表：{signals_table}）...", flush=True)
    dbmod.save_signals_to_mssql(
        df, server, database, user, password, table_name=signals_table
    )
    save_time = time.time() - save_start

    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'分析結果已儲存至 {output_path}', flush=True)

    total_time = time.time() - total_start_time
    print(f"\n總執行時間: {total_time:.2f} 秒", flush=True)
    print(
        f"- 資料讀取: {read_time:.2f}秒 ({read_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 指標計算: {calc_time:.2f}秒 ({calc_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 訊號生成: {signal_time:.2f}秒 ({signal_time/total_time*100:.1f}%)",
        flush=True)
    print(
        f"- 資料儲存: {save_time:.2f}秒 ({save_time/total_time*100:.1f}%)",
        flush=True)

    tradesmod.print_analysis_summary(df)
