# -*- coding: utf-8 -*-
"""資料庫相關讀寫函式（MSSQL）"""

import pandas as pd


def read_ohlcv_from_mssql(
    server, database, table, user, password, chunk_size=50000
):
    import pyodbc

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={user};PWD={password};"
        f"Trusted_Connection=no;Connection Timeout=30;"
        f"Application Name=TechnicalAnalysis"
    )

    try:
        with pyodbc.connect(conn_str) as conn:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')

            count_query = f"SELECT COUNT(*) FROM {table}"
            cursor = conn.cursor()
            row_count = cursor.execute(count_query).fetchval()
            print(f"資料表 {table} 共有 {row_count:,} 筆資料", flush=True)

            if row_count <= chunk_size:
                query = f"SELECT * FROM {table} ORDER BY datetime"
                df = pd.read_sql(query, conn)
                print(f"已一次讀取全部 {len(df):,} 筆資料", flush=True)
            else:
                print("資料量較大，使用分塊讀取...", flush=True)
                date_range_query = (
                    f"SELECT MIN(datetime) as min_date, "
                    f"MAX(datetime) as max_date FROM {table}"
                )
                date_range = pd.read_sql(date_range_query, conn)
                min_date = date_range['min_date'].iloc[0]
                max_date = date_range['max_date'].iloc[0]

                chunks = []
                current_date = min_date
                end_date = max_date

                while current_date <= end_date:
                    next_date = (
                        pd.to_datetime(current_date) + pd.DateOffset(months=3)
                    )
                    chunk_query = (
                        "SELECT * FROM {} WHERE datetime >= '{}' "
                        "AND datetime < '{}' ORDER BY datetime"
                    ).format(table, current_date, next_date)
                    chunk = pd.read_sql(chunk_query, conn)
                    chunks.append(chunk)
                    print(
                        f"已讀取 {current_date} 至 {next_date} 期間的 "
                        f"{len(chunk):,} 筆資料", flush=True
                    )
                    current_date = next_date

                df = pd.concat(chunks, ignore_index=True)
                print(f"共讀取 {len(df):,} 筆資料", flush=True)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    except Exception as e:
        print(f"讀取資料時發生錯誤: {str(e)}", flush=True)
        return pd.DataFrame()


def save_signals_to_mssql(
    df, server, database, user, password, table_name='trade_signals'
):
    import time
    import pyodbc
    start_time = time.time()

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={user};PWD={password};"
        f"Trusted_Connection=no"
    )

    conn = pyodbc.connect(conn_str, timeout=30)
    cursor = conn.cursor()

    try:
        check_table_query = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}')
        BEGIN
            CREATE TABLE {table_name} (
                id INT IDENTITY(1,1) PRIMARY KEY,
                datetime DATETIME,
                symbol NVARCHAR(20),
                close_price FLOAT,
                Trade_Signal NVARCHAR(50),
                Signal_Strength NVARCHAR(50),
                Buy_Signals FLOAT,
                Sell_Signals FLOAT,
                MA_Cross NVARCHAR(50),
                BB_Signal NVARCHAR(50),
                MACD_Cross NVARCHAR(50),
                Trend NVARCHAR(50),
                MACD_Div NVARCHAR(50),
                RSI_Signal NVARCHAR(50),
                KD_Signal NVARCHAR(50),
                SR_Signal NVARCHAR(50),
                Volume_Anomaly NVARCHAR(50),
                EMA_Cross NVARCHAR(50),
                CCI_Signal NVARCHAR(50),
                WILLR_Signal NVARCHAR(50),
                MOM_Signal NVARCHAR(50),
                Anomaly NVARCHAR(50),
                INDEX idx_datetime (datetime),
                INDEX idx_symbol (symbol)
            )
        END
        """
        cursor.execute(check_table_query)
        conn.commit()

        required_columns = [
            'datetime', 'symbol', 'close_price', 'Trade_Signal',
            'Signal_Strength', 'Buy_Signals', 'Sell_Signals',
            'MA_Cross', 'BB_Signal', 'MACD_Cross', 'Trend', 'MACD_Div',
            'RSI_Signal', 'KD_Signal', 'SR_Signal', 'Volume_Anomaly',
            'EMA_Cross', 'CCI_Signal', 'WILLR_Signal', 'MOM_Signal', 'Anomaly'
        ]

        for col in required_columns:
            if col not in df.columns:
                if col == 'symbol' and 'symbol' not in df.columns:
                    df['symbol'] = 'Unknown'
                else:
                    df[col] = ''

        # 不執行預刪除，改為使用 staging + MERGE 做 upsert
        # 這樣會保留歷史紀錄，只對相同 (symbol, datetime) 做更新或插入
        print("使用 MERGE 進行 upsert，不會先刪除歷史紀錄", flush=True)

        # 使用暫存 staging table 與 MERGE 做 upsert（更新或插入）
        batch_size = 1000
        total_rows = len(df)

        # 建立暫存表 #staging（與主表結構相符，不建立索引）
        create_staging_sql = (
            "IF OBJECT_ID('tempdb..#staging') IS NOT NULL DROP TABLE #staging;"
            "CREATE TABLE #staging ("
            "datetime DATETIME,"
            "symbol NVARCHAR(20),"
            "close_price FLOAT,"
            "Trade_Signal NVARCHAR(50),"
            "Signal_Strength NVARCHAR(50),"
            "Buy_Signals FLOAT,"
            "Sell_Signals FLOAT,"
            "MA_Cross NVARCHAR(50),"
            "BB_Signal NVARCHAR(50),"
            "MACD_Cross NVARCHAR(50),"
            "Trend NVARCHAR(50),"
            "MACD_Div NVARCHAR(50),"
            "RSI_Signal NVARCHAR(50),"
            "KD_Signal NVARCHAR(50),"
            "SR_Signal NVARCHAR(50),"
            "Volume_Anomaly NVARCHAR(50),"
            "EMA_Cross NVARCHAR(50),"
            "CCI_Signal NVARCHAR(50),"
            "WILLR_Signal NVARCHAR(50),"
            "MOM_Signal NVARCHAR(50),"
            "Anomaly NVARCHAR(50)"
            ");"
        )
        cursor.execute(create_staging_sql)

        insert_staging_sql = (
            "INSERT INTO #staging (datetime, symbol, close_price, "
            "Trade_Signal, Signal_Strength, Buy_Signals, Sell_Signals,"
            " MA_Cross,"
            " BB_Signal, MACD_Cross, Trend, MACD_Div, RSI_Signal,"
            " KD_Signal, SR_Signal, Volume_Anomaly, EMA_Cross,"
            " CCI_Signal, WILLR_Signal, MOM_Signal, Anomaly) VALUES ("
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
            ")"
        )

        # 插入 staging（分批）
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_data = []
            for _, row in batch_df.iterrows():
                record = (
                    row['datetime'],
                    row.get('symbol', 'Unknown'),
                    row['close_price'],
                    row['Trade_Signal'],
                    row['Signal_Strength'],
                    row['Buy_Signals'],
                    row['Sell_Signals'],
                    row['MA_Cross'],
                    row['BB_Signal'],
                    row['MACD_Cross'],
                    row['Trend'],
                    row['MACD_Div'],
                    row['RSI_Signal'],
                    row['KD_Signal'],
                    row['SR_Signal'],
                    row['Volume_Anomaly'],
                    row['EMA_Cross'],
                    row['CCI_Signal'],
                    row['WILLR_Signal'],
                    row['MOM_Signal'],
                    row.get('Anomaly', '')
                )
                batch_data.append(record)

            cursor.fast_executemany = True
            cursor.executemany(insert_staging_sql, batch_data)
            conn.commit()

            progress = min(i + batch_size, total_rows)
            print(
                f"已處理 {progress}/{total_rows} 筆資料 (已寫入暫存表) "
                f"({progress/total_rows*100:.1f}%)", flush=True
            )

        # 使用 MERGE 從 #staging 合併到主表，根據 symbol + datetime 做匹配
        merge_sql = (
            "MERGE INTO {table} AS target\n"
            "USING #staging AS src\n"
            "ON target.symbol = src.symbol "
            "AND target.datetime = src.datetime\n"
            "WHEN MATCHED THEN\n"
            "    UPDATE SET\n"
            "        close_price = src.close_price,\n"
            "        Trade_Signal = src.Trade_Signal,\n"
            "        Signal_Strength = src.Signal_Strength,\n"
            "        Buy_Signals = src.Buy_Signals,\n"
            "        Sell_Signals = src.Sell_Signals,\n"
            "        MA_Cross = src.MA_Cross,\n"
            "        BB_Signal = src.BB_Signal,\n"
            "        MACD_Cross = src.MACD_Cross,\n"
            "        Trend = src.Trend,\n"
            "        MACD_Div = src.MACD_Div,\n"
            "        RSI_Signal = src.RSI_Signal,\n"
            "        KD_Signal = src.KD_Signal,\n"
            "        SR_Signal = src.SR_Signal,\n"
            "        Volume_Anomaly = src.Volume_Anomaly,\n"
            "        EMA_Cross = src.EMA_Cross,\n"
            "        CCI_Signal = src.CCI_Signal,\n"
            "        WILLR_Signal = src.WILLR_Signal,\n"
            "        MOM_Signal = src.MOM_Signal,\n"
            "        Anomaly = src.Anomaly\n"
            "WHEN NOT MATCHED BY TARGET THEN\n"
            "    INSERT (datetime, symbol, close_price, Trade_Signal,\n"
            "        Signal_Strength, Buy_Signals, Sell_Signals, MA_Cross,\n"
            "        BB_Signal, MACD_Cross, Trend, MACD_Div, RSI_Signal,\n"
            "        KD_Signal, SR_Signal, Volume_Anomaly, EMA_Cross,\n"
            "        CCI_Signal, WILLR_Signal, MOM_Signal, Anomaly)\n"
            "    VALUES (src.datetime, src.symbol, src.close_price, "
            "src.Trade_Signal, src.Signal_Strength, src.Buy_Signals, "
            "src.Sell_Signals, src.MA_Cross, src.BB_Signal, src.MACD_Cross, "
            "src.Trend, src.MACD_Div, src.RSI_Signal, src.KD_Signal, "
            "src.SR_Signal, src.Volume_Anomaly, src.EMA_Cross, "
            "src.CCI_Signal, src.WILLR_Signal, src.MOM_Signal, src.Anomaly);"
        )
        cursor.execute(merge_sql.format(table=table_name))
        conn.commit()

        elapsed_time = time.time() - start_time
        print(
            "成功將 {} 筆資料 upsert 至 {} 資料表，耗時 {:.2f} 秒".format(
                total_rows, table_name, elapsed_time
            ),
            flush=True
        )

    except Exception as e:
        conn.rollback()
        print(f"\n[錯誤] 儲存資料至MSSQL時發生錯誤: {str(e)}", flush=True)
    finally:
        cursor.close()
        conn.close()
