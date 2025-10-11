# -*- coding: utf-8 -*-"""

from dotenv import load_dotenv
import os
import argparse
import re as _re
from signals.analyzer import analyze_signals_from_db_with_symbol

env_local = '.env.local'
if os.path.exists(env_local):
    load_dotenv(env_local, override=True)
else:
    load_dotenv()

# 環境變數預設
default_server = os.getenv('MSSQL_SERVER')
default_database = os.getenv('MSSQL_DATABASE')
default_table = os.getenv('MSSQL_TABLE')
default_user = os.getenv('MSSQL_USER')
default_password = os.getenv('MSSQL_PASSWORD')
default_output = os.getenv('OUTPUT_CSV', '')

parser = argparse.ArgumentParser(description='分析交易訊號')
# 支援舊式位置參數 symbol，也支援 -s/--symbol
parser.add_argument(
    'symbol_pos', nargs='*', help='位置參數：symbol(s)', default=None
)
parser.add_argument(
    '-s', '--symbol', nargs='+', help='股票代號或 symbol(s)', default=None
)
parser.add_argument(
    '-d', '--database', help='要讀取的 MSSQL 資料庫', default=None
)
parser.add_argument(
    '-t', '--table', help='要讀取的資料表', default=None
)
parser.add_argument(
    '--server', help='MSSQL server', default=None
)
parser.add_argument(
    '--user', help='MSSQL user', default=None
)
parser.add_argument(
    '--password', help='MSSQL password', default=None
)
parser.add_argument(
    '--output',
    nargs='?',
    const='__DEFAULT_OUTPUT__',
    help='輸出 CSV 路徑，若不帶參數則輸出到 ./output/',
    default=None,
)

# 新增簡短旗標：period（--1d/--1h 等）與 region (--us/--tw 等)
period_group = parser.add_mutually_exclusive_group()
period_group.add_argument(
    '--1d',
    dest='period',
    action='store_const',
    const='1d',
    help='使用日線',
)
period_group.add_argument(
    '--1h',
    dest='period',
    action='store_const',
    const='1h',
    help='使用小時線',
)

region_group = parser.add_mutually_exclusive_group()
region_group.add_argument(
    '--us',
    dest='region',
    action='store_const',
    const='market_stock_us',
    help='使用美股資料庫',
)
region_group.add_argument(
    '--tw',
    dest='region',
    action='store_const',
    const='market_stock_tw',
    help='使用台股資料庫',
)
region_group.add_argument(
    '--etf',
    dest='region',
    action='store_const',
    const='market_etf',
    help='使用 ETF 資料庫',
)
region_group.add_argument(
    '--index',
    dest='region',
    action='store_const',
    const='market_index',
    help='使用指數資料庫',
)
region_group.add_argument(
    '--forex',
    dest='region',
    action='store_const',
    const='market_forex',
    help='使用外匯資料庫',
)
region_group.add_argument(
    '--crypto',
    dest='region',
    action='store_const',
    const='market_crypto',
    help='使用加密貨幣資料庫',
)
region_group.add_argument(
    '--futures',
    dest='region',
    action='store_const',
    const='market_futures',
    help='使用期貨資料庫',
)


# 重新 parse（因為我們在建立群組後，必須重新解析命令列）
args = parser.parse_args()


def _normalize_symbols(pos_list, s_list):
    vals = []
    if pos_list:
        for item in pos_list:
            if not item:
                continue
            for part in str(item).split(','):
                p = part.strip()
                if p:
                    vals.append(p)
    if s_list:
        for item in s_list:
            if not item:
                continue
            for part in str(item).split(','):
                p = part.strip()
                if p:
                    vals.append(p)
    if not vals:
        return ['2317']
    return vals


symbols = _normalize_symbols(args.symbol_pos, args.symbol)

# 決定使用的 database：命令列 --database > region flag > 環境預設
if args.database is not None:
    database = args.database
elif getattr(args, 'region', None):
    database = args.region
else:
    database = default_database

# 決定使用的 table：命令列 --table > period flag > 環境預設
if args.table is not None:
    table = args.table
elif getattr(args, 'period', None):
    table = f"stock_data_{args.period}"
else:
    table = default_table

# server/user/password/output 使用者指定優先，否則回退到環境變數
server = args.server or default_server
user = args.user or default_user
password = args.password or default_password

# 處理 output：若使用者傳入 --output 而未帶值，預設輸出至 ./output/ 目錄
if args.output == '__DEFAULT_OUTPUT__':
    out_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(out_dir, exist_ok=True)
    # 傳回目錄，之後 per-symbol 會在該目錄下產生檔案
    output = out_dir
elif args.output:
    output = args.output
else:
    output = default_output or None


def _resolve_output_for_symbol(
    args_output, default_output, symbol, table, multiple
):
    if args_output == '__DEFAULT_OUTPUT__':
        out_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(out_dir, exist_ok=True)
        safe_symbol = _re.sub(r'[^0-9A-Za-z]+', '_', symbol)
        safe_table = _re.sub(r'[^0-9A-Za-z]+', '_', table)
        filename = f"{safe_symbol}_{safe_table}.csv"
        return os.path.join(out_dir, filename)

    elif args_output:
        path = args_output
        if multiple:
            # 如果是目錄，放到目錄下；如果是檔案，加入 symbol
            if os.path.isdir(path):
                safe_symbol = _re.sub(r'[^0-9A-Za-z]+', '_', symbol)
                filename = (
                    f"{safe_symbol}_{table}.csv"
                )
                return os.path.join(path, filename)
            else:
                base, ext = os.path.splitext(path)
                ext = ext if ext else '.csv'
                return f"{base}_{symbol}{ext}"
        else:
            return path

    else:
        return default_output or None


print(
    f"symbols={symbols}, database={database}, table={table}", flush=True
)

multiple = len(symbols) > 1
for symbol in symbols:
    out_path = _resolve_output_for_symbol(
        args.output, default_output, symbol, table, multiple
    )
    print(f"開始分析 symbol={symbol}", flush=True)
    analyze_signals_from_db_with_symbol(
        server,
        database,
        table,
        user,
        password,
        out_path,
        symbol,
    )
