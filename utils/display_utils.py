"""
工具模組
提供統計資訊顯示等輔助功能
"""

from typing import Dict, Any


def print_statistics(stats: Dict[str, Any]):
    """列印指定間隔的資料庫統計資訊"""
    if not stats:
        print("無法獲取統計資訊", flush=True)
        return

    table_name = stats.get('table_name', '未知表格')
    interval = stats.get('interval', '未知間隔')

    print(f"\n📊 資料庫統計資訊 ({table_name} - {interval})", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"總記錄數: {stats.get('total_records', 0):,}", flush=True)
    print(f"股票數量: {stats.get('unique_symbols', 0)}", flush=True)

    date_range = stats.get('date_range', {})
    if date_range.get('earliest') and date_range.get('latest'):
        print(
            f"日期範圍: {date_range['earliest']} ~ {date_range['latest']}",
            flush=True)

    symbols = stats.get('symbols', [])
    if symbols:
        print("\n📋 各股票詳情:", flush=True)
        for symbol_info in symbols[:10]:  # 只顯示前10個
            print(f"  {symbol_info['symbol']}: {symbol_info['records']:,} 筆 "
                  f"({symbol_info['start_date']} ~ {symbol_info['end_date']})",
                  flush=True)

        if len(symbols) > 10:
            print(f"  ... 還有 {len(symbols) - 10} 個股票", flush=True)


def print_all_statistics(all_stats: Dict[str, Dict[str, Any]]):
    """列印所有間隔表的統計資訊"""
    if not all_stats:
        print("無任何資料表統計資訊", flush=True)
        return

    print("\n📊 所有資料表統計資訊", flush=True)
    print(f"{'='*70}", flush=True)

    # 按間隔排序顯示
    interval_order = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']

    for interval in interval_order:
        if interval in all_stats:
            stats = all_stats[interval]
            table_name = stats.get('table_name', f'stock_data_{interval}')
            total_records = stats.get('total_records', 0)
            unique_symbols = stats.get('unique_symbols', 0)

            print(f"📈 {table_name} ({interval}):", flush=True)
            print(f"   記錄數: {total_records:,} 筆", flush=True)
            print(f"   股票數: {unique_symbols} 個", flush=True)

            date_range = stats.get('date_range', {})
            if date_range.get('earliest') and date_range.get('latest'):
                print(
                    f"   範圍: {date_range['earliest']} ~ "
                    f"{date_range['latest']}", flush=True)
            print(flush=True)

    # 總結
    total_tables = len(all_stats)
    total_all_records = sum(stats.get('total_records', 0)
                            for stats in all_stats.values())
    print(f"📋 總結: {total_tables} 個資料表，共 {total_all_records:,} 筆記錄", flush=True)


def format_processing_summary(results: Dict[str, Any]) -> str:
    """格式化處理結果摘要"""
    success_count = sum(1 for r in results.values() if r.success)
    total_new = sum(r.new_records for r in results.values())
    total_updated = sum(r.updated_records for r in results.values())
    total_indicators = sum(r.indicator_updates for r in results.values())

    summary = f"""
處理結果摘要:
✅ 成功處理: {success_count}/{len(results)} 個股票
📊 新增記錄: {total_new:,} 筆
🔄 更新記錄: {total_updated:,} 筆
📈 技術指標更新: {total_indicators:,} 筆
"""
    return summary
