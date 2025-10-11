# -*- coding: utf-8 -*-
"""買賣訊號彙整與分析報告"""

from .config import SIGNAL_WEIGHTS


def generate_trade_signals(df, min_signals=3):
    df['Buy_Signals'] = 0.0
    df['Sell_Signals'] = 0.0

    buy_conditions = [
        (df['MA_Cross'] == '突破MA20', SIGNAL_WEIGHTS['MA_Cross']),
        (df['MACD_Cross'] == '黃金交叉', SIGNAL_WEIGHTS['MACD_Cross']),
        (df['EMA_Cross'] == 'EMA黃金交叉', SIGNAL_WEIGHTS['EMA_Cross']),
        (df['KD_Signal'] == 'K上穿D', SIGNAL_WEIGHTS['KD_Cross']),
        (df['RSI_Signal'] == '超賣', SIGNAL_WEIGHTS['RSI_Oversold']),
        (df['RSI_Signal'] == '接近超賣', SIGNAL_WEIGHTS['RSI_Near']),
        (df['BB_Signal'] == '突破下軌', SIGNAL_WEIGHTS['BB_Break']),
        (df['MACD_Div'] == '底背離', SIGNAL_WEIGHTS['MACD_Div']),
        (df['Trend'] == '偏多', SIGNAL_WEIGHTS['Trend']),
        (df['Volume_Anomaly'] == '量能異常', SIGNAL_WEIGHTS['Volume']),
        (df['CCI_Signal'] == 'CCI超賣', SIGNAL_WEIGHTS['CCI']),
        (df['CCI_Signal'] == 'CCI上穿零軸', SIGNAL_WEIGHTS['CCI']),
        (df['WILLR_Signal'] == 'WILLR超賣', SIGNAL_WEIGHTS['WILLR']),
        (df['MOM_Signal'] == '動量轉正', SIGNAL_WEIGHTS['MOM']),
        (df['KD_Signal'] == 'KD超賣', SIGNAL_WEIGHTS['KD_Cross'])
    ]

    for condition, weight in buy_conditions:
        df.loc[condition, 'Buy_Signals'] += weight

    sell_conditions = [
        (df['MA_Cross'] == '跌破MA20', SIGNAL_WEIGHTS['MA_Cross']),
        (df['MACD_Cross'] == '死亡交叉', SIGNAL_WEIGHTS['MACD_Cross']),
        (df['EMA_Cross'] == 'EMA死亡交叉', SIGNAL_WEIGHTS['EMA_Cross']),
        (df['KD_Signal'] == 'K下穿D', SIGNAL_WEIGHTS['KD_Cross']),
        (df['RSI_Signal'] == '超買', SIGNAL_WEIGHTS['RSI_Oversold']),
        (df['RSI_Signal'] == '接近超買', SIGNAL_WEIGHTS['RSI_Near']),
        (df['BB_Signal'] == '突破上軌', SIGNAL_WEIGHTS['BB_Break']),
        (df['MACD_Div'] == '頂背離', SIGNAL_WEIGHTS['MACD_Div']),
        (df['Trend'] == '偏空', SIGNAL_WEIGHTS['Trend']),
        (df['KD_Signal'] == 'KD超買', SIGNAL_WEIGHTS['KD_Cross']),
        (df['CCI_Signal'] == 'CCI超買', SIGNAL_WEIGHTS['CCI']),
        (df['CCI_Signal'] == 'CCI下穿零軸', SIGNAL_WEIGHTS['CCI']),
        (df['WILLR_Signal'] == 'WILLR超買', SIGNAL_WEIGHTS['WILLR']),
        (df['MOM_Signal'] == '動量轉負', SIGNAL_WEIGHTS['MOM'])
    ]

    for condition, weight in sell_conditions:
        df.loc[condition, 'Sell_Signals'] += weight

    df['Trade_Signal'] = ''
    df['Signal_Strength'] = ''

    strong_buy = df['Buy_Signals'] >= min_signals + 1
    buy = (df['Buy_Signals'] >= min_signals) & (~strong_buy)
    strong_sell = df['Sell_Signals'] >= min_signals + 1
    sell = (df['Sell_Signals'] >= min_signals) & (~strong_sell)

    df.loc[strong_buy, 'Trade_Signal'] = '強烈買入'
    df.loc[buy, 'Trade_Signal'] = '買入'
    df.loc[strong_sell, 'Trade_Signal'] = '強烈賣出'
    df.loc[sell, 'Trade_Signal'] = '賣出'

    df.loc[df['Trade_Signal'].isin(['買入', '強烈買入']), 'Signal_Strength'] = (
        '多頭' + df.loc[df['Trade_Signal'].isin(['買入', '強烈買入']), 'Buy_Signals']
        .map(lambda v: f'{v:.1f}分')
    )
    df.loc[df['Trade_Signal'].isin(['賣出', '強烈賣出']), 'Signal_Strength'] = (
        '空頭' + df.loc[df['Trade_Signal'].isin(['賣出', '強烈賣出']), 'Sell_Signals']
        .map(lambda v: f'{v:.1f}分')
    )

    return df


def print_analysis_summary(df):
    print("\n=== 交易訊號分析報告 ===", flush=True)
    total_records = len(df)
    signal_records = len(df[df['Trade_Signal'] != ''])

    print(f"總資料筆數: {total_records:,}", flush=True)
    if total_records > 0:
        print(
            f"有訊號筆數: {signal_records:,} "
            f"({signal_records/total_records*100:.1f}%)",
            flush=True
        )

    trade_counts = df['Trade_Signal'].value_counts()
    print("\n交易訊號統計：", flush=True)
    for signal, count in trade_counts.items():
        if signal != '':
            percentage = count/total_records*100 if total_records > 0 else 0
            print(f"  {signal}: {count:,} 次 ({percentage:.2f}%)", flush=True)

    buy_signals = df[df['Trade_Signal'].isin(['買入', '強烈買入'])]
    sell_signals = df[df['Trade_Signal'].isin(['賣出', '強烈賣出'])]

    if len(buy_signals) > 0:
        avg_buy_strength = buy_signals['Buy_Signals'].mean()
        max_buy_strength = buy_signals['Buy_Signals'].max()
        print(
            f"\n多頭訊號強度: 平均 {avg_buy_strength:.1f}分, "
            f"最高 {max_buy_strength:.1f}分", flush=True
        )

    if len(sell_signals) > 0:
        avg_sell_strength = sell_signals['Sell_Signals'].mean()
        max_sell_strength = sell_signals['Sell_Signals'].max()
        print(
            f"空頭訊號強度: 平均 {avg_sell_strength:.1f}分, "
            f"最高 {max_sell_strength:.1f}分", flush=True
        )

    latest_signals = df[df['Trade_Signal'] != ''].tail(3)
    if len(latest_signals) > 0:
        print("\n最近3個交易訊號:", flush=True)
        for _, row in latest_signals.iterrows():
            print(
                f"  {row['datetime'].strftime('%Y-%m-%d')}: "
                f"{row['Trade_Signal']} ({row['Signal_Strength']})", flush=True
            )
