import pandas as pd
import talib
import os
from typing import List, Tuple

# 所有K線型態函數與中文註解對應表
candle_patterns: List[Tuple[str, str]] = [
    ('CDL2CROWS', '兩隻烏鴉'),
    ('CDL3BLACKCROWS', '三隻烏鴉'),
    ('CDL3INSIDE', '三內部漲跌'),
    ('CDL3LINESTRIKE', '三線打擊'),
    ('CDL3OUTSIDE', '三外部漲跌'),
    ('CDL3STARSINSOUTH', '南方三星'),
    ('CDL3WHITESOLDIERS', '三白兵'),
    ('CDLABANDONEDBABY', '棄嬰'),
    ('CDLADVANCEBLOCK', '大敵當前'),
    ('CDLBELTHOLD', '捉腰帶線'),
    ('CDLBREAKAWAY', '脫離'),
    ('CDLCLOSINGMARUBOZU', '光頭光腳(單頭腳判定)'),
    ('CDLCONCEALBABYSWALL', '藏嬰吞沒'),
    ('CDLCOUNTERATTACK', '反擊線'),
    ('CDLDARKCLOUDCOVER', '烏雲蓋頂'),
    ('CDLDOJI', '十字'),
    ('CDLDOJISTAR', '十字星'),
    ('CDLDRAGONFLYDOJI', '蜻蜓十字'),
    ('CDLENGULFING', '吞沒形態'),
    ('CDLEVENINGDOJISTAR', '十字暮星'),
    ('CDLEVENINGSTAR', '暮星'),
    ('CDLGAPSIDESIDEWHITE', '缺口上漲'),
    ('CDLGRAVESTONEDOJI', '墓碑十字'),
    ('CDLHAMMER', '錘頭'),
    ('CDLHANGINGMAN', '上吊線'),
    ('CDLHARAMI', '孕線'),
    ('CDLHARAMICROSS', '十字孕線'),
    ('CDLHIGHWAVE', '風高浪大線'),
    ('CDLHIKKAKE', 'Hikkake 陷阱'),
    ('CDLHIKKAKEMOD', 'Hikkake Modified'),
    ('CDLHOMINGPIGEON', '家鴿'),
    ('CDLIDENTICAL3CROWS', '三胞胎烏鴉'),
    ('CDLINNECK', '頸內線'),
    ('CDLINVERTEDHAMMER', '倒錘頭'),
    ('CDLKICKING', '反沖形態'),
    ('CDLKICKINGBYLENGTH', '反沖-長短判斷'),
    ('CDLLADDERBOTTOM', '梯形底部'),
    ('CDLLONGLEGGEDDOJI', '長腿十字'),
    ('CDLLONGLINE', '長蠟燭'),
    ('CDLMARUBOZU', '光頭光腳'),
    ('CDLMATCHINGLOW', '匹配低點'),
    ('CDLMATHOLD', '鋪墊'),
    ('CDLMORNINGDOJISTAR', '十字晨星'),
    ('CDLMORNINGSTAR', '晨星'),
    ('CDLONNECK', '頸上線'),
    ('CDLPIERCING', '刺穿形態'),
    ('CDLRICKSHAWMAN', '黃包車夫'),
    ('CDLRISEFALL3METHODS', '上升/下降三法'),
    ('CDLSEPARATINGLINES', '分離線'),
    ('CDLSHOOTINGSTAR', '射擊星'),
    ('CDLSHORTLINE', '短蠟燭'),
    ('CDLSPINNINGTOP', '紡錘線'),
    ('CDLSTALLEDPATTERN', '停滯形態'),
    ('CDLSTICKSANDWICH', '三明治'),
    ('CDLTAKURI', '探水竿'),
    ('CDLTASUKIGAP', '跳空並列(月缺)'),
    ('CDLTHRUSTING', '向上突破'),
    ('CDLTRISTAR', '三星'),
    ('CDLUNIQUE3RIVER', '奇特三河床'),
    ('CDLUPSIDEGAP2CROWS', '向上跳空兩隻烏鴉'),
    ('CDLXSIDEGAP3METHODS', '跳空三法'),
]

# 需 penetration 參數的型態
penetration_patterns = {
    'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
    'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR'
}

# 特殊訊號對應表
special_cases = {
    'CDLCLOSINGMARUBOZU': {1: '光頭', -1: '光腳'},
    'CDLMARUBOZU': {1: '上漲光頭光腳', -1: '下跌光頭光腳'},
    'CDL3INSIDE': {1: '三內部上漲', -1: '三內部下跌'},
    'CDL3LINESTRIKE': {1: '三線打擊上漲', -1: '三線打擊下跌'},
    'CDL3OUTSIDE': {1: '三外部上漲', -1: '三外部下跌'},
    'CDLABANDONEDBABY': {1: '棄嬰上漲', -1: '棄嬰下跌'},
    'CDLRISEFALL3METHODS': {1: '上升三法', -1: '下降三法'},
    'CDLXSIDEGAP3METHODS': {1: '上升跳空三法', -1: '下降跳空三法'},
    'CDLENGULFING': {1: '多頭吞沒', -1: '空頭吞沒'},
}


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    open_, high_, low_, close_ = (
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values)
    for func_name, _ in candle_patterns:
        if not hasattr(talib, func_name):
            continue
        func = getattr(talib, func_name)
        if func_name in penetration_patterns:
            df[func_name[3:]] = func(open_, high_, low_, close_, penetration=0)
        else:
            df[func_name[3:]] = func(open_, high_, low_, close_)
    return df


def combine_patterns(row: pd.Series) -> str:
    signals = []
    for func_name, zh in candle_patterns:
        val = row.get(func_name[3:], 0)
        # 特殊判斷
        if func_name in special_cases and val != 0:
            signals.append(special_cases[func_name].get(int(val/abs(val)), zh))
            continue
        if val != 0:
            signals.append(zh)
    return ','.join(signals) if signals else ''


def load_csv(input_csv: str) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到檔案: {input_csv}")
    df = pd.read_csv(input_csv)
    rename_map = {
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
        'volume': 'volume'
    }
    df = df.rename(columns=rename_map)
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位: {col}")
    return df


def main(input_csv: str = '2330.csv', output_csv: str = 'output.csv') -> None:
    try:
        df = load_csv(input_csv)
    except Exception as e:
        print(f"讀取資料失敗: {e}", flush=True)
        return
    df = detect_patterns(df)
    df['PatternSignals'] = df.apply(combine_patterns, axis=1)
    result = df[df['PatternSignals'] != '']
    show_cols = [c for c in [
        'symbol', 'datetime', 'open', 'high',
        'low', 'close', 'PatternSignals'
    ] if c in result.columns]
    print(result[show_cols].head(10), flush=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n已儲存 {len(result)} 筆有型態訊號的資料到 {output_csv}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='K線型態自動辨識')
    parser.add_argument('--input', default='2330.csv', help='輸入CSV檔名')
    parser.add_argument('--output', default='output.csv', help='輸出CSV檔名')
    args = parser.parse_args()
    main(args.input, args.output)
