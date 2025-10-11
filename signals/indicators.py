# -*- coding: utf-8 -*-
"""技術指標與訊號計算函式集合"""

import numpy as np


def ma_cross_signal(df):
    signal = (df['ma5'] > df['ma20']) & (
        df['ma5'].shift(1) <= df['ma20'].shift(1))
    df['MA_Cross'] = np.where(signal, '突破MA20', '')
    signal = (df['ma5'] < df['ma20']) & (
        df['ma5'].shift(1) >= df['ma20'].shift(1))
    df['MA_Cross'] = np.where(signal, '跌破MA20', df['MA_Cross'])
    return df


def bollinger_signal(df):
    df['BB_Signal'] = ''
    df.loc[df['close_price'] > df['bb_upper'], 'BB_Signal'] = '突破上軌'
    df.loc[df['close_price'] < df['bb_lower'], 'BB_Signal'] = '突破下軌'
    return df


def macd_signal(df):
    df['MACD_Cross'] = ''
    cross_up = (df['dif'] > df['macd']) & (
        df['dif'].shift(1) <= df['macd'].shift(1))
    cross_down = (df['dif'] < df['macd']) & (
        df['dif'].shift(1) >= df['macd'].shift(1))
    df.loc[cross_up, 'MACD_Cross'] = '黃金交叉'
    df.loc[cross_down, 'MACD_Cross'] = '死亡交叉'
    return df


def trend_signal(df):
    df['Trend'] = np.where(df['close_price'] > df['ma20'], '偏多', '偏空')
    return df


def macd_divergence(df, lookback=10):
    df = df.copy()
    df['MACD_Div'] = ''
    price_shift = df['close_price'].shift(1)
    dif_shift = df['dif'].shift(1)
    price_min = price_shift.rolling(
        window=lookback, min_periods=lookback
    ).min()
    price_max = price_shift.rolling(
        window=lookback, min_periods=lookback
    ).max()
    macd_min = dif_shift.rolling(
        window=lookback, min_periods=lookback
    ).min()
    macd_max = dif_shift.rolling(
        window=lookback, min_periods=lookback
    ).max()

    cond_bottom = (df['close_price'] < price_min) & (df['dif'] > macd_min)
    cond_top = (df['close_price'] > price_max) & (df['dif'] < macd_max)

    df.loc[cond_bottom, 'MACD_Div'] = '底背離'
    df.loc[cond_top, 'MACD_Div'] = '頂背離'
    return df


def anomaly_detection(df, window=20, threshold=3):
    df = df.copy()
    df['Return'] = df['close_price'].pct_change()
    roll_mean = df['Return'].rolling(window=window, min_periods=window).mean()
    roll_std = df['Return'].rolling(window=window, min_periods=window).std()
    df['ZScore'] = (df['Return'] - roll_mean) / roll_std
    df['ZScore'] = df['ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Anomaly'] = np.where(abs(df['ZScore']) > threshold, 'Anomaly', '')
    df.drop(columns=['Return', 'ZScore'], inplace=True, errors='ignore')
    return df


def rsi_signal(df, rsi_col='rsi_14', overbought=70, oversold=30, near=5):
    df['RSI_Signal'] = ''
    df.loc[df[rsi_col] >= overbought, 'RSI_Signal'] = '超買'
    df.loc[df[rsi_col] <= oversold, 'RSI_Signal'] = '超賣'
    df.loc[
        (df[rsi_col] < overbought) & (df[rsi_col] >= overbought - near),
        'RSI_Signal'
    ] = '接近超買'
    df.loc[
        (df[rsi_col] > oversold) & (df[rsi_col] <= oversold + near),
        'RSI_Signal'
    ] = '接近超賣'
    return df


def kd_signal(df, k_col='k_value', d_col='d_value'):
    df['KD_Signal'] = ''
    cross_up = (
        (df[k_col] > df[d_col])
        & (df[k_col].shift(1) <= df[d_col].shift(1))
    )
    cross_down = (
        (df[k_col] < df[d_col])
        & (df[k_col].shift(1) >= df[d_col].shift(1))
    )
    df.loc[cross_up, 'KD_Signal'] = 'K上穿D'
    df.loc[cross_down, 'KD_Signal'] = 'K下穿D'
    df.loc[(df[k_col] > 80) & (df[d_col] > 80), 'KD_Signal'] = 'KD超買'
    df.loc[(df[k_col] < 20) & (df[d_col] < 20), 'KD_Signal'] = 'KD超賣'
    return df


def support_resistance_signal(df):
    df['SR_Signal'] = ''
    df.loc[(df['close_price'] >= df['bb_upper'] * 0.98), 'SR_Signal'] = '接近壓力位'
    df.loc[(df['close_price'] <= df['bb_lower'] * 1.02), 'SR_Signal'] = '接近支撐位'
    return df


def volume_anomaly_signal(df, window=20, threshold=1.5):
    df = df.copy()
    vol_ma = df['volume'].rolling(window=window, min_periods=1).mean()
    cond = df['volume'] > vol_ma * threshold
    df['Volume_Anomaly'] = np.where(cond, '量能異常', '')
    return df


def ema_cross_signal(df):
    df['EMA_Cross'] = ''
    cross_up = (
        (df['ema12'] > df['ema26'])
        & (df['ema12'].shift(1) <= df['ema26'].shift(1))
    )
    cross_down = (
        (df['ema12'] < df['ema26'])
        & (df['ema12'].shift(1) >= df['ema26'].shift(1))
    )
    df.loc[cross_up, 'EMA_Cross'] = 'EMA黃金交叉'
    df.loc[cross_down, 'EMA_Cross'] = 'EMA死亡交叉'
    return df


def cci_signal(df, cci_col='cci', overbought=100, oversold=-100):
    df['CCI_Signal'] = ''
    df.loc[df[cci_col] >= overbought, 'CCI_Signal'] = 'CCI超買'
    df.loc[df[cci_col] <= oversold, 'CCI_Signal'] = 'CCI超賣'
    cross_up = (df[cci_col] > 0) & (df[cci_col].shift(1) <= 0)
    cross_down = (df[cci_col] < 0) & (df[cci_col].shift(1) >= 0)
    df.loc[cross_up, 'CCI_Signal'] = 'CCI上穿零軸'
    df.loc[cross_down, 'CCI_Signal'] = 'CCI下穿零軸'
    return df


def willr_signal(df, willr_col='willr', overbought=-20, oversold=-80):
    df['WILLR_Signal'] = ''
    df.loc[df[willr_col] >= overbought, 'WILLR_Signal'] = 'WILLR超買'
    df.loc[df[willr_col] <= oversold, 'WILLR_Signal'] = 'WILLR超賣'
    return df


def momentum_signal(df, mom_col='mom'):
    df['MOM_Signal'] = ''
    cross_up = (df[mom_col] > 0) & (df[mom_col].shift(1) <= 0)
    cross_down = (df[mom_col] < 0) & (df[mom_col].shift(1) >= 0)
    df.loc[cross_up, 'MOM_Signal'] = '動量轉正'
    df.loc[cross_down, 'MOM_Signal'] = '動量轉負'
    return df
