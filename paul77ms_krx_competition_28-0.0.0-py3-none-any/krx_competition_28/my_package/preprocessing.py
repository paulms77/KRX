import pandas as pd
import numpy as np
from talib import abstract
import talib as ta
from typing import Tuple
import re

def get_technical_indicators(dfs:pd.DataFrame(), symbol:str, value:int)->pd.DataFrame():
    """
    기술분석 수행 함수
    
    :param pd.DataFrame dfs: 기술분석을 수행할 데이터
    :param str symbol: 종목코드
    :param int value: 결측치 처리 기능 타입
    :return: 기술분석을 수행 후 결과
    :rtype: pd.DataFrame
    """
    temp = dfs[dfs.SYMBOL == symbol]
    O = temp.OPEN
    L = temp.LOW
    H = temp.HIGH
    C = temp.CLOSE
    V = temp.VOLUME
    A = temp.AMOUNT
    Period = pd.to_datetime(temp.DATE)

    # Cycle Indicator Functions
    indicator_names = ta.get_function_groups()['Cycle Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name == 'HT_PHASOR':
            result_df['INPHASE'], result_df['QUADRATURE'] = indicator_func(C)
        elif indicator_name == 'HT_SINE':
            result_df['SINE'], result_df['LEADSINE'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Math Operator Functions
    indicator_names = ['ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MULT', 'SUB', 'SUM']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['ADD', 'DIV', 'MULT', 'SUB']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name in ['MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'SUM']:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Momentum Indicator Functions
    indicator_names = ta.get_function_groups()['Momentum Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['BOP']:
            result_df[indicator_name] = indicator_func(O, H, L, C)
        elif indicator_name in ['MFI']:
            result_df[indicator_name] = indicator_func(H, L, C, V)
        elif indicator_name in ['ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'PLUS_DI', 'ULTOSC', 'WILLR']:
            result_df[indicator_name] = indicator_func(H, L, C)
        elif indicator_name == 'STOCH':
            result_df['SLOWK'], result_df['SLOWD'] = indicator_func(H, L, C)
        elif indicator_name == 'STOCHF':
            result_df['FASTK'], result_df['FASTD'] = indicator_func(H, L, C)
        elif indicator_name in ['AROONOSC', 'MINUS_DM', 'PLUS_DM']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name == 'AROON':
            result_df['ARROONDOWN'], result_df['AROONUP'] = indicator_func(H, L)
        elif indicator_name in ['MACD', 'MACDEXT', 'MACDFIX']:
            result_df[indicator_name], result_df[indicator_name+'SIGNAL'], result_df[indicator_name+'HIST'] = indicator_func(C)
        elif indicator_name == 'STOCHRSI':
            result_df[indicator_name+'FASTK'], result_df[indicator_name+'FASTD'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)
    
    # Overlap Studies Functions
    indicator_names = ta.get_function_groups()['Overlap Studies']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['MIDPRICE', 'SAR', 'SAREXT']:
            result_df[indicator_name] = indicator_func(H, L)
        elif indicator_name == 'BBANDS':
            result_df['UPPERBAND'], result_df['MIDDLEBAND'], result_df['LOWERBAND'] = indicator_func(C)
        elif indicator_name == 'MAVP':
            result_df[indicator_name] = indicator_func(C, Period)
        elif indicator_name == 'MAMA':
            result_df[indicator_name], result_df['FAMA'] = indicator_func(C)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)
    
    # Pattern Recognition Functions
    indicator_names = ta.get_function_groups()['Pattern Recognition']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        result_df[indicator_name] = indicator_func(O, H, L, C)
    temp = pd.concat([temp, result_df], axis=1)

    # Price Transform Functions
    indicator_names = ta.get_function_groups()['Price Transform']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name == 'AVGPRICE':
            result_df[indicator_name] = indicator_func(O, H, L, C)
        elif indicator_name in ['TYPPRICE', 'WCLPRICE']:
            result_df[indicator_name] = indicator_func(H, L, C)
        else:
            result_df[indicator_name] = indicator_func(H, L)
    temp = pd.concat([temp, result_df], axis=1)

    # Statistic Functions
    indicator_names = ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['BETA', 'CORREL']:
            result_df[indicator_name] = indicator_func(H, L)
        else:
            result_df[indicator_name] = indicator_func(C)
    temp = pd.concat([temp, result_df], axis=1)

    # Volatility Indicators
    indicator_names = ta.get_function_groups()['Volatility Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        result_df[indicator_name] = indicator_func(H, L, C)
    temp = pd.concat([temp, result_df], axis=1)

    # Volume Indicators
    indicator_names = ta.get_function_groups()['Volume Indicators']
    result_df = pd.DataFrame()
    for indicator_name in indicator_names:
        indicator_func = abstract.Function(indicator_name)
        if indicator_name in ['AD', 'ADOSC']:
            result_df[indicator_name] = indicator_func(H, L, C, V)
        else:
            result_df[indicator_name] = indicator_func(C, V)
    temp = pd.concat([temp, result_df], axis=1)

    # Add Feature Engineering
    # 이동 평균 (Rolling)
    timeperiods = [5, 6, 7, 10, 20, 21, 50, 60, 100, 120]
    for timeperiod in timeperiods:
        temp[f'MA{timeperiod}'] = C.rolling(window = timeperiod).mean()
        
    # 과거 종가 (Lagging)
    temp[f'CLOSE_BEFORE1'] = C.shift(-1)
    temp[f'CLOSE_BEFORE3'] = C.shift(-3)
    temp[f'CLOSE_BEFORE5'] = C.shift(-5)
    temp[f'CLOSE_BEFORE7'] = C.shift(-7)
    
    # 누적 거래대금
    temp['CUMULATIVE_AMOUNT'] = A.cumsum()
    
    # 누적 거래량
    temp['CUMULATIVE_VOLUME'] = V.cumsum()
    
    # HIGH/OPEN
    temp['HIGH/OPEN'] = H / O
    
    # LOW/OPEN
    temp['LOW/OPEN'] = L / O
    
    # CLOSE/OPEN
    temp['CLOSE/OPEN'] = C / O

    # 결측치 처리 기능
    temp = temp.replace([np.inf, -np.inf], np.nan)
    if isinstance(value, int) and value == -1:
        temp = temp.fillna(method="ffill").fillna(method='bfill')
    else:
        temp = temp.fillna(method="ffill").fillna(value)
    
    return temp

def train_test_split(dataset:pd.DataFrame(), close_name:str)->tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """
    Custom train test split (역할: 학습 데이터, 테스트 데이터 분리)
    
    :param pd.DataFrame dataset: train test split를 수행할 데이터프레임
    :param str close_name: target에 해당하는 열 이름
    :return: 학습 데이터 (X_train, y_train) 튜플, 테스트 데이터 (X_test, y_test) 튜플
    :rtype: tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]
    """
    dataset = dataset.copy()
    y = dataset[close_name]
    X = dataset.drop([close_name], axis = 1)
    train_samples = int(np.ceil(0.8 * X.shape[0])) # train size 0.8

    # Split 수행
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]
    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)