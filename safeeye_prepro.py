#================================================================================================
#--------------------------------------   데이터 전처리 API    --------------------------------------
#------------------------------------------------------------------------------------------------
#------- By : SafeEye (이광준, 임헌규)
#------- Last Updated : 2022.06.01
#================================================================================================

import pandas as pd
import numpy  as np


#------------------------------------------------------------------------------------------------
#-------------------------------- FFT (고속 프리에 변환) ------------------------------
#  파라미터 :  
#              df : Data Frame
#              col_name : FFT 대상 Column 이름
#               * 한번에 한 Column에 대해서만 가능함
#  처리결과 : spectrum, freq
#             spectrum : 지정된 컬럼 col_name의 값에 대한 FFT 실행 결과
#             freq : spectrum에 대한 Frequency
#------------------------------------------------------------------------------------------------
def call_FFT(df, col_name, win_size=0) :
#------------------------------------------------------------------------------------------------
    _signal = df[col_name].to_numpy()
    if win_size == 0 :
        win_size = len(_signal) 
    
    n = len(_signal) 
    k = np.arange(n)
    Fs = 1/0.001
    T = n/Fs
    freq = k/T 
    freq = freq[range(int(n/2))]

    spectrum = np.fft.fft(_signal, win_size)/n 
    spectrum = abs(spectrum[range(int(n/2))])

    return spectrum, freq


#------------------------------------------------------------------------------------------------
#-------------------------------- Moving Average (이동평균값 추정) ------------------------------
#  파라미터 :  
#              df : Data Frame
#              col_name : 이동평균값을 구할 대상 Column
#              win_size : Rolling 이동평균값을 구하기 위한 Window Size (이전 Data 개수) -- SMA에서만 이용
#              datetime_col : Datetime 컬럼 명 (이동평균을 구할 주기를 판단하는 기준)
#              type_ma : SMA or EMA or CMA
#                         * SMA : Simple Moving Average --> 이전 K 개의 Data에 대한 단순 평균
#                         * CMA : Cumulative Moving Average --> 모든 이전 Data에 대한 평균
#                         * EMA : Exponential Moving Average --> 가까운 Data에 Weight를 크게 부여
#  처리결과 : df_ma --> DataFrame
#             win_size - 1만큼의 데이터 수 감소
#             Return Type : Dataframe
#------------------------------------------------------------------------------------------------
def call_moving_average(df, col_name, win_size=30, datetime_col=None, type_ma='sma') :
#------------------------------------------------------------------------------------------------
    if datetime_col != None :
        df_ma = df[datetime_col].to_frame()
        df_ma[col_name] = df[col_name]
    else :
        df_ma = df[col_name].to_frame()
    
    if type_ma == 'sma' or type_ma == 'SMA' :                 # Simple Moving Average 구하기
        df_ma[f"ma_{col_name}"]  = df_ma[col_name].rolling(win_size).mean()

    if type_ma == 'ema' or type_ma == 'EMA' :                 # Cumulative Moving Average 구하기
        df_ma[f"ma_{col_name}"]  = df_ma[col_name].ewm(span=win_size).mean()
    
    if type_ma == 'cma' or type_ma == 'CMA' :                 # Cumulative Moving Average 구하기
        df_ma[f"ma_{col_name}"]  = df_ma[col_name].expanding().mean()
    
    df_ma.dropna(inplace=True)
    
    return df_ma
    

#------------------------------------------------------------------------------------------------
#-------------------------------- Variance (분산값 추정) ------------------------------
#  파라미터 :  
#              df : Data Frame
#              col_name : 분산값 추정 대상 column 이름
#              skipna : None 데이터 Skip 여부
# 
#  처리결과 : variance
#             분산 값
#             Return Type : float
#------------------------------------------------------------------------------------------------
def call_variance(df, col_name, _skipna=True) :
#------------------------------------------------------------------------------------------------
    _var = df[col_name].var(skipna=_skipna)
    # _var = df[col_name].var(ddof=_ddof)  # ddof : Delta Degree of Freedom
    return _var
    
#------------------------------------------------------------------------------------------------
#-------------------------------- Histogram (Histogram) ------------------------------
#  파라미터 :  
#              df : Data Frame
#              col_name : 분산값 추정 대상 column 이름
#              bins : the Number of Buckets (X-축 값의 종류)
#              figsize : None --> 그림을 그리지 않음, (16, 4) 그림을 그림 --> Jupyter Notebook에서만
# 
#  처리결과 : df_hist   --> Dataframe ['bucket', 'count']
#------------------------------------------------------------------------------------------------
def call_histogram(df, col_name, bins=20, figsize=None) :
#------------------------------------------------------------------------------------------------
    if figsize :
        hist=df.hist(column=col_name, bins=bins, grid=False, figsize=figsize)
    
    col_arr = df[col_name].to_numpy()
    count, division = np.histogram(col_arr, bins=bins)
    df_hist = pd.DataFrame()
    df_hist['bucket'] = division[1:]
    df_hist['count'] = count
    return df_hist
 