from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, SequentialFeatureSelector
from sklearn.decomposition import PCA

import pandas as pd
import os
import pandas_ta as ta
import requests
from stockstats import StockDataFrame, wrap
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import seaborn as sns
from datetime import datetime,timedelta
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler,Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pickle import dump
from requests import HTTPError


from django.conf import settings


best_features= ['supertrend', 'kst', 'coppock', 'ndi', 'ao', 'supertrend_lb', 'macd', 'qqel', 'wt1','leading_spanA', 'aroon', 'wt2', 'pdi', 'supertrend_ub', 'dma', 'qqe', 'cr', 'trix', 'macds','ppos', 'base_line', 'qqes', 'leading_spanB', 'ppo', 'rsi']

def get_data(stock_symbol, start_day=None, end_day=None):
    if end_day is None:
        ts_prev_day = (datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d')
    if start_day is None:
        ts_last_weeek = (datetime.now() + timedelta(days=-7)).strftime('%Y-%m-%d')

    try:
        response = requests.get(fr'https://api.twelvedata.com/time_series?apikey=e5edc2b5e6444ae89c9235070b420a08&interval=15min&order=ASC&previous_close=true&symbol={stock_symbol}&type=stock&adjust=all&start_date={ts_last_weeek}%2000:00:00&end_date={ts_prev_day}%2023:59:00&format=CSV')
    except HTTPError as e:
        raise e
    else:
    
        output_dir = os.path.join (settings.BASE_DIR, 'data')
        output_file_name = f'{output_dir}/{stock_symbol}_{ts_prev_day}.csv'
        with open(output_file_name, 'w') as f:
            f.write(response.text)

    return ts_prev_day

def prepare_df(symbols, file_date=None):
    if file_date is None:
        file_date = (datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d')
    output = pd.DataFrame()
    feature_list = set()
    for s in symbols:
        filename =f'data/{s}_{file_date}.csv'
        dataset = pd.read_csv(filename, sep=';')
        dataset['symbol'] = s
        dataset.set_index( [pd.DatetimeIndex(dataset['datetime']),'symbol'], inplace=True)
        dataset.drop(columns=['datetime'], inplace=True)
        output = pd.concat( [output, dataset],axis=0)
        os.remove(filename)
    return output


def add_technical_indicators(dataset):
    dataset = StockDataFrame(dataset)

    dataset.init_all()


    # trading signals generation using BUY for conversionline>baseline & close > leadingspanA & rsi > 50
    # and SELL for conversionline<baseline & close < leadingspanA & rsi >< 50
    dataset['conversion_line'] = (dataset['high'].rolling(9).max() + dataset['low'].rolling(9).min()) / 2
    dataset['base_line'] = (dataset['high'].rolling(26).max() + dataset['low'].rolling(26).min()) / 2
    dataset['leading_spanA'] = ((dataset['conversion_line'] + dataset['base_line']) / 2).shift(26)
    dataset['leading_spanB'] = ((dataset['high'].rolling(52).max() + dataset['low'].rolling(52).min()) / 2).shift(26)

    dataset['trading_signal'] = 0
    dataset.loc[(dataset.close > dataset.leading_spanA) & (dataset.conversion_line > dataset.base_line) & (
                dataset.rsi > 50), 'trading_signal'] = 1
    dataset.loc[(dataset.close < dataset.leading_spanA) & (dataset.conversion_line < dataset.base_line) & (
            dataset.rsi < 50), 'trading_signal'] = -1
    
    
    ##Perform some cleaning and fill missing values
    dataset.isna().sum()

    dataset.ffill(inplace=True)
    dataset.bfill(inplace=True)
    

    return dataset


def prepare_input(df):
    X = df[best_features]

    std_scaler = StandardScaler()
    std_scaler.fit(X)
    # std_scaler.set_output(transform='pandas')


    X_normalized = std_scaler.fit_transform(X)
    y = df['trading_signal']
    y = y.astype(np.float64)

    '''Set the data input steps and output steps, 
    we use 5 days data to predict 1 day price here, 
    reshape it to (None, input_step, number of features) used for LSTM input'''
    n_steps_in = 5
    n_features = X_normalized.shape[1]
    n_steps_out = 1

    # Get X/y dataset

    X = list()
    y = list()
    yc = list()

    length = len(X_normalized)
    for i in range(0, length, 1):
        X_value = X_normalized[i: i + n_steps_in][:, :]
        
        
        # print(len(X_value))

        if len(X_value) == 5 :
            X.append(X_value)


    return np.array(X)


def get_y_index(dataset, X_train, n_steps_in, n_steps_out):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index

def run_preprocessing(stock_symbol, n_steps_in=5, n_steps_out=1):
    ts_end = get_data(stock_symbol)
    df = prepare_df(symbols=[stock_symbol], file_date='2024-05-01')

    df_tech_ind = add_technical_indicators(df)

    inputs = prepare_input(df_tech_ind)

    return inputs, df_tech_ind

    
