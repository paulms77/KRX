#!/usr/bin/env python
# coding: utf-8

# In[4]:


import kquant as kq
import numpy as np
import pandas as pd
import multiprocessing
import datetime as dt
from datetime import datetime
import logging
import joblib
import re
import os
import math
import time
import pkg_resources
from typing import Tuple

from talib import abstract
import talib as ta
import pandas_ta as pta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from .my_package.fetch_data import fetch_data
from .my_package.metrics import MAPE, MAE, RMSE
from .my_package.preprocessing import get_technical_indicators, train_test_split
from .my_package.model import Stacked_VAE, BiLSTM
from .my_package.train import train_vae, train_bilstm, bilstm_inference

def trade_func(
    date: dt.date,
    dict_df_result: dict[str, pd.DataFrame],
    dict_df_position: dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> list[tuple[str, int]]:
    r"""주식매매 지시함수

    주식매매 지시함수에 대한 설명
    
    :param dt.date date: 매매일 날짜
    :param dict[str, pd.DataFrame] dict_df_result: 매매일까지의 주문 및 체결 정보
    :param dict[str, pd.DataFrame] dict_df_position: 매매일의 주식 보유 정보
    :param logging.Logger logger: 로거
    :return list[tuple[str, int]]: 주식매매 지시
    """
    if (date == parse('20231004').date()) | (date == parse('20231004')) | (date == dt.date(2023, 10, 4)) | (date == '2023-10-04') | (date == '20231004'):
        try:
            # (2022-09-27 ~ 2023-09-27) 1년치 종목들의 데이터를 불러오기 위한 설정 값
            load_date = '2023-09-27'
            load_date = str(load_date).replace('-', '')
            load_date = dt.datetime.strptime(load_date, "%Y%m%d")
            end_dt = load_date
            end_date = str(end_dt).split(' ')[0].replace('-', '')
            start_dt = load_date - relativedelta(years = 1)
            start_date = str(start_dt).split(' ')[0].replace('-', '')
            print(start_date, end_date)
            
            max_requests_per_minute = 20 # 데이터를 batch size 만큼 불러오기 위해 설정한 값
    
            # 코스닥 종목 리스트 불러오기
            df_symbols = kq.symbol_stock()
            kosdaq_symbols_list = list(df_symbols[df_symbols.MARKET == '코스닥'].SYMBOL.unique())
            
            start = time.time()
            # 병렬처리를 이용한 종목의 일간정보 데이터 가져오기
            pool = multiprocessing.Pool()
            
            symbols_args = [(symbol_id, start_date, end_date) for symbol_id in kosdaq_symbols_list]
            
            dfs = []
            for args_batch in [symbols_args[i:i+max_requests_per_minute] for i in range(0, len(symbols_args), max_requests_per_minute)]:
                batch_dfs = pool.map(fetch_data, args_batch)
                dfs.extend(batch_dfs)
            
            pool.close()
            pool.join()
            
            print('일간정보 병렬처리 작업시간: ', time.time() - start)
           
            # 종목 정보들을 하나로 통합하고 'DATE', 'SYMBOL'을 기준으로 오름차순 정렬
            dfs = [df for df in dfs if df is not None]
            info_dfs = pd.concat(dfs, axis = 0)
            info_dfs = info_dfs.sort_values(['DATE', 'SYMBOL'], ascending = True)
            
            # VOLUME이 0이라는 것은 즉 거래가 없는 종목이므로 투자 종목 리스트에 포함하지 않음
            zero_volume_symbols = []
            for symbol_id in info_dfs.SYMBOL.unique().tolist():
                symbol_temp = info_dfs[info_dfs.SYMBOL == symbol_id]
                if symbol_temp.VOLUME.sum() == 0:
                    zero_volume_symbols.append(symbol_id)
                    info_dfs = info_dfs[info_dfs.SYMBOL != symbol_id]
            
            info_dfs[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']] = info_dfs[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']].astype(float) # float형 변환

            # 거래량이 상위 종목 3위부터 199위 데이터를 추출
            sorted_df = info_dfs[['SYMBOL', 'VOLUME']].sort_values(by = 'VOLUME', ascending = False)
            symbols_volume = sorted_df.SYMBOL.unique()[3:200].tolist()
            filtered_info_dfs = info_dfs[info_dfs.SYMBOL.isin(symbols_volume)]
        
        except Exception as e:
            package_dir = os.path.dirname(__file__)
            try:
                parquet_file_path = os.path.join(package_dir, 'my_data/filtered_info_dfs.parquet')
            except Exception as e:
                parquet_file_path = pkg_resources.resource_filename('krx_competition_28', 'my_data/filtered_info_dfs.parquet')
                
            info_dfs = pd.read_parquet(parquet_file_path)
    
            # 거래량이 상위 종목 3위부터 199위 데이터를 추출
            sorted_df = info_dfs[['SYMBOL', 'VOLUME']].sort_values(by = 'VOLUME', ascending = False)
            symbols_volume = sorted_df.SYMBOL.unique()[3:200].tolist()
            filtered_info_dfs = info_dfs[info_dfs.SYMBOL.isin(symbols_volume)]

        # ***************************** #
        #  3. 기술 분석 및 추가 피처 엔지니어링 #
        # ***************************** #
        
        start = time.time()
        merge_info_dfs = pd.concat([get_technical_indicators(group, symbol_code, -1) for symbol_code, group in filtered_info_dfs.groupby('SYMBOL')], axis=0)
        print('기술분석 작업시간: ', time.time() - start)

        # 'CHG_TYPE_NOTADJ'열 int형 변환
        merge_info_dfs['CHG_TYPE_NOTADJ'] = merge_info_dfs['CHG_TYPE_NOTADJ'].astype(int)
        # 학습에 불필요한 'DATE', 'EX_EVENT'열 제거
        merge_info_dfs.drop(['DATE', 'EX_EVENT'], axis = 1, inplace = True)

        # ************ #
        #  4. 기능 공학   #
        # ************ #
        
        start = time.time()
        final_dfs = merge_info_dfs
        
        n_epoch = 200
        batch_size = 64
        learning_rate = 0.001
        
        n_hidden = 64
        n_latent = 10
        n_layers = 2

        main_dir = os.path.dirname(__file__)
        
        #save_dir = 'my_path/xgb'
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)
        
        # 4-1. 기능 중요도 (XGBoost)
        stacked_symbol_id = []
        stacked_vae_filter_data = []
        zero_divisions = []
        for symbol_id in final_dfs['SYMBOL'].unique().tolist():
            print('-'*20)
            print(f'symbol_id: {symbol_id}')
            print('-'*20)
        
            final_temp = final_dfs[final_dfs['SYMBOL'] == symbol_id]
        
            print('Before feature importances', final_temp.shape)
        
            num_training_days = int(np.ceil(0.8 * final_temp.shape[0]))
            
            if final_temp[num_training_days:].shape[0] == 0:
                print(f'{symbol_id} is Small Data')
                zero_divisions.append(symbol_id)
                continue
        
            columns_list = final_temp.drop(['SYMBOL'], axis = 1).columns.tolist()
        
            scaler = StandardScaler()
            scaler = scaler.fit(final_temp.drop(['SYMBOL', 'CLOSE'], axis = 1))
            final_temp_scaled = scaler.transform(final_temp.drop(['SYMBOL', 'CLOSE'], axis = 1))
            print(final_temp_scaled.shape, final_temp['CLOSE'].shape)
            final_temp_scaled = np.hstack((final_temp_scaled, np.array(final_temp['CLOSE']).reshape(-1, 1)))
            final_temp_scaled_df = pd.DataFrame(final_temp_scaled, columns = columns_list)
        
            temp = final_dfs[final_dfs['SYMBOL'] == symbol_id]
            temp = temp.drop(['SYMBOL'], axis = 1)
        
            (X_train_fi, y_train_fi), (X_test_fi, y_test_fi) = train_test_split(temp, 'CLOSE')
            #xgbregressor = xgb.XGBRegressor()
            #Training (학습)
            #xgbregressor_fit = xgbregressor.fit(X_train_fi, y_train_fi, eval_set = [(X_train_fi, y_train_fi), (X_test_fi, y_test_fi)], verbose = False)
            
            xgb_path = f'my_path/xgb/{symbol_id}_xgb.joblib'
        
            # XGBoost 학습 모델 불러오기
            xgbregressor = joblib.load(os.path.join(main_dir, xgb_path))
        
            fi = X_train_fi.columns[np.array(xgbregressor.feature_importances_) > 0.0].tolist()
            print('len:', len(fi))
            print(fi)
            fi.append('CLOSE')
        
            temp = final_dfs[final_dfs['SYMBOL'] == symbol_id]
            temp = temp[list(set(fi))]
            print('after feature importances', temp.shape)
        
            if temp.shape[1] < 2:
                print(f'{symbol_id} is zero')
                stacked_vae_filter_data.append(torch.Tensor(temp['CLOSE'].values))
                continue
        
            scaler = MinMaxScaler(feature_range = (0, 1))
            scaler = scaler.fit(temp.drop(['CLOSE'], axis = 1))
            temp_scaled = scaler.transform(temp.drop(['CLOSE'], axis = 1))
        
            num_training_days = int(np.ceil(0.8 * temp.shape[0]))
                
            feats_temp = temp_scaled[:, :]
            feats_train = temp_scaled[:num_training_days, :]
            feats_test = temp_scaled[num_training_days:, :]
            
            data_close = torch.tensor(temp['CLOSE'].values)
        
            # 4-2. 상위 수준 기능 추출 (Stacked AutoEncoder)
            feats_train_tensor = torch.tensor(feats_train, dtype = torch.float32)
            feats_test_tensor = torch.tensor(feats_test, dtype = torch.float32)
        
            train_loader = DataLoader(feats_train_tensor,  batch_size = batch_size, shuffle = False)
            test_loader = DataLoader(feats_test_tensor, batch_size = batch_size, shuffle = False)
        
            if feats_train.shape[1] > 64:
                n_hidden = 64
            else:
                n_hidden = 32

            # train and eval = True, eval = False
            model = train_vae(train_loader, test_loader, feats_train, symbol_id, n_hidden, n_latent, n_layers, learning_rate, n_epoch, main_dir, False)
        
            feats_data = torch.tensor(feats_temp, dtype = torch.float32)
            mu, lv = model.encoded(feats_data)
            z = model.reparameterize(mu, lv)
            decoded_data = model.decoded(z)
            print(decoded_data.shape)
            
            decoded_data = torch.cat((torch.tensor(scaler.inverse_transform(decoded_data.detach().numpy())), data_close.view(data_close.shape[0], 1)), dim = 1)
            print(decoded_data.shape)
            stacked_symbol_id.append(symbol_id)
            stacked_vae_filter_data.append(decoded_data)
        
        stacked_vae_filter_data = {symbol_id:torch.tensor(data) for symbol_id, data in zip(stacked_symbol_id, stacked_vae_filter_data)}
        
        print('기능 중요도 및 상위 수준 기능 추출 작업시간: ', time.time() - start)
        
        print(torch.cuda.is_available())
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        
        start = time.time()
        
        # ****************************** #
        # 5. bi-LSTM (양방향 장단기 기억 장치) #
        # ****************************** #
        
        prediction_window = 30 # 예측할 길이
        window_size = 60 # 윈도우 사이즈
        
        num_epochs = 50
        learning_rate = 0.001
        batch_size = 64
        
        hidden_size = 64
        n_layers = 3
        dropout = 0.1
        
        model_predictions = {}
        returns_dailys = {}
        zero_sequences = []
        
        for symbol_id in stacked_vae_filter_data.keys():
            print('-'*20)
            print(f'symbol_id: {symbol_id}')
            print('-'*20)
            temp = stacked_vae_filter_data[symbol_id]
            last_return = temp[-1][-1]
        
            sequences = []
            targets = []
            # 주가에 도움이 되는 정보가 없는 경우
            if len(temp.shape) < 2:
                print(f'{symbol_id} is Small Data')
                scaler = MinMaxScaler(feature_range = (0, 1))
                temp = scaler.fit_transform(temp.detach().numpy().reshape(-1, 1))
                temp = torch.tensor(temp, dtype = torch.float32)
                temp = temp.squeeze()
        
                for i in range(len(temp) - window_size - prediction_window + 1):
                    sequences.append(temp[i: i+window_size])
                    targets.append(temp[i+window_size: i+window_size+prediction_window])
                    
                input_size = 1
        
                if len(sequences) == 0:
                    zero_sequences.append(symbol_id)
                    continue
                
                sequences = torch.stack(sequences)
                targets = torch.stack(targets)
        
                print('sequences shape: ', sequences.shape)
                print('targets shape: ', targets.shape)
        
                train_dataset = TensorDataset(sequences, targets)
                train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
            
                output_size = prediction_window

                # Training (학습)
                # model = train_bilstm(train_loader, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, num_epochs, learning_rate, device, True)

                # Inference (추론)
                train_outputs = bilstm_inference(sequences, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, window_size, device, main_dir, True)
        
                train_outputs = scaler.inverse_transform(train_outputs.cpu().numpy().reshape(-1, 1))
                model_prediction = train_outputs.reshape(-1, output_size)[-1]
                model_prediction = np.array(model_prediction)

                # 30일 동안 총 예상 수익률 계산
                returns_daily = (model_prediction[-1] - last_return) / last_return
                
                model_predictions[symbol_id] = model_prediction
                returns_dailys[symbol_id] = returns_daily
        
            else:
                feats_temp = temp[:, :-1].detach().numpy()
        
                # 4-3. scaling (5번 단계에서 진행)
                feats_temp_scaled = scaler.fit_transform(feats_temp)
        
                # 4-3. pca (5번 단계에서 진행)
                pca = PCA(n_components = .8)
                feats_temp_pca = pca.fit_transform(feats_temp_scaled)
                print('pca n_components_', feats_temp_pca.shape[1])
        
                feats_temp = torch.tensor(feats_temp_pca, dtype = torch.float32)
                temp = torch.tensor(temp, dtype = torch.float32)
        
                for i in range(len(temp) - window_size - prediction_window + 1):
                    sequences.append(feats_temp[i: i+window_size, :])
                    targets.append(temp[i+window_size: i+window_size+prediction_window, -1])
        
                input_size = feats_temp.shape[1]
        
                if len(sequences) == 0:
                    zero_sequences.append(symbol_id)
                    continue
                
                sequences = torch.stack(sequences)
                targets = torch.stack(targets)

                scaler = MinMaxScaler(feature_range = (0, 1))
                normalized_targets = scaler.fit_transform(targets.reshape(-1, 1))
                normalized_targets = normalized_targets.reshape(-1, prediction_window)
                normalized_targets = torch.tensor(normalized_targets, dtype = torch.float32)
        
                print('sequences shape: ', sequences.shape)
                print('targets shape: ', targets.shape)
        
                train_dataset = TensorDataset(sequences, normalized_targets)
                train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
        
                output_size = prediction_window

                # Training (학습)
                # model = train_bilstm(train_loader, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, num_epochs, learning_rate, device, False)

                # Inference (추론)
                train_outputs = bilstm_inference(sequences, symbol_id, input_size, hidden_size, n_layers, dropout, output_size, window_size, device, main_dir, False)
        
                train_outputs = scaler.inverse_transform(train_outputs.cpu().numpy().reshape(-1, 1))
                model_prediction = train_outputs.reshape(-1, output_size)[-1]
                model_prediciton = np.array(model_prediction)

                # 30일 동안 총 예상 수익률 계산
                returns_daily = (model_prediction[-1] - last_return) / last_return
        
                model_predictions[symbol_id] = model_prediction
                returns_dailys[symbol_id] = returns_daily
        
        print('bi-LSTM 작업시간: ', time.time() - start)

        # AI 알고리즘을 통한 추천 종목 포트폴리오
        except_zere_symbols = set(final_dfs.SYMBOL.unique()).difference(set(zero_divisions), set(zero_sequences))
        symbol_id_list = list(except_zere_symbols)
        results_data = []
        for idx, symbol_id in enumerate(symbol_id_list):
            try:
                results_data.append({'SYMBOL': symbol_id, 'FINAL_RETURN': np.array(returns_dailys[symbol_id], dtype = float)})
            except Exception as e:
                print(f'Symbol {symbol_id} Error {str(e)} is not exist')
                continue
        
        results_df = pd.DataFrame(results_data, columns = ['SYMBOL', 'FINAL_RETURN'])
        
        results_df['RANK'] = results_df['FINAL_RETURN'].rank(method = 'first', ascending = False).astype(int)
        results_df = results_df.sort_values(by = 'RANK', ascending = True)

        # 구매 의향 종목 목록
        buy_symbols_rank = results_df.SYMBOL.head(3).tolist() # AI 추천 종목 상위 3개
        print(buy_symbols_rank)
        
        sorted_df = info_dfs[['SYMBOL', 'VOLUME']].sort_values(by = 'VOLUME', ascending = False)
        buy_symbols_volume = sorted_df.SYMBOL.unique()[:3].tolist() # 우량주 종목 상위 3개
        print(buy_symbols_volume)

    else:
        # 첫 거래일에 추출한 거래량이 높은 우량주 종목 상위 3개
        buy_symbols_volume = ['096040', '024810', '053590']

    # ***************** #
    # 6. 주식 매매 알고리즘 #
    # ***************** #

    # 초기금액 (10억원) 및 보유금액 정의
    try:
        init_cash = float(dict_df_result['TOTAL'].CASH) # 보유금액
        #print(f'{date} 보유금액: ', init_cash)
    except Exception as e:
        init_cash = 1000000000 # 초기금액 (10억원)

    # 첫 거래일인 경우
    if (date == parse('20231004').date()) | (date == parse('20231004')) | (date == dt.date(2023, 10, 4)) | (date == '2023-10-04') | (date == '20231004'):
        symbols_and_orders = []

        # AI 추천 종목
        order_count = 20000
        for buy_symbol_id in buy_symbols_rank:
            try:
                buy_symbol_info_now = kq.daily_stock(buy_symbol_id, date, date)
            except:
                continue
            trading_amount = float(buy_symbol_info_now.CLOSE) * order_count
            trading_charge = float(buy_symbol_info_now.CLOSE) * order_count * 0.001
            trading_cash = trading_amount + trading_charge
            if trading_cash < init_cash:
                init_cash = init_cash - trading_cash
                buy_symbol_count = (buy_symbol_id, order_count)
                symbols_and_orders.append(buy_symbol_count)
            else:
                pass

        # 우량주
        order_count = 20000
        for buy_symbol_id in buy_symbols_volume:
            try:
                buy_symbol_info_now = kq.daily_stock(buy_symbol_id, date, date)
            except:
                continue 

            trading_amount = float(buy_symbol_info_now.CLOSE) * order_count
            trading_charge = float(buy_symbol_info_now.CLOSE) * order_count * 0.001
            trading_cash = trading_amount + trading_charge
            if trading_cash < init_cash:
                init_cash = init_cash - trading_cash
                buy_symbol_count = (buy_symbol_id, order_count)
                symbols_and_orders.append(buy_symbol_count)
            else:
                pass
        
        return symbols_and_orders

    # 첫 거래일이 아닌 경우
    else:
        symbols_and_orders = []

        # 마지막 거래일인 경우
        if (date == parse('20231031').date()) | (date == parse('20231031')) | (date == dt.date(2023, 10, 31)) | (date == '2023-10-31') | (date == '20231031'):
            for symbol_id in dict_df_position.keys():
                sell_count = -dict_df_position[symbol_id].QTY.sum()
                sell_symbol_count = (symbol_id, sell_count)
                symbols_and_orders.append(sell_symbol_count)
            return symbols_and_orders

        else:
            # 보유 주식이 아예 없는 경우 (당장은 투자 시기가 아님으로 판단, 매매 로직 작동 x)
            if len(dict_df_position) == 0:
                return symbols_and_orders

            # 보유 주식이 하나라도 존재하는 경우
            else:
                not_give_symbols = []
                give_symbols = []
                total_symbols = list(dict_df_result.keys())
                symbols_without_total = [symbol for symbol in total_symbols if symbol != 'TOTAL']
                for buy_symbol_id in symbols_without_total:
                    if len(dict_df_position[buy_symbol_id]) == 0: # 해당 주식을 보유하지 않았을 경우
                        not_give_symbols.append(buy_symbol_id)
                    else: # 해당 주식을 보유했을 경우
                        give_symbols.append(buy_symbol_id)

                # 매도 기능
                sells = []
                add_buys = []
                buys = []
                for give_symbol_id in give_symbols:
                    trade_cash = 0
                    trade_order = 0
                    for i in range(len(dict_df_result[give_symbol_id])):
                        price = float(dict_df_result[give_symbol_id].iloc[i].PRICE)
                        order = dict_df_result[give_symbol_id].iloc[i].ORDER
                        cash = price * order - (price * order * 0.001)
                        if order < 0:
                            trade_cash = 0
                            trade_order = 0
                        else:
                            trade_cash += cash
                            trade_order += order
                    
                    buy_symbol_info_now = kq.daily_stock(give_symbol_id, date, date)
                    
                    now_price = float(buy_symbol_info_now.CLOSE)

                    # 공휴일을 제외한 전날 일간정보
                    for i in range(1, 5):
                        try:
                            buy_symbol_info_before = kq.daily_stock(give_symbol_id, date-dt.timedelta(days = i), date-dt.timedelta(days = i))
                            if buy_symbol_info_before is not None:
                                break
                                
                        except Exception as e:
                            continue
        
                    # 액면분할 및 액면병합 이벤트 처리    
                    split_ratio = 1
                    merge_ratio = 1
                    if now_price <= 0.5 * float(buy_symbol_info_before.CLOSE):
                        split_ratio = float(buy_symbol_info_before.CLOSE) / now_price

                    if now_price >= 2.0 * float(buy_symbol_info_before.CLOSE):
                        merge_ratio = now_price / float(buy_symbol_info_before.CLOSE)

                    trade_order = trade_order * split_ratio / merge_ratio
                    trade_cash = trade_cash / split_ratio * merge_ratio
                    now_price = now_price / split_ratio * merge_ratio

                    sell_charge = (now_price * trade_order * 0.001) + (now_price + trade_order * 0.002)
                    expectation_cash = now_price * trade_order - sell_charge # 매도시, 수수료 + 증권거래세

                    # 수익률 계산
                    try:
                        profit_pct = (expectation_cash - trade_cash) / trade_cash * 100
                    except:
                        profit_pct = 0
        
                    # 보유한 종목이 우량주일 경우
                    if give_symbol_id in buy_symbols_volume:
                        if profit_pct > 1.0: # 수익률 1%
                            # 전량매도
                            sell_count = -dict_df_position[give_symbol_id].QTY.sum()
                            sell_symbol_count = (give_symbol_id, sell_count)
                            symbol_total_profit = expectation_cash - trade_cash
                            init_cash = init_cash + symbol_total_profit
                            sells.append(sell_symbol_count)
                        elif profit_pct < -1.0: # 수익률 -1%
                            # 전량매도
                            sell_count = -dict_df_position[give_symbol_id].QTY.sum()
                            sell_symbol_count = (give_symbol_id, sell_count)
                            symbol_total_profit = expectation_cash - trade_cash
                            init_cash = init_cash + symbol_total_profit
                            sells.append(sell_symbol_count)
                        else:
                            # 추가매수
                            give_symbols_cash = sum(float(dict_df_result[give_symbol_id].iloc[i].PRICE) * dict_df_result[give_symbol_id].iloc[i].ORDER for i in range(len(dict_df_result[give_symbol_id])))
                            add_buy_symbol_cash = give_symbols_cash * 0.2
                            buy_symbol_info_now = kq.daily_stock(give_symbol_id, date, date)
                            order_v20 = add_buy_symbol_cash//float(buy_symbol_info_now.CLOSE) # 보유 종목 비중의 20%
                            trading_amount_v20 = float(buy_symbol_info_now.CLOSE) * order_v20
                            trading_charge_v20 = float(buy_symbol_info_now.CLOSE) * order_v20 * 0.001 # 매수시, 수수료
                            trading_cash_v20 = trading_amount_v20 + trading_charge_v20
                            if trading_cash_v20 < init_cash:
                                init_cash = init_cash - trading_cash_v20
                                buy_symbol_count = (give_symbol_id, order_v20)
                                add_buys.append(buy_symbol_count)
                            else:
                                print(f'보유 금액 부족, {buy_symbol_id} 추가 매수 x')
                                continue
                                
                    # 보유종목이 AI 추천 종목인 경우
                    else:
                        if profit_pct > 3.0: # 수익률 3%
                            # 전량매도
                            sell_count = -dict_df_position[give_symbol_id].QTY.sum()
                            sell_symbol_count = (give_symbol_id, sell_count)
                            symbol_total_profit = expectation_cash - trade_cash - (now_price * trade_order * 0.001)
                            init_cash = init_cash + symbol_total_profit
                            sells.append(sell_symbol_count)
                        elif profit_pct < -3.0: # 수익률 -3%
                            # 전량매도
                            sell_count = -dict_df_position[give_symbol_id].QTY.sum()
                            sell_symbol_count = (give_symbol_id, sell_count)
                            symbol_total_profit = expectation_cash - trade_cash - (now_price * trade_order * 0.001)
                            init_cash = init_cash + symbol_total_profit
                            sells.append(sell_symbol_count)
                        else:
                            # 추가매수
                            order20 = 2000
                            buy_symbol_info_now = kq.daily_stock(give_symbol_id, date, date)
                            trading20_amount = float(buy_symbol_info_now.CLOSE) * order20
                            trading20_charge = float(buy_symbol_info_now.CLOSE) * order20 * 0.001
                            trading20_cash = trading20_amount + trading20_charge
                            if trading20_cash < init_cash:
                                init_cash = init_cash - trading20_cash
                                buy_symbol_count = (give_symbol_id, order20)
                                add_buys.append(buy_symbol_count)
                            else:
                                print(f'보유 금액 부족, {buy_symbol_id} 추가 매수 x')
                                continue
                        
                # 매수 기능
                for not_give_symbol_id in not_give_symbols:
                    # 보유하지 않은 종목이 우량주일 경우 (대량 매수)
                    if not_give_symbol_id in buy_symbols_volume:
                        order_count = 20000
                        buy_symbol_info_now = kq.daily_stock(not_give_symbol_id, date, date)
                        trading_amount = float(buy_symbol_info_now.CLOSE) * order_count
                        trading_charge = float(buy_symbol_info_now.CLOSE) * order_count * 0.001
                        trading_cash = trading_amount + trading_charge

                        if trading_cash < init_cash:
                            init_cash = init_cash - trading_cash
                            buy_symbol_count = (not_give_symbol_id, order_count)
                            buys.append(buy_symbol_count)
                        else:
                            print(f'보유 금액 부족, {not_give_symbol_id} 매수 x')
                            continue
                            
                    else:
                        # 보유하지 않은 종목이 AI 추천 종목일 경우 (대량 매수)
                        order_count = 20000
                        buy_symbol_info_now = kq.daily_stock(not_give_symbol_id, date, date)
                        trading_amount = float(buy_symbol_info_now.CLOSE) * order_count
                        trading_charge = float(buy_symbol_info_now.CLOSE) * order_count * 0.001
                        trading_cash = trading_amount + trading_charge

                        if trading_cash < init_cash:
                            init_cash = init_cash - trading_cash
                            buy_symbol_count = (not_give_symbol_id, order_count)
                            buys.append(buy_symbol_count)
                        else:
                            print(f'보유 금액 부족, {not_give_symbol_id} 매수 x')
                            continue

                symbols_and_orders = sells + add_buys + buys
                return symbols_and_orders