# -*- coding: utf-8 -*-

import os
import sys
from queue import Queue
from typing import Dict
from dateutil import tz
import datetime as dt
from system.database.fre_database import FREDatabase
import numpy as np
import pandas as pd
import csv
from system.market_data.fre_market_data import EODMarketData
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal

sys.path.append('../')

database = FREDatabase()
eod_market_data = EODMarketData(os.environ.get("EOD_API_KEY"), database)

# Stock Info class based on Bollinger Bands Trading Strategy
class BollingerBandsStocksInfo:    
    def __init__(self, ticker, h=20, k1=2, notional=10000,
                 price_queue=Queue(int(20 / 5)), ):
        self.Ticker = ticker
        self.H = h
        self.K1 = k1
        self.Notional = notional
        self.price_queue = price_queue
        self.Std = "null"
        self.MA = "null"
        self.position = 0
        self.Qty = 0
        self.current_price_buy = 0
        self.current_price_sell = 1e6
        self.Tradelist = []
        self.PnLlist = []
        self.PnL = 0

class BBDmodelStockSelector:
    # Initialize A Stock Info Dictionary
    @staticmethod
    def bollingerbands_stkinfo_init(stock_list) -> Dict[str, BollingerBandsStocksInfo]:
        stock_info_dict = {stk: BollingerBandsStocksInfo(stk) for stk in stock_list}
        return stock_info_dict
    
    # @staticmethod
    # def EDTtoUnixTime(EDTdatetime):
    #     utcTime = EDTdatetime.replace(tzinfo = tz.gettz('EDT')).astimezone(tz=datetime.timezone.utc)
    #     unixTime = utcTime.timestamp()
    #     return str(int(unixTime))
    
    # @staticmethod
    # def get_sp500_component(number_of_stocks=16):
    #     select_st = "SELECT symbol FROM sp500;"
    #     result_df = database.execute_sql_statement(select_st)
    #     print(result_df.symbol.values)
    #     randomIndex = np.random.randint(0, len(result_df.symbol.values), (number_of_stocks,)).tolist()
    #     print(randomIndex)
    #     with open('system/csv/server_symbols.csv', 'w') as f:
    #         write = csv.writer(f)
    #         write.writerow(result_df.symbol.values[randomIndex]) 
    #     return result_df.symbol.values[randomIndex]
    
    # @staticmethod
    # def get_selected_stock_list():
    #     sp500_symbol_list = BBDmodelStockSelector.get_sp500_component()
    #     selected_stk, stk_df = BBDmodelStockSelector.select_highvol_stock(sp500_symbol_list)
    #     return selected_stk, stk_df
    
    @staticmethod
    def select_highvol_stock(end_date=None, stock_list=None, interval='1m', number_of_stocks=2, lookback_window=14):       
        std_resultdf = pd.DataFrame(index=stock_list)
        std_resultdf['std'] = 0.0
        for stk in stock_list:
            try:
                start_date = end_date + dt.timedelta(-lookback_window)
                print(start_date, end_date)
                start_time = int(start_date.replace(tzinfo=dt.timezone.utc).timestamp())
                end_time = int(end_date.replace(tzinfo=dt.timezone.utc).timestamp())
                print('good1')
                stk_data = pd.DataFrame(eod_market_data.get_intraday_data(stk, start_time, end_time))
                std = stk_data.close.pct_change().shift(-1).std()
                std_resultdf.loc[stk,'std'] = std
                print('Volatility of return over stock: ' + stk + ' is: ' + str(std))
            except:
                print('Cannot get data of Stock:' + stk)
        stock_selected = list(std_resultdf['std'].sort_values().index[-number_of_stocks:])
        print('selected stock list:', stock_selected)
        selected_df = std_resultdf.loc[stock_selected]
        return stock_selected, selected_df


class BBDmodelTrainer:
    stockdf = None
    
    @classmethod
    def build_trading_model(cls, stk_list=None, start_date=None):
        if not stk_list:
            print('stk_list_empty')
            last_bday = dt.datetime.today()
            nyse = mcal.get_calendar('NYSE')
            start_bday = last_bday + dt.timedelta(-29)
            train_end_date = (nyse.schedule(start_date=start_bday, end_date=last_bday).index[0] - dt.timedelta(1)).date()
            symbols = pd.read_csv('system/csv/server_symbols.csv')
            tickers = pd.concat([symbols["Ticker1"], symbols["Ticker2"]], ignore_index=True)
            server_stock = tickers.drop_duplicates(keep='first').tolist()
            stk_list, _ = BBDmodelStockSelector.select_highvol_stock(train_end_date, server_stock) 
        H_list = [40,50,60,70,80,90]
        K1_list = [1.5,1.8,2.0,2.2,2.5]
        cls.stockdf = cls.train_params_DBBD(stk_list, H_list, K1_list, start_bday, period='14')
        
        return cls.stockdf
    
    @classmethod
    def train_params_DBBD(cls, stk_list, H_list, K1_list, train_end_date, period='14'):
        train_start_date = train_end_date - dt.timedelta(days=int(period))
        train_start = dt.datetime(train_start_date.year,train_start_date.month,train_start_date.day,9,30)
        
        #!TODO: NEED TO CORRECTLY SET THE TRAIN START & END TIME IN ORDER TO DOWNLOAD INTRADAY DATA 
        train_start_time = int(train_start_date.replace(tzinfo=dt.timezone.utc).timestamp())
        train_end_time = int(train_end_date.replace(tzinfo=dt.timezone.utc).timestamp())
        #TEMPORARY TRAIN TIME
        
        mkt_opentime = dt.datetime.strptime('09:30','%H:%M').time()
        mkt_closetime = dt.datetime.strptime('16:00','%H:%M').time()
        print(mkt_closetime)
        stocks = pd.DataFrame(stk_list,columns=['Ticker'])
        stocks["H"] = 0
        stocks["K1"] = 0.0
        stocks['Notional'] = 1000000.00 / 10
        stocks["Profit_Loss_in_Training"] = 0.0
        stocks['Return_in_Training'] = 0.0
        stocks["Profit_Loss"] = 0.0
        stocks['Return'] = 0.0
        for stk in stk_list:
            print("Training params for: " + stk +' ...')
            train_data = pd.DataFrame(eod_market_data.get_intraday_data(stk, train_start_time, train_end_time))
            ### Convert UTC to EST
            train_data.datetime = pd.to_datetime(train_data.datetime) - dt.timedelta(hours=5)
            ### Select during Trading Hour and within selected period
            # print(train_data)
            # print(train_data.datetime.dt.date)
            # print('train end date', train_end_date.date())
            # print('train start date', train_start_date.date())
            train_data = train_data[(train_data.datetime.dt.time>=mkt_opentime) & (train_data.datetime.dt.time<=mkt_closetime)]
            train_data = train_data[(train_data.datetime.dt.date>=train_start_date.date()) & (train_data.datetime.dt.date<train_end_date.date())]
            IR_df = pd.DataFrame(index=H_list,columns=K1_list)
            CumPnLdf = pd.DataFrame(index=H_list,columns=K1_list)
            try:
                print(stocks)
                for H in H_list:
                    for K1 in K1_list:
                        IR, CumPnL = cls.GridSearchinDBBD(train_data,H,K1)
                        IR_df.loc[H,K1] = IR
                        CumPnLdf.loc[H,K1] = CumPnL
                        print(stk + ':H,K pair:(' + str(H) + ',' + str(K1) + ')done, with CumPnL:' + str(CumPnL))
                ### select the pair from IR
                H0 = CumPnLdf.mean(axis=1).idxmax()
                K10 = CumPnLdf.mean().idxmax()
                ### delete those with negative PnL within training period
                if CumPnLdf.loc[H0,K10] <= 0:
                    print('Training performance bad, delete stk:{}.'.format(stk))
                    stocks = stocks.drop(stocks[stocks.Ticker==stk].index)
                else:
                    stocks.loc[stocks[stocks.Ticker==stk].index,'H'] = H0
                    stocks.loc[stocks[stocks.Ticker==stk].index,'K1'] = K10
                    stocks.loc[stocks[stocks.Ticker==stk].index,'Profit_Loss_in_Training'] = CumPnLdf.loc[H0,K10]
                    stocks.loc[stocks[stocks.Ticker==stk].index,'Return_in_Training'] = CumPnLdf.loc[H0,K10] * 10 / 1000000.00
            except:
                print("Deleted. Missing data for stk: " + stk)
                stocks = stocks.drop(stocks[stocks.Ticker==stk].index)
        return stocks
    
    @classmethod
    def GridSearchinDBBD(cls, stkdata,H,K1):
        data = stkdata.copy()
        Notional = 1000000.00 / 10
        data['SMA'] = data['close'].rolling(H).mean()
        data['rstd'] = data['close'].rolling(H).std()
        data['Up1'] = data['SMA'] + K1 * data['rstd']
        data['Down1'] = data['SMA'] - K1 * data['rstd']
        
        ### signals
        data['signal'] = 0
        data.loc[data['close'] >= data['Up1'],'signal'] = -1
        data.loc[(data['close'] < data['Up1']) & (data['close'] > data['Down1']),'signal'] = 0
        data.loc[data['close'] <= data['Down1'],'signal'] = 1
        data.signal = data.signal.shift().fillna(0)
        data['trade'] = data.signal.diff()
        
        ### PnL cal
        data['pre_trade_pos'] = 0
        data['target_pos'] = 0
        data['realized_pnl_d'] = 0
        last_target_pos = 0
        for index, row in data.iterrows():
            data.loc[index,'pre_trade_pos'] = last_target_pos
            if row.trade != 0:
                data.loc[index,'target_pos'] = row.signal * int(Notional / row['close'])
                last_target_pos = row.signal * int(Notional / row['close'])
            
            if abs(row.signal) < abs(row.trade):
                if row.trade < 0:
                    data.loc[index,'realized_pnl_d'] = data.loc[index,'pre_trade_pos'] * row['close'] - Notional 
                else:
                    data.loc[index,'realized_pnl_d'] = Notional + data.loc[index,'pre_trade_pos'] * row['close']
        
        data['realized_pnl_p'] = data['realized_pnl_d'] / Notional
        data['cum_pnl_p'] = data['realized_pnl_p'].cumsum() + 1
        data['cum_pnl_p_diff'] = data['cum_pnl_p'].diff()
        
        ### IR calculation
        data.index = data.datetime
        daily_return = pd.DataFrame(data.realized_pnl_p).copy()
        daily_return['date'] = daily_return.index
        daily_return['date'] = daily_return['date'].apply(lambda x:x.date())
        df_whole = daily_return.groupby(['date']).sum()
        # performance strat
        cumulative_return = df_whole.cumsum()
        Annualized_return = cumulative_return.iloc[-1,:]* 252 / len(df_whole)
        Annualized_vol = df_whole.std() * np.sqrt(252)
        Information_ratio = (Annualized_return / Annualized_vol)[0]
        CumPnL = cumulative_return.iloc[-1,:][0] * Notional
        return Information_ratio, CumPnL
    
