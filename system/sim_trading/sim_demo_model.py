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

sys.path.append('../')

database = FREDatabase()
eod_market_data = EODMarketData(os.environ.get("EOD_API_KEY"), database)

# Stock Info class based on Bollinger Bands Trading Strategy
class BollingerBandsStocksInfo:
    def __init__(self, ticker, h=20, k1=2, notional=0,
                 price_queue=Queue(int(20 / 5))):
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