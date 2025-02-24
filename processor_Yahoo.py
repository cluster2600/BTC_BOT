from typing import List
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from datetime import datetime
try:
    import exchange_calendars as tc
except:
    print('Cannot import exchange_calendars.', 
          'If you are using python>=3.7, please install it.')
    import trading_calendars as tc
    print('Use trading_calendars instead for yahoofinance processor..')
from processor_Base import _Base

class Yahoofinance(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    def download_data(self, ticker_list: List[str]):
        self.dataframe = pd.DataFrame()
        for tic in ticker_list:
            print_info(f"Downloading data for {tic} from {self.start_date} to {self.end_date} with interval {self.time_interval}")
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date, interval=self.time_interval)
            temp_df["tic"] = tic
            self.dataframe = pd.concat([self.dataframe, temp_df])  # Replaced append with pd.concat
        self.dataframe.reset_index(inplace=True)
        try:
            self.dataframe.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "tic",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        print_info(f"Shape of DataFrame: {self.dataframe.shape}")
        self.dataframe.sort_values(by=['date', 'tic'], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        print_info(f"Downloaded data shape: {self.dataframe.shape}, columns: {self.dataframe.columns.tolist()}")
        return self.dataframe

    # Rest of the methods unchanged...