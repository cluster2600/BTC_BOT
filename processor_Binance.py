"""
Reference: https://github.com/AI4Finance-LLC/FinRL

This code defines a class BinanceProcessor which is used to process financial data from the Binance exchange. It
utilizes the Client class from the binance library to interact with the Binance API and the talib library to perform
technical analysis on the data.
"""

import pandas as pd
from datetime import datetime
import numpy as np
from binance.client import Client
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE
import logging

from config_api import *
import datetime as dt
from processor_Yahoo import Yahoofinance
from fracdiff.sklearn import FracdiffStat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

binance_client = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)

class BinanceProcessor:
    def __init__(self):
        """Initialize BinanceProcessor with API credentials and default settings."""
        logger.debug("Initializing BinanceProcessor instance")
        self.end_date = None
        self.start_date = None
        self.tech_indicator_list = None
        self.correlation_threshold = 0.9
        self.binance_api_key = API_KEY_BINANCE
        self.binance_api_secret = API_SECRET_BINANCE
        self.binance_client = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        logger.info("BinanceProcessor initialized with API key and secret")

    def run(self, ticker_list, start_date, end_date, time_interval, technical_indicator_list, if_vix):
        """Run the data processing pipeline with verbose logging."""
        logger.info(f"Starting data processing pipeline for tickers: {ticker_list}")
        logger.info(f"Parameters - Start Date: {start_date}, End Date: {end_date}, Interval: {time_interval}, Indicators: {technical_indicator_list}, Include VIX: {if_vix}")
        
        self.start_date = start_date
        self.end_date = end_date
        logger.debug("Set start_date and end_date instance variables")

        logger.info("Downloading data from Binance...")
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        logger.info("Download completed. Starting data transformation...")

        logger.info("Cleaning data...")
        data = self.clean_data(data)
        logger.debug(f"Cleaned data shape: {data.shape}")

        logger.info("Dropping 'time' column...")
        data = data.drop(columns=['time'])
        logger.debug("Dropped 'time' column")

        logger.info("Converting server timestamps to datetime...")
        data['timestamp'] = self.servertime_to_datetime(data['timestamp'])
        logger.debug("Converted timestamps")

        logger.info("Setting timestamp as index...")
        data = data.set_index('timestamp')
        logger.debug(f"Data indexed by timestamp, shape: {data.shape}")

        logger.info("Adding technical indicators...")
        data = self.add_technical_indicator(data, technical_indicator_list)
        logger.debug(f"Added technical indicators, new columns: {data.columns.tolist()}")

        logger.info("Dropping correlated features...")
        data = self.drop_correlated_features(data)
        logger.debug(f"Features after dropping correlations: {data.columns.tolist()}")

        if if_vix:
            logger.info("Adding VIX data...")
            data = self.add_vix(data)
            logger.debug(f"Added VIX, new columns: {data.columns.tolist()}")

        logger.info("Converting DataFrame to arrays...")
        price_array, tech_array, time_array = self.df_to_array(data, if_vix)
        logger.debug(f"Price array shape: {price_array.shape}, Tech array shape: {tech_array.shape}, Time array length: {len(time_array)}")

        logger.info("Handling NaN values in technical array...")
        tech_nan_positions = np.isnan(tech_array)
        if np.any(tech_nan_positions):
            logger.debug(f"Found {np.sum(tech_nan_positions)} NaN values in tech array")
            tech_array[tech_nan_positions] = 0
            logger.debug("Replaced NaN values with 0")

        # Uncomment if fracdiff is needed
        # logger.info("Applying fractional differentiation to tech array...")
        # tech_array = self.frac_diff_features(tech_array)
        # logger.debug(f"Tech array after fracdiff: shape {tech_array.shape}")

        logger.info("Data processing pipeline completed successfully")
        return data, price_array, tech_array, time_array

    def download_data(self, ticker_list, start_date, end_date, time_interval):
        """Download historical data from Binance with detailed logging."""
        logger.debug(f"Starting data download for tickers: {ticker_list}")
        self.start_time = start_date
        self.end_time = end_date
        self.interval = time_interval
        self.ticker_list = ticker_list

        final_df = pd.DataFrame()
        for i in ticker_list:
            logger.info(f"Fetching historical data for {i} from {start_date} to {end_date} with interval {time_interval}")
            hist_data = self.get_binance_bars(self.start_time, self.end_time, self.interval, symbol=i)
            logger.debug(f"Downloaded {len(hist_data)} rows for {i}")
            
            df = hist_data.iloc[:-1]
            logger.debug(f"Trimmed last row, remaining rows: {len(df)}")
            
            df = df.dropna()
            logger.debug(f"Dropped NaN rows, remaining rows: {len(df)}")
            
            df['tic'] = i
            logger.debug(f"Added ticker column 'tic' with value {i}")
            
            final_df = pd.concat([final_df, df], ignore_index=True)
            logger.debug(f"Concatenated data for {i}, current final_df shape: {final_df.shape}")

        logger.info(f"Data download completed for all tickers, final shape: {final_df.shape}")
        return final_df

    def frac_diff_features(self, array):
        """Apply fractional differentiation with logging."""
        logger.info("Starting fractional differentiation of technical array")
        array = FracdiffStat().fit_transform(array)
        logger.debug(f"Fractional differentiation completed, array shape: {array.shape}")
        return array

    def clean_data(self, df):
        """Clean the DataFrame with logging."""
        logger.info(f"Cleaning DataFrame with initial shape: {df.shape}")
        df = df.dropna()
        logger.debug(f"After dropping NaN values, shape: {df.shape}")
        logger.info("Data cleaning completed")
        return df

    def add_technical_indicator(self, df, tech_indicator_list):
        """Add technical indicators with detailed logging."""
        logger.info(f"Adding technical indicators: {tech_indicator_list}")
        final_df = pd.DataFrame()
        for i in df.tic.unique():
            logger.info(f"Processing technical indicators for ticker: {i}")
            coin_df = df[df.tic == i].copy()
            logger.debug(f"Extracted data for {i}, shape: {coin_df.shape}")
            
            coin_df = self.get_TALib_features_for_each_coin(coin_df)
            logger.debug(f"Added TA-Lib features for {i}, new columns: {coin_df.columns.tolist()}")
            
            final_df = pd.concat([final_df, coin_df], ignore_index=True)
            logger.debug(f"Concatenated {i} data, current final_df shape: {final_df.shape}")

        logger.info(f"Technical indicator addition completed, final shape: {final_df.shape}")
        return final_df

    def drop_correlated_features(self, df):
        """Drop correlated features with logging."""
        logger.info("Analyzing feature correlations for dropping")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        logger.debug(f"Selected numeric columns: {numeric_df.columns.tolist()}")
        
        corr_matrix = numeric_df.corr().abs()
        logger.debug("Computed correlation matrix")
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        logger.info(f"According to correlation analysis (threshold {self.correlation_threshold}), features to drop: {cols_to_drop}")
        
        if 'close' in cols_to_drop:
            cols_to_drop.remove('close')
            logger.debug("Removed 'close' from drop list to preserve it")
        
        real_drop = ['high', 'low', 'open', 'macd', 'cci', 'roc', 'willr']
        logger.info(f"Dropping features for model consistency: {real_drop}")
        
        df_uncorrelated = df.drop(real_drop, axis=1)
        logger.debug(f"Features after dropping: {df_uncorrelated.columns.tolist()}")
        logger.info(f"Feature dropping completed, new shape: {df_uncorrelated.shape}")
        return df_uncorrelated

    def add_turbulence(self, df):
        """Placeholder for turbulence addition with logging."""
        logger.info("Turbulence calculation not supported yet")
        logger.debug("Returning original DataFrame unchanged")
        return df

    def add_5m_CVIX(self, df):
        """Add 5-minute CVIX data from Yahoo with logging."""
        logger.info("Adding 5-minute CVIX data")
        trade_start_date = self.start_date[:10]
        trade_end_date = self.end_date[:10]
        TIME_INTERVAL = '60m'
        logger.debug(f"Initializing YahooProcessor for CVIX from {trade_start_date} to {trade_end_date} with interval {TIME_INTERVAL}")
        
        YahooProcessor = Yahoofinance('yahoofinance', trade_start_date, trade_end_date, TIME_INTERVAL)
        logger.info("Downloading CVIX data from Yahoo...")
        CVOL_df = YahooProcessor.download_data(['CVOL-USD'])
        logger.debug(f"CVIX data downloaded, shape: {CVOL_df.shape}")
        
        CVOL_df.set_index('date', inplace=True)
        logger.debug("Set index to 'date'")
        
        CVOL_df = CVOL_df.resample('5Min').interpolate(method='linear')
        logger.debug(f"Resampled CVIX to 5-minute intervals, shape: {CVOL_df.shape}")
        
        df['CVIX'] = CVOL_df['close']
        logger.debug(f"Added 'CVIX' column, new shape: {df.shape}")
        logger.info("CVIX addition completed")
        return df

    def df_to_array(self, df, if_vix):
        """Convert DataFrame to arrays with verbose logging."""
        logger.info("Converting DataFrame to arrays")
        self.tech_indicator_list = list(df.columns)
        self.tech_indicator_list.remove('tic')
        logger.info(f"Technical indicators (count: {len(self.tech_indicator_list)}): {self.tech_indicator_list}")

        unique_ticker = df.tic.unique()
        logger.debug(f"Unique tickers: {unique_ticker}")
        
        if_first_time = True
        for tic in unique_ticker:
            logger.info(f"Processing array conversion for ticker: {tic}")
            if if_first_time:
                price_array = df[df.tic == tic][['close']].values
                tech_array = df[df.tic == tic][self.tech_indicator_list].values
                if_first_time = False
                logger.debug(f"Initialized arrays for {tic}: price shape {price_array.shape}, tech shape {tech_array.shape}")
            else:
                price_array = np.hstack([price_array, df[df.tic == tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic == tic][self.tech_indicator_list].values])
                logger.debug(f"Stacked arrays for {tic}: price shape {price_array.shape}, tech shape {tech_array.shape}")

            time_array = df[df.tic == self.ticker_list[0]].index
            logger.debug(f"Time array length for {tic}: {len(time_array)}")

        assert price_array.shape[0] == tech_array.shape[0], "Price and tech array lengths mismatch"
        logger.info(f"Array conversion completed: price shape {price_array.shape}, tech shape {tech_array.shape}, time length {len(time_array)}")
        return price_array, tech_array, time_array

    def stringify_dates(self, date: datetime):
        """Convert datetime to string for Binance API."""
        logger.debug(f"Stringifying date: {date}")
        result = str(int(date.timestamp() * 1000))
        logger.debug(f"Stringified date: {result}")
        return result

    def servertime_to_datetime(self, timestamp):
        """Convert server timestamp to datetime with logging."""
        logger.info(f"Converting {len(timestamp)} timestamps to datetime")
        list_regular_stamps = [0] * len(timestamp)
        for indx, ts in enumerate(timestamp):
            list_regular_stamps[indx] = dt.datetime.fromtimestamp(ts / 1000)
            if indx % 1000 == 0:  # Log every 1000th conversion to avoid spam
                logger.debug(f"Converted timestamp {ts} to {list_regular_stamps[indx]}")
        logger.info("Timestamp conversion completed")
        return list_regular_stamps

    def get_binance_bars(self, start_date, end_date, kline_size, symbol):
        """Fetch historical candlestick data from Binance with logging."""
        logger.info(f"Fetching bars for {symbol} from {start_date} to {end_date} with kline size {kline_size}")
        data_df = pd.DataFrame()
        klines = self.binance_client.get_historical_klines(symbol, kline_size, start_date, end_date)
        logger.debug(f"Fetched {len(klines)} kline entries for {symbol}")
        
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        logger.debug(f"Created DataFrame with columns: {data.columns.tolist()}")
        
        data = data.drop(labels=['close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'], axis=1)
        logger.debug(f"Dropped unnecessary columns, remaining: {data.columns.tolist()}")
        
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = pd.concat([data_df, temp_df], ignore_index=True)
            logger.debug(f"Appended data, new shape: {data_df.shape}")
        else:
            data_df = data
            logger.debug(f"Initialized data_df, shape: {data_df.shape}")

        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        logger.debug("Converted data to numeric types")
        
        data_df['time'] = [datetime.fromtimestamp(x / 1000.0) for x in data_df.timestamp]
        logger.debug("Added 'time' column from timestamps")
        
        data_df.index = [x for x in range(len(data_df))]
        logger.debug(f"Set index, final shape: {data_df.shape}")
        
        logger.info(f"Data fetch completed for {symbol}, shape: {data_df.shape}")
        return data_df

    def get_TALib_features_for_each_coin(self, tic_df):
        """Calculate TA-Lib features for a single coin with logging."""
        logger.info(f"Calculating TA-Lib features for DataFrame with shape: {tic_df.shape}")
        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        logger.debug("Added RSI indicator")
        
        tic_df['macd'], _, _ = MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        logger.debug("Added MACD indicator")
        
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        logger.debug("Added CCI indicator")
        
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        logger.debug("Added DX indicator")
        
        tic_df['roc'] = ROC(tic_df['close'], timeperiod=10)
        logger.debug("Added ROC indicator")
        
        tic_df['ultosc'] = ULTOSC(tic_df['high'], tic_df['low'], tic_df['close'])
        logger.debug("Added ULTOSC indicator")
        
        tic_df['willr'] = WILLR(tic_df['high'], tic_df['low'], tic_df['close'])
        logger.debug("Added WILLR indicator")
        
        tic_df['obv'] = OBV(tic_df['close'], tic_df['volume'])
        logger.debug("Added OBV indicator")
        
        tic_df['ht_dcphase'] = HT_DCPHASE(tic_df['close'])
        logger.debug("Added HT_DCPHASE indicator")
        
        logger.info(f"TA-Lib feature calculation completed, new columns: {tic_df.columns.tolist()}")
        return tic_df