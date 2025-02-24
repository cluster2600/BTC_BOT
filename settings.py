# settings.py
from datetime import datetime, timedelta
import numpy as np
import operator as op
from functools import reduce
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

# General Training Settings
trade_start_date = '2022-04-30 00:00:00'
trade_end_date = '2022-06-27 00:00:00'

SEED_CFG = 2390408
TIMEFRAME = '5m'
H_TRIALS = 50
KCV_groups = 5
K_TEST_GROUPS = 2
NUM_PATHS = 4
N_GROUPS = NUM_PATHS + 1
NUMBER_OF_SPLITS = nCr(N_GROUPS, N_GROUPS - K_TEST_GROUPS)

logging.debug(f"Calculated NUMBER_OF_SPLITS: {NUMBER_OF_SPLITS}")

no_candles_for_train = 20000
no_candles_for_val = 5000

TICKER_LIST = ['AAVEUSDT', 'AVAXUSDT', 'BTCUSDT', 'NEARUSDT', 'LINKUSDT', 
               'ETHUSDT', 'LTCUSDT', 'MATICUSDT', 'UNIUSDT', 'SOLUSDT']

ALPACA_LIMITS = np.array([0.01, 0.10, 0.0001, 0.1, 0.1, 0.001, 0.01, 10, 0.1, 0.01])

TECHNICAL_INDICATORS_LIST = ['open', 'high', 'low', 'close', 'volume', 'macd', 
                             'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']

def calculate_start_end_dates(candlewidth):
    candle_to_no_minutes = {'1m': 1, '5m': 5, '10m': 10, '30m': 30, '1h': 60, 
                            '2h': 2*60, '4h': 4*60, '12h': 12*60}
    no_minutes = candle_to_no_minutes[candlewidth]
    logging.debug(f"Using {no_minutes} minutes per candle for timeframe {candlewidth}")

    trade_start_date_datetimeObj = datetime.strptime(trade_start_date, "%Y-%m-%d %H:%M:%S")

    train_start_date = (trade_start_date_datetimeObj 
                        - timedelta(minutes=no_minutes * (no_candles_for_train + no_candles_for_val))).strftime("%Y-%m-%d %H:%M:%S")
    train_end_date = (trade_start_date_datetimeObj 
                      - timedelta(minutes=no_minutes * (no_candles_for_val + 1))).strftime("%Y-%m-%d %H:%M:%S")
    val_start_date = (trade_start_date_datetimeObj 
                      - timedelta(minutes=no_minutes * no_candles_for_val)).strftime("%Y-%m-%d %H:%M:%S")
    val_end_date = (trade_start_date_datetimeObj 
                    - timedelta(minutes=no_minutes * 1)).strftime("%Y-%m-%d %H:%M:%S")

    logging.debug(f"Train Start: {train_start_date}, Train End: {train_end_date}")
    logging.debug(f"Val Start: {val_start_date}, Val End: {val_end_date}")
    return train_start_date, train_end_date, val_start_date, val_end_date

TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE = calculate_start_end_dates(TIMEFRAME)