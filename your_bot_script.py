#!/usr/bin/env python3
"""
Main Trading Bot with Optimized Strategy and Error Handling for Leverage Issues.
Integrated with BinanceProcessor and Yahoofinance for data processing.
"""

import os
import sys
import time
import json
import math
import urllib.parse
from datetime import datetime
from typing import Any, Dict, Optional, List
import signal

import ccxt
import numpy as np
import pandas as pd
import requests
import websocket
import logging
from logging.handlers import RotatingFileHandler
import colorlog
import coremltools as ct
import ta
import dotenv
from telebot import TeleBot, types

from ensemble_models import load_ydf_model, load_mlx_model, get_ensemble_decision
from processor_Binance import BinanceProcessor
from processor_Yahoo import Yahoofinance

# Load environment variables
dotenv.load_dotenv()

# Global Configuration
TEST_MODE = False
USE_TESTNET = False
MLX_SERVER_URL = "http://localhost:1234/v1/completions"
NN_MODEL_PATH = "/Users/maxime/pump_project/NNModel.mlpackage"

# Trading Parameters
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.03
ADX_THRESHOLD = 25
CONFIDENCE_THRESHOLD = 0.8
TAKER_FEE = 0.0006
MAKER_FEE = 0.0002
MIN_USDT = 100.0
MIN_BTC = 0.00105
TRADE_PERCENTAGE = 0.10
HIGH_CONFIDENCE_TRADE_PCT = 0.20
COOLDOWN = 300.0
SLEEP_INTERVAL = 0.5
FUNDING_CHECK_INTERVAL = 3600
LEVERAGE = 3
DEFAULT_LEVERAGE = 1

# Logger Setup
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True

# Processor Parameters
TICKER_LIST = ["BTCUSDT"]
START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIME_INTERVAL = "1m"
TECHNICAL_INDICATOR_LIST = ["rsi", "macd", "cci", "dx"]
IF_VIX = True

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("TradingBot")
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "bold_white", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red"}
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if LOG_TO_FILE:
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        file_handler = RotatingFileHandler(f"TradingBot_{current_datetime}.log", maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

log = setup_logger()

def print_error(message: str) -> None:
    log.error(message)

def print_info(message: str) -> None:
    log.info(message)

# Load environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    print_error("Missing required environment variables.")
    sys.exit(1)

# Telegram Bot Setup
bot = TeleBot(TELEGRAM_TOKEN)

def telegram_notify(message: str, retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={urllib.parse.quote(message)}"
            response = requests.get(url, timeout=10)
            print_info(f"Telegram notification sent: {message}")
            return response.json()
        except Exception as e:
            print_error(f"Telegram error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(5)
    print_error(f"Failed to send Telegram notification after {retries} attempts: {message}")
    return None

# Binance Exchange Instances
custom_headers = {'User-Agent': 'Mozilla/5.0', 'X-MBX-APIKEY': BINANCE_API_KEY}
exchange_futures = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'urls': {'api': {'public': 'https://fapi.binance.com/fapi/v1', 'private': 'https://fapi.binance.com/fapi/v1'}},
    'headers': custom_headers
})

# Global State Variables
initial_futures_balance: Optional[float] = None
current_position: Optional[float] = None
entry_price: Optional[float] = None
previous_position: Optional[float] = None
last_close_time: float = 0
last_ohlcv_fetch_time: float = 0
ohlcv_cache: Optional[pd.DataFrame] = None
last_funding_check: float = 0
cumulative_funding_fees: float = 0.0
current_leverage: int = DEFAULT_LEVERAGE

def initialize_funds() -> None:
    global initial_futures_balance, current_position
    try:
        print_info("Fetching initial balance from Binance futures...")
        fut_bal = exchange_futures.fetch_balance({'type': 'future'})
        initial_futures_balance = float(fut_bal.get('USDT', {}).get("total", 0))
        current_position = float(fut_bal.get('BTC', {}).get("free", 0))
        if initial_futures_balance < MIN_USDT:
            print_error(f"Initial Futures USDT balance {initial_futures_balance:.2f} below minimum {MIN_USDT} USDT.")
            sys.exit(1)
        print_info(f"Initial Futures USDT: {initial_futures_balance:.2f}, BTC: {current_position:.8f}")
    except Exception as e:
        print_error(f"Error initializing funds: {e}")
        sys.exit(1)

def display_portfolio() -> None:
    try:
        print_info("Fetching portfolio balance...")
        bal = exchange_futures.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        btc = float(bal.get('BTC', {}).get("free", 0))
        print_info(f"[PORTFOLIO] USDT: {usdt:.8f}, BTC: {btc:.8f}")
    except Exception as e:
        print_error(f"Error displaying portfolio: {e}")

def display_profit(initial_balance: float) -> None:
    try:
        print_info("Calculating profit/loss...")
        bal = exchange_futures.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("total", 0))
        profit = current_usdt - initial_balance - cumulative_funding_fees
        profit_str = f"Profit: {profit:.2f} USDT" if profit >= 0 else f"Loss: {profit:.2f} USDT"
        print_info(f"[OVERALL] {profit_str} (Funding Fees Paid: {cumulative_funding_fees:.2f} USDT)")
    except Exception as e:
        print_error(f"Error displaying profit: {e}")

def check_funding_fees(symbol: str) -> None:
    global cumulative_funding_fees, last_funding_check
    now = time.time()
    if now - last_funding_check < FUNDING_CHECK_INTERVAL:
        print_info(f"Funding check skipped, last check: {now - last_funding_check:.0f}s ago")
        return
    try:
        print_info(f"Fetching funding fees for {symbol}...")
        funding_history = exchange_futures.fetch_funding_history(symbol, limit=100)
        for entry in funding_history:
            fee = float(entry['info'].get('income', 0))
            cumulative_funding_fees += fee
        last_funding_check = now
        print_info(f"Updated cumulative funding fees: {cumulative_funding_fees:.2f} USDT")
    except Exception as e:
        print_error(f"Error fetching funding fees: {e}")

def get_ohlcv(symbol: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
    try:
        print_info(f"Fetching OHLCV data for {symbol}, timeframe: {timeframe}, limit: {limit}")
        ohlcv = exchange_futures.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print_info(f"OHLCV fetched, shape: {df.shape}, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print_error(f"Error fetching OHLCV: {e}")
        return pd.DataFrame()

def get_ohlcv_cached(symbol: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
    global last_ohlcv_fetch_time, ohlcv_cache
    now = time.time()
    if now - last_ohlcv_fetch_time > 60 or ohlcv_cache is None:
        print_info("OHLCV cache expired or empty, refreshing...")
        ohlcv_cache = get_ohlcv(symbol, timeframe, limit)
        last_ohlcv_fetch_time = now
    else:
        print_info("Using cached OHLCV data")
    print_info(f"Cached OHLCV shape: {ohlcv_cache.shape if not ohlcv_cache.empty else 'empty'}")
    return ohlcv_cache

def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
    try:
        print_info(f"Calculating indicators from DataFrame, shape: {df.shape}, columns: {df.columns.tolist()}")
        indicators = {}
        required_ohlcv = ['open', 'high', 'low', 'close']

        # Calculate SMA, ATR, ADX from OHLCV
        if all(col in df.columns for col in required_ohlcv):
            print_info("Full OHLCV data available, calculating indicators...")
            indicators['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            indicators['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
            indicators['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator().iloc[-1] if len(df) >= 200 else np.nan
            indicators['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
            indicators['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().iloc[-1]
        else:
            print_info("Missing full OHLCV, setting defaults for SMA, ATR, ADX")
            indicators['rsi'] = 0.0
            indicators['sma_20'] = 0.0
            indicators['sma_200'] = np.nan
            indicators['atr'] = 0.0
            indicators['adx'] = 0.0

        print_info(f"Indicators calculated: {indicators}")
        return indicators
    except Exception as e:
        print_error(f"Error calculating indicators: {e}")
        return {'rsi': 0.0, 'sma_20': 0.0, 'sma_200': np.nan, 'atr': 0.0, 'adx': 0.0}

def get_current_position(symbol: str) -> float:
    try:
        print_info(f"Fetching current position for {symbol}...")
        positions = exchange_futures.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and pos['side'] == 'long':
                contracts = float(pos['contracts'])
                print_info(f"Current position: {contracts} contracts")
                return contracts
        print_info("No long position found")
        return 0.0
    except Exception as e:
        print_error(f"Error fetching position: {e}")
        return 0.0

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float) -> Optional[float]:
    global current_position, current_leverage
    try:
        print_info(f"Adjusting quantity for {symbol}, side: {side}, qty: {calculated_qty}, price: {current_price}")
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        lot_step = next((float(f.get('stepSize')) for f in filters if f.get('filterType') == 'LOT_SIZE'), 1e-8)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        free_usdt = float(bal.get('USDT', {}).get("free", 0))
        if free_usdt < MIN_USDT:
            print_info(f"Insufficient USDT balance ({free_usdt:.2f} < {MIN_USDT}), skipping trade")
            return None
        margin_required = (calculated_qty * current_price) / current_leverage
        if side.upper() == 'BUY':
            final_qty = max(calculated_qty, MIN_USDT / current_price)
            if margin_required > free_usdt:
                final_qty = (free_usdt * current_leverage) / current_price
                print_info(f"Adjusted qty due to margin: {final_qty:.8f}")
            final_qty = min(final_qty, free_usdt / current_price)
        elif side.upper() == 'SELL' and current_position > 0:
            final_qty = min(calculated_qty, current_position)
            if final_qty <= 0:
                print_info(f"Invalid SELL qty: {final_qty}")
                return None
        else:
            print_info(f"No adjustment for {side} with position {current_position}")
            return None
        adjusted_qty = math.floor(final_qty / lot_step) * lot_step
        print_info(f"Adjusted quantity: {adjusted_qty}")
        return adjusted_qty
    except Exception as e:
        print_error(f"Error adjusting quantity: {e}")
        return None

def execute_buy_order(symbol: str, final_quantity: float, current_price: float) -> Optional[Dict[str, Any]]:
    global current_position, entry_price
    try:
        print_info(f"Executing BUY order for {symbol}: {final_quantity} @ {current_price}")
        order = exchange_futures.create_limit_buy_order(symbol, final_quantity, current_price, params={"reduceOnly": False})
        order_info = exchange_futures.fetch_order(order['id'], symbol)
        current_position = final_quantity
        entry_price = float(order_info['price']) if order_info['price'] else current_price
        print_info(f"[TRADE] BUY {final_quantity} units at {entry_price} USDT (Limit Order)")
        telegram_notify(f"[TRADE] BUY {final_quantity} units at {entry_price} USDT (Limit Order)")
        return order
    except Exception as e:
        print_error(f"Error executing BUY order: {e}")
        telegram_notify(f"ALERT: Error executing BUY order: {e}")
        return None

try:
    print_info("Loading YDF model...")
    ydf_model = load_ydf_model("model_rf.ydf")
    print_info("Loading Core ML model...")
    nn_model = ct.models.MLModel(NN_MODEL_PATH)
    print_info("Loading MLX model...")
    load_mlx_model()
    print_info("Ensemble models loaded successfully.")
except Exception as e:
    print_error(f"Error loading models: {e}")
    sys.exit(1)

def get_latest_price(symbol: str) -> Optional[float]:
    try:
        print_info(f"Fetching latest price for {symbol}")
        price = exchange_futures.fetch_ticker(symbol).get('last')
        print_info(f"Latest price: {price}")
        return price
    except Exception as e:
        print_error(f"Error fetching price: {e}")
        return None

def trading_logic(symbol: str, binance_data: pd.DataFrame, yahoo_data: pd.DataFrame) -> None:
    global previous_position, last_close_time, current_position, entry_price
    print_info(f"Entering trading logic for {symbol}")

    # Log data details
    print_info(f"Binance data shape: {binance_data.shape if not binance_data.empty else 'empty'}, columns: {binance_data.columns.tolist() if not binance_data.empty else 'none'}")
    print_info(f"Yahoo data shape: {yahoo_data.shape if not yahoo_data.empty else 'empty'}, columns: {yahoo_data.columns.tolist() if not yahoo_data.empty else 'none'}")

    # Use cached OHLCV for raw data
    ohlcv = get_ohlcv_cached(symbol)
    if ohlcv.empty or len(ohlcv) < 200:
        print_info(f"Insufficient OHLCV data, rows: {len(ohlcv)}")
        return

    indicators = calculate_indicators(ohlcv)
    # Overlay processed indicators from binance_data if available
    if not binance_data.empty and len(binance_data) > 0:
        for ind in ['rsi', 'macd', 'cci', 'dx']:
            if ind in binance_data.columns:
                indicators[ind] = float(binance_data[ind].iloc[-1])
                print_info(f"Overlaid processed {ind}: {indicators[ind]}")
    # Use dx as ADX if available
    if 'dx' in indicators:
        indicators['adx'] = indicators['dx']
        print_info(f"Using overlaid dx as adx: {indicators['adx']}")
    if np.isnan(indicators['sma_200']):
        print_info("SMA_200 is NaN, skipping trade")
        return

    current_price = get_latest_price(symbol)
    if current_price is None:
        print_info("Failed to fetch current price, skipping trade")
        return

    current_position = get_current_position(symbol)
    if current_position > 0 and entry_price is None:
        print_info("Active position detected, setting entry price")
        positions = exchange_futures.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and pos['side'] == 'long':
                entry_price = float(pos['entryPrice'])
                print_info(f"Entry price set to {entry_price}")
                break
    elif current_position == 0:
        entry_price = None
        print_info("No position, entry price reset to None")

    if previous_position is not None and previous_position > 0 and current_position == 0:
        last_close_time = time.time()
        try:
            print_info("Position closed, fetching trade history for PnL")
            trades = exchange_futures.fetch_my_trades(symbol, limit=10)
            realized_pnl = 0.0
            for trade in trades[::-1]:
                if trade['info']['positionSide'] == 'LONG' and trade['info']['side'] == 'SELL':
                    realized_pnl = float(trade['info'].get('realizedPnl', 0))
                    break
            if realized_pnl != 0.0:
                result_str = f"Profit: {realized_pnl:.2f} USDT" if realized_pnl >= 0 else f"Loss: {realized_pnl:.2f} USDT"
                print_info(f"[TRADE CLOSED] Position closed. Realized {result_str}")
                telegram_notify(f"[TRADE CLOSED] Position closed. Realized {result_str}")
            else:
                print_info("[TRADE CLOSED] Position closed. No realized PnL found.")
        except Exception as e:
            print_error(f"Error fetching realized PnL: {e}")
            if entry_price:
                approx_profit = (current_price - entry_price) * previous_position
                result_str = f"Profit: {approx_profit:.2f} USDT" if approx_profit >= 0 else f"Loss: {approx_profit:.2f} USDT"
                print_info(f"[TRADE CLOSED] Position closed. Approximate {result_str}")
                telegram_notify(f"[TRADE CLOSED] Position closed. Approximate {result_str}")

    elapsed_since_close = time.time() - last_close_time
    if current_position == 0 and elapsed_since_close > COOLDOWN:
        print_info(f"Cooldown expired ({elapsed_since_close:.0f}s > {COOLDOWN}s), evaluating trade conditions")
        print_info(f"Conditions - Price: {current_price}, SMA_200: {indicators['sma_200']}, SMA_20: {indicators['sma_20']}, ADX: {indicators['adx']}, RSI: {indicators['rsi']}, ATR: {indicators['atr']}")

        conditions = {
            "ATR <= 0.01 * Price": indicators['atr'] <= 0.01 * current_price,
            "ADX > 20": indicators['adx'] > 20,
            "Price > SMA_200 or SMA_20": (current_price > indicators['sma_200']) or (current_price > indicators['sma_20']),
            "RSI < 70": indicators['rsi'] < 70
        }
        for cond, met in conditions.items():
            print_info(f"{cond}: {'Met' if met else 'Not Met'} (Value: {conditions[cond]})")

        features = {
            "price": current_price,
            "rsi": indicators['rsi'],
            "sma_200": indicators['sma_200'],
            "sma_20": indicators['sma_20'],
            "atr": indicators['atr'],
            "adx": indicators['adx']
        }
        if not yahoo_data.empty and 'close' in yahoo_data.columns:
            features['vix'] = yahoo_data['close'].iloc[-1] if len(yahoo_data) > 0 else 0.0
            print_info(f"Added VIX to features: {features['vix']}")
        print_info(f"Features for ensemble model: {features}")

        decision, confidence = get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)
        print_info(f"Ensemble decision: {decision}, Confidence: {confidence:.4f}")

        conditions.update({
            "Decision == BUY": decision == "BUY",
            "Confidence > 0.7": confidence > 0.7
        })
        print_info(f"Decision == BUY: {'Met' if conditions['Decision == BUY'] else 'Not Met'}")
        print_info(f"Confidence > 0.7: {'Met' if conditions['Confidence > 0.7'] else 'Not Met'} (Value: {confidence:.4f})")

        if all(conditions.values()):
            print_info("All trade conditions met, preparing BUY order")
            bal = exchange_futures.fetch_balance({'type': 'future'})
            free_usdt = float(bal.get('USDT', {}).get("free", 0))
            if free_usdt < MIN_USDT:
                print_info(f"Insufficient USDT balance ({free_usdt:.2f} < {MIN_USDT}), skipping trade")
                return
            trade_pct = HIGH_CONFIDENCE_TRADE_PCT if confidence > 0.9 else TRADE_PERCENTAGE
            qty = trade_pct * (free_usdt / current_price)
            final_qty = adjust_quantity(symbol, "BUY", qty, current_price)
            if final_qty and final_qty > 0:
                order = execute_buy_order(symbol, final_qty, current_price)
                if order:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    try:
                        print_info(f"Placing SL at {stop_loss_price:.2f}, TP at {take_profit_price:.2f}")
                        sl_order = exchange_futures.create_order(
                            symbol, 'stop', 'sell', current_position,
                            params={'stopPrice': stop_loss_price, 'reduceOnly': True}
                        )
                        tp_order = exchange_futures.create_order(
                            symbol, 'limit', 'sell', current_position, take_profit_price,
                            params={'reduceOnly': True}
                        )
                        print_info(f"Placed SL order at {stop_loss_price:.2f} USDT and TP order at {take_profit_price:.2f} USDT")
                    except Exception as e:
                        print_error(f"Error placing SL/TP orders: {e}")
                        telegram_notify(f"ALERT: Error placing SL/TP orders: {e}")
            else:
                print_info("No valid quantity for BUY order")
        else:
            print_info("Trade conditions not met, skipping trade")
    else:
        print_info(f"Position active ({current_position}) or cooldown active ({COOLDOWN - elapsed_since_close:.0f}s remaining)")

    previous_position = current_position
    check_funding_fees(symbol)

def main() -> None:
    print_info("Starting trading bot execution...")
    binance_processor = BinanceProcessor()
    yahoo_processor = Yahoofinance('yahoofinance', START_DATE, END_DATE, '1h')

    try:
        print_info(f"Running Binance data processing for {TICKER_LIST} from {START_DATE} to {END_DATE} with interval {TIME_INTERVAL}")
        binance_data, binance_price_array, binance_tech_array, binance_time_array = binance_processor.run(
            ticker_list=TICKER_LIST,
            start_date=START_DATE,
            end_date=END_DATE,
            time_interval=TIME_INTERVAL,
            technical_indicator_list=TECHNICAL_INDICATOR_LIST,
            if_vix=False
        )
        print_info(f"Binance data processing completed successfully, shape: {binance_data.shape}, columns: {binance_data.columns.tolist()}")
    except Exception as e:
        print_error(f"Error during Binance data processing: {e}")
        binance_data = pd.DataFrame()

    try:
        print_info(f"Fetching Yahoo VIX data for ['^VIX'] from {START_DATE} to {END_DATE} with interval '1h'")
        vix_data = yahoo_processor.download_data(['^VIX'])
        print_info(f"Yahoo VIX data processing completed successfully, shape: {vix_data.shape}, columns: {vix_data.columns.tolist()}")
    except Exception as e:
        print_error(f"Error during Yahoo VIX data processing: {e}")
        vix_data = pd.DataFrame()

    global previous_position, last_close_time, current_leverage
    initialize_funds()
    display_portfolio()
    symbol = "BTC/USDT:USDT"
    print_info(f"Starting trading bot for {symbol}...")
    try:
        print_info(f"Attempting to set leverage to {LEVERAGE}x for {symbol}")
        exchange_futures.set_leverage(LEVERAGE, symbol)
        current_leverage = LEVERAGE
        print_info(f"Leverage set to {LEVERAGE}x for {symbol}")
    except ccxt.OperationRejected as e:
        print_error(f"Failed to set leverage to {LEVERAGE}x: {str(e)}. Falling back to {DEFAULT_LEVERAGE}x.")
        current_leverage = DEFAULT_LEVERAGE
        try:
            exchange_futures.set_leverage(DEFAULT_LEVERAGE, symbol)
            print_info(f"Leverage set to {DEFAULT_LEVERAGE}x for {symbol}")
        except Exception as e:
            print_error(f"Failed to set fallback leverage: {e}. Continuing with default exchange leverage.")
    except Exception as e:
        print_error(f"Unexpected error setting leverage: {e}. Continuing with default exchange leverage.")

    previous_position = current_position
    last_close_time = 0  # Reset cooldown for immediate testing

    initial_balance = initial_futures_balance or 0
    last_profit_time = time.time()

    while True:
        try:
            trading_logic(symbol, binance_data, vix_data)
            if time.time() - last_profit_time >= 60:
                display_profit(initial_balance)
                last_profit_time = time.time()
            print_info(f"Sleeping for {SLEEP_INTERVAL} seconds...")
            time.sleep(SLEEP_INTERVAL)
        except Exception as e:
            print_error(f"Trading loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()