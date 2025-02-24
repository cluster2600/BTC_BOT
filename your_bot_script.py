#!/usr/bin/env python3
"""
Main Trading Bot with Optimized Strategy and Error Handling for Leverage Issues.
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

# Load environment variables
dotenv.load_dotenv()

# Global Configuration
TEST_MODE = False
USE_TESTNET = False
MLX_SERVER_URL = "http://localhost:1234/v1/completions"
NN_MODEL_PATH = "/Users/maxime/pump_project/NNModel.mlpackage"

# Trading Parameters
STOP_LOSS_PCT = 0.015       # 1.5% stop-loss
TAKE_PROFIT_PCT = 0.03      # 3% take-profit
ADX_THRESHOLD = 25          # Stronger trend requirement
CONFIDENCE_THRESHOLD = 0.8  # High confidence for entries
TAKER_FEE = 0.0006          # 0.06% taker fee
MAKER_FEE = 0.0002          # 0.02% maker fee
MIN_USDT = 100.0            # Minimum USDT balance to trade
MIN_BTC = 0.00105
TRADE_PERCENTAGE = 0.10     # Default 10% of balance
HIGH_CONFIDENCE_TRADE_PCT = 0.20  # 20% for high-confidence trades
COOLDOWN = 300.0            # 5 minutes cooldown
SLEEP_INTERVAL = 0.5
FUNDING_CHECK_INTERVAL = 3600  # Check funding fees hourly
LEVERAGE = 3                # Target leverage
DEFAULT_LEVERAGE = 1        # Fallback leverage if setting fails

# Logger Setup
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True

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
current_leverage: int = DEFAULT_LEVERAGE  # Track current leverage

def initialize_funds() -> None:
    global initial_futures_balance, current_position
    try:
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
        bal = exchange_futures.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        btc = float(bal.get('BTC', {}).get("free", 0))
        print_info(f"[PORTFOLIO] USDT: {usdt:.8f}, BTC: {btc:.8f}")
    except Exception as e:
        print_error(f"Error displaying portfolio: {e}")

def display_profit(initial_balance: float) -> None:
    try:
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
        return
    try:
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
        ohlcv = exchange_futures.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print_error(f"Error fetching OHLCV: {e}")
        return pd.DataFrame()

def get_ohlcv_cached(symbol: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
    global last_ohlcv_fetch_time, ohlcv_cache
    now = time.time()
    if now - last_ohlcv_fetch_time > 60:
        ohlcv_cache = get_ohlcv(symbol, timeframe, limit)
        last_ohlcv_fetch_time = now
    return ohlcv_cache

def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
    try:
        indicators = {}
        indicators['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        indicators['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
        indicators['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator().iloc[-1] if len(df) >= 200 else np.nan
        indicators['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
        indicators['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().iloc[-1]
        return indicators
    except Exception as e:
        print_error(f"Error calculating indicators: {e}")
        return {'rsi': 0.0, 'sma_20': 0.0, 'sma_200': np.nan, 'atr': 0.0, 'adx': 0.0}

def get_current_position(symbol: str) -> float:
    try:
        positions = exchange_futures.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and pos['side'] == 'long':
                return float(pos['contracts'])
        return 0.0
    except Exception as e:
        print_error(f"Error fetching position: {e}")
        return 0.0

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float) -> Optional[float]:
    global current_position, current_leverage
    try:
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        lot_step = next((float(f.get('stepSize')) for f in filters if f.get('filterType') == 'LOT_SIZE'), 1e-8)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        free_usdt = float(bal.get('USDT', {}).get("free", 0))
        if free_usdt < MIN_USDT:
            print_info(f"Insufficient free USDT balance ({free_usdt:.2f} < {MIN_USDT:.2f}), skipping trade.")
            return None
        margin_required = (calculated_qty * current_price) / current_leverage
        if side.upper() == 'BUY':
            final_qty = max(calculated_qty, MIN_USDT / current_price)
            if margin_required > free_usdt:
                final_qty = (free_usdt * current_leverage) / current_price
                print_info(f"Adjusted quantity to {final_qty:.8f} BTC due to insufficient margin.")
            final_qty = min(final_qty, free_usdt / current_price)
        elif side.upper() == 'SELL' and current_position > 0:
            final_qty = min(calculated_qty, current_position)
            if final_qty <= 0:
                print_info(f"Calculated SELL quantity {final_qty} is invalid for {symbol}")
                return None
        else:
            return None
        return math.floor(final_qty / lot_step) * lot_step
    except Exception as e:
        print_error(f"Error adjusting quantity: {e}")
        return None

def execute_buy_order(symbol: str, final_quantity: float, current_price: float) -> Optional[Dict[str, Any]]:
    global current_position, entry_price
    try:
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
    ydf_model = load_ydf_model("model_rf.ydf")
    nn_model = ct.models.MLModel(NN_MODEL_PATH)
    load_mlx_model()
    print_info("Ensemble models loaded successfully.")
except Exception as e:
    print_error(f"Error loading models: {e}")
    sys.exit(1)

def get_latest_price(symbol: str) -> Optional[float]:
    try:
        return exchange_futures.fetch_ticker(symbol).get('last')
    except Exception as e:
        print_error(f"Error fetching price: {e}")
        return None

def trading_logic(symbol: str) -> None:
    global previous_position, last_close_time, current_position, entry_price

    ohlcv = get_ohlcv_cached(symbol)
    if ohlcv.empty or len(ohlcv) < 200:
        print_info("Insufficient data for indicators.")
        return

    indicators = calculate_indicators(ohlcv)
    if np.isnan(indicators['sma_200']):
        print_info("SMA_200 not available yet.")
        return

    current_price = get_latest_price(symbol)
    if current_price is None:
        return

    current_position = get_current_position(symbol)
    if current_position > 0 and entry_price is None:
        positions = exchange_futures.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and pos['side'] == 'long':
                entry_price = float(pos['entryPrice'])
                break
    elif current_position == 0:
        entry_price = None

    if previous_position is not None and previous_position > 0 and current_position == 0:
        last_close_time = time.time()
        try:
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

    # Check market conditions and buying opportunities
    if current_position == 0 and time.time() - last_close_time > COOLDOWN:
        if indicators['atr'] > 0.01 * current_price:
            print_info("High volatility detected (ATR > 1% of price), skipping trade.")
            return
        if indicators['adx'] < 15:
            print_info("Sideways market detected (ADX < 15), pausing trades.")
            return

        features = {
            "price": current_price,
            "rsi": indicators['rsi'],
            "sma_200": indicators['sma_200'],
            "sma_20": indicators['sma_20'],
            "atr": indicators['atr'],
            "adx": indicators['adx']
        }
        decision, confidence = get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)

        if (current_price > indicators['sma_200'] and
            current_price > indicators['sma_20'] and
            indicators['adx'] > ADX_THRESHOLD and
            decision == "BUY" and
            confidence > CONFIDENCE_THRESHOLD and
            indicators['rsi'] < 30):
            bal = exchange_futures.fetch_balance({'type': 'future'})
            free_usdt = float(bal.get('USDT', {}).get("free", 0))
            if free_usdt < MIN_USDT:
                print_info(f"Free USDT balance too low ({free_usdt:.2f} < {MIN_USDT:.2f}), skipping trade.")
                return
            trade_pct = HIGH_CONFIDENCE_TRADE_PCT if confidence > 0.9 else TRADE_PERCENTAGE  # Adjusted to 0.9 for high confidence
            qty = trade_pct * (free_usdt / current_price)
            final_qty = adjust_quantity(symbol, "BUY", qty, current_price)
            if final_qty and final_qty > 0:
                order = execute_buy_order(symbol, final_qty, current_price)
                if order:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    try:
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

    previous_position = current_position
    check_funding_fees(symbol)

def main() -> None:
    global previous_position, last_close_time, current_leverage
    initialize_funds()
    display_portfolio()
    symbol = "BTC/USDT:USDT"
    print_info(f"Starting trading bot for {symbol}...")
    try:
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
    last_close_time = 0

    initial_balance = initial_futures_balance or 0
    last_profit_time = time.time()

    while True:
        try:
            trading_logic(symbol)
            if time.time() - last_profit_time >= 60:
                display_profit(initial_balance)
                last_profit_time = time.time()
            time.sleep(SLEEP_INTERVAL)
        except Exception as e:
            print_error(f"Trading loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()