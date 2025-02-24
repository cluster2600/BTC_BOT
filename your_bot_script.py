#!/usr/bin/env python3
"""
Main Trading Bot with Ensemble Model Integration including MLX via LM Studio and Core ML NN,
Forced Minimum Order Sizes, and Binance Futures Production Support.
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

# Logger Setup
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True

def setup_logger() -> logging.Logger:
    """Set up the logger with console and file handlers."""
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
    """Send a notification to Telegram."""
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

# Fund Initialization
initial_futures_balance: Optional[float] = None
current_position: Optional[float] = None  # BTC held
entry_price: Optional[float] = None  # Price at which position was opened

def initialize_funds() -> None:
    """Initialize funds and set initial balance."""
    global initial_futures_balance, current_position
    try:
        fut_bal = exchange_futures.fetch_balance({'type': 'future'})
        initial_futures_balance = float(fut_bal.get('USDT', {}).get("free", 0))
        current_position = float(fut_bal.get('BTC', {}).get("free", 0))
        if initial_futures_balance < MIN_USDT:
            print_error(f"Initial Futures USDT balance {initial_futures_balance:.2f} below minimum {MIN_USDT} USDT.")
            sys.exit(1)
        print_info(f"Initial Futures USDT: {initial_futures_balance:.2f}, BTC: {current_position:.8f}")
    except Exception as e:
        print_error(f"Error initializing funds: {e}")
        sys.exit(1)

# Portfolio and Profit Display
def display_portfolio() -> None:
    """Display current portfolio balances."""
    try:
        bal = exchange_futures.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        btc = float(bal.get('BTC', {}).get("free", 0))
        print_info(f"[PORTFOLIO] USDT: {usdt:.8f}, BTC: {btc:.8f}")
    except Exception as e:
        print_error(f"Error displaying portfolio: {e}")

def display_profit(initial_balance: float) -> None:
    """Display profit since initialization."""
    try:
        bal = exchange_futures.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("free", 0))
        profit = current_usdt - initial_balance
        profit_str = f"Profit: {profit:.2f} USDT"
        print_info(f"[PROFIT] {profit_str}" if profit >= 0 else f"[LOSS] {profit_str}")
    except Exception as e:
        print_error(f"Error displaying profit: {e}")

# Technical Indicators
def calculate_indicators(prices: List[float]) -> Dict[str, float]:
    """Calculate technical indicators from price data."""
    try:
        df = pd.DataFrame(prices, columns=['close'])
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        return df.iloc[-1].to_dict()
    except Exception as e:
        print_error(f"Error calculating indicators: {e}")
        return {'rsi': 0.0}

# Order Management
MIN_USDT = 100.0
MIN_BTC = 0.00105
TRADE_PERCENTAGE = 0.10

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float) -> Optional[float]:
    """Adjust quantity to comply with minimums and balance constraints."""
    try:
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        lot_step = next((float(f.get('stepSize')) for f in filters if f.get('filterType') == 'LOT_SIZE'), 1e-8)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        if side.upper() == 'BUY':
            final_qty = max(calculated_qty, MIN_USDT / current_price)
            available_usdt = float(bal.get('USDT', {}).get("free", 0))
            final_qty = min(final_qty, available_usdt / current_price)
        elif side.upper() == 'SELL':
            final_qty = max(calculated_qty, max(MIN_BTC, MIN_USDT / current_price))
            available_btc = float(bal.get('BTC', {}).get("free", 0))
            final_qty = min(final_qty, available_btc)
        else:
            return None
        return math.floor(final_qty / lot_step) * lot_step
    except Exception as e:
        print_error(f"Error adjusting quantity: {e}")
        return None

def execute_order(symbol: str, decision: str, final_quantity: float, current_price: float) -> Optional[Dict[str, Any]]:
    """Execute a market order."""
    global current_position, entry_price
    try:
        if decision.upper() == "BUY":
            order = exchange_futures.create_market_buy_order(symbol, final_quantity, params={"reduceOnly": False})
            current_position = final_quantity
            entry_price = current_price
        elif decision.upper() == "SELL":
            order = exchange_futures.create_market_sell_order(symbol, final_quantity, params={"reduceOnly": False})
            current_position = 0.0
            entry_price = None
        print_info(f"[TRADE] {decision.upper()} {final_quantity} units at {current_price} USDT")
        telegram_notify(f"[TRADE] {decision.upper()} {final_quantity} units at {current_price} USDT")
        return order
    except Exception as e:
        print_error(f"Error executing {decision.upper()} order: {e}")
        return None

# Model Loading
try:
    ydf_model = load_ydf_model("model_rf.ydf")
    nn_model = ct.models.MLModel(NN_MODEL_PATH)
    load_mlx_model()
    print_info("Ensemble models loaded successfully.")
except Exception as e:
    print_error(f"Error loading models: {e}")
    sys.exit(1)

# Trading Logic
historical_prices: Dict[str, List[float]] = {}
WINDOW_SIZE = 50
COOLDOWN = 5.0
SLEEP_INTERVAL = 0.5

def get_latest_price(symbol: str) -> Optional[float]:
    """Fetch the latest price."""
    try:
        return exchange_futures.fetch_ticker(symbol).get('last')
    except Exception as e:
        print_error(f"Error fetching price: {e}")
        return None

def trading_logic(symbol: str, last_trade_time: Dict[str, float]) -> Dict[str, float]:
    """Main trading logic with position tracking, RSI, and profit/loss rules."""
    global current_position, entry_price
    current_price = get_latest_price(symbol)
    if current_price is None:
        return last_trade_time

    if symbol not in historical_prices:
        historical_prices[symbol] = []
    historical_prices[symbol].append(current_price)
    if len(historical_prices[symbol]) > WINDOW_SIZE:
        historical_prices[symbol].pop(0)

    indicators = calculate_indicators(historical_prices[symbol])
    rsi = indicators.get('rsi', 0.0)
    print_info(f"[DATA] {symbol} - Price: {current_price:.2f} USDT | RSI: {rsi:.2f}")

    features = {"price": current_price, "rsi": rsi}  # Simplified for brevity
    decision = get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)

    # RSI Thresholds
    if rsi < 30 and decision != "SELL":
        decision = "BUY"
    elif rsi > 70 and decision != "BUY":
        decision = "SELL"

    # Profit-Taking and Stop-Loss
    if current_position and entry_price:
        profit_pct = (current_price - entry_price) / entry_price * 100
        if profit_pct > 2:  # 2% profit
            decision = "SELL"
        elif profit_pct < -1:  # 1% loss
            decision = "SELL"

    if decision in ["BUY", "SELL"]:
        now = time.time()
        if now - last_trade_time.get(symbol, 0) < COOLDOWN:
            print_info(f"Cooldown active for {symbol}.")
            return last_trade_time

        if decision == "BUY" and current_position:
            print_info(f"Already holding position in {symbol}. No BUY executed.")
            return last_trade_time
        if decision == "SELL" and not current_position:
            print_info(f"No position to sell in {symbol}.")
            return last_trade_time

        bal = exchange_futures.fetch_balance({'type': 'future'})
        qty = TRADE_PERCENTAGE * (float(bal.get('USDT', {}).get("free", 0)) / current_price if decision == "BUY" else current_position)
        final_qty = adjust_quantity(symbol, decision, qty, current_price)
        if final_qty and final_qty > 0:
            execute_order(symbol, decision, final_qty, current_price)
            last_trade_time[symbol] = now
    else:
        print_info(f"HOLD predicted for {symbol}.")
    return last_trade_time

# Main Loop
def main() -> None:
    """Main trading loop."""
    initialize_funds()
    display_portfolio()
    symbol = "BTC/USDT:USDT"
    last_trade_time = {symbol: 0}
    print_info(f"Starting trading bot for {symbol}...")
    exchange_futures.set_leverage(20, symbol)

    initial_balance = initial_futures_balance or 0
    last_profit_time = time.time()

    while True:
        try:
            last_trade_time = trading_logic(symbol, last_trade_time)
            if time.time() - last_profit_time >= 60:
                display_profit(initial_balance)
                last_profit_time = time.time()
            time.sleep(SLEEP_INTERVAL)
        except Exception as e:
            print_error(f"Trading loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()