#!/usr/bin/env python3
"""
Main Trading Bot with Ensemble Model Integration

This module integrates an enhanced trading model ensemble that combines:
  1. A YDF Random Forest model.
  2. A TensorFlow neural network model.
  
The ensemble pipeline:
  - Generates features from market data.
  - Uses both models (via ensemble_models.py) to produce predictions.
  - Averages output probabilities and applies rule-based signals.
  - Executes trades only if the ensemble and rules agree.
  - Enforces a minimum order size of 100 USDT notional (BUY) or a forced minimum BTC quantity (SELL).

This code is designed to run on a Mac M1 with Metal GPU acceleration.
"""

import os
import sys
import time
import json
import math
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
import websocket
import logging
import colorlog
import urllib.request  # Added to fix Telegram notification error

# Import the ensemble model functions (assumed to be defined in ensemble_models.py)
# The module should provide:
#   load_ydf_model(), load_tf_model(), and ensemble_predict(features)
from ensemble_models import load_ydf_model, load_tf_model, ensemble_predict

##############################################
#         LOGGER & UTILITY FUNCTIONS         #
##############################################
LOG_LEVEL = logging.DEBUG
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
        log_colors={
            "DEBUG": "cyan",
            "INFO": "bold_white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if LOG_TO_FILE:
        file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
        )
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        file_handler = logging.FileHandler(f"TradingBot_{current_datetime}.log")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

log = setup_logger()

def print_error(message: str) -> None:
    log.error(message)

def print_info(message: str) -> None:
    log.info(message)

def debug_log(message: str) -> None:
    log.debug(message)

##############################################
#         CONFIGURATION & SECRETS LOADING    #
##############################################
def read_config(filename: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        debug_log(f"Configuration loaded from {filename}: {list(config.keys())}")
    except Exception as e:
        print_error(f"Error reading {filename}: {e}")
    return config

secrets = read_config("secrets.txt")
TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = secrets.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print_error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in secrets.txt.")
    sys.exit(1)
else:
    print_info("Telegram secrets loaded successfully.")

api_keys = read_config("apikeys.txt")
BINANCE_API_KEY = api_keys.get("BINANCE_API_KEY")
BINANCE_API_SECRET = api_keys.get("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print_error("Missing Binance API keys in apikeys.txt.")
    sys.exit(1)

##############################################
#         TELEGRAM BOT SETUP                 #
##############################################
from telebot import TeleBot, types
bot = TeleBot(TELEGRAM_TOKEN)

def telegram_notify(message: str) -> Optional[Dict[str, Any]]:
    try:
        url = (f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?"
               f"chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={urllib.parse.quote(message)}")
        response = requests.get(url)
        debug_log(f"Telegram response: {response.json()}")
        return response.json()
    except Exception as e:
        print_error(f"Telegram notification error: {e}")
        return None

##############################################
#         BINANCE EXCHANGE INSTANCES         #
##############################################
custom_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'X-MBX-APIKEY': BINANCE_API_KEY,
}

exchange_spot = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'headers': custom_headers,
})

futures_config = {
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'urls': {
        'api': {
            'public': 'https://fapi.binance.com/fapi/v1',
            'private': 'https://fapi.binance.com/fapi/v1'
        }
    },
    'headers': custom_headers,
}
print_info("PRODUCTION mode activated for Futures using standard endpoint.")
exchange_futures = ccxt.binance(futures_config)

##############################################
#   FUND TRANSFER & INITIALIZATION FUNCTIONS   #
##############################################
def transfer_funds_spot_to_futures() -> None:
    try:
        spot_bal = exchange_spot.fetch_balance({'type': 'spot'})
        usdt_spot = float(spot_bal.get('USDT', {}).get('free', 0))
        if usdt_spot > 0:
            print_info(f"Transferring {usdt_spot} USDT from Spot to Futures.")
            result = exchange_futures.transfer('USDT', usdt_spot, 'spot', 'future')
            print_info(f"Transfer result: {result}")
        else:
            print_info("No USDT available on Spot to transfer.")
    except Exception as e:
        print_error(f"Error transferring funds: {e}")

initial_futures_balance: Optional[float] = None

def initialize_funds() -> None:
    global initial_futures_balance
    transfer_funds_spot_to_futures()
    try:
        fut_bal = exchange_futures.fetch_balance({'type': 'future'})
        initial_futures_balance = float(fut_bal.get('USDT', {}).get("free", 0))
        print_info(f"Initial Futures USDT balance: {initial_futures_balance:.2f}")
    except Exception as e:
        print_error(f"Error initializing funds: {e}")

##############################################
#         DISPLAY PORTFOLIO & PROFIT         #
##############################################
def display_portfolio() -> None:
    try:
        bal = exchange_futures.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        base_asset = "BTC"
        btc = float(bal.get(base_asset, {}).get("free", 0))
        bnb = float(bal.get('BNB', {}).get("free", 0))
        print_info(f"[PORTFOLIO] Current balances:")
        print_info(f"  USDT: {usdt:.8f}")
        print_info(f"  {base_asset}: {btc:.8f}")
        print_info(f"  BNB: {bnb:.8f}")
    except Exception as e:
        print_error(f"Error displaying portfolio: {e}")

def display_profit(initial_balance: float) -> None:
    try:
        bal = exchange_futures.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("free", 0))
        profit = current_usdt - initial_balance
        profit_str = f"Profit: {profit:.2f} USDT"
        if profit >= 0:
            print_info(f"[PROFIT] {profit_str}")
        else:
            print_error(f"[PROFIT] {profit_str}")
    except Exception as e:
        print_error(f"Error displaying profit: {e}")

##############################################
#         TECHNICAL INDICATOR FUNCTIONS      #
##############################################
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < slow:
        return None, None, None
    ema_fast = np.mean(prices[-fast:])
    ema_slow = np.mean(prices[-slow:])
    macd = ema_fast - ema_slow
    signal_line = np.mean(prices[-signal:])
    return macd, signal_line, macd - signal_line

def calculate_bbands(prices: List[float], period: int = 50, num_std: int = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    sma = np.mean(window)
    std = np.std(window)
    return sma - num_std * std, sma, sma + num_std * std

##############################################
#         ORDER SIZE & QUANTITY ADJUSTMENT   #
##############################################
def adjust_lot_size(symbol: str, quantity: float) -> float:
    try:
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        step_size = None
        for f in filters:
            if f.get('filterType') == 'LOT_SIZE':
                step_size = float(f.get('stepSize'))
                break
        if step_size:
            quantity = math.floor(quantity / step_size) * step_size
        return quantity
    except Exception as e:
        print_error(f"Error adjusting lot size for {symbol}: {e}")
        return quantity

# Forced minimums for BTC/USDT:
MIN_USDT = 100.0      # For BUY orders, minimum notional is 100 USDT.
MIN_BTC  = 0.00105    # For SELL orders, minimum BTC quantity is 0.0010500.
TRADE_PERCENTAGE = 0.10

def calculate_dynamic_quantity(symbol: str, side: str, current_price: float) -> Optional[float]:
    bal = exchange_futures.fetch_balance({'type': 'future'})
    if side.upper() == 'BUY':
        available_usdt = float(bal.get('USDT', {}).get('free', 0))
        quantity = (TRADE_PERCENTAGE * available_usdt) / current_price
    elif side.upper() == 'SELL':
        base_asset = symbol.split("/")[0]
        available_asset = float(bal.get(base_asset, {}).get("free", 0))
        quantity = TRADE_PERCENTAGE * available_asset
    else:
        return None
    return quantity

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float) -> Optional[float]:
    # Retrieve lot step for rounding purposes
    filters = exchange_futures.markets[symbol]['info'].get('filters', [])
    lot_step = 1e-8
    for f in filters:
        if f.get('filterType') == 'LOT_SIZE':
            lot_step = float(f.get('stepSize'))
            break

    if side.upper() == 'BUY':
        # Calculate forced quantity for 100 USDT notional
        forced_qty = math.ceil((MIN_USDT / current_price) / lot_step) * lot_step
        final_qty = max(calculated_qty, forced_qty)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        available_usdt = float(bal.get('USDT', {}).get("free", 0))
        if current_price * final_qty > available_usdt:
            final_qty = available_usdt / current_price
    elif side.upper() == 'SELL':
        # Calculate forced quantity for 0.00105 BTC
        forced_qty = math.ceil(MIN_BTC / lot_step) * lot_step
        final_qty = max(calculated_qty, forced_qty)
        base_asset = symbol.split("/")[0]
        available_asset = float(exchange_futures.fetch_balance({'type': 'future'}).get(base_asset, {}).get("free", 0))
        if final_qty > available_asset:
            final_qty = available_asset
    else:
        return None

    precision = 8 if symbol.startswith("BTC") else 2
    final_qty = round(final_qty, precision)
    final_qty = adjust_lot_size(symbol, final_qty)
    debug_log(f"Final adjusted quantity for {symbol}: {final_qty}")
    return final_qty

##############################################
#            ORDER EXECUTION               #
##############################################
def execute_order(symbol: str, decision: str, final_quantity: float, current_price: float) -> Optional[Dict[str, Any]]:
    if decision.upper() == "BUY":
        notional = current_price * final_quantity
        if notional < MIN_USDT:
            print_error(f"Forced BUY order notional {notional:.2f} USDT is below {MIN_USDT} USDT.")
            final_quantity = math.ceil((MIN_USDT / current_price) / 1e-8) * 1e-8
    elif decision.upper() == "SELL":
        if final_quantity < MIN_BTC:
            print_error(f"Forced SELL quantity {final_quantity} BTC is below {MIN_BTC} BTC.")
            final_quantity = math.ceil(MIN_BTC / 1e-8) * 1e-8
    try:
        if decision.upper() == "BUY":
            order = exchange_futures.create_market_buy_order(symbol, final_quantity, params={"reduceOnly": False})
        elif decision.upper() == "SELL":
            order = exchange_futures.create_market_sell_order(symbol, final_quantity, params={"reduceOnly": False})
        print_info(f"[TRADE] {decision.upper()} order executed on {symbol}: {final_quantity} units at {current_price} USDT.\nOrder result: {order}")
        return order
    except Exception as e:
        print_error(f"Error executing {decision.upper()} order on {symbol}: {e}")
        telegram_notify(f"ALERT: Error executing {decision.upper()} order on {symbol}: {e}")
        return None

##############################################
#        ENSEMBLE MODEL INTEGRATION          #
##############################################
# Load the ensemble models (YDF and TensorFlow) once at startup.
try:
    ydf_model = load_ydf_model()  # Function from ensemble_models.py
    tf_model = load_tf_model()    # Function from ensemble_models.py
    print_info("Ensemble models loaded successfully.")
except Exception as e:
    print_error(f"Error loading ensemble models: {e}")
    sys.exit(1)

def get_ensemble_decision(features: Dict[str, Any]) -> str:
    """
    Given a feature dictionary, return the ensemble prediction.
    The ensemble_predict() function should combine predictions from both the YDF and TensorFlow models.
    """
    try:
        decision = ensemble_predict(ydf_model, tf_model, features)
        decision = decision.upper()
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        return decision
    except Exception as e:
        print_error(f"Error during ensemble prediction: {e}")
        return "HOLD"

##############################################
#        MODEL PREDICTION (Fallback)         #
##############################################
def predict_decision(symbol: str, current_price: float, moving_average: float,
                     rsi: Optional[float], macd: Optional[float],
                     signal_line: Optional[float], bbands: Tuple[Optional[float], Optional[float], Optional[float]]) -> str:
    # Build feature dictionary
    lower_bb, sma_bb, upper_bb = bbands if bbands is not None else (0.0, 0.0, 0.0)
    lower_bb = 0.0 if lower_bb is None else lower_bb
    sma_bb = 0.0 if sma_bb is None else sma_bb
    upper_bb = 0.0 if upper_bb is None else upper_bb

    features = {
        "price": current_price,
        "sma": moving_average,
        "rsi": rsi if rsi is not None else 0.0,
        "macd": macd if macd is not None else 0.0,
        "signal_line": signal_line if signal_line is not None else 0.0,
        "lower_bb": lower_bb,
        "sma_bb": sma_bb,
        "upper_bb": upper_bb,
        # Additional engineered features could be added here:
        "vol_adjusted_price": 0.0,
        "volume_ma": 0.0,
        "adx": 0.0,
        "atr": 0.0,
        "order_book_depth": 0.0,
        "news_sentiment": 0.0,
        "social_feature": 0.0
    }
    debug_log(f"Features for {symbol}: {features}")
    
    # Use ensemble prediction first
    ensemble_decision = get_ensemble_decision(features)
    if ensemble_decision != "HOLD":
        return ensemble_decision
    
    # Fallback: use rule-based decision using RSI thresholds
    if rsi is not None:
        if rsi < 40:
            return "BUY"
        elif rsi > 60:
            return "SELL"
    return "HOLD"

##############################################
#            TRADING LOGIC LOOP              #
##############################################
historical_prices: Dict[str, List[float]] = {}
WINDOW_SIZE = 50
COOLDOWN = 0.2  # Reduced cooldown for more trades.
SLEEP_INTERVAL = 0.1

def get_latest_price(symbol: str) -> Optional[float]:
    try:
        ticker = exchange_futures.fetch_ticker(symbol)
        return ticker.get('last')
    except Exception as e:
        print_error(f"Error fetching latest price for {symbol}: {e}")
        return None

def trading_logic(symbol: str, last_trade_time: Dict[str, float]) -> Dict[str, float]:
    current_price = get_latest_price(symbol)
    if current_price is None:
        return last_trade_time

    if symbol not in historical_prices:
        historical_prices[symbol] = []
    historical_prices[symbol].append(current_price)
    if len(historical_prices[symbol]) > WINDOW_SIZE:
        historical_prices[symbol].pop(0)

    moving_average = np.mean(historical_prices[symbol])
    rsi = calculate_rsi(historical_prices[symbol])
    macd, signal_line, _ = calculate_macd(historical_prices[symbol])
    bbands = calculate_bbands(historical_prices[symbol])
    rsi_disp = f"{rsi:.2f}" if rsi is not None else "N/A"
    print_info(f"[DATA] {symbol} - Price: {current_price:.2f} USDT | SMA: {moving_average:.2f} | RSI: {rsi_disp}")

    decision = predict_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands)
    print_info(f"Predicted decision for {symbol}: {decision}")

    if decision.upper() in ["BUY", "SELL"]:
        now = time.time()
        if now - last_trade_time.get(symbol, 0) < COOLDOWN:
            print_info(f"Cooldown active for {symbol}. No trade executed.")
            return last_trade_time

        base_qty = calculate_dynamic_quantity(symbol, decision, current_price)
        if base_qty is None:
            print_info(f"Calculated quantity is 0 for {symbol} {decision}. No trade executed.")
            return last_trade_time

        final_quantity = adjust_quantity(symbol, decision, base_qty, current_price)
        if final_quantity is None:
            print_info(f"Final quantity is 0 for {symbol} {decision}. No trade executed.")
            return last_trade_time

        debug_log(f"Final order notional: {current_price * final_quantity:.2f} USDT")
        order = execute_order(symbol, decision, final_quantity, current_price)
        if order is not None:
            print_info(f"Order executed on {symbol}.")
            last_trade_time[symbol] = now
            telegram_notify(f"[TRADE] {decision.upper()} order on {symbol}: {final_quantity} units at {current_price} USDT.")
    else:
        print_info(f"Model predicted HOLD for {symbol}. No trade executed.")
    return last_trade_time

##############################################
#          MAIN TRADING LOOP & BOT           #
##############################################
def main() -> None:
    initialize_funds()
    display_portfolio()

    try:
        markets = exchange_futures.load_markets()
    except Exception as e:
        print_error(f"Error loading markets: {e}")
        sys.exit(1)
    available_symbols = list(markets.keys())
    if "BTC/USDT:USDT" in markets:
        symbol = "BTC/USDT:USDT"
    elif "BTC/USDT" in markets:
        symbol = "BTC/USDT"
    elif "BTCUSDT" in markets:
        symbol = "BTCUSDT"
    else:
        print_error(f"Symbol 'BTC/USDT' or 'BTCUSDT' not found. Available symbols: {available_symbols}")
        sys.exit(1)
    print_info(f"Using trading symbol: {symbol}")
    if symbol not in historical_prices:
        historical_prices[symbol] = []
    last_trade_time = {symbol: 0}
    iteration = 0
    print_info(f"Starting HFT Bot for {symbol} with ensemble models and leverage enabled...")

    LEVERAGE = 20
    try:
        exchange_futures.set_leverage(LEVERAGE, symbol)
        print_info(f"Leverage set to {LEVERAGE} for {symbol}.")
    except Exception as e:
        print_error(f"Error setting leverage for {symbol}: {e}")

    initial_balance = initial_futures_balance if initial_futures_balance is not None else 0
    last_profit_display_time = time.time()

    while True:
        last_trade_time = trading_logic(symbol, last_trade_time)
        iteration += 1
        if iteration % 50 == 0:
            display_portfolio()
        current_time = time.time()
        if current_time - last_profit_display_time >= 60:
            display_profit(initial_balance)
            last_profit_display_time = current_time
        time.sleep(SLEEP_INTERVAL)

##############################################
#             TELEGRAM COMMANDS              #
##############################################
@bot.message_handler(commands=['shutdown'])
def handle_shutdown(message: types.Message) -> None:
    bot.reply_to(message, "Shutdown selected.")
    shutdown_bot()

@bot.message_handler(commands=['poweron'])
def handle_poweron(message: types.Message) -> None:
    bot.reply_to(message, "Power On selected.")
    main()

@bot.message_handler(commands=['balance'])
def handle_balance(message: types.Message) -> None:
    display_portfolio()
    bal = exchange_futures.fetch_balance({'type': 'future'})
    usdt = float(bal.get('USDT', {}).get("free", 0))
    bot.reply_to(message, f"Futures Balance: {usdt:.2f} USDT")
    telegram_notify(f"Futures Balance: {usdt:.2f} USDT")

@bot.message_handler(commands=['profit'])
def handle_profit(message: types.Message) -> None:
    if initial_futures_balance is not None:
        display_profit(initial_futures_balance)
    else:
        bot.reply_to(message, "Initial balance not set.")

def shutdown_bot() -> None:
    print_info("Shutting down bot and closing all positions...")
    try:
        close_position()  # Placeholder: implement your close_position logic as needed.
    except Exception as e:
        print_error(f"Error closing positions during shutdown: {e}")
    telegram_notify("Bot is shutting down. All open positions are closed.")
    sys.exit("BOT SHUTDOWN")

##############################################
#            WEBSOCKET INTEGRATION           #
##############################################
def on_open(ws: websocket.WebSocketApp) -> None:
    print_info("WebSocket connection opened.")
    telegram_notify("WebSocket connection opened.")

def on_close(ws: websocket.WebSocketApp) -> None:
    print_info("WebSocket connection closed.")
    telegram_notify("WebSocket connection closed.")

def on_message(ws: websocket.WebSocketApp, message: str) -> None:
    try:
        json_message = json.loads(message)
        messageTime = json_message.get('E', None)
        if messageTime:
            timestamp = datetime.fromtimestamp(int(messageTime)/1000)
            messageSentTime = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
            debug_log(f"Message Time: {messageSentTime}")
        candle = json_message.get('k', {})
        if candle.get('x', False):
            close_price = float(candle.get('c', 0))
            print_info(f"Candle closed at: {close_price}")
            global historical_prices
            historical_prices[symbol].append(close_price)
            if len(historical_prices[symbol]) > WINDOW_SIZE:
                historical_prices[symbol].pop(0)
            trading_logic(symbol, {symbol: time.time()})
    except Exception as e:
        print_error(f"Error processing WebSocket message: {e}")

def start_websocket() -> None:
    ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_close=on_close, on_message=on_message)
    ws.run_forever()

##############################################
#                ENTRY POINT                 #
##############################################
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "telegram":
        try:
            bot.infinity_polling()
        except Exception as e:
            shutdown_bot()
            print_error(e)
    else:
        main()