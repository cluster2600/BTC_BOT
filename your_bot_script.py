#!/usr/bin/env python3
"""
DISCLAIMER:
This code is for educational purposes only. I am not a financial advisor.
High-frequency trading involves significant risk. Always test on the Binance Futures
Testnet before going live.

This bot implements a HFT strategy for Binance Futures (BTC/USDT) using a pre-trained
Random Forest model (saved as 'model_rf.pkl'). It calculates technical indicators,
predicts trade decisions (BUY/SELL/HOLD), manages fund transfers between Spot and Futures,
executes market orders (without leverage by default), and logs profit in blue while errors
are shown in red. Telegram notifications are sent for major events.
"""

import ccxt
import numpy as np
import pandas as pd
import time, os, sys, json, logging, requests, datetime as dt
import joblib
from telebot import TeleBot
from telebot import types

# ----------------- UTILITY FUNCTIONS FOR COLOR PRINTING -----------------
def print_error(message):
    print(f"\033[91m{message}\033[0m")

def print_profit(message):
    print(f"\033[94m{message}\033[0m")

# ----------------- SECRETS LOADING -----------------
def read_secrets(filename):
    secrets = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        secrets[key.strip()] = value.strip()
    except Exception as e:
        print_error(f"[ERREUR] Reading {filename}: {e}")
    return secrets

# Load Telegram credentials from secrets.txt
secrets = read_secrets("secrets.txt")
TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = secrets.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print_error("[ERREUR] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in secrets.txt.")
    sys.exit(1)
else:
    print("[INFO] Telegram secrets loaded successfully.")

# Initialize TeleBot using the loaded credentials
bot = TeleBot(TELEGRAM_TOKEN)

def telegram_notify(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={message}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print_error(f"[ERREUR] Telegram notification: {e}")

# ----------------- LOGGING CONFIGURATION -----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('bot_logfile.log')
formatter = logging.Formatter('%(asctime)s %(message)s','%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Date                Price             Profit             Decision")

# ----------------- BINANCE FUTURES SETUP -----------------
# Read API keys from file (apikeys.txt should contain BINANCE_API_KEY and BINANCE_API_SECRET)
def read_api_keys(filename):
    keys = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        keys[key.strip()] = value.strip()
    except Exception as e:
        print_error(f"[ERREUR] Reading API keys: {e}")
    return keys

api_keys = read_api_keys("apikeys.txt")
BINANCE_API_KEY = api_keys.get("BINANCE_API_KEY")
BINANCE_API_SECRET = api_keys.get("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print_error("[ERREUR] Missing Binance API keys in apikeys.txt.")
    sys.exit(1)

USE_TESTNET = True  # True for testnet mode

exchange_config = {
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
}
if USE_TESTNET:
    exchange_config['urls'] = {
        'api': {
            'public': 'https://testnet.binancefuture.com/fapi/v1',
            'private': 'https://testnet.binancefuture.com/fapi/v1'
        }
    }
    print("[INFO] TESTNET mode activated.")

exchange = ccxt.binance(exchange_config)
try:
    exchange.load_markets()
    print("[INFO] Connected to Binance Futures and markets loaded.")
except Exception as e:
    print_error(f"[ERREUR] Loading markets: {e}")
    sys.exit(1)

def check_server_time():
    try:
        server_time = exchange.fetch_time()
        print(f"[INFO] Server time: {server_time} ms")
    except Exception as e:
        print_error(f"[ERREUR] Server time: {e}")

check_server_time()

# ----------------- SYMBOL SELECTION -----------------
DEFAULT_SYMBOL = "BTC/USDT"
if DEFAULT_SYMBOL in exchange.markets:
    SYMBOL = DEFAULT_SYMBOL
else:
    SYMBOL = None
    for m in exchange.markets:
        if "BTC" in m and "USDT" in m:
            SYMBOL = m
            print(f"[WARNING] {DEFAULT_SYMBOL} not found, using {SYMBOL}")
            break
    if not SYMBOL:
        print_error("[ERREUR] No BTC/USDT symbol found.")
        sys.exit(1)
SYMBOLS = [SYMBOL]

# ----------------- GLOBAL VARIABLES -----------------
WINDOW_SIZE = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
COOLDOWN = 0.5
SLEEP_INTERVAL = 0.1
initial_futures_balance = None
historical_prices = {symbol: [] for symbol in SYMBOLS}

# ----------------- TECHNICAL INDICATORS -----------------
def calculate_rsi(prices, period=RSI_PERIOD):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(prices) < slow:
        return None, None, None
    ema_fast = np.mean(prices[-fast:])
    ema_slow = np.mean(prices[-slow:])
    macd = ema_fast - ema_slow
    signal_line = np.mean(prices[-signal:])  # Simplified calculation
    return macd, signal_line, macd - signal_line

def calculate_bbands(prices, period=WINDOW_SIZE, num_std=2):
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    sma = np.mean(window)
    std = np.std(window)
    return sma - num_std * std, sma, sma + num_std * std

# ----------------- LOT SIZE ADJUSTMENT -----------------
def adjust_lot_size(symbol, quantity):
    try:
        filters = exchange.markets[symbol]['info'].get('filters', [])
        step_size = None
        for f in filters:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                break
        if step_size:
            quantity = np.floor(quantity / step_size) * step_size
        return quantity
    except Exception as e:
        print_error(f"[ERREUR] LOT_SIZE adjustment for {symbol}: {e}")
        return quantity

# ----------------- PORTFOLIO MANAGEMENT -----------------
def display_portfolio():
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        print("\n[PORTFOLIO] Current:")
        for asset in ["USDT", "BTC", "BNB"]:
            asset_info = bal.get(asset, {})
            free = asset_info.get("free", 0)
            total = asset_info.get("total", 0)
            print(f"  {asset}: Total = {total}, Free = {free}")
        print("")
    except Exception as e:
        print_error(f"[ERREUR] Fetching portfolio: {e}")

def display_profit(initial_balance):
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get('free', 0))
        profit = current_usdt - initial_balance
        print_profit(f"[PROFIT] Realized Profit: {profit:.2f} USDT")
    except Exception as e:
        print_error(f"[ERREUR] Profit display: {e}")

# ----------------- PRICE RETRIEVAL -----------------
def get_latest_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print_error(f"[ERREUR] Price for {symbol}: {e}")
        return None

# ----------------- QUANTITY CALCULATION -----------------
def calculate_dynamic_quantity(symbol, side, current_price):
    bal = exchange.fetch_balance({'type': 'future'})
    market = exchange.markets[symbol]
    min_cost = float(market['limits']['cost']['min'])
    if side.upper() == 'BUY':
        available_usdt = float(bal.get('USDT', {}).get('free', 0))
        min_order_amt = float(market['limits']['amount']['min'])
        if available_usdt < current_price * min_order_amt:
            print_error(f"[ALERTE] Insufficient funds for BUY: {current_price * min_order_amt:.2f} USDT needed, available: {available_usdt} USDT.")
            return None
        print(f"[DEBUG] Available USDT for BUY: {available_usdt}")
        quantity = available_usdt / current_price
        print(f"[DEBUG] Raw BUY quantity: {quantity}")
    elif side.upper() == 'SELL':
        asset = symbol.replace("/", "")
        available_asset = float(bal.get(asset, {}).get("free", 0))
        print(f"[DEBUG] Available {asset} for SELL: {available_asset}")
        quantity = available_asset
    else:
        return None

    if current_price * quantity < min_cost:
        print_error(f"[ALERTE] Notional too low for {symbol} {side}: {current_price * quantity:.2f} USDT, min {min_cost} USDT.")
        return None
    return quantity

def adjust_quantity(symbol, side, quantity, current_price):
    market = exchange.markets[symbol]
    min_order_amt = float(market['limits']['amount']['min'])
    min_cost = float(market['limits']['cost']['min'])
    required_by_cost = min_cost / current_price
    valid_quantity = max(quantity, required_by_cost, min_order_amt)
    
    bal = exchange.fetch_balance({'type': 'future'})
    if side.upper() == 'BUY':
        available_usdt = float(bal.get('USDT', {}).get('free', 0))
        if current_price * valid_quantity > available_usdt:
            valid_quantity = available_usdt / current_price
    elif side.upper() == 'SELL':
        asset = symbol.replace("/", "")
        available_asset = float(bal.get(asset, {}).get("free", 0))
        if valid_quantity > available_asset:
            valid_quantity = available_asset

    if current_price * valid_quantity < min_cost:
        print_error(f"[ALERTE] Final quantity too low for {symbol}: {current_price * valid_quantity:.2f} USDT < {min_cost} USDT")
        return None

    precision = 8 if symbol.startswith("BTC") else 2
    rounded_qty = round(valid_quantity, precision)
    final_qty = adjust_lot_size(symbol, rounded_qty)
    if final_qty < 0.001:
        print_error(f"[ERREUR] Final adjusted quantity ({final_qty}) is below the minimum (0.001 BTC) for {symbol}.")
        return None

    print(f"[DEBUG] Final adjusted quantity for {symbol}: {final_qty}")
    return final_qty

# ----------------- ORDER EXECUTION (NO LEVERAGE) -----------------
def execute_order(symbol, decision, final_quantity, current_price):
    try:
        if decision.upper() == "BUY":
            order = exchange.create_market_buy_order(symbol, final_quantity, params={"reduceOnly": False})
            stop_price = current_price * 0.99  # Example stop loss at 1% below current price
            try:
                order_stop = exchange.create_order(
                    symbol,
                    type="stop_market",
                    side="sell",
                    amount=final_quantity,
                    params={"stopPrice": stop_price, "reduceOnly": False}
                )
                print(f"[STOP-LOSS] Stop-loss order placed at {stop_price:.2f} USDT for {symbol}.")
            except Exception as e:
                print_error(f"[ERREUR] Stop-loss for {symbol}: {e}")
        elif decision.upper() == "SELL":
            order = exchange.create_market_sell_order(symbol, final_quantity, params={"reduceOnly": False})
        print(f"[TRADE] {decision.upper()} order executed on {symbol} with quantity {final_quantity} at price {current_price} USDT.\nOrder result: {order}")
        return order
    except Exception as e:
        print_error(f"[ERREUR] Order execution {decision.upper()} on {symbol}: {e}")
        telegram_notify(f"[ALERTE] Order {decision.upper()} on {symbol} error: {e}")
        return None

# ----------------- MODEL PREDICTION -----------------
def predict_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands):
    lower_bb, sma_bb, upper_bb = bbands if bbands is not None else (0, 0, 0)
    features = [
        current_price,
        moving_average,
        rsi if rsi is not None else 0,
        macd if macd is not None else 0,
        signal_line if signal_line is not None else 0,
        lower_bb,
        sma_bb,
        upper_bb,
        0  # Default social feature for BTC
    ]
    print(f"[DEBUG] Features for {symbol}: {features}")
    try:
        model = joblib.load("model_rf.pkl")
        size_mb = os.path.getsize("model_rf.pkl") / (1024 * 1024)
        print(f"[DEBUG] Model 'model_rf.pkl' loaded, size: {size_mb:.2f} MB.")
    except Exception as e:
        print_error(f"[ERREUR] Model loading: {e}")
        return "HOLD"
    feature_names = ["price", "sma", "rsi", "macd", "signal", "lower_bb", "sma_bb", "upper_bb", "social_feature"]
    df_features = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(df_features)
    if prediction[0] == 1:
        return "BUY"
    elif prediction[0] == -1:
        return "SELL"
    else:
        return "HOLD"

# ----------------- TRADING LOGIC -----------------
def trading_logic(symbol, last_trade_time):
    current_price = get_latest_price(symbol)
    if current_price is None:
        return last_trade_time

    historical_prices[symbol].append(current_price)
    if len(historical_prices[symbol]) > WINDOW_SIZE:
        historical_prices[symbol].pop(0)
    moving_average = np.mean(historical_prices[symbol])
    rsi = calculate_rsi(historical_prices[symbol])
    macd, signal_line, _ = calculate_macd(historical_prices[symbol])
    bbands = calculate_bbands(historical_prices[symbol])
    rsi_display = f"{rsi:.2f}" if rsi is not None else "N/A"
    print(f"[DATA] {symbol} - Price: {current_price:.2f} USDT | SMA: {moving_average:.2f} | RSI: {rsi_display}")

    decision = predict_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands)
    print(f"[INFO] Predicted decision for {symbol}: {decision}")

    # Fallback: if SELL but insufficient BTC, force BUY
    if decision.upper() == "SELL":
        bal = exchange.fetch_balance({'type': 'future'})
        available_btc = float(bal.get("BTC", {}).get("free", 0))
        min_order_amt = float(exchange.markets[symbol]['limits']['amount']['min'])
        if available_btc < min_order_amt:
            decision = "BUY"
            print(f"[FALLBACK] BTC balance ({available_btc}) insufficient (< {min_order_amt}), forcing BUY.")
            print(f"[INFO] Fallback decision for {symbol}: {decision}")

    # Additional fallback based on RSI if decision is HOLD
    if decision.upper() == "HOLD" and rsi is not None:
        bal = exchange.fetch_balance({'type': 'future'})
        available_btc = float(bal.get("BTC", {}).get("free", 0))
        if available_btc < float(exchange.markets[symbol]['limits']['amount']['min']):
            decision = "BUY"
            print("[FALLBACK] Insufficient BTC for SELL, forcing BUY.")
        else:
            decision = "BUY" if rsi <= 50 else "SELL"
            print(f"[FALLBACK] RSI {'<= 50' if rsi <= 50 else '> 50'}, forcing {decision}.")
        print(f"[INFO] Fallback decision for {symbol}: {decision}")

    if decision.upper() in ["BUY", "SELL"]:
        now = time.time()
        if now - last_trade_time.get(symbol, 0) < COOLDOWN:
            print(f"[INFO] Cooldown active for {symbol}. No trade executed.")
            return last_trade_time

        quantity_calculated = calculate_dynamic_quantity(symbol, decision, current_price)
        if not quantity_calculated:
            print(f"[INFO] Calculated quantity = 0, no trade executed for {symbol} {decision}.")
            return last_trade_time

        final_quantity = adjust_quantity(symbol, decision, quantity_calculated, current_price)
        if not final_quantity:
            print(f"[INFO] Final quantity = 0, no trade executed for {symbol} {decision}.")
            return last_trade_time

        # For SELL, check minimum order amount constraint
        min_order_amt = float(exchange.markets[symbol]['limits']['amount']['min'])
        if decision.upper() == "SELL" and final_quantity < min_order_amt:
            print_error(f"[ERREUR] Final quantity ({final_quantity}) for SELL is below minimum ({min_order_amt}). No trade executed.")
            return last_trade_time

        notional = current_price * final_quantity
        min_cost = float(exchange.markets[symbol]['limits']['cost']['min'])
        print(f"[DEBUG] Order notional: {notional:.2f} USDT (min required: {min_cost} USDT)")

        order = execute_order(symbol, decision, final_quantity, current_price)
        if order is not None:
            print(f"[INFO] Order executed on {symbol}.")
            last_trade_time[symbol] = now
            telegram_notify(f"[TRADE] {decision.upper()} order on {symbol}: {final_quantity} at {current_price} USDT.")
    else:
        print(f"[INFO] Model predicts HOLD for {symbol}. No trade executed.")
    return last_trade_time

# ----------------- FUND TRANSFERS -----------------
def transfer_bnb_to_futures():
    try:
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        bnb_spot = float(spot_bal.get('BNB', {}).get('free', 0))
        bnb_futures = float(futures_bal.get('BNB', {}).get('free', 0))
        print(f"[INFO] Spot BNB: {bnb_spot}, Futures BNB: {bnb_futures}")
        if bnb_spot > 0:
            print(f"[INFO] Transferring {bnb_spot} BNB from Spot to Futures for fee reduction.")
            result = exchange.transfer('BNB', bnb_spot, 'spot', 'future')
            print(f"[INFO] BNB Transfer result: {result}")
        else:
            print("[INFO] No BNB to transfer.")
    except Exception as e:
        print_error(f"[ERREUR] BNB Transfer: {e}")

def initialize_funds():
    global initial_futures_balance
    try:
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        
        usdt_spot = float(spot_bal.get('USDT', {}).get('free', 0))
        usdt_futures = float(futures_bal.get('USDT', {}).get('free', 0))
        print(f"[INFO] Spot USDT: {usdt_spot}, Futures USDT: {usdt_futures}")
        if usdt_spot > 0:
            print(f"[INFO] Transferring {usdt_spot} USDT from Spot to Futures.")
            result_usdt = exchange.transfer('USDT', usdt_spot, 'spot', 'future')
            print(f"[INFO] USDT Transfer result: {result_usdt}")
        else:
            print("[INFO] No USDT to transfer.")
        
        btc_spot = float(spot_bal.get('BTC', {}).get('free', 0))
        btc_futures = float(futures_bal.get('BTC', {}).get('free', 0))
        print(f"[INFO] Spot BTC: {btc_spot}, Futures BTC: {btc_futures}")
        if btc_spot > 0:
            print(f"[INFO] Transferring {btc_spot} BTC from Spot to Futures.")
            result_btc = exchange.transfer('BTC', btc_spot, 'spot', 'future')
            print(f"[INFO] BTC Transfer result: {result_btc}")
        else:
            print("[INFO] No BTC to transfer.")
        
        transfer_bnb_to_futures()
        
        futures_bal = exchange.fetch_balance({'type': 'future'})
        initial_futures_balance = float(futures_bal.get('USDT', {}).get('free', 0))
        print(f"[INFO] Initial Futures USDT: {initial_futures_balance:.2f}")
    except Exception as e:
        print_error(f"[ERREUR] Initializing funds: {e}")

# ----------------- MAIN LOOP -----------------
def main():
    last_trade_time = {symbol: 0 for symbol in SYMBOLS}
    iteration = 0
    iterations_per_minute = int(60 / SLEEP_INTERVAL)
    print("[INFO] Starting HFT Trading Bot for BTC with Random Forest (no leverage)...\n")
    
    initialize_funds()
    display_portfolio()
    while True:
        for symbol in SYMBOLS:
            last_trade_time = trading_logic(symbol, last_trade_time)
        iteration += 1
        if iteration % 50 == 0:
            display_portfolio()
        if iteration % iterations_per_minute == 0 and initial_futures_balance is not None:
            display_profit(initial_futures_balance)
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()
