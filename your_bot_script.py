#!/usr/bin/env python3
"""
Binance Futures HFT Trading Bot – Extended Production Code
============================================================
This bot does the following:
  - Tests connectivity using the ping endpoint.
  - Loads API keys and secrets from files (apikeys.txt and secrets.txt).
  - Connects to Binance Futures (or testnet if USE_TESTNET=True).
  - Retrieves public endpoint data (open interest, order book, historical trades, option mark price/Greeks, index price, klines).
  - Computes technical indicators (RSI, MACD, Bollinger Bands).
  - Adjusts order quantities according to market filters.
  - Executes market orders (BUY/SELL) based on a pre‑trained Random Forest model.
  - Displays portfolio balances and profit (errors in red, profits in blue).
  - Transfers funds from Spot to Futures (including BNB transfers for fee reduction).
  - Provides Telegram notifications and command handlers.
  - Listens to live WebSocket candlestick data and runs trading logic.
  - Also includes endpoints for old trades and option mark price/Greeks.
  
WARNING: This code is for educational purposes only. Use testnet mode for testing before live deployment.
"""

##############################################
#           IMPORTS & GLOBAL SETUP           #
##############################################
import ccxt, os, sys, json, time, logging, requests, datetime as dt, datetime, hmac, hashlib, urllib.parse
import numpy as np
import pandas as pd
import joblib
from telebot import TeleBot, types
import websocket
from pprint import pprint

# Uncomment if you need additional technical indicator libraries:
# import tulipy as ti

##############################################
#           UTILITY FUNCTIONS                #
##############################################
def print_error(message):
    print(f"\033[91m{message}\033[0m")  # Red text

def print_profit(message):
    print(f"\033[94m{message}\033[0m")  # Blue text

def debug_log(message):
    print(f"[DEBUG] {message}")

##############################################
#         CONFIG & SECRETS LOADING           #
##############################################
def read_file(filename):
    data = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        data[key.strip()] = value.strip()
    except Exception as e:
        print_error(f"[ERREUR] Reading {filename}: {e}")
    return data

# Load Telegram secrets from secrets.txt
secrets = read_file("secrets.txt")
TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = secrets.get("TELEGRAM_CHAT_ID")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print_error("[ERREUR] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in secrets.txt.")
    sys.exit(1)
else:
    print("[INFO] Telegram secrets loaded successfully.")

# Load Binance API keys from apikeys.txt
api_keys = read_file("apikeys.txt")
BINANCE_API_KEY = api_keys.get("BINANCE_API_KEY")
BINANCE_API_SECRET = api_keys.get("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print_error("[ERREUR] Missing Binance API keys in apikeys.txt.")
    sys.exit(1)

##############################################
#          TELEGRAM BOT SETUP                #
##############################################
bot = TeleBot(TELEGRAM_TOKEN)
def telegram_notify(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={urllib.parse.quote(message)}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print_error(f"[ERREUR] Telegram notification: {e}")

##############################################
#           LOGGING CONFIGURATION            #
##############################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_handler = logging.FileHandler('bot_logfile.log')
log_formatter = logging.Formatter('%(asctime)s %(message)s','%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)
logger.info("Date                Price             SAR             %K                 %D                 Open Profit                 Total Profit                 Pos Status                 Profit (Previous Trade)")

##############################################
#         BINANCE FUTURES SETUP              #
##############################################
USE_TESTNET = True  # Set to False for live trading
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
exchange.options['fapiV2'] = True

##############################################
#      SYMBOL CHECKING & MARKET LOADING      #
##############################################
try:
    exchange.load_markets()
except Exception as e:
    print_error(f"[ERREUR] Loading markets: {e}")
    sys.exit(1)

# Print available market symbols for debugging
available_symbols = list(exchange.markets.keys())
print(f"[INFO] Available symbols: {available_symbols}")

# Determine the correct symbol key
if "BTC/USDT" in available_symbols:
    symbol = "BTC/USDT"
elif "BTCUSDT" in available_symbols:
    symbol = "BTCUSDT"
else:
    # Fallback: choose the first symbol containing both 'BTC' and 'USDT'
    symbol = None
    for sym in available_symbols:
        if "BTC" in sym and "USDT" in sym:
            symbol = sym
            break
    if not symbol:
        print_error("Symbol 'BTC/USDT' or 'BTCUSDT' not found in markets.")
        sys.exit(1)

print(f"[INFO] Using symbol: {symbol}")

##############################################
#      PUBLIC ENDPOINT FUNCTIONS             #
##############################################
def test_connectivity():
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com/fapi/v1"
        url = f"{base_url}/ping"
    else:
        base_url = "https://eapi.binance.com"
        url = f"{base_url}/eapi/v1/ping"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("[INFO] Ping successful. API connectivity is OK.")
        print("LET'S POP SOME CHERRIES")
    except Exception as e:
        print_error(f"[ERREUR] Ping failed: {e}")
        sys.exit(1)

def get_open_interest(underlying_asset, expiration):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/openInterest"
    params = {"underlyingAsset": underlying_asset, "expiration": expiration}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        oi = response.json()
        print(f"[INFO] Open Interest for {underlying_asset} exp {expiration}:")
        pprint(oi)
        return oi
    except Exception as e:
        print_error(f"[ERREUR] Open interest: {e}")
        return None

def get_order_book(symbol, limit=100):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        order_book = response.json()
        print(f"[INFO] Order Book for {symbol} (limit {limit}):")
        pprint(order_book)
        return order_book
    except Exception as e:
        print_error(f"[ERREUR] Order Book for {symbol}: {e}")
        return None

def get_recent_block_trades(symbol=None, limit=100):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/blockTrades"
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        trades = response.json()
        print(f"[INFO] Recent Block Trades (limit {limit}):")
        pprint(trades)
        return trades
    except Exception as e:
        print_error(f"[ERREUR] Recent block trades: {e}")
        return None

def get_historical_trades(symbol, fromId=None, limit=100):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/historicalTrades"
    params = {"symbol": symbol, "limit": limit}
    if fromId:
        params["fromId"] = fromId
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        trades = response.json()
        print(f"[INFO] Historical Trades for {symbol} (limit {limit}):")
        pprint(trades)
        return trades
    except Exception as e:
        print_error(f"[ERREUR] Historical Trades for {symbol}: {e}")
        return None

def get_index_price(underlying):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/index"
    params = {"underlying": underlying}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        index_data = response.json()
        print(f"[INFO] Index Price for {underlying}: {index_data.get('indexPrice', 'N/A')} USDT at time {index_data.get('time')}")
        return index_data
    except Exception as e:
        print_error(f"[ERREUR] Index Price for {underlying}: {e}")
        return None

def get_klines(symbol, interval, startTime=None, endTime=None, limit=500):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if startTime:
        params["startTime"] = startTime
    if endTime:
        params["endTime"] = endTime
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        klines = response.json()
        print(f"[INFO] Retrieved {len(klines)} klines for {symbol} on interval {interval}.")
        return klines
    except Exception as e:
        print_error(f"[ERREUR] Get klines for {symbol}: {e}")
        return None

def get_option_mark_price(symbol=None):
    base_url = "https://eapi.binance.com"
    if USE_TESTNET:
        base_url = "https://testnet.binancefuture.com"
    url = f"{base_url}/eapi/v1/mark"
    params = {}
    if symbol:
        params["symbol"] = symbol
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        mark_data = response.json()
        print(f"[INFO] Option Mark Price & Greeks for {symbol if symbol else 'all symbols'}:")
        pprint(mark_data)
        return mark_data
    except Exception as e:
        print_error(f"[ERREUR] Option Mark Price: {e}")
        return None

def get_historical_trades_old(symbol, fromId=None, limit=100):
    return get_historical_trades(symbol, fromId, limit)

##############################################
#         TECHNICAL INDICATOR CALCS          #
##############################################
def calculate_rsi(prices, period=14):
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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow:
        return None, None, None
    ema_fast = np.mean(prices[-fast:])
    ema_slow = np.mean(prices[-slow:])
    macd = ema_fast - ema_slow
    signal_line = np.mean(prices[-signal:])  # Simplified signal calculation
    return macd, signal_line, macd - signal_line

def calculate_bbands(prices, period=50, num_std=2):
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    sma = np.mean(window)
    std = np.std(window)
    return sma - num_std * std, sma, sma + num_std * std

##############################################
#         LOT SIZE & QUANTITY ADJUSTMENTS    #
##############################################
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

def calculate_dynamic_quantity(symbol, side, current_price):
    bal = exchange.fetch_balance({'type': 'future'})
    market = exchange.markets[symbol]
    min_cost = float(market['limits']['cost']['min'])
    if side.upper() == 'BUY':
        available_usdt = float(bal.get('USDT', {}).get('free', 0))
        min_order_amt = float(market['limits']['amount']['min'])
        if available_usdt < current_price * min_order_amt:
            print_error(f"[ALERTE] Insufficient funds for BUY: need at least {current_price * min_order_amt:.2f} USDT, have {available_usdt} USDT.")
            return None
        debug_log(f"Available USDT for BUY: {available_usdt}")
        quantity = available_usdt / current_price
    elif side.upper() == 'SELL':
        asset = symbol.replace("/", "")
        available_asset = float(bal.get(asset, {}).get("free", 0))
        debug_log(f"Available {asset} for SELL: {available_asset}")
        quantity = available_asset
    else:
        return None

    if current_price * quantity < min_cost:
        print_error(f"[ALERTE] Notional too low for {symbol} {side}: {current_price * quantity:.2f} USDT, minimum required is {min_cost} USDT.")
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
        available_usdt = float(bal.get('USDT', {}).get("free", 0))
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

    precision = 8 if "BTC" in symbol.upper() else 2
    rounded_qty = round(valid_quantity, precision)
    final_qty = adjust_lot_size(symbol, rounded_qty)
    if final_qty < 0.001:
        print_error(f"[ERREUR] Final adjusted quantity ({final_qty}) is below the minimum (0.001) for {symbol}.")
        return None

    debug_log(f"Final adjusted quantity for {symbol}: {final_qty}")
    return final_qty

##############################################
#         ORDER EXECUTION FUNCTIONS          #
##############################################
def execute_order(symbol, decision, final_quantity, current_price):
    try:
        if decision.upper() == "BUY":
            order = exchange.create_market_buy_order(symbol, final_quantity, params={"reduceOnly": False})
        elif decision.upper() == "SELL":
            order = exchange.create_market_sell_order(symbol, final_quantity, params={"reduceOnly": False})
        print(f"[TRADE] {decision.upper()} order executed on {symbol} with quantity {final_quantity} at price {current_price} USDT.\nOrder result: {order}")
        return order
    except Exception as e:
        print_error(f"[ERREUR] Order execution {decision.upper()} on {symbol}: {e}")
        telegram_notify(f"[ALERTE] Order {decision.upper()} on {symbol} error: {e}")
        return None

##############################################
#          MODEL PREDICTION FUNCTION         #
##############################################
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
        0  # Default social feature placeholder for BTC
    ]
    debug_log(f"Features for {symbol}: {features}")
    try:
        model = joblib.load("model_rf.pkl")
        size_mb = os.path.getsize("model_rf.pkl") / (1024 * 1024)
        debug_log(f"Model 'model_rf.pkl' loaded, size: {size_mb:.2f} MB.")
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

##############################################
#          PORTFOLIO & PROFIT DISPLAY        #
##############################################
def display_portfolio():
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        btc = float(bal.get('BTC', {}).get("free", 0))
        bnb = float(bal.get('BNB', {}).get("free", 0))
        print(f"[PORTFOLIO] Current:")
        print(f"  USDT: Total = {usdt:.8f}, Free = {usdt:.8f}")
        print(f"  BTC: Total = {btc:.8f}")
        print(f"  BNB: Total = {bnb:.8f}")
    except Exception as e:
        print_error(f"[ERREUR] Display portfolio: {e}")

def display_profit(initial_balance):
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("free", 0))
        profit = current_usdt - initial_balance
        profit_str = f"Profit: {profit:.2f} USDT"
        if profit >= 0:
            print_profit(f"[PROFIT] {profit_str}")
        else:
            print_error(f"[PROFIT] {profit_str}")
    except Exception as e:
        print_error(f"[ERREUR] Display profit: {e}")

##############################################
#          FUND TRANSFER FUNCTIONS           #
##############################################
def transfer_bnb_to_futures():
    try:
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        bnb_spot = float(spot_bal.get('BNB', {}).get("free", 0))
        bnb_futures = float(futures_bal.get('BNB', {}).get("free", 0))
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
        exchange.options['fapiV2'] = True
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        usdt_spot = float(spot_bal.get('USDT', {}).get("free", 0))
        usdt_futures = float(futures_bal.get('USDT', {}).get("free", 0))
        print(f"[INFO] Spot USDT: {usdt_spot}, Futures USDT: {usdt_futures}")
        if usdt_spot > 0:
            print(f"[INFO] Transferring {usdt_spot} USDT from Spot to Futures.")
            result_usdt = exchange.transfer('USDT', usdt_spot, 'spot', 'future')
            print(f"[INFO] USDT Transfer result: {result_usdt}")
        else:
            print("[INFO] No USDT to transfer.")
        btc_spot = float(spot_bal.get('BTC', {}).get("free", 0))
        btc_futures = float(futures_bal.get('BTC', {}).get("free", 0))
        print(f"[INFO] Spot BTC: {btc_spot}, Futures BTC: {btc_futures}")
        if btc_spot > 0:
            print(f"[INFO] Transferring {btc_spot} BTC from Spot to Futures.")
            result_btc = exchange.transfer('BTC', btc_spot, 'spot', 'future')
            print(f"[INFO] BTC Transfer result: {result_btc}")
        else:
            print("[INFO] No BTC to transfer.")
        transfer_bnb_to_futures()
        futures_bal = exchange.fetch_balance({'type': 'future'})
        initial_futures_balance = float(futures_bal.get('USDT', {}).get("free", 0))
        print(f"[INFO] Initial Futures USDT: {initial_futures_balance:.2f}")
    except Exception as e:
        print_error(f"[ERREUR] Initializing funds: {e}")

##############################################
#          TRADING LOGIC FUNCTIONS           #
##############################################
WINDOW_SIZE = 50
COOLDOWN = 0.5
SLEEP_INTERVAL = 0.1
initial_futures_balance = None
historical_prices = {"BTC/USDT": []}  # Use our chosen symbol key

def get_latest_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print_error(f"[ERREUR] Get latest price for {symbol}: {e}")
        return None

def trading_logic(symbol, last_trade_time):
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
    rsi_display = f"{rsi:.2f}" if rsi is not None else "N/A"
    print(f"[DATA] {symbol} - Price: {current_price:.2f} USDT | SMA: {moving_average:.2f} | RSI: {rsi_display}")
    decision = predict_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands)
    print(f"[INFO] Predicted decision for {symbol}: {decision}")

    # Fallback: if SELL is predicted but insufficient BTC available, force BUY.
    if decision.upper() == "SELL":
        bal = exchange.fetch_balance({'type': 'future'})
        available_btc = float(bal.get("BTC", {}).get("free", 0))
        min_order_amt = float(exchange.markets[symbol]['limits']['amount']['min'])
        if available_btc < min_order_amt:
            decision = "BUY"
            print_error(f"[FALLBACK] BTC balance ({available_btc}) insufficient (< {min_order_amt}), forcing BUY.")
            print(f"[INFO] Fallback decision for {symbol}: {decision}")

    # Additional fallback based on RSI if HOLD is predicted
    if decision.upper() == "HOLD" and rsi is not None:
        bal = exchange.fetch_balance({'type': 'future'})
        available_btc = float(bal.get("BTC", {}).get("free", 0))
        if available_btc < float(exchange.markets[symbol]['limits']['amount']['min']):
            decision = "BUY"
            print_error("[FALLBACK] Insufficient BTC for SELL, forcing BUY.")
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

        if decision.upper() == "SELL":
            min_order_amt = float(exchange.markets[symbol]['limits']['amount']['min'])
            if final_quantity < min_order_amt:
                print_error(f"[ERREUR] Final quantity ({final_quantity}) for SELL is below minimum ({min_order_amt}). No trade executed.")
                return last_trade_time

        notional = current_price * final_quantity
        min_cost = float(exchange.markets[symbol]['limits']['cost']['min'])
        debug_log(f"Order notional: {notional:.2f} USDT (min required: {min_cost} USDT)")
        order = execute_order(symbol, decision, final_quantity, current_price)
        if order is not None:
            print(f"[INFO] Order executed on {symbol}.")
            last_trade_time[symbol] = now
            telegram_notify(f"[TRADE] {decision.upper()} order on {symbol}: {final_quantity} at {current_price} USDT.")
    else:
        print(f"[INFO] Model predicts HOLD for {symbol}. No trade executed.")
    return last_trade_time

##############################################
#          PORTFOLIO & PROFIT DISPLAY        #
##############################################
def display_portfolio():
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        usdt = float(bal.get('USDT', {}).get("free", 0))
        btc = float(bal.get('BTC', {}).get("free", 0))
        bnb = float(bal.get('BNB', {}).get("free", 0))
        print(f"[PORTFOLIO] Current:")
        print(f"  USDT: Total = {usdt:.8f}, Free = {usdt:.8f}")
        print(f"  BTC: Total = {btc:.8f}")
        print(f"  BNB: Total = {bnb:.8f}")
    except Exception as e:
        print_error(f"[ERREUR] Display portfolio: {e}")

def display_profit(initial_balance):
    try:
        bal = exchange.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("free", 0))
        profit = current_usdt - initial_balance
        profit_str = f"Profit: {profit:.2f} USDT"
        if profit >= 0:
            print_profit(f"[PROFIT] {profit_str}")
        else:
            print_error(f"[PROFIT] {profit_str}")
    except Exception as e:
        print_error(f"[ERREUR] Display profit: {e}")

##############################################
#          FUND TRANSFER FUNCTIONS           #
##############################################
def transfer_bnb_to_futures():
    try:
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        bnb_spot = float(spot_bal.get('BNB', {}).get("free", 0))
        bnb_futures = float(futures_bal.get('BNB', {}).get("free", 0))
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
        exchange.options['fapiV2'] = True
        spot_bal = exchange.fetch_balance({'type': 'spot'})
        futures_bal = exchange.fetch_balance({'type': 'future'})
        usdt_spot = float(spot_bal.get('USDT', {}).get("free", 0))
        usdt_futures = float(futures_bal.get('USDT', {}).get("free", 0))
        print(f"[INFO] Spot USDT: {usdt_spot}, Futures USDT: {usdt_futures}")
        if usdt_spot > 0:
            print(f"[INFO] Transferring {usdt_spot} USDT from Spot to Futures.")
            result_usdt = exchange.transfer('USDT', usdt_spot, 'spot', 'future')
            print(f"[INFO] USDT Transfer result: {result_usdt}")
        else:
            print("[INFO] No USDT to transfer.")
        btc_spot = float(spot_bal.get('BTC', {}).get("free", 0))
        btc_futures = float(futures_bal.get('BTC', {}).get("free", 0))
        print(f"[INFO] Spot BTC: {btc_spot}, Futures BTC: {btc_futures}")
        if btc_spot > 0:
            print(f"[INFO] Transferring {btc_spot} BTC from Spot to Futures.")
            result_btc = exchange.transfer('BTC', btc_spot, 'spot', 'future')
            print(f"[INFO] BTC Transfer result: {result_btc}")
        else:
            print("[INFO] No BTC to transfer.")
        transfer_bnb_to_futures()
        futures_bal = exchange.fetch_balance({'type': 'future'})
        initial_futures_balance = float(futures_bal.get('USDT', {}).get("free", 0))
        print(f"[INFO] Initial Futures USDT: {initial_futures_balance:.2f}")
    except Exception as e:
        print_error(f"[ERREUR] Initializing funds: {e}")

##############################################
#          MAIN TRADING LOOP                 #
##############################################
def main():
    test_connectivity()  # Test API connectivity at startup
    # Ensure that the symbol key we use exists in the markets
    if symbol not in exchange.markets:
        print_error(f"Symbol {symbol} not found in markets.")
        sys.exit(1)
    last_trade_time = {symbol: 0}
    iteration = 0
    iterations_per_minute = int(60 / SLEEP_INTERVAL)
    print(f"[INFO] Starting HFT Trading Bot for {symbol} with Random Forest (NO LEVERAGE)...\n")
    
    initialize_funds()
    display_portfolio()
    initial_balance = initial_futures_balance if initial_futures_balance is not None else 0

    while True:
        last_trade_time = trading_logic(symbol, last_trade_time)
        iteration += 1
        if iteration % 50 == 0:
            display_portfolio()
        if iteration % iterations_per_minute == 0 and initial_balance:
            display_profit(initial_balance)
        time.sleep(SLEEP_INTERVAL)

##############################################
#          TELEGRAM BOT HANDLERS             #
##############################################
@bot.message_handler(commands=['shutdown'])
def handle_shutdown(message):
    bot.reply_to(message, "Shutdown selected.")
    shutdown_bot()

@bot.message_handler(commands=['poweron'])
def handle_poweron(message):
    bot.reply_to(message, "Power On selected.")
    main()

@bot.message_handler(commands=['balance'])
def handle_balance(message):
    display_portfolio()
    bal = exchange.fetch_balance({'type': 'future'})
    usdt = float(bal.get('USDT', {}).get("free", 0))
    bot.reply_to(message, f"Futures Balance: {usdt:.2f} USDT")
    telegram_notify(f"Futures Balance: {usdt:.2f} USDT")

@bot.message_handler(commands=['profit'])
def handle_profit(message):
    if initial_futures_balance:
        display_profit(initial_futures_balance)
    else:
        bot.reply_to(message, "Initial balance not set.")

# Additional command handlers can be added here...

##############################################
#          SHUTDOWN FUNCTION                 #
##############################################
def shutdown_bot():
    print("Shutting down bot and closing all positions...")
    try:
        close_position()
    except Exception as e:
        print_error(f"[ERREUR] Closing positions during shutdown: {e}")
    telegram_notify("Bot is shutting down. All open positions are closed.")
    sys.exit("BOT SHUTDOWN")

##############################################
#          WEBSOCKET HANDLERS (OPTIONAL)       #
##############################################
def on_open(ws):
    print("WebSocket connection opened.")
    telegram_notify("WebSocket connection opened.")

def on_close(ws):
    print("WebSocket connection closed.")
    telegram_notify("WebSocket connection closed.")

def on_message(ws, message):
    try:
        json_message = json.loads(message)
        messageTime = json_message.get('E', None)
        if messageTime:
            timestamp = dt.datetime.fromtimestamp(int(messageTime)/1000)
            messageSentTime = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
            debug_log(f"Message Time: {messageSentTime}")
        candle = json_message.get('k', {})
        if candle.get('x', False):  # Candle closed
            close_price = float(candle.get('c', 0))
            print(f"[INFO] Candle closed at price: {close_price}")
            global historical_prices
            historical_prices["BTC/USDT"].append(close_price)
            if len(historical_prices["BTC/USDT"]) > WINDOW_SIZE:
                historical_prices["BTC/USDT"].pop(0)
            trading_logic("BTC/USDT", {"BTC/USDT": time.time()})
    except Exception as e:
        print_error(f"[ERREUR] Processing WebSocket message: {e}")

def start_websocket():
    ws_url = f"wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_close=on_close, on_message=on_message)
    ws.run_forever()

##############################################
#          RUN MAIN OR TELEGRAM BOT          #
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
