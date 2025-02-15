#!/usr/bin/env python3
"""
DISCLAIMER:
Ce code est fourni à titre éducatif uniquement. Je ne suis pas un conseiller financier.
L'utilisation de ce système en conditions réelles, notamment avec effet de levier et trading haute fréquence,
peut entraîner des pertes importantes. Testez ce système en simulation ou sur le testnet de Binance Futures avant toute utilisation en réel.
Objectif très ambitieux : atteindre 1 BTC en 1 mois (objectif extrêmement risqué).

Ce script s'inspire de la méthodologie présentée dans la thèse
"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"
de l'Université d'Amsterdam. Il se concentre ici uniquement sur le trading de BTC/USDT en utilisant
des indicateurs financiers (SMA, RSI, MACD, Bollinger Bands) et un modèle Random Forest préalablement entraîné
(sauvegardé sous le nom 'model_rf.pkl'). Avant de commencer, une fonction d'initialisation transfère
la totalité des fonds du compte Spot vers le compte Futures (USDT, BTC et BNB) en vérifiant que les minimums requis sont respectés.
"""

import ccxt
import numpy as np
import time
import feedparser
import joblib
import os
import pandas as pd

# ----- Paramètres globaux pour le trading -----
WINDOW_SIZE = 50        # Nombre de points historiques pour les indicateurs
RSI_PERIOD = 14         # Période pour calculer le RSI
MACD_FAST = 12          # Période EMA rapide pour MACD
MACD_SLOW = 26          # Période EMA lente pour MACD
MACD_SIGNAL = 9         # Période de la ligne signal pour MACD
COOLDOWN = 0.5          # Cooldown en secondes entre deux trades sur une même paire
SLEEP_INTERVAL = 0.1    # Intervalle de boucle

# Paramètres pour l'effet de levier et stop-loss
LEVERAGE = 20           # Par exemple, 20x de levier
STOP_LOSS_PERCENT = 1.0 # Stop-loss à 1% en dessous du prix d'entrée (optionnel)

# ----- Paramètres pour les actualités via RSS -----
NEWS_INTERVAL = 300     # Actualiser les actualités toutes les 5 minutes
NEWS_SOURCES = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "CryptoSlate": "https://cryptoslate.com/feed/"
}

# ----- Fonction pour lire les clés API depuis un fichier texte -----
def lire_cle_api(fichier):
    cles = {}
    try:
        with open(fichier, 'r') as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne and not ligne.startswith('#'):
                    if '=' in ligne:
                        key, value = ligne.split('=', 1)
                    elif ':' in ligne:
                        key, value = ligne.split(':', 1)
                    else:
                        continue
                    cles[key.strip()] = value.strip()
    except Exception as e:
        print(f"[ERREUR] Lecture du fichier {fichier} : {e}")
    return cles

# ----- Lecture des clés API -----
cles_api = lire_cle_api("apikeys.txt")
BINANCE_API_KEY = cles_api.get("BINANCE_API_KEY")
BINANCE_API_SECRET = cles_api.get("BINANCE_API_SECRET")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("[ERREUR] Clés API manquantes dans apikeys.txt.")
    exit(1)

# ----- Configuration de Binance Futures via ccxt -----
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
exchange.load_markets()
print("[INFO] Connexion à Binance Futures établie et marchés chargés.")

# ----- On se concentre uniquement sur BTC/USDT -----
SYMBOLS = ["BTC/USDT"]

# ----- Historique des prix pour chaque symbole -----
historique_prix = {symbol: [] for symbol in SYMBOLS}

# ----- Dictionnaire pour le suivi du leverage -----
leverage_set = {symbol: False for symbol in SYMBOLS}

# ----- Fonction pour récupérer les actualités via RSS -----
def recuperer_news():
    news_summary = ""
    for source, url in NEWS_SOURCES.items():
        flux = feedparser.parse(url)
        if flux.entries:
            titres = [entry.get("title", "Titre non disponible") for entry in flux.entries[:2]]
            news_summary += f"{source}: " + " | ".join(titres) + "\n"
    return news_summary

# ----- Fonctions techniques pour calculer les indicateurs financiers -----
def calculer_rsi(prices, period=RSI_PERIOD):
    if len(prices) < period + 1:
        return None
    diffs = np.diff(prices)
    gains = np.maximum(diffs, 0)
    pertes = np.abs(np.minimum(diffs, 0))
    moyenne_gain = np.mean(gains[-period:])
    moyenne_perte = np.mean(pertes[-period:])
    if moyenne_perte == 0:
        return 100
    rs = moyenne_gain / moyenne_perte
    return 100 - (100 / (1 + rs))

def calculer_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(prices) < slow:
        return None, None, None
    ema_fast = np.mean(prices[-fast:])
    ema_slow = np.mean(prices[-slow:])
    macd = ema_fast - ema_slow
    signal_line = np.mean(prices[-signal:])  # simplification
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculer_bbands(prices, period=WINDOW_SIZE, num_std=2):
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    sma = np.mean(window)
    std = np.std(window)
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return lower_band, sma, upper_band

# ----- Gestion du portefeuille -----
def afficher_portefeuille():
    try:
        balance = exchange.fetch_balance({'type': 'future'})
        print("\n[PORTFOLIO] Actuel:")
        for asset in ["USDT", "BTC", "BNB"]:
            info = balance.get(asset, {})
            free = info.get("free", 0)
            total = info.get("total", 0)
            print(f"  {asset}: Total = {total}, Disponible = {free}")
        print("")
    except Exception as e:
        print(f"[ERREUR] Récupération du portefeuille : {e}")

# ----- Récupérer le dernier prix -----
def recuperer_dernier_prix(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"[ERREUR] Récupération du prix pour {symbol} : {e}")
        return None

# ----- Calcul de la quantité à trader -----
def calculer_quantite_dyn(symbol, side, current_price):
    """
    Pour BUY : utiliser 100 % du solde USDT disponible en Futures.
    Pour SELL : vendre la totalité de la position BTC.
    """
    balance = exchange.fetch_balance({'type': 'future'})
    market = exchange.markets[symbol]
    min_cost = market['limits']['cost']['min']
    if side.lower() == 'buy':
        available_usdt = balance.get('USDT', {}).get('free', 0)
        print(f"[DEBUG] Disponible USDT pour BUY: {available_usdt}")
        quantite = available_usdt / current_price
        print(f"[DEBUG] Quantité brute calculée pour BUY: {quantite}")
    elif side.lower() == 'sell':
        asset = symbol.split('/')[0]
        available_asset = balance.get(asset, {}).get('free', 0)
        print(f"[DEBUG] Disponible {asset} pour SELL: {available_asset}")
        quantite = available_asset
    else:
        return None
    if current_price * quantite < min_cost:
        print(f"[ALERTE] Quantité calculée insuffisante pour {symbol} {side.upper()} (Notional: {current_price * quantite:.2f} USDT < min_cost {min_cost} USDT)")
        return None
    return quantite

# ----- Ajustement et arrondi de la quantité -----
def ajuster_quantite(symbol, side, quantite, current_price):
    market = exchange.markets[symbol]
    # Forcer un seuil minimum personnalisé pour éviter l'alerte (ici 1e-8)
    min_amount = 1e-8  
    min_cost = market['limits']['cost']['min']
    required_by_cost = min_cost / current_price
    quantite_valide = max(quantite, required_by_cost, min_amount)
    
    balance = exchange.fetch_balance({'type': 'future'})
    if side.lower() == 'buy':
        available_usdt = balance.get('USDT', {}).get('free', 0)
        if current_price * quantite_valide > available_usdt:
            quantite_valide = available_usdt / current_price
    elif side.lower() == 'sell':
        asset = symbol.split('/')[0]
        available_asset = balance.get(asset, {}).get('free', 0)
        if quantite_valide > available_asset:
            quantite_valide = available_asset
    
    if current_price * quantite_valide < min_cost:
        print(f"[ALERTE] Quantité finale insuffisante pour {symbol}: {current_price * quantite_valide:.2f} USDT < min_cost {min_cost} USDT")
        return None

    # Pour BTC, fixer la précision à 8 décimales
    precision = 8 if symbol.startswith("BTC") else 2
    quantite_finale = round(quantite_valide, precision)
    if quantite_finale < min_amount:
        print(f"[ALERTE] Quantité arrondie trop faible pour {symbol}: {quantite_finale} < min_amount {min_amount}")
        return None
    print(f"[DEBUG] Quantité finale ajustée pour {symbol}: {quantite_finale}")
    return quantite_finale

# ----- Prédiction de la décision via Random Forest pour BTC -----
def predire_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands):
    """
    Construit un vecteur de caractéristiques et utilise un modèle Random Forest préalablement entraîné,
    sauvegardé dans 'model_rf.pkl', pour prédire la direction du marché.
    Caractéristiques utilisées (9 features) :
      - Indicateurs financiers : prix actuel, SMA, RSI, MACD, Signal, Bollinger Bands (Lower, SMA, Upper)
      - Une caractéristique sociale : pour BTC, la valeur par défaut est 0.
    La prédiction renvoie 1 (BUY), -1 (SELL) ou 0 (HOLD).
    """
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
        0  # social_feature par défaut pour BTC
    ]
    
    print(f"[DEBUG] Caractéristiques pour {symbol} : {features}")
    
    try:
        model = joblib.load("model_rf.pkl")
        file_size_bytes = os.path.getsize("model_rf.pkl")
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"[DEBUG] Modèle 'model_rf.pkl' chargé, taille : {file_size_mb:.2f} MB.")
    except Exception as e:
        print(f"[ERREUR] Chargement du modèle : {e}")
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

# ----- Logique de trading pour BTC -----
def logique_trading(symbol, last_trade_time):
    current_price = recuperer_dernier_prix(symbol)
    if current_price is None:
        return last_trade_time

    historique_prix[symbol].append(current_price)
    if len(historique_prix[symbol]) > WINDOW_SIZE:
        historique_prix[symbol].pop(0)

    moving_average = np.mean(historique_prix[symbol])
    rsi = calculer_rsi(historique_prix[symbol])
    macd, signal_line, _ = calculer_macd(historique_prix[symbol])
    bbands = calculer_bbands(historique_prix[symbol])
    
    rsi_display = f"{rsi:.2f}" if rsi is not None else "N/A"
    print(f"[DATA] {symbol} - Prix: {current_price:.2f} USDT | SMA: {moving_average:.2f} | RSI: {rsi_display}")
    
    decision = predire_decision(symbol, current_price, moving_average, rsi, macd, signal_line, bbands)
    print(f"[INFO] Décision prédite pour {symbol} : {decision}")
    
    # Logique de fallback : si le modèle prédit HOLD, forcer un trade selon le RSI
    if decision == "HOLD" and rsi is not None:
        balance = exchange.fetch_balance({'type': 'future'})
        available_btc = balance.get("BTC", {}).get("free", 0)
        if available_btc <= 0:
            decision = "BUY"
            print("[FALLBACK] Aucun BTC disponible pour SELL, forçant BUY.")
        else:
            if rsi <= 50:
                decision = "BUY"
                print("[FALLBACK] RSI <= 50, forçant BUY.")
            else:
                decision = "SELL"
                print("[FALLBACK] RSI > 50, forçant SELL.")
        print(f"[INFO] Décision modifiée par fallback pour {symbol} : {decision}")
    
    # Si le modèle prédit SELL et qu'il n'y a pas de BTC disponible, forcer BUY pour utiliser les fonds USDT
    if decision == "SELL":
        balance = exchange.fetch_balance({'type': 'future'})
        available_btc = balance.get("BTC", {}).get("free", 0)
        if available_btc <= 0:
            decision = "BUY"
            print("[FALLBACK] Aucune quantité BTC pour SELL, forçant BUY.")
            print(f"[INFO] Décision modifiée par fallback pour {symbol} : {decision}")
    
    if decision in ["BUY", "SELL"]:
        now = time.time()
        if now - last_trade_time.get(symbol, 0) < COOLDOWN:
            print(f"[INFO] Cooldown actif pour {symbol}. Aucun trade n'est passé.")
            return last_trade_time
        
        if not leverage_set[symbol]:
            try:
                exchange.set_leverage(LEVERAGE, symbol)
                leverage_set[symbol] = True
                print(f"[INFO] Leverage {LEVERAGE}x appliqué pour {symbol}.")
            except Exception as e:
                print(f"[ERREUR] Échec de configuration du leverage pour {symbol}: {e}")
        
        quantite_calculee = calculer_quantite_dyn(symbol, decision, current_price)
        if quantite_calculee is None or quantite_calculee == 0:
            print(f"[INFO] Quantité calculée = 0, aucun trade exécuté pour {symbol} {decision}.")
            return last_trade_time
        
        quantite_finale = ajuster_quantite(symbol, decision, quantite_calculee, current_price)
        if quantite_finale is None or quantite_finale == 0:
            print(f"[INFO] Quantité finale = 0, aucun trade exécuté pour {symbol} {decision}.")
            return last_trade_time
        
        # Debug: calcul de la valeur notionnelle
        notional = current_price * quantite_finale
        print(f"[DEBUG] Notional de l'ordre: {notional:.2f} USDT (min requis: {exchange.markets[symbol]['limits']['cost']['min']} USDT)")
        
        try:
            if decision == "BUY":
                ordre = exchange.create_market_buy_order(symbol, quantite_finale, params={"reduceOnly": False})
                try:
                    stop_price = current_price * (1 - STOP_LOSS_PERCENT / 100)
                    ordre_stop = exchange.create_order(
                        symbol, 
                        type="stop_market", 
                        side="sell", 
                        amount=quantite_finale, 
                        params={"stopPrice": stop_price, "reduceOnly": False}
                    )
                    print(f"[STOP-LOSS] Ordre stop-loss placé pour {symbol} à {stop_price:.2f} USDT.")
                except Exception as e:
                    print(f"[ERREUR] Placement du stop-loss pour {symbol} : {e}")
            elif decision == "SELL":
                ordre = exchange.create_market_sell_order(symbol, quantite_finale, params={"reduceOnly": False})
            print(f"[TRADE] Ordre {decision} exécuté sur {symbol} avec {quantite_finale} (Prix: {current_price})\nRésultat: {ordre}")
            last_trade_time[symbol] = now
        except Exception as e:
            print(f"[ERREUR] Exécution du trade {decision} sur {symbol}: {e}")
    else:
        print(f"[INFO] Le modèle prédit HOLD pour {symbol}. Aucun trade n'est exécuté.")
    return last_trade_time

# ----- Transfert de BNB vers Futures -----
def transfer_bnb_to_futures():
    try:
        spot_balance = exchange.fetch_balance({'type': 'spot'})
        futures_balance = exchange.fetch_balance({'type': 'future'})
        bnb_spot = spot_balance.get('BNB', {}).get('free', 0)
        bnb_futures = futures_balance.get('BNB', {}).get('free', 0)
        print(f"[INFO] Solde Spot BNB: {bnb_spot}, Solde Futures BNB: {bnb_futures}")
        if bnb_spot > 0:
            print(f"[INFO] Transfert de {bnb_spot} BNB du compte Spot vers Futures pour réduction des frais.")
            result_bnb = exchange.transfer('BNB', bnb_spot, 'spot', 'future')
            print(f"[INFO] Résultat du transfert BNB: {result_bnb}")
        else:
            print("[INFO] Aucune BNB à transférer.")
    except Exception as e:
        print(f"[ERREUR] Transfert de BNB: {e}")

# ----- Fonction d'initialisation des fonds (transfert Spot -> Futures) -----
def initialize_funds():
    try:
        spot_balance = exchange.fetch_balance({'type': 'spot'})
        futures_balance = exchange.fetch_balance({'type': 'future'})
        
        # Transfert USDT : transférer tout le solde USDT du Spot vers Futures
        usdt_spot = spot_balance.get('USDT', {}).get('free', 0)
        usdt_futures = futures_balance.get('USDT', {}).get('free', 0)
        print(f"[INFO] Solde Spot USDT: {usdt_spot}, Solde Futures USDT: {usdt_futures}")
        if usdt_spot > 0:
            print(f"[INFO] Transfert de {usdt_spot} USDT du compte Spot vers Futures.")
            result_usdt = exchange.transfer('USDT', usdt_spot, 'spot', 'future')
            print(f"[INFO] Résultat du transfert USDT: {result_usdt}")
        else:
            print("[INFO] Aucune USDT à transférer.")
        
        # Transfert BTC : transférer tout le solde BTC du Spot vers Futures
        btc_spot = spot_balance.get('BTC', {}).get('free', 0)
        btc_futures = futures_balance.get('BTC', {}).get('free', 0)
        print(f"[INFO] Solde Spot BTC: {btc_spot}, Solde Futures BTC: {btc_futures}")
        if btc_spot > 0:
            print(f"[INFO] Transfert de {btc_spot} BTC du compte Spot vers Futures.")
            result_btc = exchange.transfer('BTC', btc_spot, 'spot', 'future')
            print(f"[INFO] Résultat du transfert BTC: {result_btc}")
        else:
            print("[INFO] Aucune BTC à transférer.")
        
        # Transfert BNB
        transfer_bnb_to_futures()
        
    except Exception as e:
        print(f"[ERREUR] Initialisation des fonds: {e}")

# ----- Fonction principale -----
def main():
    global leverage_set
    leverage_set = {symbol: False for symbol in SYMBOLS}
    last_trade_time = {symbol: 0 for symbol in SYMBOLS}
    iteration = 0
    last_news_time = 0
    print("[INFO] Démarrage du système de trading HFT pour BTC avec Random Forest...\n")
    
    initialize_funds()
    afficher_portefeuille()
    while True:
        current_time = time.time()
        if current_time - last_news_time > NEWS_INTERVAL:
            print("\n[NEWS] Actualités récentes:")
            print(recuperer_news())
            last_news_time = current_time
        for symbol in SYMBOLS:
            last_trade_time = logique_trading(symbol, last_trade_time)
        iteration += 1
        if iteration % 50 == 0:
            afficher_portefeuille()
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()