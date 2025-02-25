# FinRL_Crypto_HFT: Deep Reinforcement Learning and Random Forest for Binance Futures Trading

## Overview  
This project integrates **Maxime Grenu's Binance Futures HFT Trading Bot** (using Random Forest) with **FinRL_Crypto** by Berend Jelmer Dirk Gort et al., blending **Deep Reinforcement Learning (DRL)** and **high-frequency trading (HFT)** for Binance Futures, targeting **BTC/USDT**. It tackles overfitting in financial RL and employs a **Random Forest model** with technical indicators (**SMA, RSI, MACD, Bollinger Bands**), enriched by Annelotte Bonenkamp's insights on combining financial and social features. **Tested across multiple cryptocurrencies and market crashes**, it aims to surpass traditional strategies.  

> **⚠ Warning:** For educational purposes only. Not production-ready without extensive validation. Leveraged trading carries high risk—use simulation or Binance Testnet first.  

## Features  
- **Hybrid Decision-Making**: Merges DRL (**PPO, DDPG, SAC via ElegantRL**) with Random Forest.  
- **Binance Futures Integration**: Uses `ccxt` for trading, with Spot-to-Futures transfers (**USDT, BTC, BNB**).  
- **Technical Indicators**: Features **SMA, RSI, MACD, Bollinger Bands**, plus `TA-Lib` extras (**CCI, DX, ROC**).  
- **Overfitting Mitigation**: Applies **Walkforward analysis, K-Fold CV, and Combinatorial Purged CV (CPCV)** with resolved timestamp handling issues.  
- **Order Management**: Enforces minimums (**0.00105 BTC, 100 USDT**), defaults to **20x leverage**.  
- **Real-Time Operations**: Utilizes **WebSocket feeds and Telegram alerts**.  
- **Backtesting**: Includes **Probability of Backtest Overfitting (PBO)** and performance metrics with stabilized log returns.  
- **Production Optimizations**: Boosts **security, error handling, and live trading efficiency**.  
- **Ensemble Models**: Integrates YDF, Core ML, and MLX models for improved decision-making.  
- **Enhanced Indicator Calculation**: Direct calculation from OHLCV data with overlaid indicators from Binance processing.  
- **Leverage and Position Management**: Dynamic leverage adjustment and improved handling of open positions.  
- **Funding Fee Tracking**: Periodic checks and accumulation of funding fees for perpetual futures.  
- **Error Handling and Logging**: Enhanced mechanisms for operations like leverage setting and order execution.  
- **Telegram Notifications**: Real-time alerts for trade executions, errors, and funding fee updates.  
- **Data Caching**: Efficient caching of OHLCV data to reduce API calls and improve performance.  
- **Cooldown and Trade Conditions**: Includes a cooldown period after closing positions and refined entry conditions.  

## Papers  
- **Deep Reinforcement Learning for Cryptocurrency Trading** by Berend Jelmer Dirk Gort et al.  
- **"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"** by Annelotte Bonenkamp (Bachelor Thesis, University of Amsterdam, June 2021).  

## Prerequisites  
- **Python 3.10** (tested with virtual environment `venv310`).  
- **Modules**: `ccxt`, `numpy`, `pandas`, `joblib`, `binance`, `talib`, `requests`, `yfinance` (optional), `telebot`, `websocket-client`, `ta`, `python-dotenv`, `torch`, `optuna`, plus `ElegantRL` dependencies.  
- **Binance Futures API access** (`API key` and `secret`).  
- **Configuration**: `.env` file with:  
  ```plaintext
  BINANCE_API_KEY=your_binance_api_key  
  BINANCE_API_SECRET=your_binance_api_secret  
  TELEGRAM_TOKEN=your_telegram_token  
  TELEGRAM_CHAT_ID=your_telegram_chat_id  
  ```

## Installation  
### Clone Repository:  
```bash
git clone https://github.com/cluster2600/BTC_BOT.git && cd BTC_BOT  
```

### Virtual Environment:  
#### Linux/macOS:  
```bash
python3.10 -m venv venv310 && source venv310/bin/activate  
```
#### Windows:  
```bash
python -m venv venv310 && venv310\Scripts\activate  
```

### Install Dependencies:  
```bash
pip install -r requirements.txt  
```
Set up `.env` with credentials (see Configuration below).  

## How to Use  
### Configuration  
Edit `config_main.py` for:  
- Validation methods: Walkforward, K-Fold CV, CPCV.  
- Candle counts: e.g., 20,000 training, 5,000 validation.  
- Tickers: e.g., `["BTC/USDT"]`.  
- Indicators and trade date ranges: e.g., 2022-02-02 to 2022-04-29.  
- Secure Loading: Credentials sourced from `.env` via `python-dotenv`.  

### Folder Structure  
- `data/`: Training/validation data (e.g., `./data/5m_25000`), with trade_data subfolder.  
- `drl_agents/`: ElegantRL DRL implementations.  
- `random_forest/`: Random Forest model scripts and training datasets.  
- `config/`: Configuration files for hyperparameters and Binance API.  
- `logs/`: Trading and backtesting logs.  
- `notebooks/`: Jupyter notebooks for exploratory analysis.  

### Running the Bot  
To run the bot in live mode, ensure API credentials are set and execute:  
```bash
python main.py --mode live  
```
For backtesting, run:  
```bash
python main.py --mode backtest  
```

## Recent Updates  
- **Ensemble Models**: Added support for YDF, Core ML, and MLX models to enhance decision-making accuracy, introducing a `get_ensemble_decision` function to combine predictions for a more robust trading strategy.  
- **Indicator Calculation**: Improved direct calculation of technical indicators (**RSI, SMA, ATR, ADX**) from OHLCV data and integrated overlaid indicators from Binance processing for consistency and accuracy.  
- **Leverage and Position Management**: Implemented dynamic leverage adjustment with a fallback to default leverage if rejected, and enhanced position management to better handle open positions, including setting entry prices and managing stop-loss/take-profit orders.  
- **Funding Fee Tracking**: Introduced periodic checks and accumulation of funding fees for perpetual futures, ensuring these costs are accounted for in profit calculations.  
- **Error Handling and Logging**: Enhanced error handling for critical operations like leverage setting and order execution, and improved logging to provide detailed insights into indicator values, trade decisions, and errors.  
- **Telegram Notifications**: Integrated real-time alerts for trade executions, errors, and funding fee updates, keeping users informed of critical events.  
- **Data Caching**: Added efficient caching of OHLCV data to reduce API calls and improve performance, with a refresh mechanism to keep data up-to-date.  
- **Cooldown and Trade Conditions**: Introduced a cooldown period after closing positions to prevent immediate re-entry, and refined trade entry conditions to include checks on **ATR, ADX, RSI, and ensemble model confidence** for better timing.  
- **Code Structure and Documentation**: Updated the code structure to clearly separate concerns (e.g., data fetching, indicator calculation, trade execution) and added comprehensive comments to explain functionality.  
- **Testing and Validation**: Conducted extensive backtesting with historical data and live testing in a controlled environment to validate the bot's performance.  

## Risk Disclaimer  
This bot is experimental and not financial advice. Use at your own risk. Consider using Binance Testnet before live trading.  

## GitHub Repository  
Access the full source code and documentation here: [GitHub Repository](https://github.com/cluster2600/BTC_BOT/tree/main).
