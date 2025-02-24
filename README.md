# FinRL_Crypto_HFT: Deep Reinforcement Learning and Random Forest for Binance Futures Trading

![banner](https://user-images.githubusercontent.com/69801109/214294114-a718d378-6857-4182-9331-20869d64d3d9.png)

This project integrates Maxime Grenu's Binance Futures HFT Trading Bot (using Random Forest) with FinRL_Crypto by Berend Jelmer Dirk Gort et al., blending **Deep Reinforcement Learning (DRL)** and **high-frequency trading (HFT)** for Binance Futures, targeting **BTC/USDT**. It tackles overfitting in financial RL and employs a Random Forest model with technical indicators (SMA, RSI, MACD, Bollinger Bands), enriched by Annelotte Bonenkamp's insights on combining financial and social features. Tested across multiple cryptocurrencies and market crashes, it aims to surpass traditional strategies.

**Warning:** For educational purposes only. Not production-ready without extensive validation. Leveraged trading carries high risk—use simulation or Binance Testnet first.

## Features

- **Hybrid Decision-Making**: Merges DRL (PPO, DDPG, SAC via ElegantRL) with Random Forest.
- **Binance Futures Integration**: Uses `ccxt` for trading, with Spot-to-Futures transfers (USDT, BTC, BNB).
- **Technical Indicators**: Features SMA, RSI, MACD, Bollinger Bands, plus TA-Lib extras (CCI, DX, ROC).
- **Overfitting Mitigation**: Applies Walkforward analysis, K-Fold CV, and Combinatorial Purged CV (CPCV).
- **Order Management**: Enforces minimums (0.00105 BTC, 100 USDT), defaults to 20x leverage.
- **Real-Time Operations**: Utilizes WebSocket feeds and Telegram alerts.
- **Backtesting**: Includes Probability of Backtest Overfitting (PBO) and performance metrics.
- **Production Optimizations**: Boosts security, error handling, and live trading efficiency.

## Papers

- [Deep Reinforcement Learning for Cryptocurrency Trading](https://arxiv.org/abs/2209.05559) by Berend Jelmer Dirk Gort et al.
- "High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features" by Annelotte Bonenkamp (Bachelor Thesis, University of Amsterdam, June 2021)

## Prerequisites

- **Python 3.x**
- **Modules**: `ccxt`, `numpy`, `pandas`, `joblib`, `binance`, `talib`, `requests`, `yfinance` (optional), `telebot`, `websocket-client`, `ta`, `python-dotenv`, plus ElegantRL dependencies.
- Binance Futures API access (API key and secret).
- **Configuration**: `.env` file with `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

## Installation

1. Clone:
   ```bash
   git clone <repository_url> && cd FinRL_Crypto_HFT
   ```
2. Virtual Environment:
   - Linux/macOS:
     ```bash
     python -m venv myenv && source myenv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv myenv && myenv\Scripts\activate
     ```
3. Install:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up `.env` with credentials (see Configuration below).

## How to Use

### Configuration

- Edit `config_main.py` for:
  - Validation methods (Walkforward, K-Fold CV, CPCV)
  - Candle counts
  - Tickers (e.g., `["BTC/USDT"]`)
  - Indicators and trade date ranges
- **Secure Loading**: Credentials sourced from `.env` via `python-dotenv`.

Example `.env`:
```plaintext
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### Folders

- `data`: Training/validation data, with `trade_data` subfolder.
- `drl_agents`: ElegantRL DRL implementations.
- `plots_and_metrics`: Analysis outputs (metrics, plots).
- `train`: DRL training utilities.
- `train_results`: Trained DRL models.
- `models`: Random Forest model (`model_rf.pkl`).

### Workflow

1. **Data Collection**:
   ```bash
   python 0_dl_trainval_data.py  # Training/validation data
   python 0_dl_trade_data.py  # Live data
   ```
2. **Training**:
   ```bash
   python 1_optimize_cpcv.py  # CPCV
   python 1_optimize_kcv.py  # K-Fold CV
   python 1_optimize_wf.py  # Walkforward
   python random_forest.py  # Random Forest
   ```
3. **Validation**:
   ```bash
   python 2_validate.py
   ```
4. **Backtesting**:
   ```bash
   python 4_backtest.py
   ```
5. **Evaluation**:
   ```bash
   python 5_pbo.py  # PBO calculation
   ```
6. **Live Trading**:
   ```bash
   python your_bot_script.py
   ```

## Training the Random Forest

Run:
```bash
python random_forest.py
```
- Trains and saves the Random Forest as `model_rf.pkl`.
- Ensure Binance data compatibility.

## Citing

```bibtex
@article{gort2022deep,
  title={Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting},
  author={Gort, Berend Jelmer Dirk and Liu, Xiao-Yang and Gao, Jiechao and Chen, Shuaiyu and Wang, Christina Dan},
  journal={AAAI Bridge on AI for Financial Services},
  year={2023}
}

@thesis{bonenkamp2021high,
  title={High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features},
  author={Bonenkamp, Annelotte},
  school={University of Amsterdam},
  type={Bachelor's Thesis},
  year={2021},
  month={June}
}
```

**Note:** Random Forest HFT adapts Maxime Grenu’s work, with social feature insights from Bonenkamp (2021).

## Limitations and Warnings

- **Order Minimums**: 0.00105 BTC, 100 USDT (auto-adjusted if below).
- **Leverage**: Default 20x—adjust with caution.
- **Fallback Logic**: Low BTC triggers BUY; HOLD uses RSI (customizable).
- **Risks**: HFT with leverage is high-risk—test thoroughly in Testnet first.

## Merging Process

Combines FinRL_Crypto (DRL, overfitting mitigation) with Maxime Grenu’s HFT Bot (Random Forest, enhanced by Bonenkamp’s social features). Unified via `config_main.py`, `.env`, `requirements.txt`, and this `README.md`.

## Production Enhancements & Optimizations

`your_bot_script.py` upgrades:
- **Security**: API keys from `.env` via `python-dotenv`.
- **Logging**: Rotating handler (10MB cap, 5 backups).
- **Telegram**: Retry logic (3 attempts, 5s delays).
- **API Limits**: `rateLimit` at 1200 for Binance Futures.
- **Transfers**: 10 USDT minimum for Spot-to-Futures.
- **Error Handling**: Robust portfolio/profit display with Telegram error reporting.
- **Indicators**: `ta` library for efficiency.
- **Orders**: Trade size capped at 25% of balance.
- **MLX Health**: Timeout-based server check.
- **Slippage**: 0.1% adjustment in trading logic.
- **Shutdown**: Graceful handling of `SIGINT`, `SIGTERM`.
- **WebSocket**: Auto-reconnect on disconnect.

## License

MIT License. See `LICENSE` for details.

## Contact

- Open repository issues for support.
- HFT/Random Forest: Contact Maxime Grenu.
- DRL: Contact FinRL_Crypto authors.

**Note:** For learning only. Authors are not liable for misuse.
