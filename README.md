# FinRL_Crypto_HFT: Deep Reinforcement Learning and Random Forest for Binance Futures Trading  
![banner](https://user-images.githubusercontent.com/69801109/214294114-a718d378-6857-4182-9331-20869d64d3d9.png)  
This project merges Maxime Grenu's Binance Futures HFT Trading Bot with Random Forest with FinRL_Crypto by Berend Jelmer Dirk Gort et al., combining Deep Reinforcement Learning (DRL) with a high-frequency trading (HFT) system for Binance Futures, focusing on **BTC/USDT**. It tackles overfitting in financial RL and uses a Random Forest model with indicators (SMA, RSI, MACD, Bollinger Bands) for BUY/SELL/HOLD decisions. Tested on multiple cryptocurrencies and market crashes, it aims to outperform traditional strategies.  
**Warning:** For educational use only. Not for production without validation. Leveraged trading is high-risk; use simulation/Testnet first.  

## Features  
- **Hybrid Decision-Making**: DRL (PPO, DDPG, SAC via ElegantRL) + Random Forest.  
- **Binance Futures**: `ccxt` API, Spot-to-Futures transfers (USDT, BTC, BNB).  
- **Indicators**: SMA, RSI, MACD, Bollinger Bands, TA-Lib (CCI, DX, ROC, etc.).  
- **Overfitting Mitigation**: Walkforward, K-Fold CV, CPCV for DRL.  
- **Order Management**: Minimums (0.00105 BTC, 100 USDT), 20x leverage.  
- **Real-Time**: WebSocket data, Telegram notifications.  
- **Backtesting**: PBO and performance metrics.  

## Paper  
[Deep Reinforcement Learning for Cryptocurrency Trading](https://arxiv.org/abs/2209.05559) by Berend Jelmer Dirk Gort et al.  

## Prerequisites  
- **Python 3.x**  
- Modules: `ccxt`, `numpy`, `pandas`, `joblib`, `binance`, `talib`, `requests`, `yfinance` (optional), `telebot`, `websocket-client`, ElegantRL dependencies.  
- Binance Futures API access.  
- Config: `apikeys.txt` (`BINANCE_API_KEY`, `BINANCE_API_SECRET`), `secrets.txt` (`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`).  

## Installation  
1. Clone: `git clone <repository_url> && cd FinRL_Crypto_HFT`  
2. Virtual env: `python -m venv myenv && source myenv/bin/activate` (Linux/MacOS) or `myenv\Scripts\activate` (Windows)  
3. Install: `pip install -r requirements.txt`  
4. Add `apikeys.txt` and `secrets.txt`.  

## How to Use  
### Configuration  
Edit `config_main.py`: validation methods, candle counts, tickers (e.g., ["BTC/USDT"]), indicators, trade dates.  
### Folders  
- `data`: Training/validation, `trade_data` subfolder.  
- `drl_agents`: ElegantRL DRL.  
- `plots_and_metrics`: Analysis outputs.  
- `train`: DRL utilities.  
- `train_results`: Trained models.  
- `models`: `model_rf.pkl`.  
### Workflow  
1. **Data**: `0_dl_trainval_data.py`, `0_dl_trade_data.py`  
2. **Train**: `1_optimize_cpcv.py`, `1_optimize_kcv.py`, `1_optimize_wf.py`, `random_forest.py` (Maxime Grenu’s)  
3. **Validate**: `2_validate.py`  
4. **Backtest**: `4_backtest.py`  
5. **Evaluate**: `5_pbo.py`  
6. **Live**: `your_bot_script.py`  
Models saved in `train_results`.  

## Training the Random Forest  
`python random_forest.py` (Maxime Grenu’s): Trains, saves `model_rf.pkl`. Adapt for Binance data.  

## Citing  
@article
{gort2022deep,
  title={Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting},
  author={Gort, Berend Jelmer Dirk and Liu, Xiao-Yang and Gao, Jiechao and Chen, Shuaiyu and Wang, Christina Dan},
  journal={AAAI Bridge on AI for Financial Services},
  year={2023}
}
Note: Random Forest HFT adapted from Maxime Grenu’s work.  

## Limitations and Warnings  
- **Minimums**: 0.00105 BTC, 100 USDT; adjusted if below.  
- **Leverage**: 20x default; adjust cautiously.  
- **Fallbacks**: BUY if low BTC, RSI if HOLD—customize.  
- **Risks**: High-risk HFT; validate in Testnet.  

## Merging Process  
Combines FinRL_Crypto (Gort et al.) DRL/overfitting solutions with Maxime Grenu’s HFT bot/Random Forest, unifying configs (`config_main.py`, `apikeys.txt`, `secrets.txt`), dependencies (`requirements.txt`), and docs.  

## License  
MIT License. See `LICENSE`.  

## Contact  
Open an issue or contact Maxime Grenu (HFT bot) or FinRL_Crypto authors (DRL).  
**Note:** For learning only. Authors disclaim liability for misuse.
