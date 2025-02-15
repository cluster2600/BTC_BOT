# Binance Futures HFT Trading Bot with Random Forest

This project implements a high-frequency trading (HFT) system for Binance Futures, focused on the **BTC/USDT** pair. The bot uses financial indicators (SMA, RSI, MACD, Bollinger Bands) as well as a pre-trained Random Forest model to make trading decisions (BUY, SELL, or HOLD).

**Warning:**  
This project is provided for educational purposes only. It is not intended for production use without thorough validation and adaptation to real market conditions. Trading on leveraged markets carries significant risks and can lead to substantial financial losses. Use this code in simulation or on Binance Futures' testnet before any live deployment.

## Features

- **Connection to Binance Futures** via the ccxt API.  
- Automatic transfer of funds from the Spot account to the Futures account (USDT, BTC, and BNB).  
- Calculation of financial indicators based on historical price data (SMA, RSI, MACD, Bollinger Bands).  
- Prediction of trading decisions (BUY, SELL, or HOLD) using a pre-trained Random Forest model (file `model_rf.pkl`).  
- Execution of market orders based on the predicted decision.  
- Verification and adjustment of order quantities to meet the minimum requirements (e.g., 0.001 BTC for BTC/USDT).  
- Fallback management: for example, if the BTC balance is insufficient for selling, the decision is forced to BUY.

## Prerequisites

- **Python 3.x**  
- Required Python modules (installable via pip):  
  - `ccxt`  
  - `numpy`  
  - `pandas`  
  - `joblib`  
  - `requests` (installed with ccxt)  
- A Binance account with Futures API access.  
- An `apikeys.txt` file containing your Binance API keys in the following format:
    ```
    BINANCE_API_KEY=YourAPIKey
    BINANCE_API_SECRET=YourAPISecret
    ```

## Installation

1. Clone this repository or download the source code.
2. Create a virtual environment (optional but recommended):
    
    (Linux/MacOS)
    ```
    python -m venv myenv
    source myenv/bin/activate
    ```
    
    (Windows)
    ```
    python -m venv myenv
    myenv\Scripts\activate
    ```
3. Install the required dependencies:
    
    ```
    pip install ccxt numpy pandas joblib
    ```
4. Place your `apikeys.txt` file in the same directory as the script.

## Training the Random Forest Model

Before running the bot, you need to train and save a Random Forest model in the file `model_rf.pkl`. You can use the provided `random_forest.py` script to train the model on a sample dataset. For example:

    python random_forest.py

This script will display the best hyperparameters, the F1-score, and save the model in `model_rf.pkl`.

## Usage

To launch the trading bot, simply run the main script:

    python your_bot_script.py

The bot will perform the following operations:
- Initialize funds: transfer USDT, BTC, and BNB from the Spot account to the Futures account.  
- Display the current portfolio.  
- Retrieve the latest price for BTC/USDT and update the price history.  
- Calculate financial indicators and predict the trading decision using the Random Forest model.  
- Verify minimum conditions (e.g., sufficient balance to buy at least 0.001 BTC) and execute market orders.

## Limitations and Warnings

- **Minimum Quantities:**  
  The bot checks that the order quantity is at least equal to the minimum required by Binance (generally 0.001 BTC for BTC/USDT). If this threshold is not met, the order will not be executed.

- **No Leverage:**  
  This code executes orders without leverage (1x). You can adapt the code if you wish to incorporate leverage, but do so with caution.

- **Decision Fallbacks:**  
  Fallback mechanisms are in place to force a BUY order if the BTC balance is insufficient for a SELL, or to enforce an order based on RSI in case of a HOLD prediction. Adapt these rules according to your strategy.

- **Financial Risks:**  
  High-frequency trading on leveraged markets carries significant risks. This project should only be used after thorough validation and a complete understanding of the associated risks.

## License

This project is distributed under the MIT license. See the LICENSE file for more information.

## Contact

For any questions or suggestions, please open an issue in the repository or contact me directly.

**Note:** This project is intended for learning and experimentation. The author disclaims any liability for improper use.
