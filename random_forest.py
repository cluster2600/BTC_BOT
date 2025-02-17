#!/usr/bin/env python3
"""
Enhanced Trading Model Ensemble using Yggdrasil Decision Forests (YDF) and TensorFlow.

This script generates a synthetic dataset with multiple technical indicators and market features,
applies feature engineering, trains:
    • a YDF Random Forest model, and
    • a TensorFlow neural network for multi-class classification (HOLD, BUY, SELL),
then ensembles their predictions via stacking.
The ensemble prediction (optionally combined with a simple rule-based signal) is used to simulate trading,
ensuring that each trade order is at least 100 USDT and that BTC orders meet the minimum quantity of 0.00105 BTC.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import ydf  # pip install ydf -U
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import Optional, Tuple
import urllib.request  # Added to fix Telegram notification error

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
debug_log = logging.debug  # helper alias

# Global indicator parameters (to be optimized on historical data)
INDICATOR_PARAMS = {
    'rsi_buy_threshold': 30,
    'rsi_sell_threshold': 70,
    'adx_threshold': 25,
    'atr_multiplier': 1.5
}

# Define the required features in the order expected by the YDF model
REQUIRED_FEATURES = [
    'price', 'sma', 'rsi', 'macd', 'signal_line',
    'lower_bb', 'sma_bb', 'upper_bb', 'social_feature',
    'adx', 'atr', 'volume', 'order_book_depth',
    'news_sentiment', 'vol_adjusted_price', 'volume_ma'
]

# Global variable for ensemble model dictionary (populated after training)
ensemble_model_dict = {}

def generate_synthetic_data(num_samples: int = 100_000) -> pd.DataFrame:
    logging.info(f"Creating a dataset with {num_samples} examples...")
    data = {
        'price': np.random.rand(num_samples) * 10000,
        'sma': np.random.rand(num_samples) * 10000,
        'rsi': np.random.rand(num_samples) * 100,
        'macd': np.random.rand(num_samples) * 10 - 5,
        'signal_line': np.random.rand(num_samples) * 10 - 5,
        'lower_bb': np.random.rand(num_samples) * 10000,
        'sma_bb': np.random.rand(num_samples) * 10000,
        'upper_bb': np.random.rand(num_samples) * 10000,
        'social_feature': np.random.randint(0, 100, num_samples),
        'adx': np.random.rand(num_samples) * 50,
        'atr': np.random.rand(num_samples) * 200,
        'volume': np.random.rand(num_samples) * 10000,
        'order_book_depth': np.random.rand(num_samples) * 1000,
        'news_sentiment': np.random.rand(num_samples) * 2 - 1
    }
    data['target'] = np.random.choice([0, 1, 2], num_samples, p=[0.1, 0.45, 0.45])
    df = pd.DataFrame(data)
    df['target'] = df['target'].astype(str)
    logging.info(f"Dataset created with shape: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Performing feature engineering...")
    df['vol_adjusted_price'] = df['price'] / (df['atr'] * INDICATOR_PARAMS['atr_multiplier'] + 1)
    df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
    logging.info("Feature engineering completed.")
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8):
    logging.info("Splitting dataset into training and test sets...")
    shuffled_df = df.sample(frac=1, random_state=42)
    train_size = int(train_ratio * len(df))
    train_df = shuffled_df.iloc[:train_size].reset_index(drop=True)
    test_df = shuffled_df.iloc[train_size:].reset_index(drop=True)
    logging.info(f"Training set shape: {train_df.shape}; Test set shape: {test_df.shape}")
    return train_df, test_df

def get_model_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    return 0

### YDF Random Forest Model Training ###
def train_random_forest(train_df: pd.DataFrame, learner_params: dict) -> any:
    logging.info("Training YDF Random Forest model...")
    learner = ydf.RandomForestLearner(label="target", **learner_params)
    start_time = time.time()
    model = learner.train(train_df)
    elapsed_time = time.time() - start_time
    logging.info(f"YDF training completed in {elapsed_time:.2f} seconds.")
    return model

### TensorFlow Neural Network Training ###
def prepare_tf_dataset(df: pd.DataFrame, feature_cols: list, batch_size: int = 100):
    X = df[feature_cols].copy()
    y = df['target'].astype(int).values
    y_onehot = to_categorical(y, num_classes=3)
    dataset = tf.data.Dataset.from_tensor_slices((X.values.astype('float32'), y_onehot))
    dataset = dataset.shuffle(buffer_size=len(X), seed=42).batch(batch_size)
    return dataset

def build_nn_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_neural_network(tf_train_ds, tf_val_ds, input_dim: int, epochs: int = 5) -> tf.keras.Model:
    logging.info("Training TensorFlow neural network model...")
    model = build_nn_model(input_dim)
    model.fit(tf_train_ds, validation_data=tf_val_ds, epochs=epochs, verbose=1)
    return model

### Rule-Based Signal Generation ###
def rule_based_signal(row: pd.Series) -> str:
    if row['rsi'] < INDICATOR_PARAMS['rsi_buy_threshold'] and row['adx'] > INDICATOR_PARAMS['adx_threshold']:
        return "BUY"
    elif row['rsi'] > INDICATOR_PARAMS['rsi_sell_threshold'] and row['adx'] > INDICATOR_PARAMS['adx_threshold']:
        return "SELL"
    else:
        return "HOLD"

def ensemble_signal(ydf_pred: np.ndarray, nn_pred: np.ndarray) -> str:
    avg_prob = (ydf_pred + nn_pred) / 2.0
    pred_class = np.argmax(avg_prob)
    return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(pred_class, "HOLD")

### Trade Execution ###
def execute_trade(signal: str, order_value: float, current_price: float, symbol: str):
    """
    Executes a trade order. For BTC orders, if the computed quantity (order_value/current_price)
    is below 0.00105 BTC, the trade is aborted.
    """
    if "BTC" in symbol:
        quantity = order_value / current_price if current_price > 0 else 0
        if quantity < 0.00105:
            logging.error(f"Forced {signal} quantity {quantity:.6f} BTC is below 0.00105 BTC.")
            return
    if order_value < 100:
        logging.info(f"Order value {order_value} USDT too low, adjusting to 100 USDT.")
        order_value = 100
    logging.info(f"Executing {signal} order for {symbol} with value {order_value} USDT.")
    # Insert your Binance API integration here

### Helper Functions for Feature Processing ###
def to_feature_dict(x, feature_cols):
    return {col: tf.convert_to_tensor(x[:, i:i+1]) for i, col in enumerate(feature_cols)}

def ensure_required_features(features: dict) -> dict:
    for key in REQUIRED_FEATURES:
        if key not in features:
            features[key] = 0.0
    return features

### NEW: Ensemble Prediction and Decision Functions ###
def get_ensemble_decision(features: dict) -> str:
    global ensemble_model_dict
    input_array = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype='float32')
    ydf_probs = ensemble_model_dict['ydf'](input_array).numpy()[0]
    nn_probs = ensemble_model_dict['nn'](input_array).numpy()[0]
    return ensemble_signal(ydf_probs, nn_probs)

def predict_decision(symbol: str, current_price: float, moving_average: float,
                     rsi: Optional[float], macd: Optional[float],
                     signal_line: Optional[float],
                     bbands: Tuple[Optional[float], Optional[float], Optional[float]]) -> str:
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
        "volume": 0.0,  # Added to satisfy the model's requirement.
        "social_feature": 0.0,
        "adx": 0.0,
        "atr": 0.0,
        "order_book_depth": 0.0,
        "news_sentiment": 0.0,
        "vol_adjusted_price": 0.0,
        "volume_ma": 0.0
    }
    debug_log(f"Features for {symbol}: {features}")
    ensemble_decision = get_ensemble_decision(features)
    if ensemble_decision != "HOLD":
        return ensemble_decision
    if rsi is not None:
        if rsi < 40:
            return "BUY"
        elif rsi > 60:
            return "SELL"
    return "HOLD"

### Simulation of Trading Using the Ensemble Model (Test Data) ###
def simulate_trading(ensemble_model, test_df: pd.DataFrame, feature_cols: list):
    logging.info("Simulating trading using ensemble predictions...")
    for idx in range(10):
        row_features = test_df.iloc[idx].to_dict()
        row_features = ensure_required_features(row_features)
        input_array = np.array([[row_features[col] for col in REQUIRED_FEATURES]], dtype='float32')
        ydf_probs = ensemble_model['ydf'](input_array).numpy()[0]
        nn_probs = ensemble_model['nn'](input_array).numpy()[0]
        final_signal = ensemble_signal(ydf_probs, nn_probs)
        rule_sig = rule_based_signal(test_df.iloc[idx])
        if final_signal == rule_sig and final_signal in ["BUY", "SELL"]:
            order_value = 50  # Dummy value in USDT
            price = row_features.get("price", 0.0)
            logging.info(f"Example {idx}: YDF={ydf_probs}, NN={nn_probs}, Rule={rule_sig}, Ensemble signal: {final_signal}")
            execute_trade(final_signal, order_value, price, "BTC/USDT:USDT")
        else:
            logging.info(f"Example {idx}: No clear actionable signal (YDF: {ydf_probs}, NN: {nn_probs}, Rule: {rule_sig}). No trade executed.")

### Main Pipeline ###
def main():
    global ensemble_model_dict
    df = generate_synthetic_data()
    df = feature_engineering(df)
    feature_cols = [col for col in df.columns if col != 'target']
    train_df, test_df = split_dataset(df)
    
    # --- Train YDF Model ---
    ydf_learner_params = {"num_trees": 800, "max_depth": 30, "random_seed": 42}
    ydf_model = train_random_forest(train_df, ydf_learner_params)
    ydf_tf = ydf_model.to_tensorflow_function()
    
    # --- Prepare TensorFlow Datasets for NN Training ---
    split_index = int(0.9 * len(train_df))
    tf_train_df = train_df.iloc[:split_index].reset_index(drop=True)
    tf_val_df = train_df.iloc[split_index:].reset_index(drop=True)
    tf_train_ds = prepare_tf_dataset(tf_train_df, feature_cols, batch_size=100)
    tf_val_ds = prepare_tf_dataset(tf_val_df, feature_cols, batch_size=100)
    
    # --- Train Neural Network Model ---
    input_dim = len(feature_cols)
    nn_model = train_neural_network(tf_train_ds, tf_val_ds, input_dim, epochs=5)
    
    # --- Build Ensemble Model Dictionary ---
    ensemble_model_dict = {
        'ydf': lambda x: ydf_tf(to_feature_dict(x, feature_cols)),
        'nn': lambda x: nn_model(x)
    }
    
    simulate_trading(ensemble_model_dict, test_df, feature_cols)
    
    # Example: Live data prediction integration
    live_features = {
        'price': 95324.0,
        'sma': 95324.0,
        'rsi': 0.0,
        'macd': 0.0,
        'signal_line': 0.0,
        'lower_bb': 0.0,
        'sma_bb': 0.0,
        'upper_bb': 0.0,
        # 'volume' intentionally missing to test default insertion
        'order_book_depth': 0.0,
        'news_sentiment': 0.0,
        'vol_adjusted_price': 0.0,
        'volume_ma': 0.0,
        'adx': 0.0,
        'atr': 0.0,
        'social_feature': 0.0
    }
    live_decision = predict_decision("BTC/USDT:USDT", live_features.get("price", 0.0),
                                     live_features.get("sma", 0.0),
                                     live_features.get("rsi", 0.0),
                                     live_features.get("macd", 0.0),
                                     live_features.get("signal_line", 0.0),
                                     (live_features.get("lower_bb", 0.0),
                                      live_features.get("sma_bb", 0.0),
                                      live_features.get("upper_bb", 0.0)))
    logging.info(f"Predicted live decision for BTC/USDT:USDT: {live_decision}")
    
    model_save_path = "model_rf.ydf"
    logging.info(f"Saving YDF model to {model_save_path}...")
    ydf_model.save(model_save_path)
    size_mb = get_model_size(model_save_path) / (1024 * 1024)
    logging.info(f"YDF model saved (size: {size_mb:.2f} MB).")
    
    nn_save_path = "nn_model.h5"
    nn_model.save(nn_save_path)
    logging.info(f"Neural network model saved to {nn_save_path}.")

if __name__ == "__main__":
    main()