#!/usr/bin/env python3
"""
Enhanced Trading Model Ensemble with DeepSeek Integration.

This script trains a Random Forest (via TensorFlow Decision Forests), a Neural Network,
and integrates a pre-trained DeepSeek model for trading decisions.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
debug_log = logging.debug

# Global parameters
INDICATOR_PARAMS = {'atr_multiplier': 1.5, 'sma_period': 20, 'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'bb_period': 20, 'bb_std': 2}

# Required features with underscores (consistent with TFDF naming)
REQUIRED_FEATURES = [
    'price', 'Order_Amount', 'sma', 'Filled', 'Total', 'future_price', 'atr', 'vol_adjusted_price',
    'volume_ma', 'macd', 'signal_line', 'lower_bb', 'sma_bb', 'upper_bb', 'news_sentiment',
    'social_feature', 'adx', 'rsi', 'order_book_depth', 'volume'
]

# Global dictionary for ensemble models
ensemble_model_dict = {}

### Data Loading and Feature Engineering ###

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(100)

def calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_bbands(series: pd.Series, period: int, num_std: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    return sma - num_std * std, sma, sma + num_std * std

def load_trade_data(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading trade data from {filepath}...")
    df = pd.read_excel(filepath)
    mapping = {'Order Price': 'price', 'AvgTrading Price': 'sma', 'Order Amount': 'Order_Amount'}
    df = df.rename(columns=mapping)
    non_numeric = ['Date(UTC)', 'orderId', 'clientOrderId', 'Pair', 'Type', 'status', 'Strategy Id', 'Strategy Type']
    for col in non_numeric:
        if col in df.columns:
            logging.info(f"Dropping non-numeric column: {col}")
            df = df.drop(columns=[col])
    if 'target' not in df.columns:
        logging.warning(f"No 'target' column found in {filepath}. Generating synthetic labels for testing purposes.")
        df['target'] = np.random.choice([0, 1, 2], size=len(df))  # 0 = HOLD, 1 = BUY, 2 = SELL
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Performing feature engineering...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    df['sma'] = calculate_sma(df['price'], INDICATOR_PARAMS['sma_period'])
    df['rsi'] = calculate_rsi(df['price'], INDICATOR_PARAMS['rsi_period'])
    df['macd'], df['signal_line'], _ = calculate_macd(df['price'], INDICATOR_PARAMS['macd_fast'], INDICATOR_PARAMS['macd_slow'], INDICATOR_PARAMS['macd_signal'])
    df['lower_bb'], df['sma_bb'], df['upper_bb'] = calculate_bbands(df['price'], INDICATOR_PARAMS['bb_period'], INDICATOR_PARAMS['bb_std'])
    df['vol_adjusted_price'] = df['price'] / (df.get('atr', 0.0) * INDICATOR_PARAMS['atr_multiplier'] + 1)
    df['volume_ma'] = calculate_sma(df.get('volume', pd.Series(0, index=df.index)), INDICATOR_PARAMS['sma_period'])
    
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    logging.info("Feature engineering completed.")
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Splitting dataset into training and test sets...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(train_ratio * len(df_shuffled))
    return df_shuffled.iloc[:split_index], df_shuffled.iloc[split_index:]

### Model Training Functions ###

def train_random_forest(train_df: pd.DataFrame, learner_params: dict) -> 'tfdf.keras.RandomForestModel':
    logging.info("Training Random Forest model using TensorFlow Decision Forests...")
    import tensorflow_decision_forests as tfdf
    model = tfdf.keras.RandomForestModel(
        num_trees=learner_params["num_trees"],
        max_depth=learner_params["max_depth"],
        random_seed=learner_params["random_seed"],
        task=tfdf.keras.Task.CLASSIFICATION
    )
    tf_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df[REQUIRED_FEATURES + ['target']], label="target")
    start_time = time.time()
    model.fit(tf_train_ds)
    elapsed_time = time.time() - start_time
    logging.info(f"TFDF training completed in {elapsed_time:.2f} seconds.")
    model.save("model_rf.ydf")
    return model

def prepare_tf_dataset(df: pd.DataFrame, feature_cols: list, batch_size: int = 100) -> tf.data.Dataset:
    X = df[feature_cols].copy()
    y = df['target'].astype(int).values
    y_onehot = to_categorical(y, num_classes=3)
    dataset = tf.data.Dataset.from_tensor_slices((X.values.astype('float32'), y_onehot))
    return dataset.shuffle(buffer_size=len(X), seed=42).batch(batch_size)

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
    model.save("nn_model.h5")
    return model

### DeepSeek Model ###

def load_deepseek_model(model_path: str = "./DeepSeek-R1-Distill-Qwen-1.5B", device: str = "cpu"):
    logging.info(f"Loading DeepSeek model from {model_path} on device {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model, device

def deepseek_generate(tokenizer, model, prompt: str, device: str = "cpu", max_new_tokens: int = 32) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"DeepSeek generation failed: {e}")
        return "HOLD"  # Fallback to HOLD if generation fails

### Ensemble Decision ###

def get_ensemble_decision(features: dict) -> str:
    # Prepare input for YDF with consistent feature names
    ydf_input = {key: tf.convert_to_tensor([features.get(key, 0.0)], dtype=tf.float32) for key in REQUIRED_FEATURES}
    ydf_probs = ensemble_model_dict['ydf'](ydf_input)[0]  # YDF predict returns NumPy array directly
    
    # Prepare input for NN (flat array)
    nn_input = np.array([[features.get(col, 0.0) for col in REQUIRED_FEATURES]], dtype='float32')
    nn_probs = ensemble_model_dict['nn'](nn_input).numpy()[0]
    
    # DeepSeek prediction
    prompt = ("Based on the following market features: " +
              ", ".join(f"{k}: {v}" for k, v in features.items()) +
              ". What is the recommended trading action? Answer BUY, SELL, or HOLD.")
    ds_output = deepseek_generate(ensemble_model_dict['ds_tokenizer'], ensemble_model_dict['ds_model'], prompt, ensemble_model_dict['ds_device'])
    logging.info(f"DeepSeek output: {ds_output}")
    ds_decision = "HOLD"
    if "BUY" in ds_output.upper():
        ds_decision = "BUY"
    elif "SELL" in ds_output.upper():
        ds_decision = "SELL"
    ds_probs = {"BUY": [0, 1, 0], "SELL": [0, 0, 1], "HOLD": [1, 0, 0]}[ds_decision]
    
    # Average probabilities
    avg_probs = (ydf_probs + nn_probs + np.array(ds_probs)) / 3.0
    return {0: "HOLD", 1: "BUY", 2: "SELL"}[np.argmax(avg_probs)]

def simulate_trading(ensemble_model: dict, test_df: pd.DataFrame, feature_cols: list):
    logging.info("Simulating trading...")
    test_df = test_df.rename(columns={'Order Amount': 'Order_Amount'})
    for idx in range(min(10, len(test_df))):
        row = test_df.iloc[idx]
        features = {col: row.get(col, 0.0) for col in REQUIRED_FEATURES}
        decision = get_ensemble_decision(features)
        logging.info(f"Example {idx}: Features: {features} -> Decision: {decision}")
        execute_trade(decision, order_value=100, current_price=features["price"], symbol="BTC/USDT")

def execute_trade(signal: str, order_value: float, current_price: float, symbol: str):
    logging.info(f"Executing {signal} order for {symbol} with value {order_value} USDT at price {current_price}")

### Main Pipeline ###

def main():
    df_trade = load_trade_data("export_trades.xlsx")
    df_trade = feature_engineering(df_trade)
    feature_cols = REQUIRED_FEATURES
    train_df, test_df = split_dataset(df_trade)

    ydf_learner_params = {"num_trees": 500, "max_depth": 20, "random_seed": 42}
    ydf_model = train_random_forest(train_df, ydf_learner_params)
    ydf_tf = lambda x: ydf_model.predict(x)

    split_index = int(0.9 * len(train_df))
    tf_train_df = train_df.iloc[:split_index].reset_index(drop=True)
    tf_val_df = train_df.iloc[split_index:].reset_index(drop=True)
    tf_train_ds = prepare_tf_dataset(tf_train_df, feature_cols)
    tf_val_ds = prepare_tf_dataset(tf_val_df, feature_cols)
    nn_model = train_neural_network(tf_train_ds, tf_val_ds, len(feature_cols))

    ds_tokenizer, ds_model, ds_device = load_deepseek_model()

    ensemble_model_dict['ydf'] = ydf_tf
    ensemble_model_dict['nn'] = nn_model
    ensemble_model_dict['ds_tokenizer'] = ds_tokenizer
    ensemble_model_dict['ds_model'] = ds_model
    ensemble_model_dict['ds_device'] = ds_device

    simulate_trading(ensemble_model_dict, test_df, feature_cols)

if __name__ == "__main__":
    main()
