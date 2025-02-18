#!/usr/bin/env python3
"""
Enhanced Trading Model Ensemble using Yggdrasil Decision Forests (YDF), TensorFlow, and DeepSeek‑Qwen‑1.5B for decision support.

This script loads trade data from an Excel file, applies feature engineering, and trains:
    • a YDF Random Forest model, and
    • a TensorFlow neural network for multi-class classification.
Their predictions are ensembled via stacking, and additional decision support is provided via the DeepSeek‑Qwen‑1.5B model.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
import ydf  # Ensure YDF is installed (e.g., pip install ydf -U)

# Global indicator parameters
INDICATOR_PARAMS = {
    'rsi_buy_threshold': 30,
    'rsi_sell_threshold': 70,
    'adx_threshold': 25,
    'atr_multiplier': 1.5
}

# List of features expected by the ensemble (and used for training)
REQUIRED_FEATURES = [
    'price', 'sma', 'rsi', 'macd', 'signal_line',
    'lower_bb', 'sma_bb', 'upper_bb', 'social_feature',
    'adx', 'atr', 'volume', 'order_book_depth',
    'news_sentiment', 'vol_adjusted_price', 'volume_ma'
]

# Global ensemble model dictionary
ensemble_model_dict = {}

# ---------------------------
# Secrets Handling
# ---------------------------
def get_huggingface_token(secrets_file="secrets.txt"):
    token = None
    if os.path.exists(secrets_file):
        with open(secrets_file, "r") as f:
            for line in f:
                if line.startswith("HUGGINGFACE_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break
    if not token:
        logging.warning("No Hugging Face token found in %s", secrets_file)
    else:
        logging.info("Hugging Face token loaded from %s", secrets_file)
    return token

# ---------------------------
# Data Loading and Processing
# ---------------------------
def load_trade_data(filepath: str) -> pd.DataFrame:
    logging.info("Loading trade data from %s...", filepath)
    df = pd.read_excel(filepath)
    
    # Map columns if necessary
    column_map = {'Order Price': 'price', 'AvgTrading Price': 'sma'}
    logging.info("Mapping columns: %s", column_map)
    df.rename(columns=column_map, inplace=True)
    
    # Drop non-numeric columns that are not needed
    drop_cols = ['Date(UTC)', 'clientOrderId', 'Pair', 'Type', 'status', 'Strategy Id', 'Strategy Type', 'orderId']
    for col in drop_cols:
        if col in df.columns:
            logging.info("Dropping column: %s", col)
            df.drop(columns=[col], inplace=True)
    
    # Ensure required numeric columns are present; add missing ones with default 0.0
    required_numeric = {'adx', 'social_feature', 'upper_bb', 'order_book_depth', 'volume',
                        'rsi', 'atr', 'lower_bb', 'volume_ma', 'sma_bb', 'news_sentiment', 'macd', 'signal_line', 'vol_adjusted_price'}
    missing_cols = required_numeric - set(df.columns)
    if missing_cols:
        logging.info("Adding missing columns with default value 0.0: %s", missing_cols)
        for col in missing_cols:
            df[col] = 0.0

    # Generate synthetic target labels if not present
    if 'target' not in df.columns:
        logging.info("No 'target' column found in trade data. Generating synthetic target labels.")
        try:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            noise = np.random.uniform(-0.05, 0.05, len(df))
            df.loc[:, 'future_price'] = df['price'] * (1 + noise)
            df.loc[:, 'predicted_return'] = ((df['future_price'] - df['price']) / df['price']) * 100
            def label_target(row):
                return 1 if row['predicted_return'] > 0 else (2 if row['predicted_return'] < 0 else 0)
            df.loc[:, 'target'] = df.apply(label_target, axis=1).astype(str)
        except Exception as e:
            logging.error("Error generating synthetic target: %s", e)
            raise e

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Performing feature engineering...")
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['atr'] = pd.to_numeric(df['atr'], errors='coerce')
    df.loc[:, 'vol_adjusted_price'] = df['price'] / (df['atr'] * INDICATOR_PARAMS['atr_multiplier'] + 1)
    df.loc[:, 'volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
    logging.info("Feature engineering completed.")
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8):
    logging.info("Splitting dataset into training and test sets...")
    shuffled_df = df.sample(frac=1, random_state=42)
    train_size = int(train_ratio * len(df))
    train_df = shuffled_df.iloc[:train_size].reset_index(drop=True)
    test_df = shuffled_df.iloc[train_size:].reset_index(drop=True)
    logging.info("Training set shape: %s; Test set shape: %s", train_df.shape, test_df.shape)
    return train_df, test_df

def prepare_tf_dataset(df: pd.DataFrame, feature_cols: list, batch_size: int = 100):
    X = df[feature_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = df['target'].astype(int).values
    y_onehot = to_categorical(y, num_classes=3)
    dataset = tf.data.Dataset.from_tensor_slices((X.values.astype('float32'), y_onehot))
    dataset = dataset.shuffle(buffer_size=len(X), seed=42).batch(batch_size)
    return dataset

# ---------------------------
# Model Training Functions
# ---------------------------
def train_random_forest(train_df: pd.DataFrame, learner_params: dict) -> any:
    logging.info("Training YDF Random Forest model...")
    learner = ydf.RandomForestLearner(label="target", **learner_params)
    start_time = time.time()
    model = learner.train(train_df)
    elapsed_time = time.time() - start_time
    logging.info("YDF training completed in %.2f seconds.", elapsed_time)
    return model

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

# ---------------------------
# DeepSeek‑Qwen‑1.5B Model Integration
# ---------------------------
def load_deepseek_model(model_path: str = "./DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Load the DeepSeek‑Qwen‑1.5B model and its tokenizer for local inference.
    Uses the Hugging Face token from secrets.txt if necessary.
    """
    token = get_huggingface_token()
    device = torch.device("mps")
    logging.info("Loading DeepSeek‑Qwen‑1.5B model on device: %s", device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=token
        ).to(device)
    except Exception as e:
        logging.error("Failed to load DeepSeek model: %s", e)
        raise e
    return tokenizer, model, device

# ---------------------------
# Ensemble and Decision Functions
# ---------------------------
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

def get_ensemble_decision(features: dict) -> str:
    global ensemble_model_dict
    # Build input array from REQUIRED_FEATURES in fixed order
    input_array = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype='float32')
    # Build input dictionary for YDF using the same fixed order
    ydf_input = {col: input_array[:, i:i+1] for i, col in enumerate(REQUIRED_FEATURES)}
    ydf_probs = ensemble_model_dict['ydf'](ydf_input).numpy()[0]
    nn_probs = ensemble_model_dict['nn'](input_array).numpy()[0]
    return ensemble_signal(ydf_probs, nn_probs)

def predict_decision(symbol: str, current_price: float, moving_average: float,
                     rsi: float, macd: float, signal_line: float,
                     bbands: tuple) -> str:
    lower_bb, sma_bb, upper_bb = bbands
    features = {
        "price": current_price,
        "sma": moving_average,
        "rsi": rsi,
        "macd": macd,
        "signal_line": signal_line,
        "lower_bb": lower_bb,
        "sma_bb": sma_bb,
        "upper_bb": upper_bb,
        "volume": 0.0,
        "social_feature": 0.0,
        "adx": 0.0,
        "atr": 0.0,
        "order_book_depth": 0.0,
        "news_sentiment": 0.0,
        "vol_adjusted_price": 0.0,
        "volume_ma": 0.0
    }
    ensemble_decision = get_ensemble_decision(features)
    if ensemble_decision != "HOLD":
        return ensemble_decision
    if rsi < 40:
        return "BUY"
    elif rsi > 60:
        return "SELL"
    return "HOLD"

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    # Load and process trade data
    df_trade = load_trade_data("export_trades.xlsx")
    df_trade = feature_engineering(df_trade)
    feature_cols = [col for col in df_trade.columns if col != 'target']
    train_df, test_df = split_dataset(df_trade)

    # Train YDF model
    ydf_learner_params = {"num_trees": 800, "max_depth": 30, "random_seed": 42}
    ydf_model = train_random_forest(train_df, ydf_learner_params)
    ydf_tf = ydf_model.to_tensorflow_function()

    # Prepare TensorFlow datasets for NN training
    split_index = int(0.9 * len(train_df))
    tf_train_df = train_df.iloc[:split_index].reset_index(drop=True)
    tf_val_df = train_df.iloc[split_index:].reset_index(drop=True)
    tf_train_ds = prepare_tf_dataset(tf_train_df, feature_cols, batch_size=100)
    tf_val_ds = prepare_tf_dataset(tf_val_df, feature_cols, batch_size=100)

    # Train NN model
    input_dim = len(feature_cols)
    nn_model = train_neural_network(tf_train_ds, tf_val_ds, input_dim, epochs=5)

    # Build ensemble dictionary using REQUIRED_FEATURES order.
    # NOTE: We now directly pass the input dictionary to ydf_tf.
    global ensemble_model_dict
    ensemble_model_dict = {
        'ydf': lambda x: ydf_tf(x),
        'nn': lambda x: nn_model(x)
    }

    # Simulate trading using ensemble predictions (example)
    logging.info("Simulating trading using ensemble predictions...")
    for idx in range(10):
        row = test_df.iloc[idx]
        features = {col: row[col] for col in REQUIRED_FEATURES}
        decision = get_ensemble_decision(features)
        logging.info("Example %d: Ensemble decision = %s", idx, decision)

    # Load the DeepSeek‑Qwen‑1.5B model for decision support
    try:
        ds_tokenizer, ds_model, ds_device = load_deepseek_model()
        prompt = "Based on current market data, what is the recommended trading decision?"
        input_ids = ds_tokenizer(prompt, return_tensors="pt").input_ids.to(ds_device)
        output = ds_model.generate(input_ids, max_new_tokens=50)
        response = ds_tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info("DeepSeek model response: %s", response)
    except Exception as e:
        logging.error("DeepSeek model integration failed: %s", e)

    # Save models if needed
    ydf_model.save("model_rf.ydf")
    nn_model.save("nn_model.keras")
    logging.info("Models saved successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()