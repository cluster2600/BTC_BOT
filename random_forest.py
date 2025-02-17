#!/usr/bin/env python3
"""
Random Forest Training using Yggdrasil Decision Forests (YDF) on a Mac with potential Metal GPU acceleration.

This script generates a synthetic dataset with 100,000 examples,
trains a more aggressive Random Forest model, evaluates it,
and saves the model for later use.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import ydf  # Ensure you have installed YDF via: pip install ydf -U

# Set up logging for clear messages
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def generate_synthetic_data(num_samples: int = 100_000) -> pd.DataFrame:
    """
    Generates a synthetic dataset with the specified number of examples.
    Distribution: HOLD=10%, BUY=45%, SELL=45%.
    """
    logging.info(f"Creating a dataset with {num_samples} examples...")
    data = {
        'price': np.random.rand(num_samples) * 10000,
        'sma': np.random.rand(num_samples) * 10000,
        'rsi': np.random.rand(num_samples) * 100,  # RSI in [0, 100]
        'macd': np.random.rand(num_samples) * 10 - 5,
        'signal': np.random.rand(num_samples) * 10 - 5,
        'lower_bb': np.random.rand(num_samples) * 10000,
        'sma_bb': np.random.rand(num_samples) * 10000,
        'upper_bb': np.random.rand(num_samples) * 10000,
        'social_feature': np.random.randint(0, 100, num_samples),
        # Reduced HOLD probability for a more aggressive stance
        'target': np.random.choice([0, 1, 2], num_samples, p=[0.1, 0.45, 0.45])
    }
    df = pd.DataFrame(data)
    # Convert target to string for classification
    df['target'] = df['target'].astype(str)
    logging.info(f"Dataset created. DataFrame shape: {df.shape}")
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Shuffles and splits the dataset into training and testing sets.
    """
    logging.info("Splitting dataset into training and test sets...")
    shuffled_df = df.sample(frac=1, random_state=42)
    train_size = int(train_ratio * len(df))
    train_df = shuffled_df.iloc[:train_size].reset_index(drop=True)
    test_df = shuffled_df.iloc[train_size:].reset_index(drop=True)
    logging.info(f"Training set: {train_df.shape}")
    logging.info(f"Test set: {test_df.shape}")
    return train_df, test_df

def get_model_size(path: str) -> int:
    """
    Calculates the total file size of the saved model, whether it's a single file or a directory.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    return 0

def train_random_forest(train_df: pd.DataFrame, learner_params: dict) -> any:
    """
    Trains a Random Forest model using YDF.
    
    Params:
        train_df: Training DataFrame.
        learner_params: Dictionary of parameters for the learner.
        
    Returns:
        The trained model.
    """
    logging.info("Starting Random Forest training with YDF...")
    learner = ydf.RandomForestLearner(label="target", **learner_params)
    start_time = time.time()
    model = learner.train(train_df)
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")
    return model

def evaluate_model(model: any, test_df: pd.DataFrame) -> dict:
    """
    Evaluates the trained model on the test dataset.
    """
    logging.info("Evaluating the model on the test set...")
    evaluation = model.evaluate(test_df)
    logging.info(f"Evaluation results: {evaluation}")
    return evaluation

def save_model(model: any, model_path: str):
    """
    Saves the model to disk and prints its size.
    """
    logging.info(f"Saving the model to '{model_path}'...")
    model.save(model_path)
    total_size = get_model_size(model_path)
    size_mb = total_size / (1024 * 1024)
    logging.info(f"Model saved at '{model_path}' (total size: {size_mb:.2f} MB).")
    if size_mb < 1:
        logging.info("Note: The file size is small because Yggdrasil Decision Forests uses a very compact storage format. This is normal.")

def execute_trade(signal: str, order_value: float):
    """
    Executes a trade order, ensuring that the order is at least 100 USDT.
    For orders below the threshold, the value is automatically adjusted to 100 USDT.
    """
    if order_value < 100:
        logging.info(f"Order value {order_value} USDT is too low, adjusting to 100 USDT.")
        order_value = 100
    logging.info(f"Executing {signal} order with a value of {order_value} USDT.")
    # Integrate your Binance API call here to execute the trade

def simulate_trading(model: any, test_df: pd.DataFrame):
    """
    Simulates trading using the trained model on the test dataset.
    For each BUY or SELL prediction, executes a trade with a default order value
    (e.g., 50 USDT which will be adjusted via execute_trade if necessary).
    """
    logging.info("Simulating trading on the test set...")
    predictions = model.predict(test_df)
    # Loop over a few predictions for demonstration
    for idx, pred in enumerate(predictions[:10]):  # simulate for 10 examples
        signal = { "0": "HOLD", "1": "BUY", "2": "SELL" }.get(pred, "HOLD")
        if signal != "HOLD":
            # Example: compute an order value based on some strategy (here a dummy value of 50 USDT)
            order_value = 50
            logging.info(f"Prediction {idx}: {signal} signal detected with an initial order value of {order_value} USDT.")
            execute_trade(signal, order_value)
        else:
            logging.info(f"Prediction {idx}: HOLD signal detected. No trade executed.")

def main():
    # Parameters
    num_samples = 100_000
    learner_params = {
        "num_trees": 800,
        "max_depth": 30,
        "random_seed": 42
    }
    model_save_path = "model_rf.ydf"

    # Generate synthetic data with a more aggressive approach
    df = generate_synthetic_data(num_samples)

    # Split into training and test sets
    train_df, test_df = split_dataset(df)

    # Train the Random Forest model
    model = train_random_forest(train_df, learner_params)

    # Evaluate the model
    evaluate_model(model, test_df)

    # Display the model's description for further insights
    logging.info("Model description:")
    logging.info(model.describe())

    # Save the model
    save_model(model, model_save_path)

    # Simulate trading to demonstrate the minimum order constraint
    simulate_trading(model, test_df)

if __name__ == "__main__":
    main()
