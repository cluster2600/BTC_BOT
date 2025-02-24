import numpy as np
import pandas as pd
import tensorflow as tf
import ydf
import requests
import coremltools as ct
import os

# List of required features for the models
REQUIRED_FEATURES = [
    "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
    "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
    "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
]

# Define the standard class order for consistency across models
CLASSES = ["BUY", "HOLD", "SELL"]

def ensure_required_features(features: dict) -> dict:
    """
    Ensure all required features are present in the input dictionary, filling missing ones with 0.0.

    Args:
        features (dict): Input feature dictionary.

    Returns:
        dict: Feature dictionary with all required features, missing ones set to 0.0.
    """
    for key in REQUIRED_FEATURES:
        if key not in features:
            features[key] = 0.0
    return {key: features[key] for key in REQUIRED_FEATURES}

def load_ydf_model(model_path: str = "model_rf.ydf"):
    """
    Load the YDF model from the specified path, supporting TensorFlow Decision Forests format.

    Args:
        model_path (str): Path to the YDF model file.

    Returns:
        ydf_model: Loaded YDF model.

    Raises:
        RuntimeError: If the YDF model fails to load.
    """
    try:
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YDF model file not found at {model_path}")
        
        # Load the model using the correct YDF method for TFDF models
        ydf_model = ydf.from_tensorflow_decision_forests(model_path)
        print(f"YDF model loaded from {model_path}")
        return ydf_model
    except Exception as e:
        raise RuntimeError(f"Failed to load YDF model from {model_path}: {e}")

def load_mlx_model():
    """
    Check connectivity to the MLX model via LM Studio server.

    Raises:
        RuntimeError: If the MLX model fails to connect.
    """
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        response.raise_for_status()
        print("MLX model loaded via LM Studio at http://localhost:1234")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to LM Studio server: {e}")

def mlx_generate(prompt: str, url: str = "http://localhost:1234/v1/completions", max_tokens: int = 10) -> str:
    """
    Generate a trading decision using the MLX model via LM Studio API.

    Args:
        prompt (str): Input prompt for the MLX model.
        url (str): URL of the LM Studio API endpoint.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: Generated text from the MLX model, defaulting to "HOLD" on failure.
    """
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"MLX generation failed: {e}")
        return "HOLD"

def parse_mlx_decision(output: str) -> str:
    """
    Extract the first occurrence of BUY, SELL, or HOLD from MLX output.

    Args:
        output (str): Generated text from the MLX model.

    Returns:
        str: Parsed decision ("BUY", "HOLD", or "SELL"), defaulting to "HOLD".
    """
    output = output.upper()
    words = output.split()
    for word in words:
        if word in CLASSES:
            return word
    return "HOLD"

def ensemble_predict(ydf_model, nn_model, features: dict, mlx_url: str = "http://localhost:1234/v1/completions"):
    """
    Predict trading decision using an ensemble of YDF, Core ML NN, and MLX models.

    Args:
        ydf_model: Loaded YDF model.
        nn_model: Loaded Core ML neural network model.
        features (dict): Input feature dictionary.
        mlx_url (str): URL of the MLX model API endpoint.

    Returns:
        tuple: (decision, confidence) where decision is "BUY", "HOLD", or "SELL",
               and confidence is the average probability for the chosen decision.
    """
    features = ensure_required_features(features)

    # YDF prediction (assuming order: BUY, HOLD, SELL)
    ydf_input = {key: tf.convert_to_tensor([float(features[key])], dtype=tf.float32) for key in REQUIRED_FEATURES}
    ydf_probs = ydf_model.predict(ydf_input)[0]
    if not isinstance(ydf_probs, np.ndarray) or ydf_probs.shape != (3,):
        print("YDF probabilities are not in the expected format.")
        ydf_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Default to HOLD

    # Core ML NN prediction
    nn_input = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype=np.float32)
    nn_pred = nn_model.predict({"features": nn_input})
    if 'classLabel_probs' in nn_pred:
        probs_dict = nn_pred['classLabel_probs']
        nn_probs = np.array([probs_dict.get("BUY", 0.0), probs_dict.get("HOLD", 0.0), probs_dict.get("SELL", 0.0)], dtype=np.float32)
    elif 'classProbability' in nn_pred:
        probs_dict = nn_pred['classProbability']
        nn_probs = np.array([probs_dict.get("BUY", 0.0), probs_dict.get("HOLD", 0.0), probs_dict.get("SELL", 0.0)], dtype=np.float32)
    elif 'predictedClass' in nn_pred:
        predicted_class = nn_pred['predictedClass']
        nn_probs = np.zeros(3, dtype=np.float32)
        if predicted_class == "BUY":
            nn_probs[0] = 1.0
        elif predicted_class == "HOLD":
            nn_probs[1] = 1.0
        elif predicted_class == "SELL":
            nn_probs[2] = 1.0
    else:
        print(f"Core ML model output format is unexpected. Full output: {nn_pred}")
        nn_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Default to HOLD

    # MLX prediction
    prompt = ("Based on the following market features: " +
              ", ".join(f"{k}: {v}" for k, v in features.items()) +
              ". What is the recommended trading action? Answer BUY, SELL, or HOLD.")
    mlx_output = mlx_generate(prompt, mlx_url, max_tokens=10)
    mlx_decision = parse_mlx_decision(mlx_output)
    mlx_probs = np.array({"BUY": [1.0, 0.0, 0.0], "HOLD": [0.0, 1.0, 0.0], "SELL": [0.0, 0.0, 1.0]}[mlx_decision], dtype=np.float32)

    # Average probabilities (assuming order: BUY, HOLD, SELL)
    avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
    decision_index = np.argmax(avg_probs)
    decision = CLASSES[decision_index]
    confidence = avg_probs[decision_index]

    print(f"YDF probs: {ydf_probs}, NN probs: {nn_probs}, MLX probs: {mlx_probs}, Avg probs: {avg_probs}, Decision: {decision}, Confidence: {confidence}")
    return decision, confidence

def get_ensemble_decision(features: dict, ydf_model, nn_model, device: str = "cpu", mlx_url: str = "http://localhost:1234/v1/completions") -> tuple:
    """
    Get the ensemble decision with error handling.

    Args:
        features (dict): Input feature dictionary.
        ydf_model: Loaded YDF model.
        nn_model: Loaded Core ML neural network model.
        device (str): Device to use (default: "cpu").
        mlx_url (str): URL of the MLX model API endpoint.

    Returns:
        tuple: (decision, confidence) where decision is "BUY", "HOLD", or "SELL",
               and confidence is the average probability for the chosen decision.
    """
    try:
        decision, confidence = ensemble_predict(ydf_model, nn_model, features, mlx_url)
        return decision.upper(), confidence
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        return "HOLD", 0.0