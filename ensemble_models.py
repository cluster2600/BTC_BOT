import numpy as np
import pandas as pd
import tensorflow as tf
import ydf
import requests
import coremltools as ct

# List of required features for the models
REQUIRED_FEATURES = [
    "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
    "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
    "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
]

def ensure_required_features(features: dict) -> dict:
    """Ensure all required features are present in the input dictionary, filling missing with 0.0."""
    for key in REQUIRED_FEATURES:
        if key not in features:
            features[key] = 0.0
    return {key: features[key] for key in REQUIRED_FEATURES}

def load_ydf_model(model_path: str = "model_rf.ydf"):
    """Load the YDF model from a TensorFlow Decision Forests format."""
    try:
        ydf_model = ydf.from_tensorflow_decision_forests(model_path)
        print(f"YDF model loaded from TFDF format at {model_path}")
        return ydf_model
    except Exception as e:
        raise RuntimeError(f"Failed to load YDF model from {model_path}: {e}")

def load_mlx_model():
    """Check connectivity to the MLX model via LM Studio server."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        response.raise_for_status()
        print("MLX model loaded via LM Studio at http://localhost:1234")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to LM Studio server: {e}")

def mlx_generate(prompt: str, url: str = "http://localhost:1234/v1/completions", max_tokens: int = 10) -> str:
    """Generate a trading decision using the MLX model via LM Studio API with reduced max_tokens for concise output."""
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
    """Extract the first occurrence of BUY, SELL, or HOLD from MLX output."""
    output = output.upper()
    words = output.split()
    for word in words:
        if word in ["BUY", "SELL", "HOLD"]:
            return word
    return "HOLD"

def ensemble_predict(ydf_model, nn_model, features: dict, mlx_url: str = "http://localhost:1234/v1/completions") -> str:
    """Predict trading decision using an ensemble of YDF, Core ML NN, and MLX models with improved parsing and validation."""
    features = ensure_required_features(features)

    # YDF prediction
    ydf_input = {key: tf.convert_to_tensor([float(features[key])], dtype=tf.float32) for key in REQUIRED_FEATURES}
    ydf_probs = ydf_model.predict(ydf_input)[0]
    if not isinstance(ydf_probs, np.ndarray) or ydf_probs.shape != (3,):
        print("YDF probabilities are not in the expected format.")
        ydf_probs = np.array([1, 0, 0], dtype=np.float32)  # Default to HOLD

    # Core ML NN prediction
    nn_input = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype=np.float32)
    nn_pred = nn_model.predict({"features": nn_input})
    if 'classProbability' in nn_pred:
        probs_dict = nn_pred['classProbability']
        nn_probs = np.array([probs_dict.get("HOLD", 0.0), probs_dict.get("BUY", 0.0), probs_dict.get("SELL", 0.0)], dtype=np.float32)
    else:
        print("Core ML model did not return class probabilities.")
        nn_probs = np.array([1, 0, 0], dtype=np.float32)  # Default to HOLD

    # MLX prediction via LM Studio API
    prompt = ("Based on the following market features: " +
              ", ".join(f"{k}: {v}" for k, v in features.items()) +
              ". What is the recommended trading action? Answer BUY, SELL, or HOLD.")
    mlx_output = mlx_generate(prompt, mlx_url, max_tokens=10)  # Reduced for concise output
    mlx_decision = parse_mlx_decision(mlx_output)
    mlx_probs = np.array({"BUY": [0, 1, 0], "SELL": [0, 0, 1], "HOLD": [1, 0, 0]}[mlx_decision], dtype=np.float32)

    # Ensure all probability arrays have shape (3,)
    if ydf_probs.shape != (3,) or nn_probs.shape != (3,) or mlx_probs.shape != (3,):
        raise ValueError("Probability arrays have inconsistent shapes.")

    # Average probabilities
    avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
    classes = ["HOLD", "BUY", "SELL"]
    final_decision = classes[np.argmax(avg_probs)]
    print(f"YDF probs: {ydf_probs}, NN probs: {nn_probs}, MLX probs: {mlx_probs}, Avg probs: {avg_probs}, Final: {final_decision}")
    return final_decision

def get_ensemble_decision(features: dict, ydf_model, nn_model, device: str = "cpu", mlx_url: str = "http://localhost:1234/v1/completions") -> str:
    """Get the ensemble decision with error handling."""
    try:
        decision = ensemble_predict(ydf_model, nn_model, features, mlx_url)
        return decision.upper()
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        return "HOLD"