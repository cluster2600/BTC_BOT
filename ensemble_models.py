# ensemble_models.py

import numpy as np
import pandas as pd
import tensorflow as tf
import ydf
import requests

# REQUIRED_FEATURES matching your_bot_script.py
REQUIRED_FEATURES = [
    "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
    "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
    "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
]

def ensure_required_features(features: dict) -> dict:
    for key in REQUIRED_FEATURES:
        if key not in features:
            features[key] = 0.0
    return {key: features[key] for key in REQUIRED_FEATURES}

def load_ydf_model(model_path: str = "model_rf.ydf"):
    try:
        # Load the TFDF model and convert to YDF
        ydf_model = ydf.from_tensorflow_decision_forests(model_path)
        print(f"YDF model loaded from TFDF format at {model_path}")
        return ydf_model
    except Exception as e:
        raise RuntimeError(f"Failed to load YDF model from {model_path}: {e}")

def load_mlx_model():
    try:
        response = requests.get("http://localhost:1234/v1/models")
        response.raise_for_status()
        print("MLX model loaded via LM Studio at http://localhost:1234")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to LM Studio server: {e}")
    return None

def mlx_generate(prompt: str, url: str = "http://localhost:1234/v1/completions", max_tokens: int = 32) -> str:
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"MLX generation failed: {e}")
        return "HOLD"

def ensemble_predict(ydf_model, nn_model, features: dict, mlx_url: str = "http://localhost:1234/v1/completions") -> str:
    features = ensure_required_features(features)
    # YDF prediction
    ydf_input = {key: tf.convert_to_tensor([float(features[key])], dtype=tf.float32) for key in REQUIRED_FEATURES}
    ydf_probs = ydf_model.predict(ydf_input)[0]  # NumPy array

    # Core ML NN prediction (ML Program format)
    nn_input = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype=np.float32)
    nn_pred = nn_model.predict({"features": nn_input})
    nn_probs = list(nn_pred.values())[0]  # Extract probabilities from Core ML output dict

    # MLX prediction via LM Studio API
    prompt = ("Based on the following market features: " +
              ", ".join(f"{k}: {v}" for k, v in features.items()) +
              ". What is the recommended trading action? Answer BUY, SELL, or HOLD.")
    mlx_output = mlx_generate(prompt, mlx_url)
    print(f"MLX output: {mlx_output}")
    mlx_decision = "HOLD"
    if "BUY" in mlx_output.upper():
        mlx_decision = "BUY"
    elif "SELL" in mlx_output.upper():
        mlx_decision = "SELL"
    mlx_probs = {"BUY": [0, 1, 0], "SELL": [0, 0, 1], "HOLD": [1, 0, 0]}[mlx_decision]

    # Average probabilities
    avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
    classes = ["HOLD", "BUY", "SELL"]
    final_decision = classes[np.argmax(avg_probs)]
    print(f"YDF probs: {ydf_probs}, NN probs: {nn_probs}, MLX decision: {mlx_decision}, Avg probs: {avg_probs}, Final: {final_decision}")
    return final_decision

def get_ensemble_decision(features: dict, ydf_model, nn_model, device: str = "cpu", mlx_url: str = "http://localhost:1234/v1/completions") -> str:
    try:
        decision = ensemble_predict(ydf_model, nn_model, features, mlx_url)
        return decision.upper()
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        return "HOLD"

if __name__ == '__main__':
    import coremltools as ct
    ydf_model = load_ydf_model("model_rf.ydf")
    nn_model = ct.models.MLModel("NNModel.mlpackage")
    live_features = {
        "price": 95280.0, "Order_Amount": 0.0, "sma": 95280.07, "Filled": 0.0, "Total": 0.0,
        "future_price": 95280.0, "atr": 0.0, "vol_adjusted_price": 95280.0, "volume_ma": 0.0,
        "macd": 0.0, "signal_line": 0.0, "lower_bb": 0.0, "sma_bb": 0.0, "upper_bb": 0.0,
        "news_sentiment": 0.0, "social_feature": 0.0, "adx": 0.0, "rsi": 0.0, "order_book_depth": 0.0,
        "volume": 0.0
    }
    decision = get_ensemble_decision(live_features, ydf_model, nn_model)
    print(f"Ensemble model decision: {decision}")