# ensemble_models.py

import numpy as np
import pandas as pd
import ydf
import tensorflow as tf

# Global list of required features in the order expected by the YDF model.
REQUIRED_FEATURES = [
    "price", "sma", "rsi", "macd", "signal_line", 
    "lower_bb", "sma_bb", "upper_bb", "social_feature", 
    "adx", "atr", "volume", "order_book_depth", "news_sentiment", 
    "vol_adjusted_price", "volume_ma"
]

# A global dictionary for ensemble model parameters or outputs.
# This variable could be set after training to store parameters used by both models.
ensemble_model_dict = {}

def ensure_required_features(features: dict) -> dict:
    """
    Ensure that the features dictionary contains all keys specified in REQUIRED_FEATURES.
    If a key is missing, it is added with a default value of 0.0.
    
    Parameters:
        features (dict): Input dictionary of features.
        
    Returns:
        dict: Updated dictionary with all required keys.
    """
    for key in REQUIRED_FEATURES:
        if key not in features:
            features[key] = 0.0
    # Optionally, re-order the dictionary (if order is important) by REQUIRED_FEATURES:
    ordered_features = {key: features[key] for key in REQUIRED_FEATURES}
    return ordered_features

def load_ydf_model(model_path: str = "model_rf.ydf"):
    """
    Load the YDF Random Forest model.
    
    Parameters:
        model_path (str): Path to the YDF model file.
    
    Returns:
        The loaded YDF model.
    """
    try:
        model = ydf.load_model(model_path)
        print(f"YDF model loaded from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YDF model from {model_path}: {e}")

def load_tf_model(model_path: str = "nn_model.h5"):
    """
    Load the TensorFlow neural network model.
    
    Parameters:
        model_path (str): Path to the TensorFlow model (HDF5 file).
    
    Returns:
        The loaded TensorFlow Keras model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"TensorFlow model loaded from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model from {model_path}: {e}")

def ensemble_predict(ydf_model, tf_model, features: dict) -> str:
    """
    Generate an ensemble prediction by averaging predictions from the YDF and TensorFlow models.
    This function first ensures that all required features are present.
    
    Parameters:
        ydf_model: Loaded YDF model.
        tf_model: Loaded TensorFlow model.
        features (dict): A dictionary of feature values.
        
    Returns:
        A string decision: "BUY", "SELL", or "HOLD".
    
    Assumptions:
      - Both models output probability distributions over the classes in the order:
        ["BUY", "SELL", "HOLD"].
    """
    # Ensure the features dictionary includes all required keys.
    features = ensure_required_features(features)
    
    # Convert features to a single-row DataFrame.
    df = pd.DataFrame([features])
    
    # Obtain predictions from the YDF model.
    try:
        ydf_pred = ydf_model.predict(df)
        # Assume the output is a list or array of probabilities.
        ydf_probs = np.array(ydf_pred)[0]
    except Exception as e:
        raise RuntimeError(f"YDF model prediction failed: {e}")
    
    # Obtain predictions from the TensorFlow model.
    try:
        tf_pred = tf_model.predict(df)
        tf_probs = tf_pred[0]
    except Exception as e:
        raise RuntimeError(f"TensorFlow model prediction failed: {e}")
    
    # Average the probabilities from both models.
    avg_probs = (np.array(ydf_probs) + np.array(tf_probs)) / 2.0

    # Map the indices to class labels (assumed order: BUY, SELL, HOLD).
    classes = ["BUY", "SELL", "HOLD"]
    final_decision = classes[np.argmax(avg_probs)]
    
    print(f"YDF probabilities: {ydf_probs}")
    print(f"TensorFlow probabilities: {tf_probs}")
    print(f"Averaged probabilities: {avg_probs}")
    print(f"Final ensemble decision: {final_decision}")
    
    return final_decision

def get_ensemble_decision(features: dict, ydf_model, tf_model) -> str:
    """
    Wrapper to get ensemble decision using ensemble_predict().
    This function returns the model's decision (BUY, SELL, or HOLD).
    """
    try:
        decision = ensemble_predict(ydf_model, tf_model, features)
        decision = decision.upper()
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        return decision
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        return "HOLD"

# Example usage in a live scenario:
if __name__ == '__main__':
    # Example: Load models and simulate a prediction with missing "volume".
    ydf_model = load_ydf_model("model_rf.ydf")
    tf_model = load_tf_model("nn_model.h5")
    
    # Sample live features (intentionally missing the "volume" key)
    live_features = {
        "price": 95280.0,
        "sma": 95280.07,
        "rsi": 0.0,
        "macd": 0.0,
        "signal_line": 0.0,
        "lower_bb": 0.0,
        "sma_bb": 0.0,
        "upper_bb": 0.0,
        # "volume" key is missing here!
        "social_feature": 0.0,
        "adx": 0.0,
        "atr": 0.0,
        "order_book_depth": 0.0,
        "news_sentiment": 0.0,
        "vol_adjusted_price": 0.0,
        "volume_ma": 0.0
    }
    # The helper ensure_required_features() will add "volume": 0.0 automatically.
    decision = get_ensemble_decision(live_features, ydf_model, tf_model)
    print(f"Ensemble model decision: {decision}")