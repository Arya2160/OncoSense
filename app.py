import os
import json
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

# --- Auto-download model from Google Drive ---
MODEL_URL = os.environ.get("MODEL_URL")  # set in Render env
MODEL_PATH = Path("leukemia_model.h5")

def ensure_model_downloaded():
    """Download model from MODEL_URL if not present."""
    if MODEL_PATH.exists():
        print("✅ Model already exists:", MODEL_PATH)
        return True
    if not MODEL_URL:
        print("❌ MODEL_URL not set in environment.")
        return False
    try:
        print("📥 Downloading model from:", MODEL_URL)
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded:", MODEL_PATH)
        return True
    except Exception as e:
        print("❌ Model download failed:", e)
        return False

# Ensure model file exists before loading
ensure_model_downloaded()

# --- TensorFlow model loading ---
TF_OK = True
MODEL = None
MODEL_ERR = None

def try_load_model():
    """Try loading TensorFlow model safely."""
    global MODEL, MODEL_ERR, TF_OK
    if MODEL is not None or MODEL_ERR is not None:
        return MODEL
    try:
        import tensorflow as tf
        MODEL = tf.keras.models.load_model("leukemia_model.h5", compile=False)
        print("✅ TF model loaded successfully!")
        return MODEL
    except Exception as e:
        TF_OK = False
        MODEL_ERR = str(e)
        print("⚠️ TensorFlow model load failed:", e)
        return None

# --- fallback heuristic if model fails ---
def fallback_score(features):
    """Rudimentary heuristic model."""
    score = 0.0
    if str(features.get("fever", "No")).lower() == "yes": score += 0.25
    if str(features.get("pallor", "No")).lower() == "yes": score += 0.25
    if str(features.get("bruises", "No")).lower() == "yes": score += 0.20
    if str(features.get("weight_loss", "No")).lower() == "yes": score += 0.15
    try:
        pulse = float(features.get("pulse", 80))
        if pulse >= 100: score += 0.10
    except: pass
    try:
        temp = float(features.get("temp_c", 37))
        if temp >= 38: score += 0.10
    except: pass
    return min(1.0, max(0.0, score))

# --- Flask app setup ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) or {}

    model = try_load_model()
    if model is not None:
        try:
            x = np.array([[
                float(data.get("age", 8)),
                float(data.get("pulse", 80)),
                1.0 if str(data.get("fever", "No")).lower()=="yes" else 0.0,
                float(data.get("temp_c", 37) or 0),
                1.0 if str(data.get("pallor", "No")).lower()=="yes" else 0.0,
                1.0 if str(data.get("bruises", "No")).lower()=="yes" else 0.0,
                1.0 if str(data.get("weight_loss", "No")).lower()=="yes" else 0.0,
            ]], dtype="float32")

            prob = float(model.predict(x, verbose=0)[0][0])
            label = "High" if prob >= 0.5 else "Low"
            return jsonify({
                "risk": label,
                "score": round(prob, 3),
                "model": "tf-keras",
                "api": "v1"
            })
        except Exception as e:
            print("⚠️ Prediction error:", e)

    # fallback
    prob = fallback_score({
        "fever": data.get("fever"),
        "pallor": data.get("pallor"),
        "bruises": data.get("bruises"),
        "weight_loss": data.get("weight_loss"),
        "pulse": data.get("pulse"),
        "temp_c": data.get("temp_c"),
    })
    label = "High" if prob >= 0.5 else "Low"
    return jsonify({
        "risk": label,
        "score": round(prob, 3),
        "model": "heuristic-fallback",
        "tf_load_error": MODEL_ERR
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8766)
