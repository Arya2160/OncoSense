from flask import Flask, request, jsonify, send_file
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np

# TensorFlow (Keras) for .h5
try:
    from tensorflow.keras.models import load_model
    MODEL = load_model("leukemia_model.h5")
except Exception as e:
    MODEL = None
    print("⚠️  Could not load Keras model:", e)

app = Flask(__name__, static_folder=".", static_url_path="")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

@app.get("/")
def home():
    # Serve the single-page UI
    return send_file("index.html")

def _vectorize(payload: dict) -> np.ndarray:
    """
    Frontend se jo fields aa rahi hain unko vector bana do.
    Tumhare form ke base par simple mapping:
      age (int), pulse (int), fever(0/1), fever_temp(float),
      pallor(0/1), bruises(0/1), weight_loss(0/1)
    """
    def yn(x): 
        return 1 if str(x).strip().lower() in ("yes","y","1","true","haan","ha","h") else 0

    age         = int(payload.get("age", 0) or 0)
    pulse       = int(payload.get("pulse", 0) or 0)
    fever       = yn(payload.get("fever"))
    fever_temp  = float(payload.get("fever_temp", 0) or 0.0)
    pallor      = yn(payload.get("pallor"))
    bruises     = yn(payload.get("bruises"))
    weight_loss = yn(payload.get("weight_loss"))

    return np.array([[age, pulse, fever, fever_temp, pallor, bruises, weight_loss]], dtype=np.float32)

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        x = _vectorize(payload)

        # Model prediction (sigmoid output assumed: 0..1)
        if MODEL is not None:
            y = MODEL.predict(x, verbose=0)
            score = float(np.clip(y.ravel()[0], 0, 1))
            model_used = "keras"
        else:
            # Fallback heuristic: fever + pallor + bruises + tachycardia
            score = 0.0
            _, pulse, fever, fever_temp, pallor, bruises, weight_loss = x.ravel()
            score += 0.25 if fever or fever_temp >= 38 else 0.0
            score += 0.25 if pallor else 0.0
            score += 0.2 if bruises else 0.0
            score += 0.15 if pulse >= 100 else 0.0
            score += 0.15 if weight_loss else 0.0
            score = float(np.clip(score, 0, 1))
            model_used = "heuristic"

        label = ("Low", "Moderate", "High")[0 if score < 0.33 else (1 if score < 0.66 else 2)]
        return jsonify({
            "ok": True,
            "risk_score": round(score, 3),
            "risk_label": label,
            "model": model_used
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=8766)
