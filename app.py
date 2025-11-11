# app.py  — OncoSense API (Railway-ready)

import os
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- Model bootstrap ----------
MODEL_KIND = "rule_based"
LEUK_MODEL = None

# Optional: load keras model if present
H5_PATH = os.path.join(os.path.dirname(__file__), "leukemia_model.h5")
if os.path.exists(H5_PATH):
    try:
        # tensorflow-cpu in requirements; Railway par lightweight
        from tensorflow import keras  # type: ignore
        LEUK_MODEL = keras.models.load_model(H5_PATH)
        MODEL_KIND = "leukemia_h5"
        print("✅ Loaded keras model:", H5_PATH)
    except Exception as e:
        print("⚠️ TF load failed, using rule-based:", e)
        MODEL_KIND = "rule_based"
        LEUK_MODEL = None

# ---------- Feature schemas ----------
CORE = ["age", "fever", "pulse", "pallor", "bruises", "weight_loss"]
ADV  = ["fatigue", "night_sweats", "frequent_infections", "bone_pain"]

def geti(d: Dict[str, Any], k: str, default: int = 0) -> int:
    try:
        return int(float(d.get(k, default)))
    except Exception:
        return default

def getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return default

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def grade(p: float) -> str:
    if p >= 0.66: return "High"
    if p >= 0.33: return "Medium"
    return "Low"

def temp_component(temp_c: float) -> float:
    """
    Temperature ko zyada izzat: continuous buckets
    <38 → 0, 38–38.9 → 0.18, 39–39.9 → 0.30, ≥40 → 0.45
    """
    if temp_c < 38.0:   return 0.00
    if temp_c < 39.0:   return 0.18
    if temp_c < 40.0:   return 0.30
    return 0.45

def pulse_component(pulse: int) -> float:
    if pulse >= 120: return 0.25
    if pulse >= 100: return 0.15
    if pulse >= 90:  return 0.05
    return 0.00

def rule_based_prob(
    age:int, fever:int, temp_c:float, pulse:int, pallor:int, bruises:int, wloss:int,
    fatigue:int, nsweats:int, infections:int, bone:int
) -> Dict[str, float]:
    contrib = {}

    # Temperature-heavy contribution
    c_temp = temp_component(temp_c)
    contrib["temperature"] = c_temp

    # Fever flag (auto / manual) — light touch
    c_fever = 0.05 * fever
    contrib["fever_flag"] = c_fever

    # Core red flags
    c_pallor  = 0.20 * pallor
    c_bruises = 0.20 * bruises
    c_wloss   = 0.20 * wloss
    contrib["pallor"]  = c_pallor
    contrib["bruises"] = c_bruises
    contrib["weight_loss"] = c_wloss

    # Pulse & age
    c_pulse = pulse_component(pulse)
    c_age   = 0.05 if age <= 6 else 0.00
    contrib["pulse"] = c_pulse
    contrib["young_age"] = c_age

    # Advanced symptoms (mild–moderate)
    c_fatigue = 0.05 * fatigue
    c_sweats  = 0.05 * nsweats
    c_inf     = 0.10 * infections
    c_bone    = 0.15 * bone
    contrib["fatigue"] = c_fatigue
    contrib["night_sweats"] = c_sweats
    contrib["infections"] = c_inf
    contrib["bone_pain"]  = c_bone

    total = sum(contrib.values())
    prob  = clamp01(total)
    return {"prob": prob, **{f"w_{k}": v for k, v in contrib.items()}}

@app.get("/health")
def health():
    return {"ok": True, "model_kind": MODEL_KIND}, 200

@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}

    # Temperature → fever auto-map if fever missing
    temp_c = getf(data, "fever_temp_c", getf(data, "temperature_c", 0.0))
    fever  = geti(data, "fever", 0)
    if "fever" not in data and temp_c > 0:
        fever = 1 if temp_c >= 38.0 else 0

    age    = geti(data, "age", 8)
    pulse  = geti(data, "pulse", 90)
    pallor = geti(data, "pallor", 0)
    bruis  = geti(data, "bruises", 0)
    wloss  = geti(data, "weight_loss", 0)

    fatigue    = geti(data, "fatigue", 0)
    nsweats    = geti(data, "night_sweats", 0)
    infections = geti(data, "frequent_infections", 0)
    bone       = geti(data, "bone_pain", 0)

    # Model path if available (trained on 6 core features)
    model_prob = None
    used_model = None
    if MODEL_KIND == "leukemia_h5" and LEUK_MODEL is not None:
        try:
            import numpy as np  # local import to keep cold start small
            x = np.array([[age, fever, pulse, pallor, bruis, wloss]], dtype="float32")
            model_prob = float(LEUK_MODEL.predict(x, verbose=0)[0][0])
            used_model = MODEL_KIND
        except Exception as e:
            print("⚠️ keras predict failed:", e)

    if model_prob is not None:
        base_prob = model_prob
        # Lightly nudge with advanced symptoms even on model path
        adj = 0.05*fatigue + 0.05*nsweats + 0.10*infections + 0.15*bone + temp_component(temp_c)*0.25
        prob = clamp01(base_prob + adj)
        out = {"risk": round(prob, 4), "class_name": grade(prob),
               "model_prob": round(base_prob, 4), "used_model": used_model}
        return jsonify(out)

    # Rule-based (tuned)
    rb = rule_based_prob(age, fever, temp_c, pulse, pallor, bruis, wloss, fatigue, nsweats, infections, bone)
    prob = rb.pop("prob")
    out = {
        "risk": round(prob, 4),
        "class_name": grade(prob),
        "used_model": None,
        "details": rb
    }
    return jsonify(out), 200

# ---------- Local run (Railway picks PORT env) ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8766))
    app.run(host="0.0.0.0", port=port, debug=False)
