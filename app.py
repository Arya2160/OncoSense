# app.py
import os, threading
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_KIND = "rule_based"
LEUK_MODEL = None

# ---- helpers ----
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def parse_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

@app.get("/health")
def health():
    return {"ok": True, "model_kind": MODEL_KIND}, 200

@app.post("/predict")
def predict():
    """
    Expected core fields:
      age (int), pulse (int), fever (0/1), pallor (0/1), bruises (0/1), weight_loss (0/1)
    Optional:
      fever_temp_c (float), fatigue/night_sweats/frequent_infections/bone_pain (0/1)
    """
    data = request.get_json(force=True) or {}

    # ---------- core ----------
    age    = int(parse_float(data.get("age", 8), 8))
    pulse  = int(parse_float(data.get("pulse", 90), 90))
    fever  = int(parse_float(data.get("fever", 0), 0))
    pallor = int(parse_float(data.get("pallor", 0), 0))
    bruis  = int(parse_float(data.get("bruises", 0), 0))
    wloss  = int(parse_float(data.get("weight_loss", 0), 0))

    # ---------- optional extras ----------
    temp_c = parse_float(data.get("fever_temp_c"), None)
    fatigue    = int(parse_float(data.get("fatigue", 0), 0))
    nsweats    = int(parse_float(data.get("night_sweats", 0), 0))
    infections = int(parse_float(data.get("frequent_infections", 0), 0))
    bone_pain  = int(parse_float(data.get("bone_pain", 0), 0))

    # ---------- temperature-first scoring ----------
    # temp severity: 37.5 -> 0, 41.0 -> 1 (linear), clamp 0..1
    if temp_c is not None:
        temp_sev = (temp_c - 37.5) / 3.5
    else:
        temp_sev = 1.0 if fever == 1 else 0.0
    temp_sev = clamp01(temp_sev)

    # pulse severity: 90 -> 0, 120 -> 1 (linear)
    if pulse <= 90:
        pulse_sev = 0.0
    elif pulse >= 120:
        pulse_sev = 1.0
    else:
        pulse_sev = (pulse - 90) / 30.0
    pulse_sev = clamp01(pulse_sev)

    # weights (sum ~1). Temp ko sabse zyada izzat.
    w_temp, w_feverflag = 0.35, 0.05
    w_pallor, w_bruises, w_wloss = 0.12, 0.12, 0.12
    w_pulse, w_age = 0.07, 0.02

    score = 0.0
    score += w_temp * temp_sev
    score += w_feverflag * (1.0 if fever==1 else 0.0)
    score += w_pallor * (1.0 if pallor==1 else 0.0)
    score += w_bruises * (1.0 if bruis==1 else 0.0)
    score += w_wloss * (1.0 if wloss==1 else 0.0)
    score += w_pulse * pulse_sev
    if age <= 6: score += w_age

    # advanced symptoms adjust (soft push)
    delta = 0.10*fatigue + 0.10*nsweats + 0.15*infections + 0.20*bone_pain
    prob = clamp01(score + 0.4*delta)

    cls = (
    "High" if prob >= 0.72 else
    "Medium" if prob >= 0.40 else
    "Low"
)

    return jsonify({
        "class_name": cls,
        "risk": round(prob,4),
        "model_prob": round(score,4),
        "used_model": MODEL_KIND,
        "debug": {
            "temp_c": temp_c,
            "temp_sev": round(temp_sev,3),
            "pulse_sev": round(pulse_sev,3)
        }
    })

# ---------- run server ----------
if __name__ == "__main__":
    # port 8766 as used by HTML
    app.run(host="0.0.0.0", port=8766, debug=False)
