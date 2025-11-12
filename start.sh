#!/usr/bin/env bash
set -e

MODEL_FILE="leukemia_model.h5"
DRIVE_ID="1QgAoKw3xEfU_YDpzxlcLQVcACSVUAqLd"

if [ ! -f "" ]; then
  echo "📥 Model not found — downloading from Drive..."
  URL="https://drive.google.com/uc?export=download&id="
  if curl -L --fail "" -o ""; then
    echo "✅ Model downloaded: "
  else
    echo "⚠️ Model download failed — continuing (app will use fallback)."
  fi
else
  echo "✅ Model already present: "
fi

echo "🚀 Starting app with gunicorn..."
exec gunicorn app:app --bind 0.0.0.0: --timeout 120
