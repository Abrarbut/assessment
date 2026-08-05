"""Predict rates for every load in data/validation.csv and write
validation_predictions.csv in the required load_id,predicted_rate format.

Run:
    python src/predict_validation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from features import build_feature_frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
OUT_PATH = Path(__file__).resolve().parent.parent / "validation_predictions.csv"


def main() -> None:
    model = joblib.load(MODEL_DIR / "model.joblib")
    city_lookup = joblib.load(MODEL_DIR / "city_lookup.joblib")
    fills = json.loads((MODEL_DIR / "fill_values.json").read_text())

    val = pd.read_csv(DATA_DIR / "validation.csv")
    X = build_feature_frame(
        val, city_lookup, fills["weight_fill"], fills["market_index_fill"], fills["quote_signal_fill"]
    )
    preds = model.predict(X)
    preds = preds.clip(min=1.0)  # guard against non-positive predictions

    template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")
    assert list(template["load_id"]) == list(val["load_id"]), "load_id order mismatch between files"
    template["predicted_rate"] = preds
    template.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(template)} predictions to {OUT_PATH}")
    print(template["predicted_rate"].describe())


if __name__ == "__main__":
    main()
