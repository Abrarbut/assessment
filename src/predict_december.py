"""Fill predicted_rate in data/december-chart-inputs.csv and write it back
out in the exact 7-column format score.py expects, ready to pass as
--december-predictions.

Note: december-chart-inputs.csv does NOT include market_index or
quote_signal (unlike train-test.csv / validation.csv). We fill those with
the training-set medians learned in train.py -- reasonable here since both
features showed weak correlation with rate (~0.03-0.08) in EDA, so the
imputation has minimal impact on the prediction.

Run:
    python src/predict_december.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from features import build_feature_frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
OUT_PATH = Path(__file__).resolve().parent.parent / "december_predictions.csv"


def main() -> None:
    model = joblib.load(MODEL_DIR / "model.joblib")
    city_lookup = joblib.load(MODEL_DIR / "city_lookup.joblib")
    fills = json.loads((MODEL_DIR / "fill_values.json").read_text())

    dec = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")
    X = build_feature_frame(
        dec, city_lookup, fills["weight_fill"], fills["market_index_fill"], fills["quote_signal_fill"]
    )
    preds = model.predict(X).clip(min=1.0)

    out = dec.copy()
    out["predicted_rate"] = preds
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} December predictions to {OUT_PATH}")
    print(out[["date", "predicted_rate"]])


if __name__ == "__main__":
    main()
