"""Train the freight rate model on data/train-test.csv.

Split strategy
--------------
The 48,000 labeled rows span 2025-01-01 through 2025-10-31, but the loads we
must ultimately price (validation.csv, december-chart-inputs.csv) fall in
November-December 2025 -- dates the model has never seen. A random/shuffled
train-test split would let the model "peek" at data from the same week it's
being tested on, which overstates how well it will generalize forward in
time. Instead we use a time-based split: train on Jan-Aug, validate on
Sep-Oct (the most recent ~2 months), which mimics the real task of
predicting rates for months that come after the training window.

Run:
    python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from features import build_city_lookup, build_feature_frame, clean_weight

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

SPLIT_DATE = "2025-09-01"  # train < this date, validate >= this date


def load_clean_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train-test.csv")
    df["weight"] = clean_weight(df["weight"])
    return df


def main() -> None:
    df = load_clean_train()
    df["date"] = pd.to_datetime(df["date"])

    city_lookup = build_city_lookup(df)
    weight_fill = df["weight"].median()
    market_index_fill = df["market_index"].median()
    quote_signal_fill = df["quote_signal"].median()

    train_mask = df["date"] < SPLIT_DATE
    train_raw, holdout_raw = df[train_mask], df[~train_mask]
    print(f"Train rows: {len(train_raw)}  ({train_raw.date.min().date()} - {train_raw.date.max().date()})")
    print(f"Holdout rows: {len(holdout_raw)}  ({holdout_raw.date.min().date()} - {holdout_raw.date.max().date()})")

    X_train = build_feature_frame(train_raw, city_lookup, weight_fill, market_index_fill, quote_signal_fill)
    y_train = train_raw["posted_rate"].values
    X_holdout = build_feature_frame(holdout_raw, city_lookup, weight_fill, market_index_fill, quote_signal_fill)
    y_holdout = holdout_raw["posted_rate"].values

    # Baseline: Ridge regression, to sanity-check that the gradient boosting
    # model is actually earning its complexity.
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_holdout)

    # Main model
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)
    gbm_pred = model.predict(X_holdout)

    def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f"{name:>22s}:  MAE=${mae:,.2f}   RMSE=${rmse:,.2f}   MAPE={mape:.2%}")
        return {"mae": mae, "rmse": rmse, "mape": mape}

    print()
    print(f"Holdout period: {SPLIT_DATE} through {holdout_raw.date.max().date()}")
    baseline_metrics = report("Ridge baseline", y_holdout, base_pred)
    gbm_metrics = report("HistGBM (final model)", y_holdout, gbm_pred)

    # Feature importance via permutation would be nicer, but HistGBM doesn't
    # expose native importances cheaply; report correlation-based signal
    # instead for a quick sanity check.
    print()
    print("Refitting final model on ALL labeled data (train+holdout) for production use...")
    X_all = build_feature_frame(df, city_lookup, weight_fill, market_index_fill, quote_signal_fill)
    y_all = df["posted_rate"].values
    final_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=model.n_iter_,  # reuse the iteration count chosen by early stopping
        l2_regularization=0.1,
        random_state=42,
    )
    final_model.fit(X_all, y_all)

    joblib.dump(final_model, MODEL_DIR / "model.joblib")
    joblib.dump(city_lookup, MODEL_DIR / "city_lookup.joblib")
    fill_values = {
        "weight_fill": float(weight_fill),
        "market_index_fill": float(market_index_fill),
        "quote_signal_fill": float(quote_signal_fill),
    }
    (MODEL_DIR / "fill_values.json").write_text(json.dumps(fill_values, indent=2))
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps({"baseline": baseline_metrics, "gbm": gbm_metrics, "split_date": SPLIT_DATE}, indent=2)
    )
    print(f"\nSaved model + lookups to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
