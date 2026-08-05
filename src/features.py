"""Shared feature engineering for the freight rate model.

Kept in one place so training, validation-prediction, and December-chart
prediction all transform raw rows identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_LEVELS = ["Dry Van", "Flatbed", "Reefer"]


def build_city_lookup(train_df: pd.DataFrame) -> pd.DataFrame:
    """Every city has one fixed (lat, lon) in the data. Build city -> coords."""
    pick = train_df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    deliv = train_df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    both = pd.concat([pick, deliv], ignore_index=True).drop_duplicates("city")
    return both.set_index("city")[["lat", "lon"]]


def clean_weight(weight: pd.Series) -> pd.Series:
    """Some weights are sign-flipped (negative). Magnitudes look valid, so
    take the absolute value rather than dropping/treating as missing."""
    return weight.abs()


def add_lane_features(df: pd.DataFrame, city_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pickup_lat" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(city_lookup["lat"])
        df["pickup_lon"] = df["pickup"].map(city_lookup["lon"])
    if "delivery_lat" not in df.columns:
        df["delivery_lat"] = df["delivery"].map(city_lookup["lat"])
        df["delivery_lon"] = df["delivery"].map(city_lookup["lon"])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    doy = dt.dt.dayofyear.astype(float)
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    return df


def build_feature_frame(
    raw: pd.DataFrame,
    city_lookup: pd.DataFrame,
    weight_fill: float,
    market_index_fill: float,
    quote_signal_fill: float | None = None,
) -> pd.DataFrame:
    """Turn a raw rows dataframe (validation.csv-style or
    december-chart-inputs.csv-style) into the numeric feature matrix the
    model expects. Missing market_index/quote_signal (as in the December
    file, which doesn't carry those columns at all) are filled with values
    learned from training data.
    """
    df = add_lane_features(raw, city_lookup)
    df = add_date_features(df)

    df["weight"] = clean_weight(df["weight"])
    df["weight"] = df["weight"].fillna(weight_fill)

    if "market_index" in df.columns:
        df["market_index"] = df["market_index"].fillna(market_index_fill)
    else:
        df["market_index"] = market_index_fill

    if "quote_signal" in df.columns:
        df["quote_signal"] = df["quote_signal"].fillna(
            quote_signal_fill if quote_signal_fill is not None else 0.0
        )
    else:
        df["quote_signal"] = quote_signal_fill if quote_signal_fill is not None else 0.0

    for level in EQUIPMENT_LEVELS:
        df[f"equipment_{level.replace(' ', '_')}"] = (df["equipment"] == level).astype(int)

    feature_cols = [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
        "month",
    ] + [f"equipment_{level.replace(' ', '_')}" for level in EQUIPMENT_LEVELS]

    return df[feature_cols]
