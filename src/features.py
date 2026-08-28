"""
Feature engineering for the freight rate model.

Design notes (these matter for the December scenario, see below):

- `pickup` / `delivery` / `equipment` are kept as pandas category dtype and
  handed to LightGBM's native categorical handling rather than one-hot
  encoded. With 64 cities on each side, one-hot would add >120 sparse
  columns for no real benefit — LightGBM splits on categories directly.
- Date is decomposed into calendar features plus cyclical (sin/cos)
  encodings of month and day-of-year so the model can pick up on
  seasonality without treating December 31 and January 1 as "far apart".
- `pickup_lat/lon` and `delivery_lat/lon` are not always available at
  inference time (the December chart inputs only give city names). A
  `CITY_COORDS` lookup, built once from the training data, backfills
  coordinates from the city name in that case.
- `market_index` and `quote_signal` are similarly absent from the
  December file. Rather than skip the columns (which would require a
  second model), we hold them at the trained-on median. This is
  documented explicitly in the report — the December curve isolates the
  seasonal/calendar effect on a fixed lane, and does not react to any
  live market signal because none is available for future dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]

NUMERIC_COLUMNS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "weight_was_missing",
    "market_index_was_missing",
]

DATE_FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + DATE_FEATURE_COLUMNS


def build_city_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per city with its lat/lon, built from whichever side
    (pickup or delivery) the city shows up on in the training data.
    Used to backfill coordinates when only a city name is available.
    """
    pickup_side = df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery_side = df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    combined = pd.concat([pickup_side, delivery_side], ignore_index=True)
    return combined.groupby("city", as_index=False).first()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def fill_missing_coords(df: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    """Backfill pickup/delivery lat-lon from the city lookup where absent."""
    df = df.copy()
    lookup = city_coords.set_index("city")[["lat", "lon"]]

    for side in ["pickup", "delivery"]:
        lat_col, lon_col = f"{side}_lat", f"{side}_lon"
        if lat_col not in df.columns:
            df[lat_col] = np.nan
        if lon_col not in df.columns:
            df[lon_col] = np.nan
        matched = df[side].map(lookup["lat"])
        df[lat_col] = df[lat_col].fillna(matched)
        matched_lon = df[side].map(lookup["lon"])
        df[lon_col] = df[lon_col].fillna(matched_lon)
    return df


def fill_missing_market_features(
    df: pd.DataFrame, fallback_values: dict[str, float]
) -> pd.DataFrame:
    """Hold market_index / quote_signal at their trained-on median if absent."""
    df = df.copy()
    for col, value in fallback_values.items():
        if col not in df.columns:
            df[col] = value
        else:
            df[col] = df[col].fillna(value)
    return df


def set_categorical_dtypes(df: pd.DataFrame, category_levels: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for col, levels in category_levels.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    city_coords: pd.DataFrame,
    category_levels: dict[str, list],
    market_fallback: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    df = fill_missing_coords(df, city_coords)
    df = fill_missing_market_features(df, market_fallback)
    df = add_date_features(df)
    df = set_categorical_dtypes(df, category_levels)
    return df[FEATURE_COLUMNS]
