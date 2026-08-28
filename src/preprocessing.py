"""
Cleaning and preprocessing for the freight rate dataset.

The raw train/validation files share the same quirks, so every fix here
is applied through one shared function to keep train and inference
consistent. Two issues showed up during EDA and are handled explicitly:

1. `weight` has a chunk of negative values (roughly 0.6% of rows in both
   train_test.csv and validation.csv). These look like a sign error at
   entry time rather than genuine negative freight weight (a negative
   trailer weight isn't physically meaningful), and the magnitude of the
   negative values sits inside the normal positive weight range. Taking
   the absolute value recovers a plausible weight instead of throwing
   the rows away.
2. `weight` and `market_index` both have a small number of true nulls
   (~0.6-1.8% depending on file/column). These are imputed with the
   training median, and a `_was_missing` flag is kept for each so the
   model can still learn if "missingness" itself carries signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_FLAG_COLUMNS = ["weight", "market_index"]


def fix_weight_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Negative weights are a sign-entry error; recover the magnitude."""
    df = df.copy()
    df["weight"] = df["weight"].abs()
    return df


def add_missing_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Flags rows where the value was null. If the column is absent entirely
    (the December chart inputs don't carry market_index/quote_signal at
    all), every row is flagged as missing for that column.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
        else:
            df[f"{col}_was_missing"] = 1
    return df


def impute_with_reference(
    df: pd.DataFrame, reference_medians: dict[str, float]
) -> pd.DataFrame:
    """Fill nulls using medians computed on the training split only."""
    df = df.copy()
    for col, median_value in reference_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_value)
    return df


def compute_reference_medians(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {col: float(df[col].median()) for col in columns if col in df.columns}


def clean_dataframe(
    df: pd.DataFrame,
    reference_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Full cleaning pass. If reference_medians is None, medians are computed
    from this dataframe (use this for the training split). If provided,
    those medians are reused (use this for validation/test/inference so
    nothing about the target split leaks into imputation).
    """
    df = df.copy()
    df = fix_weight_sign(df)
    df = add_missing_flags(df, MISSING_FLAG_COLUMNS)

    if reference_medians is None:
        reference_medians = compute_reference_medians(df, MISSING_FLAG_COLUMNS)

    df = impute_with_reference(df, reference_medians)
    return df, reference_medians
