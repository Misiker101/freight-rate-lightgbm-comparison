"""
Trains the freight rate model on data/train_test.csv and writes:
  - models/model.pkl          the fitted LightGBM model + metadata
  - models/model_baseline.pkl a linear regression baseline (for the report)
  - report/validation_metrics.json
  - report/holdout_predictions.csv   (for residual plots in the report)

Validation strategy
--------------------
train_test.csv covers Jan-Oct 2025. The real task is to predict Nov-Dec
2025 (validation.csv) and then a fixed lane across all of December. That
is a forecasting problem, not an interpolation problem, so a random
k-fold split would be misleading here — it would let the model see
loads from late October right next to loads from early October and
report a validation score that has no relationship to how well it
extrapolates two months forward.

Instead we hold out the most recent slice of train_test.csv by date
(the last 2 months, September-October) as the internal test set, and
train on everything before that. This mirrors the actual deployment
gap between train_test.csv and validation.csv and gives a much more
honest read on generalization to unseen future dates.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import features as feat
import preprocessing as prep

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

HOLDOUT_START = "2025-09-01"  # last two months held out as the internal test set
TARGET = "posted_rate"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def time_split(df: pd.DataFrame):
    cutoff = pd.Timestamp(HOLDOUT_START)
    train = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()
    return train, holdout


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    raw = load_raw()
    train_raw, holdout_raw = time_split(raw)
    print(f"Train rows: {len(train_raw):,} (through {train_raw['date'].max().date()})")
    print(f"Holdout rows: {len(holdout_raw):,} ({holdout_raw['date'].min().date()} to {holdout_raw['date'].max().date()})")

    # --- cleaning: fit imputation on train only, reuse on holdout ---
    train_clean, ref_medians = prep.clean_dataframe(train_raw, reference_medians=None)
    holdout_clean, _ = prep.clean_dataframe(holdout_raw, reference_medians=ref_medians)

    # --- feature engineering setup, also fit on train only ---
    city_coords = feat.build_city_coords(train_clean)
    category_levels = {
        "pickup": sorted(train_clean["pickup"].unique()),
        "delivery": sorted(train_clean["delivery"].unique()),
        "equipment": sorted(train_clean["equipment"].unique()),
    }
    market_fallback = {
        "market_index": ref_medians["market_index"],
        "quote_signal": float(train_clean["quote_signal"].median()),
    }

    X_train = feat.build_feature_matrix(train_clean, city_coords, category_levels, market_fallback)
    X_holdout = feat.build_feature_matrix(holdout_clean, city_coords, category_levels, market_fallback)
    y_train = np.log1p(train_clean[TARGET].values)
    y_holdout = holdout_clean[TARGET].values

    # --- baseline: plain linear regression on one-hot features ---
    # Gives us a number to say "the gradient booster is meaningfully better
    # than a naive linear fit," not just a black box we trust blindly.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = ohe.fit_transform(X_train[feat.CATEGORICAL_COLUMNS].astype(str))
    cat_holdout = ohe.transform(X_holdout[feat.CATEGORICAL_COLUMNS].astype(str))
    num_cols = feat.NUMERIC_COLUMNS + feat.DATE_FEATURE_COLUMNS
    baseline_X_train = np.hstack([cat_train, X_train[num_cols].values])
    baseline_X_holdout = np.hstack([cat_holdout, X_holdout[num_cols].values])

    baseline = LinearRegression()
    baseline.fit(baseline_X_train, y_train)
    baseline_pred = np.expm1(baseline.predict(baseline_X_holdout))
    baseline_metrics = evaluate(y_holdout, baseline_pred)
    print("Baseline (linear regression) holdout metrics:", baseline_metrics)

    # --- main model: LightGBM gradient boosted trees ---
    # Chosen because the features are a mix of high-cardinality categoricals
    # (64 cities x2) and numeric/date features with likely non-linear and
    # interacting effects (e.g. rate-per-mile depends on equipment type
    # AND market_index together, not either alone). LightGBM handles both
    # natively without manual interaction terms or one-hot blowup.
    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=feat.CATEGORICAL_COLUMNS, free_raw_data=False
    )
    holdout_set = lgb.Dataset(
        X_holdout, label=np.log1p(y_holdout), categorical_feature=feat.CATEGORICAL_COLUMNS,
        reference=train_set, free_raw_data=False,
    )

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[holdout_set],
        valid_names=["holdout"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=0),
            lgb.record_evaluation(lgb_eval_history := {}),
        ],
    )

    lgb_pred = np.expm1(model.predict(X_holdout, num_iteration=model.best_iteration))
    lgb_metrics = evaluate(y_holdout, lgb_pred)
    print("LightGBM holdout metrics:", lgb_metrics)
    print("Best iteration:", model.best_iteration)

    # feature importance for the report / loom walkthrough
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))

    # --- refit on the FULL train_test.csv before shipping the model ---
    # The holdout above exists purely to pick hyperparameters/iterations and
    # report an honest generalization number. Once we trust that number, we
    # refit on all labeled data (train + holdout) so the model that actually
    # produces validation_predictions.csv has seen as much history as
    # possible, right up to the edge of the forecast gap.
    full_clean, full_medians = prep.clean_dataframe(raw, reference_medians=None)
    full_city_coords = feat.build_city_coords(full_clean)
    full_category_levels = {
        "pickup": sorted(full_clean["pickup"].unique()),
        "delivery": sorted(full_clean["delivery"].unique()),
        "equipment": sorted(full_clean["equipment"].unique()),
    }
    full_market_fallback = {
        "market_index": full_medians["market_index"],
        "quote_signal": float(full_clean["quote_signal"].median()),
    }
    X_full = feat.build_feature_matrix(full_clean, full_city_coords, full_category_levels, full_market_fallback)
    y_full = np.log1p(full_clean[TARGET].values)
    full_set = lgb.Dataset(X_full, label=y_full, categorical_feature=feat.CATEGORICAL_COLUMNS, free_raw_data=False)

    final_model = lgb.train(
        params,
        full_set,
        num_boost_round=model.best_iteration,
    )

    artifact = {
        "model": final_model,
        "reference_medians": full_medians,
        "city_coords": full_city_coords,
        "category_levels": full_category_levels,
        "market_fallback": full_market_fallback,
        "feature_columns": feat.FEATURE_COLUMNS,
        "holdout_metrics": lgb_metrics,
        "baseline_metrics": baseline_metrics,
        "best_iteration": model.best_iteration,
    }
    joblib.dump(artifact, MODEL_DIR / "model.pkl")

    metrics_out = {
        "holdout_period": [str(holdout_raw["date"].min().date()), str(holdout_raw["date"].max().date())],
        "n_train": int(len(train_raw)),
        "n_holdout": int(len(holdout_raw)),
        "baseline_linear_regression": baseline_metrics,
        "lightgbm": lgb_metrics,
        "best_iteration": int(model.best_iteration),
    }
    with open(REPORT_DIR / "validation_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    holdout_out = holdout_raw[["load_id", "date", "equipment", "distance"]].copy()
    holdout_out["actual_rate"] = y_holdout
    holdout_out["predicted_rate"] = lgb_pred
    holdout_out["abs_error"] = np.abs(holdout_out["actual_rate"] - holdout_out["predicted_rate"])
    holdout_out.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    print("\nSaved model.pkl, validation_metrics.json, holdout_predictions.csv, feature_importance.csv")

    # per-round training curve (MAE on log1p(target), the objective's own scale)
    # saved so it can be plotted alongside HGBR's curve in compare_models.py
    # without needing to retrain LightGBM again.
    curve = pd.DataFrame({
        "iteration": range(1, len(lgb_eval_history["holdout"]["l1"]) + 1),
        "holdout_mae_log": lgb_eval_history["holdout"]["l1"],
    })
    curve.to_csv(REPORT_DIR / "lightgbm_training_curve.csv", index=False)
    print(f"Saved lightgbm_training_curve.csv ({len(curve)} rounds)")


if __name__ == "__main__":
    main()
