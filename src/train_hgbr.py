"""
Same pipeline as train.py — same cleaning (preprocessing.py), same feature
engineering (features.py), same time-based train/holdout split, same
evaluate() function — with HistGradientBoostingRegressor (HGBR) swapped in
for LightGBM. This exists purely to answer one question: does a different
gradient boosting implementation change the result on this data, or was
LightGBM's edge over the linear baseline (see train.py) really about the
model family and not implementation details.

Why a manual warm-start loop instead of HGBR's built-in early_stopping
-----------------------------------------------------------------------
HGBR has its own early_stopping option, but it works by randomly carving a
validation_fraction out of whatever X you hand it — that would validate on
a random slice of Jan-Aug data, not on the Sep-Oct holdout. That's exactly
the random-split problem train.py's docstring already argues against for
this forecasting task. So early_stopping is turned off here, and instead
the same explicit Sep-Oct holdout is checked every `check_every` rounds
via warm_start (each fit() call just adds more trees, it doesn't restart),
mirroring what LightGBM's early_stopping(stopping_rounds=75) does, just
implemented by hand.

Outputs (all in report/, none of this touches train.py's files):
  - report/hgbr_training_curve.csv      round-by-round train/holdout MAE
  - report/hgbr_metrics.json             final holdout metrics
  - models/model_hgbr.pkl                fitted HGBR model + same lookups
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features as feat
import preprocessing as prep
from train import DATA_DIR, HOLDOUT_START, MODEL_DIR, REPORT_DIR, TARGET, evaluate, load_raw, time_split

CHECK_EVERY = 20      # add this many trees between checkpoints
PATIENCE_CHECKS = 4   # stop if holdout MAE hasn't improved in this many checkpoints (~80 rounds, close to LightGBM's 75)
MAX_ROUNDS = 1500


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # --- identical data prep to train.py ---
    raw = load_raw()
    train_raw, holdout_raw = time_split(raw)
    print(f"Train rows: {len(train_raw):,} (through {train_raw['date'].max().date()})")
    print(f"Holdout rows: {len(holdout_raw):,} ({holdout_raw['date'].min().date()} to {holdout_raw['date'].max().date()})")

    train_clean, ref_medians = prep.clean_dataframe(train_raw, reference_medians=None)
    holdout_clean, _ = prep.clean_dataframe(holdout_raw, reference_medians=ref_medians)

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
    y_train_log = np.log1p(train_clean[TARGET].values)
    y_holdout_log = np.log1p(holdout_clean[TARGET].values)
    y_holdout_dollars = holdout_clean[TARGET].values

    # --- HGBR setup ---
    # Parameters chosen to mirror train.py's LightGBM params as closely as
    # the two APIs allow: same learning_rate, same L1/MAE objective, same
    # leaf-count cap, same min-samples-per-leaf, same L2 term, same seed.
    #   LightGBM num_leaves=63        -> HGBR max_leaf_nodes=63
    #   LightGBM min_data_in_leaf=30  -> HGBR min_samples_leaf=30
    #   LightGBM lambda_l2=1.0        -> HGBR l2_regularization=1.0
    #   LightGBM objective="regression_l1" -> HGBR loss="absolute_error"
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_iter=CHECK_EVERY,
        warm_start=True,
        early_stopping=False,   # we drive stopping ourselves against the time-based holdout
        categorical_features="from_dtype",  # uses the pandas category dtype set in features.py
        random_state=42,
    )

    best_holdout_mae = np.inf
    best_iter = 0
    no_improve_checks = 0
    rounds, train_curve, holdout_curve = [], [], []

    print("\nTraining HGBR with manual warm-start early stopping against the Sep-Oct holdout...")
    for target_iter in range(CHECK_EVERY, MAX_ROUNDS + 1, CHECK_EVERY):
        model.max_iter = target_iter
        model.fit(X_train, y_train_log)  # warm_start=True: continues from current trees, doesn't restart

        train_pred_log = model.predict(X_train)
        holdout_pred_log = model.predict(X_holdout)
        train_mae_log = float(np.mean(np.abs(y_train_log - train_pred_log)))
        holdout_mae_log = float(np.mean(np.abs(y_holdout_log - holdout_pred_log)))

        rounds.append(target_iter)
        train_curve.append(train_mae_log)
        holdout_curve.append(holdout_mae_log)

        improved = holdout_mae_log < best_holdout_mae - 1e-5
        if improved:
            best_holdout_mae = holdout_mae_log
            best_iter = target_iter
            no_improve_checks = 0
        else:
            no_improve_checks += 1

        print(f"  iter {target_iter:4d}  train_mae(log)={train_mae_log:.5f}  holdout_mae(log)={holdout_mae_log:.5f}"
              f"{'  <- best' if improved else ''}")

        if no_improve_checks >= PATIENCE_CHECKS:
            print(f"  no improvement in {PATIENCE_CHECKS} checkpoints, stopping at iter {target_iter} "
                  f"(best was {best_iter})")
            break

    # refit fresh at exactly best_iter so the shipped model matches the
    # checkpoint that scored best on the holdout, not whatever iteration
    # the patience loop happened to stop at
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_iter=best_iter,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=42,
    )
    model.fit(X_train, y_train_log)

    holdout_pred_dollars = np.expm1(model.predict(X_holdout))
    hgbr_metrics = evaluate(y_holdout_dollars, holdout_pred_dollars)
    print(f"\nBest iteration: {best_iter}")
    print("HGBR holdout metrics (dollar space):", hgbr_metrics)

    # --- save training curve + metrics, same shape as train.py's outputs ---
    curve_df = pd.DataFrame({"iteration": rounds, "train_mae_log": train_curve, "holdout_mae_log": holdout_curve})
    curve_df.to_csv(REPORT_DIR / "hgbr_training_curve.csv", index=False)

    metrics_out = {
        "holdout_period": [str(holdout_raw["date"].min().date()), str(holdout_raw["date"].max().date())],
        "n_train": int(len(train_raw)),
        "n_holdout": int(len(holdout_raw)),
        "hgbr": hgbr_metrics,
        "best_iteration": int(best_iter),
    }
    with open(REPORT_DIR / "hgbr_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # --- refit on full labeled data before shipping, same as train.py does for LightGBM ---
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
    y_full_log = np.log1p(full_clean[TARGET].values)

    final_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_iter=best_iter,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=42,
    )
    final_model.fit(X_full, y_full_log)

    artifact = {
        "model": final_model,
        "reference_medians": full_medians,
        "city_coords": full_city_coords,
        "category_levels": full_category_levels,
        "market_fallback": full_market_fallback,
        "feature_columns": feat.FEATURE_COLUMNS,
        "holdout_metrics": hgbr_metrics,
        "best_iteration": int(best_iter),
    }
    joblib.dump(artifact, MODEL_DIR / "model_hgbr.pkl")
    print("\nSaved model_hgbr.pkl, hgbr_metrics.json, hgbr_training_curve.csv")


if __name__ == "__main__":
    main()
