"""
Loads the trained model and produces:
  - validation_predictions.csv       (12,000 rows, load_id + predicted_rate)
  - data/december_chart_inputs.csv   (filled predicted_rate column, in place)

December note
--------------
The December chart inputs file only carries 7 columns: pickup, delivery,
distance, equipment, weight, date, predicted_rate. It has no lat/lon and
no market_index/quote_signal. That's handled by features.py already
(city-name coordinate lookup + median fallback for the two market
columns) — see the module docstring there for why. Practically, this
means the December curve is driven entirely by the calendar (day of
week, day of year / seasonality) since every other input is held fixed
for all 31 rows. That's the correct read: it's a seasonality curve for
one lane, not a market forecast.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features as feat
import preprocessing as prep

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def load_artifact():
    return joblib.load(MODEL_DIR / "model.pkl")


def predict_validation(artifact) -> pd.DataFrame:
    val_raw = pd.read_csv(DATA_DIR / "validation.csv")
    val_clean, _ = prep.clean_dataframe(val_raw, reference_medians=artifact["reference_medians"])
    X_val = feat.build_feature_matrix(
        val_clean,
        artifact["city_coords"],
        artifact["category_levels"],
        artifact["market_fallback"],
    )
    preds = np.expm1(artifact["model"].predict(X_val))
    preds = np.clip(preds, 1.0, None)  # guard against any non-positive edge case
    out = pd.DataFrame({"load_id": val_raw["load_id"], "predicted_rate": preds})
    return out


def predict_december(artifact) -> pd.DataFrame:
    dec_raw = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    dec_clean, _ = prep.clean_dataframe(dec_raw, reference_medians=artifact["reference_medians"])
    X_dec = feat.build_feature_matrix(
        dec_clean,
        artifact["city_coords"],
        artifact["category_levels"],
        artifact["market_fallback"],
    )
    preds = np.expm1(artifact["model"].predict(X_dec))
    preds = np.clip(preds, 1.0, None)
    out = dec_raw.copy()
    out["predicted_rate"] = preds
    return out


def main() -> None:
    artifact = load_artifact()

    val_preds = predict_validation(artifact)
    val_out_path = ROOT / "validation_predictions.csv"
    val_preds.to_csv(val_out_path, index=False)
    print(f"Wrote {len(val_preds):,} rows to {val_out_path}")
    print(val_preds["predicted_rate"].describe())

    dec_preds = predict_december(artifact)
    dec_out_path = DATA_DIR / "december_chart_inputs.csv"
    dec_preds.to_csv(dec_out_path, index=False)
    print(f"\nWrote filled December predictions to {dec_out_path}")
    print(dec_preds[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
