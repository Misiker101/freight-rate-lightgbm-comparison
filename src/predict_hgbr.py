"""
Same idea as predict.py, pointed at models/model_hgbr.pkl instead of
models/model.pkl. Outputs are named separately so they never overwrite
the LightGBM outputs:
  - validation_predictions_hgbr.csv
  - data/december_chart_inputs_hgbr.csv

See predict.py's docstring for why the December file needs the city
coordinate lookup and market_index/quote_signal fallback — same logic,
same features.py functions, applies here unchanged.
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
    return joblib.load(MODEL_DIR / "model_hgbr.pkl")


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
    preds = np.clip(preds, 1.0, None)
    out = pd.DataFrame({"load_id": val_raw["load_id"], "predicted_rate": preds})
    return out


def predict_december(artifact) -> pd.DataFrame:
    # always start from the original (unfilled) december file so this
    # never accidentally reads back in an already-filled version
    dec_raw = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    dec_raw = dec_raw.drop(columns=["predicted_rate"]).assign(predicted_rate=np.nan)
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
    val_out_path = ROOT / "validation_predictions_hgbr.csv"
    val_preds.to_csv(val_out_path, index=False)
    print(f"Wrote {len(val_preds):,} rows to {val_out_path}")
    print(val_preds["predicted_rate"].describe())

    dec_preds = predict_december(artifact)
    dec_out_path = DATA_DIR / "december_chart_inputs_hgbr.csv"
    dec_preds.to_csv(dec_out_path, index=False)
    print(f"\nWrote HGBR December predictions to {dec_out_path}")
    print(dec_preds[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
