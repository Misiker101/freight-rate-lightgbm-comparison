"""
Loads the outputs of train.py (LightGBM) and train_hgbr.py (HGBR) —
does not retrain anything — and produces:
  - report/model_comparison_training_curve.png
  - a metrics comparison printed to stdout and saved to
    report/model_comparison_metrics.json

Run train.py and train_hgbr.py first.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"

LGB_COLOR = "#064A56"
HGBR_COLOR = "#C0392B"


def load_curves():
    lgb_curve = pd.read_csv(REPORT_DIR / "lightgbm_training_curve.csv")
    hgbr_curve = pd.read_csv(REPORT_DIR / "hgbr_training_curve.csv")
    return lgb_curve, hgbr_curve


def load_metrics():
    with open(REPORT_DIR / "validation_metrics.json") as f:
        lgb_metrics = json.load(f)
    with open(REPORT_DIR / "hgbr_metrics.json") as f:
        hgbr_metrics = json.load(f)
    return lgb_metrics, hgbr_metrics



def print_and_save_metrics_table(lgb_metrics: dict, hgbr_metrics: dict) -> None:
    lgb_m = lgb_metrics["lightgbm"]
    hgbr_m = hgbr_metrics["hgbr"]

    rows = [
        ("MAE ($)", lgb_m["mae"], hgbr_m["mae"]),
        ("RMSE ($)", lgb_m["rmse"], hgbr_m["rmse"]),
        ("MAPE", lgb_m["mape"], hgbr_m["mape"]),
        ("R\u00b2", lgb_m["r2"], hgbr_m["r2"]),
        ("Best iteration", lgb_metrics["best_iteration"], hgbr_metrics["best_iteration"]),
    ]

    print(f"\n{'Metric':<16}{'LightGBM':>14}{'HGBR':>14}")
    print("-" * 44)
    for label, lgb_val, hgbr_val in rows:
        if isinstance(lgb_val, float):
            print(f"{label:<16}{lgb_val:>14.4f}{hgbr_val:>14.4f}")
        else:
            print(f"{label:<16}{lgb_val:>14}{hgbr_val:>14}")

    out = {
        "lightgbm": {"mae": lgb_m["mae"], "rmse": lgb_m["rmse"], "mape": lgb_m["mape"], "r2": lgb_m["r2"], "best_iteration": lgb_metrics["best_iteration"]},
        "hgbr": {"mae": hgbr_m["mae"], "rmse": hgbr_m["rmse"], "mape": hgbr_m["mape"], "r2": hgbr_m["r2"], "best_iteration": hgbr_metrics["best_iteration"]},
    }
    with open(REPORT_DIR / "model_comparison_metrics.json", "w") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    lgb_curve, hgbr_curve = load_curves()
    lgb_metrics, hgbr_metrics = load_metrics()
    plot_curves(lgb_curve, hgbr_curve)
    print_and_save_metrics_table(lgb_metrics, hgbr_metrics)


if __name__ == "__main__":
    main()
