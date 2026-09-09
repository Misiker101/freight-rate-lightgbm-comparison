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


def plot_curves(lgb_curve: pd.DataFrame, hgbr_curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=170)

    ax = axes[0]
    ax.plot(lgb_curve["iteration"], lgb_curve["holdout_mae_log"], color=LGB_COLOR, linewidth=1.8, label="LightGBM")
    ax.plot(hgbr_curve["iteration"], hgbr_curve["holdout_mae_log"], color=HGBR_COLOR, linewidth=1.8, label="HGBR")
    lgb_best = lgb_curve.loc[lgb_curve["holdout_mae_log"].idxmin()]
    hgbr_best = hgbr_curve.loc[hgbr_curve["holdout_mae_log"].idxmin()]
    ax.scatter([lgb_best["iteration"]], [lgb_best["holdout_mae_log"]], color=LGB_COLOR, zorder=5, s=35)
    ax.scatter([hgbr_best["iteration"]], [hgbr_best["holdout_mae_log"]], color=HGBR_COLOR, zorder=5, s=35)
    ax.set_xlim(0, 300)  # zoomed to where the two curves actually separate visually
    ax.set_ylim(0.038, 0.10)
    ax.set_xlabel("Boosting round (zoomed to first 300)")
    ax.set_ylabel("Holdout MAE (log1p target)")
    ax.set_title("Holdout error vs. boosting round", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(lgb_curve["iteration"], lgb_curve["holdout_mae_log"], color=LGB_COLOR, linewidth=1.8, label="LightGBM \u2014 holdout")
    if "train_mae_log" in lgb_curve.columns:
        ax.plot(lgb_curve["iteration"], lgb_curve["train_mae_log"], color=LGB_COLOR, linewidth=1.2, linestyle="--", label="LightGBM \u2014 train")
    ax.plot(hgbr_curve["iteration"], hgbr_curve["holdout_mae_log"], color=HGBR_COLOR, linewidth=1.8, label="HGBR \u2014 holdout")
    ax.plot(hgbr_curve["iteration"], hgbr_curve["train_mae_log"], color=HGBR_COLOR, linewidth=1.2, linestyle="--", label="HGBR \u2014 train")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("MAE (log1p target)")
    ax.set_title("Train vs. holdout (overfit check)", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = REPORT_DIR / "model_comparison_training_curve.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    lgb_curve, hgbr_curve = load_curves()
    lgb_metrics, hgbr_metrics = load_metrics()
    plot_curves(lgb_curve, hgbr_curve)
    print_and_save_metrics_table(lgb_metrics, hgbr_metrics)


if __name__ == "__main__":
    main()
