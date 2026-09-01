# Freight Rate Prediction

Solution for the Spotter Machine Learning Engineer take-home. Predicts `posted_rate`
for freight loads using a LightGBM model trained on `data/train_test.csv`, then scores
every load in `data/validation.csv` and a fixed-lane December 2025 forecast.

Full writeup (EDA findings, data-quality issues, validation approach, results) is in
`report/Freight_Rate_Assessment_Report.docx`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Tested on Python 3.11+.

## Run

```bash
cd src
python train.py            # LightGBM: cleans data, time-based split, trains, evaluates,
                            # refits on full data, saves models/model.pkl,
                            # saves report/lightgbm_training_curve.csv
python predict.py          # writes validation_predictions.csv, fills data/december_chart_inputs.csv
cd ..
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

### HGBR comparison (optional)

Same cleaning (`preprocessing.py`) and same feature engineering (`features.py`) as the
LightGBM run above, same time-based Sept–Oct holdout, only the model swapped for
`sklearn.ensemble.HistGradientBoostingRegressor`:

```bash
cd src
python train_hgbr.py       
python compare_models.py   
python predict_hgbr.py     
cd ..
python score.py --predictions validation_predictions_hgbr.csv \
  --december-predictions data/december_chart_inputs_hgbr.csv \
  --output-dir scorer_results_hgbr
```

Run `train.py` before `train_hgbr.py`/`compare_models.py` — `train_hgbr.py` imports
`load_raw`, `time_split`, and `evaluate` straight from `train.py` so both models are
scored with exactly the same logic, and `compare_models.py` reads both training-curve
CSVs from disk rather than retraining anything.

`score.py` validates both output files and writes `scorer_results/candidate_december.png`
(or `scorer_results_hgbr/candidate_december.png`), which is embedded in the report.

## Layout

```
data/            train_test.csv, validation.csv, december_chart_inputs.csv (filled by predict.py)
models/          model.pkl (LightGBM) and model_hgbr.pkl (HGBR) — each bundles the fitted
                 model with its fitted imputation medians, city coordinate lookup, and
                 category levels, so predict.py / predict_hgbr.py don't need to touch
                 train_test.csv again
report/          validation_metrics.json, hgbr_metrics.json, model_comparison_metrics.json,
                 feature_importance.csv, holdout_predictions.csv,
                 lightgbm_training_curve.csv, hgbr_training_curve.csv,
                 diagnostic charts + model_comparison_training_curve.png,
                 and the final .docx report
src/
  preprocessing.py    weight sign fix, missing-value imputation, missing-value flags
  features.py          calendar features, city coordinate lookup, feature matrix assembly
  train.py              LightGBM: time-based split, trains + evaluates + refits
  predict.py             LightGBM inference: validation_predictions.csv + December chart
  train_hgbr.py          HGBR: same split/eval logic as train.py, model swapped
  predict_hgbr.py         HGBR inference: validation_predictions_hgbr.csv + December chart
  compare_models.py       loads both training curves + metrics, plots comparison, no retraining
score.py           provided scorer (unmodified)
requirements.txt
```

## Notes on the two output files

- **validation_predictions.csv**: `load_id,predicted_rate` for all 12,000 rows in
  `data/validation.csv`, in the format the scorer expects.
- **data/december_chart_inputs.csv**: same 7 columns as provided, with `predicted_rate`
  filled in. That file doesn't carry `market_index`, `quote_signal`, or lat/lon, so
  `predict.py` backfills city coordinates from a lookup built off the training data and
  holds the two market signals at their training-set median — see section 3 of the report
  for why that's the right call here.
