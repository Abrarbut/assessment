# Freight Rate Prediction — Solution

Predicts freight load rates using a HistGradientBoostingRegressor trained on
`data/train-test.csv`, applied to `data/validation.csv` and
`data/december-chart-inputs.csv`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run everything

```bash
cd src
python train.py                 # trains model, prints holdout metrics, saves model/
python predict_validation.py    # writes ../validation_predictions.csv
python predict_december.py      # writes ../december_predictions.csv
cd ..
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer writes `scorer_results/candidate_december.png`.

## Approach summary

**Data cleaning**
- `weight` had 292 sign-flipped negative values (magnitudes matched the
  valid range) — corrected with `abs()` rather than dropped.
- Missing `weight` (300 rows) and `market_index` (374 rows) — median-imputed.
- City coordinates are 1:1 with city name across the dataset, so a
  `city -> (lat, lon)` lookup built from `train-test.csv` is reused to add
  coordinates to `december-chart-inputs.csv`, which doesn't include them.
- `december-chart-inputs.csv` also lacks `market_index`/`quote_signal`
  entirely; both showed weak correlation with rate (~0.03–0.08) in EDA, so
  they're filled with training medians with minimal impact on predictions.

**Validation split**
Time-based, not random: train on Jan–Aug 2025, hold out Sep–Oct 2025. The
loads we ultimately need to price (Nov–Dec 2025) are *after* the training
window, so validating on the most recent months mimics that forward-looking
task far better than a shuffled split would.

**Model**
HistGradientBoostingRegressor (scikit-learn), compared against a Ridge
regression baseline. On the Sep–Oct holdout:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Ridge baseline | $199.04 | $656.36 | 10.94% |
| HistGBM (final) | $131.30 | $638.43 | 5.67% |

The final model is refit on all labeled data before predicting
`validation.csv` / `december-chart-inputs.csv`.

**Features**: distance, weight, pickup/delivery lat-lon, market_index,
quote_signal, equipment (one-hot), day-of-week, month, and sin/cos of
day-of-year (lets the model treat December as seasonally close to
January/February despite having no direct Nov/Dec training examples).

## Repo layout

```
src/features.py            shared feature engineering (train + both predict scripts)
src/train.py                trains + evaluates + saves model/
src/predict_validation.py   scores data/validation.csv -> validation_predictions.csv
src/predict_december.py     scores data/december-chart-inputs.csv -> december_predictions.csv
model/                       saved model, city lookup, imputation values
data/                        provided assessment data
score.py                     provided scorer
scorer_results/               scorer output (chart)
```
