# Models checkpoint — through item 4.6

Generated from `backend/results.json` (run 2026-10-23, after item 4.6). Regenerate
from the repo root with:

```bash
python backend/evaluate.py
```

This runs the full day-ahead backtest (four expanding folds) and overwrites
`backend/results.json`. It reads the raw CSVs and the committed weather cache; it
does not touch the network. The full run takes about 5 minutes on this machine,
most of it Optuna tuning, SARIMA refits, and quantile fits.

## Protocol recap

- Forecast origin is 00:00 IST of each test day; the target is mean Wh/h per
  reporting stable apartment.
- All models are scored on the same evaluation mask: target present and
  seasonal-naive-24 available.
- Baselines: persistence, seasonal naive 24 h, same hour last week, mean of the
  same hour on the previous four same weekdays.
- Learned models: one pooled XGBoost per fold, trained only on that fold's training
  window labels; one rolling SARIMA per fold, refit weekly from pre-origin actuals;
  and three quantile XGBoost models (P10/P50/P90) on the same information set.
  None reads test-window labels.

## Feature set (XGBoost)

| Group | Features |
|---|---|
| Load lags | `lag_24` … `lag_47`, `lag_168`, `lag_336` |
| Lag-block aggregates | mean/std/min/max of the 24–47 h block |
| Calendar | hour, day of week, weekend, month, day-of-year sin/cos |
| Holiday | hand-curated major national/festival dates in the data period |
| Semester | Jan–Apr and Jul–Nov in session; May–Jun and Dec breaks |
| Reporting | `n_reporting_lag_24`, mean of the 24–47 h reporting counts |
| Weather | six lag-24 Mumbai weather values, 24–47 h mean temperature, 24–47 h precipitation sum |

Every load, reporting, and weather feature is at least 24 h old at the target hour.
Calendar/holiday/semester flags are known in advance. Target-hour weather is never
used because it would not be known at the forecast origin. Training labels are
clipped at a train-only Q3 + 3×IQR cap; predictions are clipped at zero.

## Tuning and early stopping

- Optuna 5.0.0, TPE sampler with seed 42, 30 trials per fold.
- Each trial is scored on three expanding-window CV splits inside the fold's
  training window. Validation blocks are contiguous, non-overlapping, and end at
  the last training day; training for a split ends one hour before its validation
  block. Validation blocks are 14 days, shortened for the 61-day training window.
- Early stopping uses patience 25 on those train-side validation slices. The
  number of trees for the final model is the median best iteration across the
  three slices, and the model is then refit on all training labels with that tree
  count.
- The test folds are never read during tuning. Hyperparameters are therefore
  allowed to differ by fold.

| Fold | CV MAE (Wh) | Best iterations | Median trees | Depth | Learning rate | Min child weight | Subsample | Colsample | λ | α |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017-07 | 82.33 | 30 / 24 / 30 | 30 | 2 | 0.0989 | 3.14 | 0.784 | 0.975 | 1.66 | 0.082 |
| 2017-08 | 63.06 | 170 / 320 / 649 | 320 | 3 | 0.0133 | 6.68 | 0.767 | 0.774 | 6.03 | 0.803 |
| 2017-09_12 | 58.41 | 284 / 253 / 244 | 253 | 5 | 0.0111 | 15.21 | 0.726 | 0.513 | 2.86 | 0.170 |
| 2018-05_06 | 139.55 | 132 / 85 / 68 | 85 | 3 | 0.0331 | 6.46 | 0.999 | 0.742 | 0.38 | 0.846 |

## SARIMA configuration

- Order (1,0,1)(0,1,1,24) on the gap-handled target, fitted by
  `statsmodels` SARIMAX with stationarity and invertibility constraints disabled.
- Refit on Mondays, on the first test day, and after a coverage gap, using the
  trailing 60 days (1,440 h) of contiguous history ending one hour before the
  origin. A refit needs at least 7 contiguous days.
- Between refits the fitted state is advanced with actuals strictly before each
  day's 00:00 IST origin. No value from the forecast day is read.
- A forecast is skipped when the newest state is more than 48 h old; negative
  predictions are clipped to zero.
- Coverage is partial by design: 744 / 744 / 1,554 / 38 evaluable hours in the
  four folds. The rest are hours where a gap left no sufficiently fresh,
  sufficiently long history.

## Quantile configuration

- Three independent XGBoost regressors with `objective=reg:quantileerror` at
  quantiles 0.1, 0.5, and 0.9.
- Same features, 24 h+ lags, training window, and train-only Q3 + 3×IQR cap as the
  point XGBoost model; fixed untuned 4.1 defaults (400 trees, depth 4, learning
  rate 0.03).
- Predictions are clipped at zero, then sorted row-wise so the quantiles cannot
  cross. Raw crossings were 0 / 0 / 0 / 1 rows in the four folds.
- Interval quality is measured by P10–P90 coverage (nominal 80%), mean width, and
  pinball loss at each quantile. P50 is also scored as a point forecast.

## Fold results (evaluation mask)

| Fold | Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | Best baseline (MAE) | Median abs-error diff | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 2017-07 | xgboost | 744 | 69.31 | 90.53 | 19.02 | 0.819 | 5.14 | 0.4150 | seasonal naive (75.01) | −4.00 | 0.0215 |
| 2017-07 | sarima | 744 | 65.66 | 87.70 | 18.02 | 0.776 | 7.89 | 0.4510 | seasonal naive (75.01) | −6.61 | 0.00012 |
| 2017-07 | xgboost_quantile (P50) | 744 | 65.99 | 87.83 | 18.11 | 0.780 | −1.48 | 0.4493 | seasonal naive (75.01) | −7.25 | 6.4e-05 |
| 2017-08 | xgboost | 744 | 59.51 | 80.87 | 16.30 | 0.732 | −12.87 | 0.4718 | mean of 4 weekdays (58.93) | −0.15 | 0.844 |
| 2017-08 | sarima | 744 | 60.60 | 81.11 | 16.60 | 0.745 | −6.53 | 0.4686 | mean of 4 weekdays (58.93) | +0.42 | 0.830 |
| 2017-08 | xgboost_quantile (P50) | 744 | 60.58 | 82.06 | 16.60 | 0.745 | −15.23 | 0.4561 | mean of 4 weekdays (58.93) | 0.00 | 0.607 |
| 2017-09_12 | xgboost | 2,243 | 167.27 | 212.55 | 69.73 | 2.089 | 111.15 | −0.1568 | persistence (116.21) | +12.37 | 2.3e-22 |
| 2017-09_12 | sarima | 1,554 | 104.23 | 146.99 | 44.17 | 1.302 | 8.14 | 0.4764 | persistence (116.21) | −0.43 | 0.0029 |
| 2017-09_12 | xgboost_quantile (P50) | 2,243 | 159.14 | 202.73 | 66.34 | 1.988 | 93.07 | −0.0523 | persistence (116.21) | +10.92 | 2.5e-19 |
| 2018-05_06 | xgboost | 182 | 120.75 | 159.57 | 25.74 | 1.176 | −87.45 | 0.1761 | seasonal naive (132.87) | +0.28 | 0.518 |
| 2018-05_06 | sarima | 38 | 93.80 | 127.08 | 17.53 | 0.914 | −35.21 | 0.7094 | seasonal naive (132.87) | −9.36 | 0.025 |
| 2018-05_06 | xgboost_quantile (P50) | 182 | 113.94 | 150.13 | 24.29 | 1.110 | −78.61 | 0.2706 | seasonal naive (132.87) | −3.52 | 0.309 |

Negative median abs-error diff means the model has the lower median absolute error.
Only the July fold shows a statistically significant XGBoost improvement over the
best baseline; August is a statistical tie, and Sep–Dec is significantly worse.
SARIMA is statistically better in July and on its overlapping Sep–Dec hours, but
the Sep–Dec median difference is only −0.43 Wh, and its 2018 p-value rests on 38
paired hours. SARIMA has fewer evaluable hours than XGBoost in the last two folds,
so their MAEs are not computed over identical hours. The quantile P50 is
significantly better than the best baseline only in July.

## Interval quality (quantile XGBoost)

| Fold | Coverage (nominal 80%) | Mean width (Wh) | Pinball P10 | Pinball P50 | Pinball P90 | Mean pinball |
|---|---:|---:|---:|---:|---:|---:|
| 2017-07 | 71.6% | 177.44 | 17.14 | 33.00 | 15.05 | 21.73 |
| 2017-08 | 78.5% | 176.75 | 13.58 | 30.29 | 15.73 | 19.87 |
| 2017-09_12 | 44.8% | 210.76 | 75.68 | 79.57 | 30.39 | 61.88 |
| 2018-05_06 | 69.2% | 229.05 | 19.72 | 56.97 | 29.07 | 35.25 |

Coverage is measured on the same evaluation mask and hours as the P50 point
metrics. The intervals are too narrow in every fold and badly undercovered in the
autumn shift, where the model trained on May–Aug never saw the December vacancy
regime. The pinball loss reflects that miss; no test-window recalibration was
applied.

## What changed across model iterations

| Fold | Fixed XGBoost (4.1) | + calendar/reporting (4.2) | + weather (4.3) | Tuned XGBoost (4.4) | SARIMA (4.5) |
|---|---:|---:|---:|---:|---:|
| 2017-07 | 65.95 | 66.46 | 65.69 | 69.31 | 65.66 |
| 2017-08 | 62.35 | 61.55 | 62.24 | 59.51 | 60.60 |
| 2017-09_12 | 151.01 | 163.22 | 170.40 | 167.27 | 104.23 |
| 2018-05_06 | 142.59 | 100.40 | 106.47 | 120.75 | 93.80 |

These are MAE in Wh. SARIMA's Sep–Dec and May–Jun values cover 1,554 and 38
evaluable hours respectively, versus 2,243 and 182 for XGBoost. The fixed configuration used 400 trees at depth 4 with
learning rate 0.03; the tuned configuration uses per-fold Optuna parameters. The
unseen-autumn fold degrades as features are added, while the year-later 2018 fold
improves sharply with calendar features and then gives some of that back under
tuning. The feature set was not edited in response to test results.

## Leakage evidence

`python -m pytest backend/tests -q` → 17 passed. Tests cover:

- every feature lag is ≥24 h;
- feature frames for a test day are unchanged when every value from that day's
  origin onward is replaced with 1e9 (load, reporting, and weather);
- all four baselines, XGBoost, and SARIMA predictions are unchanged under the same
  mutation;
- SARIMA contiguous history stops at the last gap, rejects runs under 7 days, and
  emits nothing from a state older than 48 h;
- quantile predictions are unchanged under the same mutation, stay ordered, and a
  hand-checked coverage/pinball example matches the formulas;
- a tuned XGBoost forecast is unchanged when test-window values are replaced;
- time-series CV splits stay inside the training window and never overlap;
- holiday/weekend/semester flags land on expected dates.

The baseline pair test from 3.3 and the schema checks remain valid.

## Limitations

- The Sep–Dec fold is a distribution shift: the model trains on May–Aug and is
  tested on autumn/winter, including the December vacancy period. Its +111 Wh bias
  means it over-predicts; the seasonal-naive and persistence baselines handle that
  regime better with far less capacity. This is the biggest model risk.
- The 2018 fold has only 182 evaluable hours across 8 days; its MAE and p-value are
  indicative, not precise.
- Weather is represented by 24 h+ lagged observations, not day-ahead weather
  forecasts. It is a weak proxy and may not add value consistently; Open-Meteo
  archive data is cached in `backend/weather_mumbai.csv` and committed.
- Holiday and semester flags are hand-curated assumptions, not an authoritative
  calendar for the campus.
- The target is a mean over reporting stable apartments, not total campus load, and
  the 24-apartment stable panel changes composition at the edges of the coverage gap.
- Optuna selects hyperparameters on short, in-window CV folds. Those folds do not
  contain an autumn/winter regime for the Sep–Dec test, so CV cannot select for it.
- SARIMA is unavailable for 689 of the 2,243 evaluable Sep–Dec hours and for 144
  of the 182 evaluable 2018 hours because target gaps leave no fresh contiguous
  history. Its 2018 metrics are therefore a 38-hour sample, and the Sep–Dec
  median improvement over persistence is only −0.43 Wh despite a lower mean MAE.
- The quantile intervals are undercovered: 44.8% versus the nominal 80% in the
  autumn fold and 69–78% elsewhere. They are useful as a relative uncertainty
  signal, not as calibrated prediction intervals.
- The quantile models use fixed untuned hyperparameters, so their P50 is not
  directly comparable in tuning effort to the Optuna-tuned point XGBoost; the
  comparison is still leakage-safe because both see the same information set.
