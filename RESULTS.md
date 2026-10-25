# ElectriCast — Backtest Results

This document reports the five-fold day-ahead backtest plus one stress test stored in
`backend/results.json`. All numbers below were read from that artifact (generated
2026-10-23T23:52:19+05:30) and can be regenerated with:

```bash
python backend/evaluate.py
```

No number in this document is hardcoded in the UI; the dashboard reads the same artifact
through the FastAPI backend.

## Protocol

- **Timezone:** Asia/Kolkata (IST). Every forecast covers 24 IST clock hours.
- **Target:** mean hourly load per reporting stable apartment, in Wh (`target_wh`).
- **Forecast origin:** 00:00 IST of each test day.
- **Information set:** target observations strictly before the origin. Lag baselines use lags of
  at least 24 h; persistence uses only the last observation before the origin; SARIMA refits
  weekly on a trailing window and advances its state with actuals up to each origin. Weather and
  reporting counts are lagged at least 24 h; calendar, holiday and semester flags are known in
  advance.
- **Folds:** expanding windows with `train_start` fixed at the earliest valid target hour,
  2016-12-14 21:00 IST. Five scoring folds test June, July, August, September–November and
  December 2017. May–June 2018 is a stress test and is excluded from every fold mean.
- **Evaluation mask:** hours where the target and the seasonal-naive-24 forecast both exist,
  so every model is scored on the same hours within a fold (subject to its own coverage).
- **Metrics:** MAE, RMSE, WAPE (%), MASE, bias, R², sMAPE (%). MASE scale is the mean absolute
  24-h seasonal difference over valid training pairs in the fold.
- **Best baseline:** lowest MASE on the all-four-baseline mask when that mask covers at least
  max(24 h, 50%) of the evaluation hours; otherwise on the evaluation mask. The selected
  baseline can therefore differ from the one that would win on the evaluation mask alone.
- **Common hours:** the evaluation mask intersected with the forecasts of all four baselines
  and all three learned models. Every row in a common-hours table is scored on exactly the same
  hours; a fold where any model has no forecasts has no common hours.
- **Significance (day level):** hourly absolute errors are aggregated to a mean absolute error
  per IST day. A two-sided Wilcoxon signed-rank test is run on the paired daily MAE differences
  against the fold's common-hours best baseline. A 95% block-bootstrap CI is computed for the
  hour-weighted MAE difference by resampling contiguous 7-day blocks with replacement (2,000
  resamples, seed 42). p-values are unadjusted; `small_sample` flags fewer than 10 days. Δ and
  the CI are hour-weighted, while the Wilcoxon test weights each day equally, so a few
  high-load days can make the two point in different directions.
- **Conformal intervals:** the raw P10/P90 interval is recalibrated per test day with rolling
  conformalized quantile regression (CQR). The conformity score is
  `max(P10 − actual, actual − P90)` over the prior 28 test days **within the same fold** (at
  least 7 days and 24 scores); q is the finite-sample 80% empirical quantile of those scores,
  floored at zero, and the calibrated interval is `[P10 − q, P90 + q]`. A day uses only
  strictly earlier days, so no actual from the scored day or later enters its interval. The
  first 7 days of each fold are warm-up and keep the raw interval.
- **Tuning:** the point XGBoost uses one frozen hyperparameter set tuned on 2016-12-14 to
  2017-05-31, a window that ends before the first scoring test month. There is no per-fold
  Optuna. Annual calendar features (`month`, `doy_sin`, `doy_cos`) are dropped when a fold's
  training span is under 365 days, including all five scoring folds. The quantile models keep
  their fixed untuned defaults.
- **Single model per family per fold:** the information set above is identical for every model;
  no model sees test-period data.

## Folds and coverage

| Fold | Train window | Test window | Train valid h | Clock h | Target-valid h | Eval h | Eval days | MASE scale (Wh) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2017-06 | 2016-12-14 21:00 to 2017-05-31 23:00 | 2017-06-01 to 2017-06-30 | 2,887 | 720 | 720 | 720 | 30 | 100.33 |
| 2017-07 | 2016-12-14 21:00 to 2017-06-30 23:00 | 2017-07-01 to 2017-07-31 | 3,607 | 744 | 744 | 744 | 31 | 96.52 |
| 2017-08 | 2016-12-14 21:00 to 2017-07-31 23:00 | 2017-08-01 to 2017-08-31 | 4,351 | 744 | 744 | 744 | 31 | 92.53 |
| 2017-09_11 | 2016-12-14 21:00 to 2017-08-31 23:00 | 2017-09-01 to 2017-11-30 | 5,095 | 2,184 | 1,781 | 1,695 | 76 | 89.99 |
| 2017-12 | 2016-12-14 21:00 to 2017-11-30 23:00 | 2017-12-01 to 2017-12-31 | 6,876 | 744 | 593 | 548 | 26 | 106.25 |
| Stress test (May–Jun 2018) | 2016-12-14 21:00 to 2017-12-31 23:00 | 2018-05-01 to 2018-06-04 | 7,469 | 840 | 214 | 182 | 8 | 103.52 |

The 2018 stress test is short and sparse: 214 of 840 clock hours have a valid target and the
evaluation mask retains 182 hours across 8 days. Training ends on 2017-12-31, January–April
2018 has no readings, and the test starts on 2018-05-01, so it measures recovery after a
four-month gap.

## Per-fold metrics

All values in Wh except WAPE, MASE, sMAPE and R². `n` is the number of hours on which that
model was scored; models with partial coverage are not imputed. The day-level table below each
metric table uses the common-hours mask, which can be smaller.

### 2017-06

Best baseline: **seasonal naive 24 h** (all-four-baseline mask, 720 h).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 720 | 222.21 | 277.49 | 53.28 | 2.215 | 203.64 | −2.257 | 44.15 |
| Seasonal naive 24 h | 720 | 83.06 | 112.12 | 19.92 | 0.828 | 3.77 | 0.468 | 22.80 |
| Same hour last week | 720 | 125.91 | 165.58 | 30.19 | 1.255 | 12.77 | −0.160 | 32.06 |
| Mean of 4 same weekdays | 720 | 96.64 | 124.43 | 23.17 | 0.963 | −1.98 | 0.345 | 24.52 |
| XGBoost | 720 | 73.13 | 96.15 | 17.53 | 0.729 | 4.70 | 0.609 | 19.18 |
| SARIMA | 720 | 76.64 | 99.85 | 18.38 | 0.764 | 3.53 | 0.578 | 20.15 |
| Quantile XGBoost (P50) | 720 | 72.65 | 95.52 | 17.42 | 0.724 | 10.73 | 0.614 | 19.14 |

Day-level comparison on the 720 common hours (all seven models forecast every evaluation
hour):

| Model | n | MAE | Δ MAE vs best (Wh) | 95% CI (Wh) | Median daily Δ (Wh) | Days | p |
|---|---:|---:|---:|---|---:|---:|---:|
| Persistence | 720 | 222.21 | +139.16 | [88.33, 184.35] | +114.11 | 30 | 3.54e−08 |
| Seasonal naive 24 h | 720 | 83.06 | reference | — | — | 30 | — |
| Same hour last week | 720 | 125.91 | +42.86 | [22.27, 65.91] | +41.89 | 30 | 2.37e−05 |
| Mean of 4 same weekdays | 720 | 96.64 | +13.58 | [−4.72, 31.55] | +14.52 | 30 | 0.0577 |
| XGBoost | 720 | 73.13 | −9.93 | [−18.36, −4.20] | −6.80 | 30 | 0.0040 |
| SARIMA | 720 | 76.64 | −6.41 | [−15.39, 3.28] | −7.57 | 30 | 0.0961 |
| Quantile XGBoost (P50) | 720 | 72.65 | −10.41 | [−18.26, −1.33] | −10.04 | 30 | 0.0017 |

XGBoost and P50 beat the seasonal baseline on both the day-level Wilcoxon test and the
bootstrap CI. SARIMA's point estimate is lower but neither test rules out no difference.

### 2017-07

Best baseline: **seasonal naive 24 h** (all-four-baseline mask, 744 h).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 744 | 119.56 | 155.91 | 32.81 | 1.239 | 62.63 | −0.735 | 31.34 |
| Seasonal naive 24 h | 744 | 75.01 | 103.36 | 20.59 | 0.777 | 0.44 | 0.237 | 23.07 |
| Same hour last week | 744 | 95.00 | 129.84 | 26.07 | 0.984 | 10.59 | −0.203 | 28.50 |
| Mean of 4 same weekdays | 744 | 86.94 | 114.40 | 23.86 | 0.901 | 30.96 | 0.066 | 24.89 |
| XGBoost | 744 | 64.98 | 86.51 | 17.83 | 0.673 | 5.55 | 0.466 | 19.60 |
| SARIMA | 744 | 65.66 | 87.70 | 18.02 | 0.680 | 7.89 | 0.451 | 19.97 |
| Quantile XGBoost (P50) | 744 | 64.56 | 86.26 | 17.72 | 0.669 | −10.34 | 0.469 | 19.46 |

Day-level comparison on the 744 common hours:

| Model | n | MAE | Δ MAE vs best (Wh) | 95% CI (Wh) | Median daily Δ (Wh) | Days | p |
|---|---:|---:|---:|---|---:|---:|---:|
| Persistence | 744 | 119.56 | +44.55 | [20.16, 81.89] | +25.38 | 31 | 4.53e−04 |
| Seasonal naive 24 h | 744 | 75.01 | reference | — | — | 31 | — |
| Same hour last week | 744 | 95.00 | +19.99 | [−0.38, 43.44] | +14.28 | 31 | 0.0110 |
| Mean of 4 same weekdays | 744 | 86.94 | +11.93 | [−7.50, 31.79] | +7.44 | 31 | 0.0305 |
| XGBoost | 744 | 64.98 | −10.03 | [−16.15, −1.87] | −9.00 | 31 | 0.0044 |
| SARIMA | 744 | 65.66 | −9.35 | [−16.85, 1.85] | −9.70 | 31 | 0.0187 |
| Quantile XGBoost (P50) | 744 | 64.56 | −10.45 | [−14.12, −5.98] | −8.21 | 31 | 0.0036 |

All three learned models have lower MAE than the seasonal baseline. The day-level Wilcoxon test
is below 0.05 for all three; the bootstrap CI excludes zero for XGBoost and P50 but not for
SARIMA.

### 2017-08

Best baseline: **mean of the last 4 same weekdays** (all-four-baseline mask, 744 h).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 744 | 105.75 | 133.17 | 28.97 | 1.143 | 18.23 | −0.432 | 28.63 |
| Seasonal naive 24 h | 744 | 76.31 | 101.77 | 20.91 | 0.825 | −0.52 | 0.164 | 22.16 |
| Same hour last week | 744 | 74.52 | 98.57 | 20.42 | 0.805 | −7.58 | 0.215 | 21.76 |
| Mean of 4 same weekdays | 744 | 58.93 | 78.10 | 16.14 | 0.637 | −8.11 | 0.507 | 17.17 |
| XGBoost | 744 | 60.24 | 81.47 | 16.50 | 0.651 | −7.14 | 0.464 | 17.12 |
| SARIMA | 744 | 60.60 | 81.11 | 16.60 | 0.655 | −6.53 | 0.469 | 17.31 |
| Quantile XGBoost (P50) | 744 | 60.12 | 81.64 | 16.47 | 0.650 | −14.61 | 0.462 | 17.15 |

Day-level comparison on the 744 common hours:

| Model | n | MAE | Δ MAE vs best (Wh) | 95% CI (Wh) | Median daily Δ (Wh) | Days | p |
|---|---:|---:|---:|---|---:|---:|---:|
| Persistence | 744 | 105.75 | +46.82 | [33.15, 58.59] | +46.75 | 31 | 9.31e−10 |
| Seasonal naive 24 h | 744 | 76.31 | +17.39 | [8.62, 25.66] | +12.24 | 31 | 1.60e−04 |
| Same hour last week | 744 | 74.52 | +15.60 | [7.61, 19.51] | +13.98 | 31 | 2.57e−06 |
| Mean of 4 same weekdays | 744 | 58.93 | reference | — | — | 31 | — |
| XGBoost | 744 | 60.24 | +1.32 | [−5.97, 8.00] | +3.36 | 31 | 0.8696 |
| SARIMA | 744 | 60.60 | +1.67 | [−5.87, 7.96] | +0.46 | 31 | 0.8092 |
| Quantile XGBoost (P50) | 744 | 60.12 | +1.20 | [−6.31, 8.39] | +4.19 | 31 | 0.8544 |

The calendar-average baseline is the best model in this month. All three learned models are
slightly worse on MAE, and no paired difference is statistically discernible.

### 2017-09_11

Best baseline: **persistence** (evaluation-mask fallback, 1,695 h; the all-four mask covers
less than 50% of evaluation hours because the long lags have gaps).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1,695 | 138.41 | 200.45 | 48.14 | 1.538 | 46.25 | −0.091 | 51.13 |
| Seasonal naive 24 h | 1,695 | 151.82 | 228.21 | 52.81 | 1.687 | 8.40 | −0.414 | 63.91 |
| Same hour last week | 1,448 | 209.73 | 269.51 | 73.78 | 2.331 | 9.79 | −0.905 | 92.53 |
| Mean of 4 same weekdays | 811 | 159.31 | 198.83 | 61.15 | 1.770 | 32.85 | −0.009 | 76.11 |
| XGBoost | 1,695 | 144.01 | 194.17 | 50.09 | 1.600 | 61.78 | −0.024 | 61.94 |
| SARIMA | 1,167 | 107.16 | 153.46 | 37.37 | 1.191 | −1.26 | 0.388 | 57.41 |
| Quantile XGBoost (P50) | 1,695 | 144.01 | 193.17 | 50.09 | 1.600 | 53.86 | −0.013 | 62.53 |

The common mask retains 555 of 1,695 evaluation hours (32.74%), because `same_hour_last_week`
and `mean_last_4_same_weekdays` have gaps and SARIMA covers 1,167 hours. Common-hours best
baseline: **persistence**.

| Model | n | MAE | Δ MAE vs best (Wh) | 95% CI (Wh) | Median daily Δ (Wh) | Days | p |
|---|---:|---:|---:|---|---:|---:|---:|
| Persistence | 555 | 130.92 | reference | — | — | 28 | — |
| Seasonal naive 24 h | 555 | 180.08 | +49.16 | [−12.16, 102.88] | +4.54 | 28 | 0.1256 |
| Same hour last week | 555 | 205.79 | +74.87 | [14.10, 160.06] | +26.95 | 28 | 0.0264 |
| Mean of 4 same weekdays | 555 | 158.13 | +27.21 | [−12.76, 80.10] | −3.51 | 28 | 0.4380 |
| XGBoost | 555 | 160.99 | +30.07 | [−10.81, 65.45] | −14.92 | 28 | 0.7793 |
| SARIMA | 555 | 112.57 | −18.35 | [−28.12, −3.33] | −20.30 | 28 | 0.0698 |
| Quantile XGBoost (P50) | 555 | 157.94 | +27.02 | [−11.60, 58.74] | −13.29 | 28 | 0.6782 |

This is a hard, heterogeneous fold. On the hours all models cover, SARIMA is the only learned
model with lower MAE than persistence, but its day-level p-value is 0.070, so the difference is
not significant at 0.05. The XGBoost and P50 hour-weighted MAE is worse than persistence even
though their median daily difference is negative: a few high-load days dominate the weighted
mean.

### 2017-12

Best baseline: **persistence** (all-four-baseline mask, 479 h).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 548 | 47.55 | 106.52 | 51.39 | 0.448 | −33.52 | 0.341 | 48.16 |
| Seasonal naive 24 h | 548 | 71.48 | 141.85 | 77.25 | 0.673 | −9.66 | −0.168 | 61.40 |
| Same hour last week | 479 | 181.77 | 260.10 | 203.06 | 1.711 | 69.83 | −2.716 | 115.71 |
| Mean of 4 same weekdays | 479 | 181.55 | 210.95 | 202.81 | 1.709 | 99.82 | −1.444 | 141.34 |
| XGBoost | 548 | 168.08 | 189.34 | 181.66 | 1.582 | 133.22 | −1.082 | 126.16 |
| SARIMA | 387 | 98.43 | 128.37 | 118.78 | 0.926 | 39.97 | 0.108 | 106.72 |
| Quantile XGBoost (P50) | 548 | 171.16 | 196.21 | 184.99 | 1.611 | 134.60 | −1.235 | 125.37 |

The common mask retains 387 of 548 evaluation hours (70.62%). Common-hours best baseline:
**persistence**.

| Model | n | MAE | Δ MAE vs best (Wh) | 95% CI (Wh) | Median daily Δ (Wh) | Days | p |
|---|---:|---:|---:|---|---:|---:|---:|
| Persistence | 387 | 48.15 | reference | — | — | 17 | — |
| Seasonal naive 24 h | 387 | 66.06 | +17.91 | [0.68, 36.22] | −1.31 | 17 | 0.9632 |
| Same hour last week | 387 | 179.27 | +131.12 | [67.53, 196.30] | +89.67 | 17 | 3.81e−04 |
| Mean of 4 same weekdays | 387 | 196.67 | +148.52 | [51.48, 173.76] | +153.74 | 17 | 6.56e−04 |
| XGBoost | 387 | 186.71 | +138.55 | [−22.55, 192.85] | +188.91 | 17 | 0.0026 |
| SARIMA | 387 | 98.43 | +50.28 | [−21.52, 65.26] | +73.95 | 17 | 0.0110 |
| Quantile XGBoost (P50) | 387 | 191.46 | +143.31 | [−34.59, 195.97] | +198.76 | 17 | 0.0026 |

December is the vacancy regime: 21 apartments are flagged suspect inactive in the month (13 of
them stable), the target level collapses, and every learned model is far worse than holding the
last observation. Persistence wins with MAE 47.55 Wh; even SARIMA, the least bad learned model,
is more than 50 Wh worse on common hours. The wide bootstrap CIs reflect only 17 days and three
7-day blocks, so the day-level test is coarse here; the direction of the loss is unambiguous.

### Stress test (May–Jun 2018)

Excluded from every fold mean. Best baseline: **seasonal naive 24 h** (evaluation-mask
fallback, 182 h; `mean_last_4_same_weekdays` has no forecasts because the preceding four
same-weekday hours are missing).

| Model | n | MAE | RMSE | WAPE % | MASE | Bias | R² | sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 182 | 274.61 | 335.85 | 58.53 | 2.653 | 185.12 | −2.650 | 51.80 |
| Seasonal naive 24 h | 182 | 132.87 | 192.27 | 28.32 | 1.284 | −67.30 | −0.196 | 40.53 |
| Same hour last week | 38 | 384.27 | 449.73 | 71.81 | 3.712 | −328.44 | −2.640 | 126.99 |
| Mean of 4 same weekdays | 0 | — | — | — | — | — | — | — |
| XGBoost | 182 | 112.70 | 150.10 | 24.02 | 1.089 | −75.52 | 0.271 | 25.49 |
| SARIMA | 38 | 93.80 | 127.08 | 17.53 | 0.906 | −35.21 | 0.709 | 24.09 |
| Quantile XGBoost (P50) | 182 | 106.22 | 141.64 | 22.64 | 1.026 | −70.14 | 0.351 | 23.95 |

There are **0 common hours** because `mean_last_4_same_weekdays` has no forecasts, so the
strict all-model mask is empty and no day-level test is possible. On its own 182 hours XGBoost
is better than the seasonal naive on MAE, but the fold is context only: it is sparse, follows a
four-month gap, and never enters a mean.

## Common-hours summary

The strict common-hours table above is the only like-for-like comparison of all seven models.
On the five scoring folds:

- **XGBoost** has lower common-hours MAE than the best baseline in June (−9.93 Wh, p = 0.0040)
  and July (−10.03 Wh, p = 0.0044). It is worse in August (+1.32 Wh, p = 0.87), September–November
  (+30.07 Wh, p = 0.78) and December (+138.55 Wh, p = 0.0026).
- **Quantile XGBoost (P50)** follows the same pattern: June (−10.41 Wh, p = 0.0017) and July
  (−10.45 Wh, p = 0.0036) are wins; August (+1.20 Wh, p = 0.85), September–November (+27.02 Wh,
  p = 0.68) and December (+143.31 Wh, p = 0.0026) are not.
- **SARIMA** has lower common-hours MAE in three folds (June −6.41, July −9.35,
  September–November −18.35) but only July has a day-level p-value below 0.05; June is p = 0.096
  and September–November p = 0.070, and the bootstrap CIs for June and July include zero.
  December is a clear loss (+50.28 Wh, p = 0.011).
- **No learned model is reliably better than the common-hours best baseline.** XGBoost and P50
  beat it in only 2 of 5 scoring folds (June and July, both significant). SARIMA is lower in 3 of
  5 (June, July, September–November), but only July is significant at 0.05. Every learned model
  loses badly in December, and none of these fold-level wins is evidence of a general advantage.

The per-model tables (each model on its own available hours) give a friendlier picture for
SARIMA, which covers only 1,167 of 1,695 hours in September–November and 387 of 548 in December.
The common-hours table removes that coverage advantage. Both are reported; neither is a
substitute for the other.

## Fold means

Unweighted mean ± sample standard deviation across the five scoring folds. The 2018 stress test
is excluded. `n` is the total evaluable hours across those folds.

| Model | Folds | Eval hours | MAE | RMSE | MASE | Beats seasonal naive |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 4,451 | 126.70 ± 63.27 | 174.71 ± 67.00 | 1.316 ± 0.642 | 2/5 |
| Seasonal naive 24 h | 5 | 4,451 | 91.54 ± 33.96 | 137.46 ± 53.23 | 0.958 ± 0.412 | reference |
| Same hour last week | 5 | 4,135 | 137.39 ± 57.19 | 184.72 ± 76.93 | 1.417 ± 0.614 | 1/5 |
| Mean of 4 same weekdays | 5 | 3,498 | 116.67 ± 51.59 | 145.34 ± 57.19 | 1.196 ± 0.512 | 1/5 |
| XGBoost | 5 | 4,451 | 102.09 ± 50.20 | 129.53 ± 57.08 | 1.047 ± 0.497 | 4/5 |
| SARIMA | 5 | 3,762 | 81.70 ± 20.35 | 110.10 ± 30.25 | 0.843 ± 0.221 | 4/5 |
| Quantile XGBoost (P50) | 5 | 4,451 | 102.50 ± 51.39 | 130.56 ± 58.77 | 1.051 ± 0.507 | 4/5 |

"Beats seasonal naive" is a descriptive count over folds where both models have data, based on
MAE; it is not a significance test and it is not the common-hours comparison. The standard
deviations are large because December is a regime break: the XGBoost fold MAE ranges from 60.24
(August) to 168.08 (December).

## Development experiments

Phase 9 tested feature and target changes under the same folds. Only the changes below were
adopted; the point-model comparisons use the full evaluation mask, not the common-hours mask.

| Variant | Decision | Why |
|---|---|---|
| Drop month/day-of-year when training <365 days | **kept** | Helped 4 of 5 scoring folds; most of the gain is December (227.67 → 196.39 Wh untuned; later tuning brought it to 168.08). Three of the four wins are under 0.6 Wh. |
| Drop weather features | rejected | Helped only 2 of 5 folds (Sep–Nov, Dec) and worsened the five-fold mean. |
| Same-hour temperature | **ORACLE only** | Reads temperature at the target hour, which is unknown at the 00:00 origin. Diagnostic only; never a reported result. |
| Level-normalized target (÷ trailing 7-day mean) | rejected | Helped only December (−44.61 Wh) and hurt the other four folds. With the near-zero feature it still helped only 1 of 5. |
| Frozen pre-scoring Optuna parameters | **kept** | One 30-trial study through 2017-05-31, reused in every fold, lowered the five-fold mean from 107.66 to 102.09 Wh (helped 3 of 5 folds). No per-fold Optuna remains. |
| Rolling conformal P10–P90 calibration | **kept** | Raises scoring hour-weighted coverage 64.46% → 75.29% without using future actuals; still below nominal in Sep–Dec (see above). |

## Prediction intervals

The quantile XGBoost models produce P10/P50/P90. Coverage is the share of the model's scored
hours with the actual inside [P10, P90]; the nominal level is 80%. Pinball loss is in Wh. These
are the raw intervals, before conformal calibration.

| Fold | n | Coverage | Mean width (Wh) | Pinball P10 | Pinball P50 | Pinball P90 | Pinball mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2017-06 | 720 | 72.64% | 206.31 | 17.71 | 36.33 | 17.39 | 23.81 |
| 2017-07 | 744 | 82.26% | 204.99 | 15.90 | 32.28 | 14.58 | 20.92 |
| 2017-08 | 744 | 80.65% | 190.97 | 13.50 | 30.06 | 15.23 | 19.60 |
| 2017-09_11 | 1,695 | 56.93% | 246.06 | 55.96 | 72.01 | 28.06 | 52.01 |
| 2017-12 | 548 | 30.84% | 352.38 | 42.50 | 85.58 | 36.78 | 54.95 |
| Stress test | 182 | 73.63% | 246.70 | 20.28 | 53.11 | 26.71 | 33.37 |
| Scoring, hour-weighted | 4,451 | 64.46% | 236.65 | — | — | — | 37.20 |
| Scoring, fold-unweighted | — | 64.66% | 240.14 | — | — | — | 34.26 |

The intervals are close to nominal in July and August, undercover in June, and severely
undercalibrated in September–December, especially December (30.84%). They are reported as
calibration diagnostics, not as production-grade uncertainty bounds.

### Conformal recalibration (rolling prior days)

Each test day's interval is expanded by the 80% conformal quantile of the previous 28 test
days' conformity scores in the same fold. "Calibrated days" excludes the 7-day warm-up, which
is unchanged; "all days" averages over every scored day. Mean width is over all hours, so it
includes the unchanged warm-up.

| Fold | Scored days | Calibrated days | Before (cal. days) | After (cal. days) | Before (all days) | After (all days) | Mean width, all (Wh) | Median q (Wh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017-06 | 30 | 23 | 71.01% | 79.71% | 72.64% | 79.31% | 206.31 → 232.62 | 18.79 |
| 2017-07 | 31 | 24 | 80.90% | 80.90% | 82.26% | 82.26% | 204.99 → 204.99 | 0.00 |
| 2017-08 | 31 | 24 | 81.60% | 82.12% | 80.65% | 81.05% | 190.97 → 193.01 | 0.00 |
| 2017-09_11 | 76 | 69 | 55.29% | 74.09% | 56.93% | 74.22% | 246.06 → 421.34 | 113.23 |
| 2017-12 | 26 | 19 | 35.00% | 71.32% | 30.84% | 56.02% | 352.38 → 453.58 | 82.46 |
| Stress test | 8 | 1 | 57.14% | 57.14% | 73.63% | 73.63% | 246.70 → 249.25 | 16.53 |
| Scoring, hour-weighted | 194 | 159 | — | — | 64.46% | 75.29% | 236.65 → 320.45 | — |

Calibration raises coverage by 8.7 points on calibrated June days, 18.8 points in
September–November and 21.3 points in December; the scoring hour-weighted coverage rises from
64.46% to 75.29%. It does not reach the 80% target in September–November or December, because
the conformity scores are not exchangeable across the vacancy regime shift: a 28-day window of
past residuals cannot anticipate a break that starts on the scored day. Because q is floored at
zero, calibration only widens intervals: July is unchanged, August moves 0.4 points, and the
mean width grows from 236.65 to 320.45 Wh on the scoring folds. The stress test has only one
calibrated day, so its after-coverage is context only.

## Data-quality inputs

From `backend/data_quality.json`:

- 39 of 39 raw files read, 196,449 rows total.
- Hourly panel 12,891 h × 39 apartments; 5,291 hours with zero reporting apartments.
- 24 apartments pass the >90% coverage stable-panel rule.
- Target valid for 7,683 hours; mean 333.50 Wh.
- 83 hours imputed in short runs (≤6 h); 5,208 hours in 31 excluded long runs; longest 3,514 h.
- 21 apartments flagged suspect inactive in December 2017 (13 of them stable).

Weather inputs come from the Open-Meteo archive for one Mumbai grid point (19.1334 N,
72.9133 E): 11,688 cached hours, six variables, no missing values, all lags ≥24 h.

## Reproduction

```bash
# from the repository root
python backend/build_data_quality.py   # writes backend/data_quality.json
python backend/evaluate.py             # writes backend/results.json
python -m pytest backend/tests -q
```

`evaluate.py` rebuilds the target, retrains the frozen-parameter point XGBoost, the quantile
XGBoost and SARIMA for every fold, and scores all baselines on the shared mask. Training
dominates the runtime; the script is not run at API startup. The backend only reads the
generated JSON.

## Caveats

- Five scoring folds only, and the 2018 stress test has 182 evaluable hours across 8 days.
  Fold-mean differences smaller than the fold-to-fold spread should not be treated as stable
  rankings. The fold means weight each fold equally, not each hour.
- The target is a per-reporting-apartment mean over the stable panel; it is not total building
  load, and changes in reporting counts can move it.
- December 2017 contains suspected vacancies, which drives the December loss and the large
  positive bias in every learned model. No learned model handles that regime.
- The common-hours table is strict: any model missing an hour removes it from every row. It
  retains 32.74% of September–November hours and 70.62% of December hours, and 0 hours in the
  stress test, so its estimates are noisier than the per-model tables.
- Day-level p-values are unadjusted for multiple comparisons and are reported per model against
  the common-hours best baseline only. The bootstrap CI is hour-weighted, while the Wilcoxon
  test weights each day equally; the two can disagree when high-load days dominate.
- December's bootstrap uses only three 7-day blocks, so its percentile CI is coarse. The
  direction of the learned-model loss is supported by both the daily median and the MAE.
- Conformal calibration expands intervals only: the conformity quantile is floored at zero, so
  overcovering days are never narrowed. Its coverage remains below the nominal 80% in
  September–November (74.09% on calibrated days) and December (71.32%) because prior-day
  residuals are not exchangeable across the vacancy regime shift. The 28-day window was chosen
  a priori as four weekly cycles, not tuned on these folds; it is the same in every fold.
- SARIMA's weekly refit cannot operate through the long 2018 gap; its stress score covers only
  38 hours.
- Weather is a single regional grid point, not building-level measurements.
- The point XGBoost tuning window ends 2017-05-31, before every scoring test period. It uses
  the same frozen parameters in every fold; there is no per-fold test tuning.
