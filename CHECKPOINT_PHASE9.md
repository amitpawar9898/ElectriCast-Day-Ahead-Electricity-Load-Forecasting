# Phase 9 checkpoint — after 9.5

Date: 2026-10-23. Branch: `improve-electricast`. Status: 9.1–9.5 complete;
9.6–9.9 pending. This checkpoint compares the pre-Phase-9 artifact
(`results_v1.json`), the current production artifact after 9.1–9.2
(`results.json`), and the point-XGBoost experiment chain through 9.5. All
values are MAE in Wh on each fold's evaluation mask unless marked otherwise.
The 2018 May–Jun fold is a stress test and is excluded from every mean.

## A. Production artifact: before Phase 9 vs after 9.1–9.2

`results_v1.json` is the pre-9.1 artifact. `results.json` is the live artifact
after the history expansion and the five-fold layout. Jul, Aug and the 2018
stress fold are directly comparable. The old Sep–Dec 2017 fold was split into
Sep–Nov and Dec; the current numbers below are recomputed on the same 2,243
evaluation hours, so the comparison is like-for-like.

| Fold | XGB v1 | XGB current | P50 v1 | P50 current | SARIMA v1 | SARIMA current | Best baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jul 2017 | 69.31 | 66.28 | 65.99 | 64.56 | 65.66 | 65.66 | 75.01 (seasonal naive) |
| Aug 2017 | 59.51 | 59.13 | 60.58 | 60.12 | 60.60 | 60.60 | 58.93 (mean of 4 same weekdays) |
| Sep–Dec 2017 (combined) | 167.27 | 151.13 | 159.14 | 150.65 | 104.23 | 104.99 | 116.21 (persistence) |
| May–Jun 2018 (stress) | 120.75 | 111.31 | 113.94 | 106.22 | 93.80 | 93.80 | 132.87 (seasonal naive) |

The expanded history lowers XGBoost and P50 MAE on every comparable window.
Sep–Dec is now two folds, and splitting December out exposes the vacancy
regime: current XGBoost MAE is 175.59 on Dec alone versus 143.22 on Sep–Nov.
The current artifact still uses the full annual calendar and per-fold Optuna;
9.3–9.5 are experiments that are not yet integrated into `results.json`.

## B. Phase 9 scoring folds: point-XGBoost experiment chain

All 9.3–9.5 variants apply the 9.3 feature rule per fold (drop month and
day-of-year when that fold's training window is shorter than 365 days; the 2018
stress fold keeps the full calendar) and the fixed untuned `DEFAULT_PARAMS`
unless noted. 9.4 normalized values are on common hours with the 9.3 winner;
all others use the full evaluation mask.

| Fold | 9.2 production (per-fold Optuna, full calendar) | 9.3 winner (untuned, no annual calendar) | 9.4 normalized (not adopted) | 9.5 clean frozen (adopted) | Best baseline |
|---|---:|---:|---:|---:|---:|
| Jun 2017 | 73.09 | 73.60 | 91.06 | **73.13** | 83.06 (seasonal naive) |
| Jul 2017 | 66.28 | 63.69 | 67.11 | **64.98** | 75.01 (seasonal naive) |
| Aug 2017 | 59.13 | 59.39 | 62.84 | **60.24** | 58.93 (mean of 4 same weekdays) |
| Sep–Nov 2017 | 143.22 | 145.25 | 153.68 | **144.01** | 138.41 (persistence) |
| Dec 2017 | 175.59 | 196.39 | 151.79 | **168.08** | 47.55 (persistence) |
| May–Jun 2018 (stress, excluded) | 111.31 | 108.87 | 157.50 | 112.70 | 132.87 (seasonal naive) |

Every 9.5 value applies the 9.3 rule per fold, so the stress fold keeps the full
calendar; the 9.4 stress value still uses the normalized variant's own config.
The stress row is context only and never enters a mean.

### Fold means and wins versus the per-fold best baseline

| Point-XGBoost variant | 5-fold mean MAE | Folds with lower MAE than the best baseline | Note |
|---|---:|---:|---|
| 9.2 production | 103.46 | 2/5 (Jun, Jul) | per-fold Optuna, full calendar |
| 9.3 winner | 107.66 | 2/5 (Jun, Jul) | untuned, no annual calendar |
| 9.4 normalized | 105.29 | 1/5 (Jul) | not adopted; helps Dec by 44.61 but hurts four folds |
| 9.5 clean frozen | **102.09** | 2/5 (Jun, Jul) | adopted; frozen parameters from the clean pre-June tune |

SARIMA remains the best learned model on the current production artifact with a
5-fold mean MAE of 81.70 Wh; it beats the best baseline on 3 of 5 folds (Jun,
Jul, Sep–Nov) and loses on Aug and Dec. No point-XGBoost variant through 9.5
beats SARIMA, and none beats the best baseline on more than 2 of 5 folds. This
is stated plainly for 9.9.

## C. Decisions carried out of 9.5

1. **Keep the 9.3 feature rule** (drop month and day-of-year when training is
   under 12 months). It helped 4 of 5 scoring folds, driven mainly by December
   (196.39 → 168.08 after tuning; 227.67 → 196.39 untuned).
2. **Reject level normalization** (9.4). It helps December by 44.61 Wh and cuts
   December bias, but helps only 1 of 5 scoring folds; the improvement is a
   regime effect, not a general gain.
3. **Use frozen tuning parameters from the clean pre-scoring window.**
   The literal largest window ends 2017-12-31 and contains every scoring test
   period, so scoring it is not clean evidence. Tuning there helped only 1/5
   folds. The largest window containing no scoring test period ends 2017-05-31;
   one 30-trial tune there, frozen across all folds, helped 3/5 scoring folds
   (Jun −0.47, Sep–Nov −1.24, Dec −28.31; Jul +1.30, Aug +0.86) and lowered the
   mean from 107.66 to 102.09 Wh. Both runs are in `tuning_results.json`.
4. **No per-fold Optuna for the point model** under the frozen configuration.
   The adopted parameters are: `max_depth=3`, `learning_rate=0.018994`,
   `min_child_weight=17.441`, `subsample=0.623`, `colsample_bytree=0.768`,
   `reg_lambda=8.204`, `reg_alpha=0.787`, `n_estimators=150`.
5. **Do not adopt the near-zero vacancy feature** in the general model. It adds
   value only in December (2.30 Wh on top of normalization) and does not change
   the 1/5 fold count.

## D. What is not yet done

- 9.3–9.5 results are experiments. `results.json`, the API, the dashboard and
  the assistant still describe the 9.2 production models. 9.6–9.9 will decide
  and implement the final artifact.
- The 2018 stress fold is context only in every table and mean above.
- Fold means are unweighted; there is no confidence interval here. Paired
  day-level tests and a block bootstrap are 9.7.
- The P10–P90 intervals are not yet conformally calibrated; that is 9.8.
- The December vacancy regime is still not forecast well by any learned model:
  the best point-XGBoost variant (168.08) loses to persistence (47.55) by a
  wide margin.

## E. Reproduce

```bash
python backend/evaluate.py
python backend/scripts/feature_ablation.py
python backend/scripts/level_experiment.py
python backend/scripts/frozen_tuning.py
python -m pytest backend/tests -q
```

Artifacts: `backend/results_v1.json` (pre-9.1), `backend/results.json` (live
production through 9.2), `backend/ablation_results.json` (9.3),
`backend/level_results.json` (9.4), `backend/tuning_results.json` (9.5).
