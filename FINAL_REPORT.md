# ElectriCast — Final Report

**Branch:** `improve-electricast` · **Base:** `main` · **Last item:** 9.9 docs, UI and assistant
**Status:** all PLAN items (0.1–9.9) `[x]`, none `[!]`
**Artifacts:** `backend/results.json` (2026-10-23T23:52:19+05:30),
`backend/data_quality.json` (2026-10-23T18:10:05+05:30)

All work is committed on `improve-electricast`. `main` was not modified, no history was
rewritten, and nothing was pushed.

## What was done

| Phase | Outcome |
|---|---|
| 0 — Hygiene | Root `.gitignore`; generated files untracked; requirements pinned to installed versions; Docker Python aligned; dead LSTM loader/Radar import removed; legacy notebook moved under `notebooks/legacy/`. |
| 1 — UI truth | Wh/kWh labels corrected; disabled LSTM filtered out of every UI path; Weekly charts moved to XGBoost; yearly chart uses backend daily means; split label computed from statistics; IST label and real dataset header; R² thresholds; layout/print fixes; real day picker; same-origin API, abortable fetches and visible errors. |
| 2 — Data pipeline | All 39 CSVs loaded with explicit error reporting; per-apartment hourly panel with `n_reporting`; UTC/IST normalization for the May-2018 `:00 UTC` files; target = mean load per reporting stable apartment; 24-apartment stable panel (>90%); gaps ≤6 h interpolated, longer/edge runs excluded; train-only Q3+3·IQR cap; December inactivity flags; `data_quality.json` plus Data Quality page. |
| 3 — Evaluation | Four day-ahead baselines; MAE/RMSE/WAPE/MASE/bias/R²/guarded sMAPE; 00:00 IST origins and a shared evaluation mask; per-fold `results.json`. |
| 4 — Models | Leakage-tested XGBoost with ≥24 h lags, calendar/holiday/semester/`n_reporting` features, cached Open-Meteo weather; rolling SARIMA(1,0,1)(0,1,1,24) refit weekly on 60 days; quantile XGBoost P10/P50/P90 with coverage and pinball loss; LSTM omitted entirely rather than shown as a placeholder. |
| 5 — Dashboard | Fold mean ± std metrics with “beats seasonal naive?”; fold-based day picker with P10–P90 band, error by hour/month and worst days; XGBoost gain importance and residuals; Limitations panel sourced from backend JSON. |
| 6 — Upload | Column mapping, unit/timezone/format selection, 256 KB preview, streamed 50 MB cap, IST conversion, validation report, 60/40 chronological split, Phase 3 baselines, zero-shot campus model and fit-on-60% model, paired significance, JSON download, progress/error states. |
| 7 — Engineering | pytest suite (metrics, split, leakage, gaps, upload validation); Makefile; Docker fixes (full runtime, `/data` mount, no-curl healthcheck, same-origin nginx, CPU-only XGBoost, no torch); README, RESULTS.md and copilot instructions rewritten for the current product. |
| 8 — Assistant | “Ask ElectriCast” drawer on every page; deterministic routing to read-only artifact tools; numeric/date/time guard with one retry; local Ollama default and optional Gemini fallback; visible unavailable/loading/error states. |
| 9 — Harder evaluation | Training starts at the first valid target hour (2016-12-14 21:00 IST) for every fold; five scoring folds (Jun, Jul, Aug, Sep–Nov, Dec 2017) plus a May–Jun 2018 stress test excluded from means; feature ablations; level-normalization test; one frozen pre-scoring Optuna study (no per-fold tuning); strict common-hours comparison; day-level Wilcoxon plus block-bootstrap CI; rolling conformal P10–P90 calibration; docs, Limitations page and assistant refreshed. |

## Final protocol

Day-ahead forecasts for 24 IST clock hours from a 00:00 IST origin. Every feature uses observations
strictly before the origin (lags ≥24 h for weather and reporting counts). Five scoring folds expand
from 2016-12-14 21:00 IST; the 2018 fold is a stress test after a four-month outage and never enters
a mean. Models are scored on the hours where their own forecast exists; the common-hours table
intersects the evaluation mask with all four baselines and all three learned models. Best baseline is
chosen by MASE. Significance is a paired Wilcoxon test on daily MAE against the fold's common-hours
best baseline, with a 95% block-bootstrap CI (2,000 resamples of contiguous 7-day blocks, seed 42).
The point XGBoost uses one frozen parameter set tuned through 2017-05-31, before every scoring test
month. Annual calendar features are dropped when training is shorter than 365 days.

## Baselines vs models

Fold means are unweighted across the five scoring folds; `n` is total evaluable hours. Source:
`backend/results.json`, reproduced in full in `RESULTS.md`.

| Model | Folds | n | MAE (Wh) | RMSE (Wh) | WAPE % | MASE | R² | Beats seasonal naive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 4,451 | 126.70 ± 63.27 | 174.71 | 42.92 | 1.316 | −0.635 | 2/5 |
| Seasonal naive 24 h | 5 | 4,451 | 91.54 ± 33.96 | 137.46 | 38.29 | 0.958 | 0.057 | reference |
| Same hour last week | 5 | 4,135 | 137.39 ± 57.19 | 184.72 | 70.70 | 1.417 | −0.754 | 1/5 |
| Mean of 4 same weekdays | 5 | 3,498 | 116.67 ± 51.59 | 145.34 | 65.43 | 1.196 | −0.107 | 1/5 |
| XGBoost (frozen) | 5 | 4,451 | 102.09 ± 50.20 | 129.53 | 56.72 | 1.047 | 0.087 | 4/5 |
| SARIMA | 5 | 3,762 | 81.70 ± 20.35 | 110.10 | 41.83 | 0.843 | 0.399 | 4/5 |
| Quantile XGBoost (P50) | 5 | 4,451 | 102.50 ± 51.39 | 130.56 | 57.34 | 1.051 | 0.059 | 4/5 |

Per-fold conclusions:

- **Jun 2017** — XGBoost and P50 beat the seasonal baseline on common hours (−9.93 and −10.41 Wh,
  p = 0.004 and 0.002). SARIMA is −6.41 Wh but p = 0.096.
- **Jul 2017** — all three learned models beat the seasonal baseline; all three day-level p-values
  are below 0.05, and the XGBoost/P50 bootstrap CIs exclude zero (SARIMA's includes zero).
- **Aug 2017** — the mean of four same weekdays wins (58.93 Wh). Learned models are +1.2 to +1.7 Wh
  worse and not significantly different.
- **Sep–Nov 2017** — persistence wins. On the 555 common hours XGBoost is +30.07 Wh (p = 0.78) and
  P50 +27.02 Wh (p = 0.68) worse; SARIMA is −18.35 Wh but p = 0.070. The strict mask retains only
  32.74% of evaluation hours.
- **Dec 2017** — persistence wins at 47.55 Wh. XGBoost is 168.08 Wh, P50 171.16 Wh and SARIMA
  98.43 Wh on common hours (+138.55, +143.31 and +50.28 Wh; all p < 0.05). The month contains 21
  apartments flagged suspect inactive (13 stable), consistent with a vacancy-driven level collapse.
- **May–Jun 2018** — 182 evaluable hours over 8 days, 0 common hours; context only.

**Common-hours bottom line:** no learned model is reliably better than the best baseline. XGBoost
and P50 beat it in only 2 of 5 scoring folds (June and July); SARIMA is lower in 3 of 5 but only
July reaches p < 0.05; every learned model loses badly in December. The fold-mean ranking is
descriptive, not a deployment recommendation.

## Prediction intervals

Raw P10–P90 coverage is 64.46% hour-weighted on the scoring folds (80% nominal). Rolling
conformalized quantile regression expands each test day's interval by the 80% quantile of the prior
28 days' conformity scores in the same fold (7-day warm-up, no future actuals, expansion only):

| Fold | Raw coverage | After CQR, calibrated days | After CQR, all days |
|---|---:|---:|---:|
| Jun 2017 | 72.64% | 79.71% | 79.31% |
| Jul 2017 | 82.26% | 80.90% | 82.26% |
| Aug 2017 | 80.65% | 82.12% | 81.05% |
| Sep–Nov 2017 | 56.93% | 74.09% | 74.22% |
| Dec 2017 | 30.84% | 71.32% | 56.02% |
| Stress test | 73.63% | 57.14% | 73.63% |
| Scoring, hour-weighted | 64.46% | — | 75.29% |

Calibration raises coverage substantially but does not reach 80% in September–November or December:
prior-day residuals are not exchangeable across the vacancy regime shift. Because q is floored at
zero, intervals never narrow; scoring mean width grows from 236.65 to 320.45 Wh. The stress fold has
one calibrated day, so its after figure is context only.

## Phase 9 decisions

1. **All valid history.** Every fold trains from 2016-12-14 21:00 IST. The pre-9.1 artifact is kept
   as `results_v1.json` for comparison.
2. **Five scoring folds + stress.** June and December are separate folds; December is no longer
   pooled with Sep–Nov. The 2018 fold is labeled a stress test and excluded from means.
3. **Feature ablation.** Dropping annual calendar features when training <365 days is kept (4/5
   folds). Dropping weather is rejected (2/5). The same-hour temperature variant is ORACLE,
   diagnostic only, and never written to `results.json`.
4. **Level normalization rejected.** It helps December by 44.61 Wh but only 1 of 5 folds.
5. **Frozen tuning.** One 30-trial Optuna study through 2017-05-31, reused in every fold, lowers the
   five-fold mean from 107.66 to 102.09 Wh. No per-fold Optuna remains.
6. **Strict common hours.** Every model is scored on exactly the same hours; no fallback mask.
7. **Day-level inference.** Daily MAE Wilcoxon plus a 7-day block bootstrap; the old hourly test is
   no longer written to the artifact.
8. **Conformal intervals.** Rolling 28-day CQR, expansion only, prior days only.

## Blocked items

None. Every PLAN item (0.1–9.9) is `[x]`. No item required the `[!]` fallback, and no dependent
item is waiting on another.

## Remaining risks

1. **Small, uneven fold set.** Five scoring folds plus one 8-day stress test. Fold-mean rankings are
   sensitive to fold composition and should not be read as stable model rankings.
2. **No reliable winner on common hours.** XGBoost/P50 win 2/5 scoring folds, SARIMA 3/5 with only
   one significant win, and all lose badly in December. The honest conclusion is that the learned
   models do not dominate the simple baselines.
3. **SARIMA coverage.** It is scored on 3,762 of 4,451 scoring hours and 38 stress hours; the
   January–April 2018 gap removes most of its 2018 forecasts. It has no reported prediction
   intervals.
4. **December vacancy regime.** 21 apartments flagged suspect inactive (13 stable); every learned
   model carries a large positive bias and loses to persistence by 50–143 Wh. Conformal calibration
   cannot fix a break that starts inside the scored fold.
5. **Target definition.** The target is a per-reporting-apartment mean over 24 stable apartments,
   not total building load; it can move with reporting counts and is not comparable to the legacy
   per-building aggregate on EDA/Weekly.
6. **Two data products.** EDA and Weekly still read the legacy 2017 single-split
   `model_results.json`; Predictions/Metrics/Explain/Quality/Limitations/Upload read
   `results.json`. The UI sources them correctly, but they are not directly comparable.
7. **Weather proxy.** One Open-Meteo grid point (Mumbai), lagged ≥24 h; not building-level and not
   a causal input.
8. **Upload domain shift.** The zero-shot model is trained on campus data and may underperform an
   uploaded series; the report labels it as zero-shot and shows the fit-on-60% alternative. Very
   short uploads (<336 valid hours) return validation only.
9. **Operational surface.** The Docker stack was built and exercised in an isolated Compose project
   and then removed. There is no CI pipeline. GNU make is not installed on this Windows host, so
   Makefile recipes were validated command-by-command.
10. **Frontend testing.** No automated frontend test runner. Pages were browser-verified against a
    live backend, and the production build passes, but regressions are not caught by CI.
11. **Assistant limits.** Single-turn, artifacts-only, and dependent on a local provider. The
    numeric guard checks provenance, not semantic truth; the default `llama3.2:3b` can phrase a
    grounded number poorly. Month-scoped answers get a deterministic grounded MAE/bias opening
    sentence from the tool result, but later prose is still the model's. The Gemini path has
    mocked tests only (no API key on this host).
12. **Upload evaluation protocol.** Upload validation still uses the Phase 3 hourly paired test and
    its own 60/40 split; it does not use the backtest's day-level test or conformal intervals.

## Exact commands

All commands are run from the repository root.

```bash
# Install
python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)

# Develop (two terminals)
(cd backend && python -m uvicorn main:app --reload --port 8000)
(cd frontend && npm run dev)
# Dashboard: http://localhost:3000   API docs: http://localhost:8000/docs

# Quality gates
python -m pytest backend/tests -q          # 79 tests
(cd frontend && npm run build)
python backend/scripts/check_chat.py       # stub-mode assistant grounding check
python backend/scripts/check_chat.py --live  # requires Ollama or Gemini

# Rebuild data-quality artifact
python backend/build_data_quality.py       # writes backend/data_quality.json

# Full retrain + five-fold backtest + stress test
python backend/evaluate.py                 # writes backend/results.json

# Phase 9 experiments
python backend/scripts/feature_ablation.py # ablation_results.json
python backend/scripts/level_experiment.py # level_results.json
python backend/scripts/frozen_tuning.py    # tuning_results.json

# Docker (same-origin SPA, backend on 8000, frontend on 3000)
docker compose up --build
# Override ports if needed:
BACKEND_PORT=18000 FRONTEND_PORT=13000 docker compose up -d --build
```

Equivalent Make targets: `make install`, `make test`, `make build`, `make data-quality`,
`make eval`, `make dev-backend`, `make dev-frontend`, `make docker-up`, `make docker-down`.

## Evidence summary

- `python -m pytest backend/tests -q` → **79 passed in 128.88 s**.
- `npm run build` → 49 modules, 412.91 kB (gzip 130.79 kB), built in 10.05 s.
- `python backend/scripts/check_chat.py` → **3/3 passed**, 21 derived values cross-checked against
  the raw artifacts; the same three questions pass in `--live` mode against Ollama
  `llama3.2:3b`, with the December answer opening on the grounded MAE/bias sentence.
- The Limitations page was browser-checked against a live backend at :8004 through the Vite dev
  server: the raw/after-CQR/calibrated-days table, frozen-tuning wording, learned-vs-baseline
  counts, conformal caveat and interval-rule footnote all render from `/api/limitations`, with no
  hardcoded numbers and a clean console. The assistant drawer was exercised in the same session
  and answered the December question with the new metrics.
- Documentation: every aggregate in this report and `README.md` was checked against
  `backend/results.json`; per-fold tables and conformal coverage are reproduced from the artifact in
  `RESULTS.md`.

## Where to look

- `RESULTS.md` — protocol, per-fold metrics, common hours, significance, intervals, experiments.
- `CHECKPOINT_PHASE9.md` — Phase 9 history/feature/tuning checkpoint (through 9.5).
- `CHECKPOINT_DATA.md`, `CHECKPOINT_EVAL.md`, `CHECKPOINT_MODELS.md` — phase checkpoints.
- `DECISIONS.md` — every conservative choice, including dependencies and data handling.
- `PROGRESS.md` — per-item commands and observed output.
- `README.md` — setup, dataset, models, upload flow, API and limitations.
