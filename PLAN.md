# ElectriCast improvement plan
Rules: one task per iteration. Tick [x] only after the VERIFY step passes.

## Phase 0 — Hygiene
- [x] 0.1 Root .gitignore; git rm --cached node_modules, .DS_Store, dist, __pycache__
- [x] 0.2 Pin requirements.txt to installed versions; align local/Docker Python
- [x] 0.3 Remove dead _load_lstm_model(), unused Radar import; move Template.ipynb to notebooks/legacy/

## Phase 1 — Stop the UI lying
- [x] 1.1 Units: Wh vs kWh fixed in every label, axis and title (EDAPage, MetricsBar, MetricsPage, charts)
- [x] 1.2 LSTM: backend emits status "disabled"; frontend filters by status ok; BEST logic ignores non-ok, handles ties, shows N/A
- [x] 1.3 WeeklyPage charts repointed from LSTM to XGBoost
- [x] 1.4 Yearly trend: backend emits daily means; frontend drops the i%3 filter and downsample=24
- [x] 1.5 Split label computed from statistics, not hardcoded 80/20
- [x] 1.6 EDA timezone label IST; header subtitle says IIT Bombay residential buildings
- [x] 1.7 MetricsBar R² color thresholds; UploadPage negative R² shown red
- [x] 1.8 Layout: min-width:0 on grid children, print CSS, compact single MetricsBar
- [x] 1.9 PredictionsPage: no setState in render; real days 1–31 in the day picker
- [x] 1.10 Contrast pass; api.js same-origin base URL; useData with AbortController and visible errors

## Phase 2 — Data pipeline (backend/data_pipeline.py)
- [x] 2.1 Load all 39 files, full date range; count read vs failed; no bare except
- [x] 2.2 Per-apartment hourly resample with sum(min_count=1); keep n_reporting
- [x] 2.3 Inspect the May-2018 :00-UTC file(s); document handling in DECISIONS.md
- [x] 2.4 Target = mean load per reporting apartment; build stable panel (>90% coverage)
- [x] 2.5 Gaps: interpolate only <=6h; longer gaps flagged and excluded from train and eval
- [x] 2.6 Outlier cap fitted on train only
- [x] 2.7 Flag suspect_inactive apartments (December mean <100 Wh/h)
- [x] 2.8 Write data_quality.json and a Data Quality page

## Phase 3 — Evaluation (backend/evaluate.py)
- [x] 3.1 Baselines: persistence, seasonal naive 24h, same-hour-last-week, mean of last 4 same weekdays
- [x] 3.2 Metrics: RMSE, MAE, WAPE, MASE, bias, R²; guarded sMAPE
- [x] 3.3 Day-ahead protocol + 4+ fold expanding-window backtest (one fold May–Jun 2018, one fold Sep–Dec 2017)
- [x] 3.4 results.json per fold/model/metric; paired significance vs best baseline

## Phase 4 — Models
- [x] 4.1 XGBoost day-ahead with lags >= 24h (or direct per-horizon); leakage pytest
- [x] 4.2 Calendar + holiday + semester flags + n_reporting features
- [x] 4.3 Weather features from Open-Meteo, cached to disk
- [x] 4.4 Optuna tuning on time-series CV inside train; early stopping on a train-side validation slice
- [x] 4.5 SARIMA/SARIMAX rolling day-ahead with weekly refit
- [x] 4.6 Quantile XGBoost P10/P50/P90 with coverage and pinball loss
- [x] 4.7 LSTM/GRU done properly, or omitted from the UI entirely

## Phase 5 — Dashboard
- [x] 5.1 Model Metrics page: fold mean ± std table, "beats seasonal naive?" column
- [x] 5.2 Predictions page: P10–P90 band, fold day picker, error by hour and month, worst-days table
- [x] 5.3 Explain page: SHAP or gain importance, residual plots
- [x] 5.4 Limitations panel

## Phase 6 — Upload & Validate redesign
- [x] 6.1 Column mapper, unit and timezone selectors, validation report
- [x] 6.2 Run the Phase 3 protocol on uploaded data (baselines, zero-shot, fit-on-60%)
- [x] 6.3 Streaming size check, .CSV accepted, progress/error states, results download

## Phase 7 — Engineering
- [x] 7.1 pytest suite: metrics, split, leakage, gaps
- [x] 7.2 Makefile; Docker fixes (models/ copied, no-curl healthcheck, same-origin API)
- [x] 7.3 README, RESULTS.md, refreshed copilot-instructions.md

## Phase 8 — Grounded assistant
- [x] 8.1 "Ask ElectriCast" chat panel: POST /api/chat with read-only artifact tools, numeric grounding guard, configurable Ollama/Gemini backend, drawer UI, tests and docs

## Phase 9 — Use all history and harder evaluation
- [x] 9.1 Use all valid history: train_start = earliest valid target hour (Dec 2016) for every fold; report 2016-12-14 → 2017-04-30 valid hours and December vacancy pattern; rerun all folds; keep `results_v1.json`
- [x] 9.2 Add Jun 2017 fold (train Dec–May), keep Jul, Aug, Sep–Nov, make Dec 2017 its own fold, relabel 2018 as stress test and exclude it from fold means
- [x] 9.3 Feature ablation on the fixed config: drop month/day-of-year when training <12 months, drop weather, oracle same-hour temperature (ORACLE only); MAE per variant per fold; keep variants helping ≥3 folds
- [x] 9.4 Level-normalized target: forecast target / trailing 7-day mean, multiply back; add lag-24 near-zero stable-apartment fraction; compare with 9.3 winner
- [x] 9.5 Tune once on the largest training window with time-series CV, freeze parameters across folds; report whether tuning helps
- [x] 9.6 Common-hours table: every model on intersected forecast hours (including SARIMA), per fold, beside existing per-model tables
- [x] 9.7 Replace hourly Wilcoxon with day-level test: daily MAE, paired day Wilcoxon + block-bootstrap CI vs best baseline; update RESULTS.md and Metrics page
- [x] 9.8 Conformal P10–P90 calibration from past residuals (rolling prior-day window); coverage before/after per fold
- [x] 9.9 Update RESULTS.md, README, Limitations page, FINAL_REPORT.md and assistant with new numbers; state plainly if no learned model beats the best baseline on common hours