# Progress log

One entry per iteration: date, item id, change, evidence, numbers.

## 2026-10-23 — 0.1 Root .gitignore; untrack generated artifacts

- Added root `.gitignore` covering Python caches, `node_modules/`, `dist/`, `.env`, OS files, and test/coverage output.
- Removed from the Git index (working-tree files kept): `frontend/node_modules/` (2,389 files), `.DS_Store`, `backend/__pycache__/main.cpython-313.pyc`.
- Evidence: `git ls-files` 2,466 → 75 tracked files; `git status --short` no longer lists `node_modules`, `.DS_Store`, or `__pycache__`; the files still exist on disk; `git check-ignore -v` reports the new rules.
- Commit: `0.1: add root gitignore and untrack generated artifacts`.

## 2026-10-23 — 0.2 Pin requirements; align Python versions

- `backend/requirements.txt`: all 9 direct dependencies now exact `==` pins matching the installed environment (fastapi 0.111.0, uvicorn 0.30.1, numpy 2.5.4, python-multipart 0.0.9, pandas 3.1.0, scikit-learn 1.9.1, statsmodels 0.15.0, xgboost 3.4.1, torch 2.14.1).
- `backend/Dockerfile`: `python:3.11-slim` → `python:3.12.10-slim`, matching local Python 3.12.10.
- Evidence: parser + `importlib.metadata` check printed `OK` for all 9 pins and `ALL_PINS_MATCH`; `pip install --dry-run --no-deps -r` reported "Requirement already satisfied" for every line; Docker tag equals `python:{local major.minor.micro}-slim` (`ALIGNED`). Docker image build not run (no cached image; network limited to pip/npm).

## 2026-10-23 — 0.3 Remove dead LSTM loader/imports; move notebook

- Deleted unused `_load_lstm_model()` from `backend/main.py` plus its now-unused `pickle`, `MinMaxScaler`, and `LSTM_WINDOW` symbols (36 lines removed).
- Removed unused `Radar` / `RadialLinearScale` imports and registration from `MetricsPage.jsx`.
- Moved `Template.ipynb` (6,302,153 B) to `notebooks/legacy/Template.ipynb` with `git mv`.
- Evidence: grep for removed symbols returns no hits in runtime code (`LSTM_WINDOW` remains only in `train_models.py` where it is written into stats); backend import + TestClient GETs `/`, `/api/metrics`, `/api/eda/statistics` all returned 200; `npm run build` succeeded (44 modules, built in 1.58s).

## 2026-10-23 — 1.1 Wh/kWh labels corrected

- Replaced every UI `kWh` label with `Wh` (values are Wh): EDAPage stats/title/axis/dataset labels, MetricsBar RMSE/MAE, MetricsPage score badges/titles/table, PredictionsPage subtitles, WeeklyPage chart titles, UploadPage metric/table/chart labels.
- Backend upload `scale_note` now states `Wh` for upload/train means.
- Evidence: `grep -rn kWh frontend/src` → none (only remaining backend hits are the input CSV column name); production bundle rebuilt successfully; TestClient upload of `smart_home_dataset.csv` returned 200 with `scale_note: ... upload mean: 54.8 Wh, train mean: 7400.2 Wh`.

## 2026-10-23 — 1.2 LSTM status filtering and BEST logic

- Backend: `load_results()` injects `status` (`lstm=disabled`, `sarima/xgboost=ok`) into `/api/metrics`; `train_models.py` writes the same statuses for future regenerations.
- Frontend: MetricsBar and MetricsPage show only `status === "ok"` models; BEST values are computed over active models only; non-finite values render N/A; ties mark all tied models. Static LSTM header badge/footer text removed.
- Evidence: TestClient `/api/metrics` → `lstm: status=disabled ... sarima: status=ok ... xgboost: status=ok`; `npm run build` passed (44 modules, 2.00s); live browser at :3000 showed MetricsBar with only SARIMA and XGBoost, Metrics page had no LSTM text and exactly 4 BEST badges, all won by XGBoost (R² 0.8238, RMSE 1968.7, MAE 1461.2, sMAPE 40.33).

## 2026-10-23 — 1.3 WeeklyPage uses XGBoost

- Removed the LSTM dataset from the selected-day profile, switched the all-days overlay and the weekday Actual-vs-model bars to XGBoost, and retitled both charts.
- Evidence: `grep -n "lstm\|LSTM" WeeklyPage.jsx` → no hits; `npm run build` passed (44 modules, 1.56s); live browser at :3000 on Weekly Pattern rendered `MON — AVERAGE 24H LOAD PROFILE (WH)`, `ALL DAYS OVERLAY — XGBOOST PREDICTIONS (WH)`, `AVERAGE DAILY LOAD BY WEEKDAY — ACTUAL VS XGBOOST (WH)` with no LSTM text.

## 2026-10-23 — 1.4 Yearly trend is daily means

- Derived 365 daily means from `data/*.csv` using the existing 2017 pipeline (UTC→IST, hourly sum, mean+3σ cap, zero→NaN linear interpolation, median fill) and added `eda.yearly_daily` to `model_results.json`. No retraining; existing `metrics`/`predictions`/other EDA structures verified identical to HEAD.
- `/api/eda/yearly` now returns `yearly_daily` with no downsample parameter; `train_models.py` emits the same structure for future runs; frontend fetches `/api/eda/yearly` and plots all 365 labels/values (removed `i % 3` and `downsample=24`).
- Numbers: 365 points, first `2017-01-01 = 4909.8 Wh`, last `2017-12-31 = 664.5 Wh`; daily mean 7108.82 Wh equals hourly mean 7108.82 Wh.
- Evidence: TestClient `/api/eda/yearly` → 200, 365 points, no `HH:MM` labels; `grep -rn "i % 3\|downsample"` → none; `npm run build` passed (44 modules, 1.63s); live browser showed `Yearly Load Trend` / `Daily mean electricity consumption trend (Wh)`.

## 2026-10-23 — 1.5 Split label computed from statistics

- PredictionsPage now fetches `/api/eda/statistics` and renders train/test percentages and hour counts from backend data; the hardcoded "80% / 20%" is gone.
- Numbers: 4,344 train / 4,416 test hours → 49.6% / 50.4%.
- Evidence: statistics endpoint returned train 4344, test 4416 (`49.6`, `50.4`); `npm run build` passed; production bundle served at :3100 and the live page rendered `Train / Test split: 49.6% / 50.4% (4,344 / 4,416 h)`.

## 2026-10-23 — 1.6 Timezone and header subtitle

- EDAPage overview now says `timezone: IST (Asia/Kolkata)` with an IIT Bombay fallback dataset name; header subtitle now reads `IIT Bombay Residential Buildings · Energy Forecasting`.
- Evidence: production build contained one `timezone: IST` and no `timezone: UTC`; live page at :3100 showed the new header and IST subtitle (`header: true`, `ist: true`, `utc: false`).

## 2026-10-23 — 1.7 R² color thresholds

- MetricsBar R²: green at ≥0.8, red only when negative, neutral otherwise (was green only >0.85, so the real 0.8238 was shown red).
- UploadPage R² card is red when the uploaded-data score is negative, green otherwise.
- Evidence: live computed styles — XGBoost `0.8238` → `rgb(22,163,74)` green, SARIMA `-1.3261` → `rgb(220,38,38)` red; uploading `smart_home_dataset.csv` through the real file input produced R² `-0.6799` rendered `rgb(220,38,38)`. Build passed (44 modules, 1.51s).

## 2026-10-23 — 1.8 Layout fixes and compact metrics bar

- `MetricsBar` is now a compact single-line card per model (inline R²/RMSE/MAE/sMAPE, description moved to the card `title`); only `status === "ok"` models render and the loading skeleton shrank from 160px to 52px.
- `index.css` adds `min-width: 0` to grid children (chart canvases can no longer widen their track) and a print stylesheet that hides header/footer/buttons, removes shadows/padding, and prevents cards breaking across pages.
- Evidence: production build passed (44 modules, CSS 0.92 kB); built CSS contains one `@media print` block; live page at :3100 measured each metric card at 48px high (previously 160px) with SARIMA −1.3261 red and XGBoost 0.8238 green; `main` scrollWidth 600 = viewport 599 with no overflowing descendants; screenshot confirms compact two-card bar above Dataset Overview.

## 2026-10-23 — 1.9 PredictionsPage render state and real day picker

- Removed the render-time `setSelectedMonth(...)`; the effective month is now `selectedMonth ?? testMonths[0]`, and the hourly fetch keys off that derived value.
- Day options come from the test-period `predDaily.labels` for the selected month (Jul/Aug/Oct/Dec 31, Sep/Nov 30) instead of a fixed 28; if the previous day does not exist in the new month, it clamps to the last available day.
- Evidence: production build passed (44 modules, 1.44s); live at :3100 — Jul options 1–31 default 15; selecting 31 then Sep gives options 1–30 and value 30; network logged `hourly?month=Jul&day=15`, `Jul&day=31`, `Sep&day=30`; `/api/predictions/hourly?month=Sep&day=30` returned 24 hours dated `Sep 30, 2017`; Vite dev build showed only the standard React DevTools message, no state-update-in-render warning.

## 2026-10-23 — 1.10 Same-origin API, abortable requests, visible errors, contrast

- `api.js` defaults to a same-origin base (relative `/api/...`); Vite dev and preview both proxy `/api` to the backend; `VITE_API_URL` still overrides. `get()` now surfaces backend `detail`/`error` messages.
- `useData` creates an `AbortController`, passes its signal to every fetcher, aborts on unmount/dependency change, and ignores abort rejections; all pages plus MetricsBar render a new `DataError` alert on failure.
- Contrast pass: muted text `#9ca3af`→`#4b5563` (2.31–2.54 → 6.87–7.56), green text `#16a34a`→`#15803d` (3.00–3.30 → 4.56–5.02), red small text on tints→`#b91c1c`; chart tick labels darkened in `chartConfig`.
- Evidence: build passed (45 modules, 1.77s); preview at :3101 loaded with all API requests same-origin (`/api/...` on localhost:3101) and zero alerts; computed-style WCAG scan found 0 failures across EDA (48 text nodes), Predictions (36), Weekly (28), Metrics (60), Upload (30); stopping the backend showed `Could not load model metrics: API error 500: /api/metrics` and `Could not load EDA data: API error 500: /api/eda/hourly`; dev StrictMode run recorded aborts for `/api/predictions/{monthly,daily,monthly_errors}` and `/api/eda/statistics`, then refetched successfully with no console warnings.

## 2026-10-23 — 2.1 Raw data loader

- Added `backend/data_pipeline.py` with `load_apartments()`: reads every `data/*.csv`, validates `TS,W1,W2,W3`, parses UTC timestamps, and returns a per-file report of rows and IST/UTC ranges.
- Missing/empty/malformed/wrong-schema files are recorded as failures with their exception type; there is no bare or swallowing `except`.
- Evidence: 39/39 files read, 0 failed, 196,449 rows, range 2016-12-14 15:30 UTC → 2018-06-04 18:00 UTC (2016-12-14 21:00 → 2018-06-04 23:30 IST); a temp-directory test with empty, wrong-column and broken CSV files returned `read_ok=1, failed=3` with `ValueError/ParserError` entries and no crash.

## 2026-10-23 — 2.2 Per-apartment hourly panel

- `apartment_hourly()` sums W1–W3 per row with `min_count=1`, converts UTC to IST, sorts, and resamples to hourly with `sum(min_count=1)`; `build_panel()` combines all 39 apartments and keeps `n_reporting` per hour.
- Evidence: panel is 12,891 hours × 39 apartments (2016-12-14 21:00 → 2018-06-04 23:00 IST); every apartment's non-null hourly count equals its raw row count (no bucket collisions); `n_reporting` ranges 0–39 (median 23; 26 reporting for 4,085 hours); 5,291 hours have zero apartments, reflecting the Dec-2017→May-2018 hole; 2017-11-26 03:00 IST has 39 reporting apartments with mean 82.34 Wh.

## 2026-10-23 — 2.3 May-2018 UTC-offset blocks

- Inspected all 39 files: 187,796 rows are stamped at :30 UTC (= exact IST hours) while 8,653 rows are at :00 UTC; 38 files contribute a May-2018 :00-UTC block (7,460 rows, 2018-05-23 19:00 → 2018-05-31 23:00 UTC). `13.csv` is only 37 rows from Nov 2017.
- Added `ist_hour_index()` (UTC→IST, floor to containing hour) and documented the decision: :00-UTC rows fall at IST :30 and join the containing IST hour; no apartment-hour bucket is duplicated (0 across all files).
- Evidence: after the explicit floor the panel is unchanged (12,891 × 39, max 39 reporting, median 23, 5,291 zero-report hours, checksum 63,669,648.7 Wh).

## 2026-10-23 — 2.4 Stable panel and target

- Added `apartment_coverage`, `stable_apartments`, and `build_target`. Stable window is 2017-05-01→2017-11-30 (the longest dense block); 24/39 apartments clear 90% coverage (90.1–91.5%), 15 are dropped.
- Target is the per-hour mean of reporting stable apartments; `n_reporting` counts them. Numbers: 7,600 non-null target hours of 12,891, mean 335.54 Wh, range 0.14–2,001.12 Wh; manual check at 2017-11-26 03:00 IST gives 24 reporters, mean 123.48 Wh = target. Fold windows have 2,370 usable hours (Sep–Dec 2017) and 214 (May–Jun 2018).

## 2026-10-23 — 2.5 Short-gap interpolation

- Added `MAX_GAP_HOURS = 6` and `fill_short_gaps()`, which records every NaN run, linearly interpolates interior runs of ≤6 h, and marks longer or edge runs in `excluded`; returns `(filled, imputed, excluded, runs)`.
- Evidence: synthetic 48-h series with a 3-h, a 10-h, and two edge gaps → exactly 3 h imputed, 12 h excluded, 4 runs recorded, filled values [10, 11, 12] between neighbours 9 and 13. Real 24-apartment target: 66 gap runs, longest 3,514 h; 83 h imputed, 5,208 h excluded, 7,683 valid hours remain (up from 7,600). Asserted no long/edge gap received a value and no hour is both imputed and excluded.

## 2026-10-23 — 2.6 Train-only outlier cap

- Added `fit_outlier_cap()` (Q3 + 3×IQR, minimum 100 samples) and `apply_outlier_cap()`, which clips only the upper tail and reports how many values were capped; the fitter accepts only the training slice by design.
- Evidence: with a provisional train slice of valid hours before 2017-10-01 (5,543 h), cap = 1,006.72 Wh; full-data cap would have been 1,070.31 Wh. Four hours exceed the cap (max 2,001.12 Wh on 2017-02-05 11:00 IST) and are clipped to 1,006.72; NaNs preserved. Mutating all post-October values to 1e6 leaves the train-fitted cap unchanged.

## 2026-10-23 — 2.7 Suspect-inactive apartments

- Added `apartment_activity()`, which computes each apartment's mean load and reading count for an IST month and flags `suspect_inactive` when the mean is <100 Wh/h with at least 24 readings.
- Evidence: using December 2017 (38/39 apartments reporting; data ends 2017-12-28), 21 apartments are flagged, including 13 of the 24 stable apartments; apartment 13 has no December data and is not flagged, and apartment 11 at 100.50 Wh/h is just above the boundary. Synthetic check confirms 99.9 flagged, 100.0/100.1 not, and a single-reading apartment not flagged.

## 2026-10-23 — 2.8 Data-quality report and page

- Added `build_data_quality.py` (writes committed `backend/data_quality.json`), `GET /api/data_quality`, the **Data Quality** page, and a responsive header fix so the sixth nav item scrolls instead of overflowing the page.
- Evidence: JSON reports 39/39 files, 196,449 rows, 12,891 panel hours, 24 stable apartments, 7,683 valid target hours, 5,208 excluded, 21 flagged inactive; page renders 6 KPI cards, 3 tables (70 rows), and a 498×280 chart with no console errors; backend stopped shows `Could not load data quality: API error 500: /api/data_quality`; body scrollWidth 600 vs viewport 604 after the nav fix. Screenshot capture was unavailable (in-app browser viewport), so verification used DOM/API checks.
- Wrote `CHECKPOINT_DATA.md` with the Phase 2 numbers, fold-window counts, regeneration commands, and remaining data risks.

## 2026-10-23 — 3.1 Baselines

- Added `backend/evaluate.py` with day-ahead baselines: persistence (last valid value before 00:00 IST, held flat), seasonal naive 24 h, same-hour-last-week (168 h), and mean of the same hour on the previous four same weekdays (all four required).
- Evidence: synthetic ramp shows persistence errors 1→24 across a day, seasonal-naive error 24, weekly error 168, four-week mean error 420; constant series yields zero errors; a missing previous day makes seasonal naive NaN while persistence falls back to the last valid value; a missing same weekday makes the four-week mean NaN. On July 2017 (744 h), MAE: persistence 119.56, seasonal naive 75.01, last-week 95.00, four-week mean 86.94 Wh.

## 2026-10-23 — 3.2 Metrics

- Added `compute_metrics()` (RMSE, MAE, WAPE, MASE, bias, R², guarded sMAPE) and `mase_scale()`, which uses the mean absolute 24-h seasonal difference over valid training pairs. Pairs are aligned on timestamps and dropped when actual or prediction is missing.
- Evidence: hand-checked example [10,20,30] vs [12,18,33] gives MAE 2.3333, RMSE 2.3805, WAPE 11.6667%, sMAPE 12.744%, bias 1.0, R² 0.915, MASE 0.4667 (scale 5); perfect forecasts give zeros and R²=1; all-zero actuals give sMAPE 0 with WAPE/R²/MASE null; NaNs reduce n correctly; a ramp has MASE scale 24. July 2017 baseline table: seasonal naive MAE 75.01, R² 0.2375 vs persistence MAE 119.56, R² -0.7352 (the fold-window MASE scale is defined in 3.3).

## 2026-10-23 — 3.3 Day-ahead backtest

- Added `FOLDS`, `PROTOCOL`, `evaluate_fold()` and `run_backtest()`: four expanding folds with train_start fixed at 2017-05-01 — test Jul 2017, Aug 2017, Sep–Dec 2017, and May–Jun 2018. Forecast origin is 00:00 IST each test day; the evaluation mask requires a target and a seasonal-naive-24 forecast, so all models share the same hours within a fold.
- Evidence: eval hours are 744 / 744 / 2,243 / 182; target-valid hours are 744 / 744 / 2,374 / 214. Best baseline by MAE is seasonal naive in Jul (75.01), mean-of-4 in Aug (58.93), persistence in Sep–Dec (116.21) and seasonal naive in May–Jun (132.87). Replacing every value from a test day's origin onward with 1e9 leaves that day's forecasts unchanged for all four baselines (6 sample days); fold boundaries satisfy train_end < test_start and expand monotonically.

## 2026-10-23 — 3.4 Fold results and paired significance

- Extended the backtest to write `backend/results.json`: per fold the train/test windows, MASE scale, all four baseline metrics on the shared evaluation mask, the best baseline, the selection mask and its metrics, and a paired Wilcoxon test of every other baseline against the best one.
- Best baseline by MASE: seasonal naive in Jul 2017 (0.887), mean-of-four-weekdays in Aug 2017 (0.725), persistence in Sep–Dec 2017 (1.292 on the 1,290-h four-baseline mask), seasonal naive in May–Jun 2018 (1.294). The four-week mean has no usable 2018 hours (all four same-weekday lags fall in the coverage gap), so it is reported as `status: no_data` with null metrics rather than omitted.
- Paired tests: Jul — best beats all three others (p ≤ 0.0031); Aug — p ≤ 2.9e-9; Sep–Dec — seasonal naive's median absolute error is 0.90 Wh lower than persistence but not significant (n = 2,243, p = 0.4068), while the other two are significantly worse; May–Jun — best beats persistence (n = 182, p = 2.0e-13) and last-week (n = 38, p = 4.8e-8).
- Edge cases verified: identical predictions → `n_pairs=30`, `reason="all paired differences are zero"`; a model with a uniform 0.5-Wh-lower absolute error over 30 pairs → `model_better=True`, p = 4.3e-8; 5 pairs → `small_sample=True`; disjoint indexes → `n_pairs=0`, `reason="no paired hours"`. A schema assertion checks all four fold ids, the nine metric keys per model, the paired keys, and that every float in the file is finite.
- Wrote `CHECKPOINT_EVAL.md` with the protocol, fold table, selection rules, and limitations.

## 2026-10-23 — 4.1 XGBoost day-ahead model

- Added `backend/forecast_model.py`: one pooled direct model per fold on `lag_24`…`lag_47`, `lag_168`, `lag_336`, four aggregates of the 24–47 h block, and hour of day. Every feature is at least 24 h old, so no feature reads the forecast day. Models are fit on the fold's training labels only; labels are clipped at the train-only Q3+3×IQR cap, and negative outputs are clipped to 0.
- Added `backend/tests/test_leakage.py` with five tests (pytest 8.4.2, newly pinned in `requirements.txt`): all lags are ≥24 h; the feature frame for a test day is unchanged when every value from that day's origin onward is replaced with 1e9; all four baselines are unchanged under the same mutation; XGBoost predictions are unchanged; and training labels end before the test day. `python -m pytest backend/tests -q` → 5 passed.
- Backtest (same four folds and evaluation mask): XGBoost MAE 65.95 in Jul (best baseline 75.01; paired median −6.85 Wh, p = 6.5e-05), 62.35 in Aug (best mean-of-4 58.93; +0.78 Wh, p = 0.168), 151.01 in Sep–Dec (persistence 116.21; +10.07 Wh, p = 2.2e-16), and 142.59 in May–Jun (seasonal naive 132.87; +11.69 Wh, p = 0.254). The first learned model beats the best baseline significantly only in the July fold; the autumn fold has a +69.88 Wh bias after training on May–Aug.
- `backend/results.json` now records the XGBoost metrics, fit metadata (`n_train`, cap, number capped, params, feature list), and its paired comparison in every fold.

## 2026-10-23 — 4.2 Calendar, holiday, semester and reporting features

- Extended `make_features()` with day of week, weekend, month, day-of-year sine/cosine, a holiday flag, a semester-in-session flag, and two reporting-count features lagged 24 h (`n_reporting_lag_24` and the mean of the 24–47 h block). `evaluate.py` now loads the stable-apartment reporting count and passes it through fit/predict; baseline forecasts and the evaluation mask are unchanged.
- Holiday list covers the data period (2016-12-25 through 2018-04-30) with major national holidays and festivals; the semester rule is Jan–Apr and Jul–Nov in session, May–Jun and Dec breaks. Both are documented approximations, not authoritative calendars.
- Leakage tests grew to 8: reporting features and XGBoost predictions stay unchanged when reporting counts from the forecast day are replaced, and holiday/weekend/semester flags land on the expected dates. `python -m pytest backend/tests -q` → 8 passed.
- Backtest: MAE 66.46 Jul (was 65.95; still beats seasonal 75.01, p = 6.1e-04), 61.55 Aug (was 62.35; 2.6 Wh behind mean-of-4, p = 0.402), 163.22 Sep–Dec (was 151.01; 47 Wh behind persistence, p = 1.6e-20), 100.40 May–Jun (was 142.59; now beats seasonal 132.87, p = 0.023). Calendar features help the year-later fold and hurt the unseen-autumn fold. The feature set is frozen here; any tuning happens on train-side CV in 4.4, not on these test folds.

## 2026-10-23 — 4.3 Open-Meteo weather features

- Added `backend/weather.py` and committed `backend/weather_mumbai.csv`: 11,688 IST hours from 2017-03-01 to 2018-06-30, six variables (temperature, relative humidity, apparent temperature, precipitation, cloud cover, wind), zero missing values, for IIT Bombay (19.1334 N, 72.9133 E) per the dataset provenance. Fetching is explicit (`python backend/weather.py`); `evaluate.py` reads only the cache, so backtests run offline.
- Weather enters the model only as lags ≥24 h: six same-hour-previous-day values, the 24–47 h mean temperature, and the 24–47 h precipitation sum. Target-hour weather would not be known at the day-ahead origin, so it is never used.
- Leakage tests grew to 10: weather feature frames and XGBoost predictions stay unchanged when weather from the forecast day is replaced. `python -m pytest backend/tests -q` → 10 passed.
- Backtest: MAE 65.69 Jul (best baseline 75.01, p = 5.6e-05), 62.24 Aug (mean-of-4 58.93, p = 0.105), 170.40 Sep–Dec (persistence 116.21, p = 1.2e-23), 106.47 May–Jun (seasonal 132.87, p = 0.254). Weather slightly helps July and hurts Sep–Dec; features stay in place and 4.4 tunes only on train-side CV.

## 2026-10-23 — 4.4 Optuna tuning and early stopping

- Added `time_series_splits()`, `tune()` and `_fit_tuned()`: 30 Optuna TPE trials per fold (seed 42) scored on three expanding-window CV splits inside the fold's training window; validation blocks are contiguous, non-overlapping, and end at the last training day. Early stopping (patience 25) runs on those train-side slices, and the final tree count is the median best iteration before refitting on all training labels. Optuna 5.0.0 is pinned in `requirements.txt`.
- Tests grew to 12: CV splits stay inside the training window and never overlap, and a tuned forecast is unchanged when every value from the test-day origin onward is replaced. `python -m pytest backend/tests -q` → 12 passed (19 s).
- Tuned backtest: XGBoost MAE 69.31 Jul (best baseline 75.01, p = 0.0215), 59.51 Aug (mean-of-4 58.93, p = 0.844), 167.27 Sep–Dec (persistence 116.21, p = 2.3e-22), 120.75 May–Jun (seasonal 132.87, p = 0.518). Tuning improved Aug (62.24 → 59.51) but worsened Jul (65.69 → 69.31) and May–Jun (106.47 → 120.75); the tuned model significantly beats the best baseline only in July.
- Wrote `CHECKPOINT_MODELS.md` with the feature set, tuning table, fold results, iteration history, leakage evidence, and limitations.








## 2026-10-23 — 4.5 Rolling SARIMA with weekly refit

- Added `backend/sarima_model.py` and registered `sarima` as a second learned model in `evaluate.py`. The model is SARIMA(1,0,1)(0,1,1,24) on the gap-handled target, refit on Mondays, the first test day, and after a coverage gap from the trailing 60 days (1,440 h) of contiguous history ending one hour before the origin. Within a week its state is advanced with actuals up to each day's origin, never with forecast-day values. A refit needs 7 contiguous days, forecasts are skipped when the state is more than 48 h stale, and negative forecasts are clipped to zero.
- Why the two guards: with the original 14-day minimum plus no staleness limit, the 2018 fold had no forecasts at all; lowering the minimum alone let a five-month-stale state emit 214 hours at MAE 249.5 Wh. The 7-day minimum plus the 48 h staleness limit yields a small but genuine 2018 forecast (96 h predicted, 38 h evaluable) instead of either nothing or a misleading full fold.
- Full backtest, evaluation mask (SARIMA hours in parentheses): SARIMA MAE 65.66 Wh in Jul (744 h; best baseline seasonal naive 75.01; paired median −6.61 Wh, p = 1.2e-04), 60.60 in Aug (744 h; mean-of-4 58.93; +0.42 Wh, p = 0.830), 104.23 in Sep–Dec (1,554 of 2,243 h; persistence 116.21; −0.43 Wh, p = 2.9e-03), and 93.80 in May–Jun (38 of 182 h; seasonal naive 132.87; −9.36 Wh, p = 0.025). XGBoost stays at 69.31 / 59.51 / 167.27 / 120.75 Wh. SARIMA is the stronger learned model in three folds, but its Sep–Dec advantage is a near-tie on the median and its 2018 sample is only 38 hours.
- Coverage is a real limitation: the Sep–Dec fold has 689 evaluable hours without a SARIMA forecast because the target gaps outrun the state, and the 2018 fold has 144. Metrics use each model's available hours, as with the lagged baselines; paired tests use overlapping hours, and coverage is recorded in `results.json`.
- Tests grew to 15: contiguous history stops at the last gap and rejects runs shorter than the minimum; SARIMA predictions are unchanged when every value from the forecast-day origin onward is replaced; a state older than 48 h emits no forecasts. `python -m pytest backend/tests -q` → 15 passed (19 s).
- Regenerated `backend/results.json` (now includes `sarima` metrics and fit metadata) and updated `CHECKPOINT_MODELS.md` through item 4.5.

## 2026-10-23 — 4.6 Quantile XGBoost P10/P50/P90

- Added `backend/quantile_model.py`: three XGBoost models with objective `reg:quantileerror` and alphas 0.1/0.5/0.9, sharing the point model's feature set, 24 h+ lags, training window, and train-only outlier cap. Hyperparameters are the fixed 4.1 defaults (400 trees, depth 4, learning rate 0.03), untuned. Predictions are clipped at zero and sorted row-wise when raw quantiles cross (zero crossings in Jul–Sep, one row in May–Jun); the P50 is also scored as a point forecast on the shared evaluation mask.
- Added `interval_metrics()` (P10–P90 coverage, mean width, pinball loss at P10/P50/P90 and their mean) and an `intervals` block to every fold in `results.json`. Protocol and learned-model descriptions were extended; no baseline, mask, or selection rule changed.
- Point results for the quantile P50: MAE 65.99 Wh in Jul (744 h; paired median −7.25 Wh vs seasonal naive, p = 6.4e-05), 60.58 in Aug (744 h; 0.00 Wh median, p = 0.607), 159.14 in Sep–Dec (2,243 h; +10.92 Wh, p = 2.5e-19), and 113.94 in May–Jun (182 h; −3.52 Wh, p = 0.309). It is better than the tuned point XGBoost in three folds but remains 43 Wh behind persistence in the shifted autumn fold.
- Interval quality: 80% nominal coverage came out at 71.6% (Jul), 78.5% (Aug), 44.8% (Sep–Dec), and 69.2% (May–Jun), with mean widths 177 / 177 / 211 / 229 Wh and mean pinball losses 21.73 / 19.87 / 61.88 / 35.25 Wh. The autumn fold is badly undercovered because the training window contains no autumn regime; this is reported as a limitation rather than fixed with test data.
- Tests grew to 17: a hand-checked coverage/pinball example (including a missed interval) and a quantile leakage test (predictions unchanged under forecast-day mutation, P10 ≤ P50 ≤ P90, no NaNs). `python -m pytest backend/tests -q` → 17 passed (18 s).
- Regenerated `backend/results.json` and updated `CHECKPOINT_MODELS.md` through item 4.6.

## 2026-10-23 — 4.7 Omit LSTM/GRU from the UI

- Took the "omitted entirely" branch: there is no trained LSTM or GRU artifact, only zero-filled placeholder data, so presenting it as a model would violate the no-placeholder rule. Training a proper deep model would need its own leakage-safe protocol and was out of scope for this item.
- Removed the remaining LSTM surfaces: MetricsBar model config/active list, MetricsPage model list, PredictionsPage model colors, visibility toggles, line datasets, monthly MAE/RMSE series, chartConfig color/meta entries, and the EDA heatmap copy that referenced LSTM window design. `grep -ri "lstm\|gru" frontend/src` → no matches.
- Evidence: `npm run build` passed (46 modules, 1.99 s). A fresh Vite server on :3005 pointed at the current backend on 127.0.0.1:8000 (the pre-existing :3000 preview and 0.0.0.0:8000 server were stale and left untouched) showed: Metrics page with only SARIMA and XGBoost and all four BEST badges on XGBoost; Predictions with Actual/SARIMA/XGBoost toggles and five canvases; Weekly with XGBoost-only overlay captions; no LSTM/GRU text on EDA, Predictions, Weekly, or Metrics.
- Backend response keys are unchanged for schema compatibility (LSTM remains `status: disabled`), but no UI path reads them. README and copilot instructions still describe LSTM and will be corrected in 7.3.

## 2026-10-23 — 5.1 Fold-level metrics in the dashboard

- Added `/api/backtest/summary` in `backend/main.py`, built at startup from the committed `backend/results.json`. It reports every baseline and learned model with per-fold metrics, an unweighted fold mean and sample standard deviation (ddof=1), the per-fold n, and a descriptive majority answer to "lower MAE than seasonal naive". The endpoint returns 503 with an explicit message when the backtest artifact is missing.
- Replaced the single-split `MetricsPage` with a four-fold view: protocol fields from the backend, fold chips with evaluation hours, a mean ± std table (MAE, RMSE, MASE, beats-naive, fold count) and a per-fold MAE table with n. The old score badges and four bar charts were removed because they mixed one split with no fold variability. `MetricsBar` now reads the same summary and shows all three learned models with their fold mean and MAE σ.
- All numbers come from the endpoint; no result is written into JSX. The page labels the beats column as a descriptive majority rule and points to the paired Wilcoxon output for significance.
- Evidence: live backend on 127.0.0.1:8010 returned 7 models and 4 folds; the endpoint's means match `results.json` (SARIMA MAE 81.1 ± 21.3, beats 4/4; XGBoost 104.2 ± 49.9, 3/4; quantile P50 99.9 ± 46.2, 3/4). A fresh Vite server on :3007 proxying that backend rendered both tables (14 rows) and the metrics bar with no console errors; the initial protocol-object render error found during this check was fixed before the build. `npm run build` passed (46 modules, 1.57 s).

## 2026-10-23 — 5.2 Fold-based Predictions page

- Extended `backend/evaluate.py` to persist the hourly evaluation rows in each fold of `results.json`: local IST timestamp, actual, seasonal-naive reference, point XGBoost, SARIMA, and quantile P10/P50/P90. Rows are limited to the existing evaluation mask (actual and seasonal-naive available), so no new data is exposed. The regenerated artifact holds 3,913 rows (744 / 744 / 2,243 / 182) and reproduces the earlier metrics exactly (SARIMA MAE 65.66 / 60.60 / 104.23 / 93.80 Wh; same coverage and pinball values).
- Added backend endpoints built once at startup: `/api/backtest/days` (folds and evaluable days), `/api/backtest/day?date=` (24-slot arrays for actual, seasonal naive, XGBoost, SARIMA, P10/P50/P90 plus per-model day MAE and n), and `/api/backtest/errors` (MAE by hour of day, MAE by calendar month, and top-10 worst days ranked by XGBoost MAE with at least 12 evaluable hours). Endpoints return 503 when the hourly artifact is missing.
- Rebuilt `PredictionsPage` around the backtest: fold chips plus a dated day picker, a day chart with the P10–P90 band and model toggles, per-model day MAE cards, MAE-by-hour and MAE-by-month bar charts, and the worst-days table with per-model n tooltips. The old single-split monthly/daily charts were removed from this page so it no longer mixes protocols; `WeeklyPage` and the legacy `/api/predictions/*` endpoints are unchanged.
- Evidence: live backend on 127.0.0.1:8011 returned 4 folds with 31 / 31 / 102 / 8 evaluable days, a 24-hour Jul 1 payload with ordered quantiles, and 10 worst-days rows. A fresh Vite server on :3009 rendered 3 canvases and the table; switching to Sep–Dec and to the 14-hour 2018-06-01 day updated the picker and showed SARIMA n=14; the SARIMA toggle changed state; no console errors. `npm run build` passed (46 modules, 1.44 s).

## 2026-10-23 — 5.3 Explain page

- Chose XGBoost gain importance rather than SHAP: the trained boosters already expose per-feature total split gain, so no retraining and no new dependency are needed. `_gain_importance()` now persists `{method, features:[{feature, gain, share}]}` in each fold's `model_fit.xgboost` block inside `backend/results.json`. Importances come from the fold's training window only; the top features are `lag_24` in all folds (share 0.314 / 0.274 / 0.177 / 0.159) and `month`, `doy_cos` rise in the 2018 fold.
- Added `/api/explain/importance` (four folds, 48 features each, 503 when the artifact or block is missing) and `/api/explain/residuals?fold=&model=` for `xgboost`, `sarima`, and `p50`. The residual endpoint returns `summary {n, mean_residual_wh, mae_wh, rmse_wh}`, predicted-vs-residual points, 24 hour buckets, and a 20-bin histogram, all derived from the existing hourly backtest rows; residual is actual − forecast, restricted to hours where both the actual and that model's forecast exist.
- New `ExplainPage` with fold chips, model chips, and four charts: top-14 gain importance, residual vs predicted with a zero line, mean residual/MAE by hour, and residual distribution. The page states that gain is descriptive and not causal, that correlated lags can share credit, and that SARIMA covers fewer hours after long gaps. Nav entry added after Model Metrics; no placeholder or client-side metric is displayed.
- Evidence: live backend on 127.0.0.1:8012 returned 4 folds × 48 features and residuals matching `results.json`: Jul XGBoost n=744, MAE 69.31 Wh, bias −5.14, RMSE 90.53; SARIMA n=744, MAE 65.66, bias −7.89, RMSE 87.7; P50 n=744, MAE 65.99, bias +1.48. A fresh Vite server on :3011 rendered the page with 4 canvases; switching to SARIMA kept n=744 and switching the fold to May–Jun 2018 showed 38 hours; no console errors. `npm run build` passed (47 modules, 381.91 kB).

## 2026-10-23 — 5.4 Limitations panel

- Added `/api/limitations`, which assembles the data, evaluation, and model caveats from `backend/data_quality.json` and `backend/results.json` at request time. It returns three groups (5 data items, 4 protocol items, 6 model items), a per-fold coverage table, eight scope caveats, and the artifact sources; no count or coverage value is written into the frontend.
- The new Limitations page (`frontend/src/pages/LimitationsPage.jsx`, nav entry after Data Quality) shows a "read before quoting" banner, severity-tagged limitation cards (high / medium / context), a fold table with evaluable hours, training labels, SARIMA hours, and P10–P90 coverage, and the "what this does not establish" caveat grid. The page covers the 3,514 h target gap, the 182-hour 2018 fold, SARIMA's 3,080/3,913 h coverage, interval undercoverage, zero-reporting hours, and suspect inactivity, with links to the artifacts in the footer.
- Evidence: live backend on 127.0.0.1:8013 returned groups with 5/4/6 items, 4 folds, and 8 caveats. Values match the artifacts: 3,913 evaluable hours / 172 days, SARIMA 3,080 h (744 / 744 / 1,554 / 38), P10–P90 coverage 71.6% / 78.5% / 44.8% / 69.2%, 7,683 valid target hours, 5,208 excluded hours in 31 runs, 24 of 39 stable apartments, 21 inactivity flags, and 11,688 cached weather hours. A fresh Vite server on :3012 rendered all three groups, the four-row fold table, and eight caveats; no console errors. `npm run build` passed (48 modules, 388.20 kB). `pytest backend/tests -q` → 17 passed.
- Also removed the stale "LSTM · SARIMA · XGBoost" phrase from the FastAPI description, since no LSTM is shipped or shown; the disabled legacy response keys remain for schema compatibility.

## 2026-10-23 — 6.1–6.3 Upload validation redesign

- Replaced the old fixed-schema upload with `backend/upload_validation.py` and three endpoints: `GET /api/upload/options`, `POST /api/upload/preview` (first 256 KB / 100 rows), and `POST /api/upload/validate` (streamed, 50 MB cap). The UI now previews column suggestions before sending the full file, then exposes timestamp, value, optional series, unit, naive-timezone, and timestamp-format selectors with a sample table.
- Parsing: epoch seconds/milliseconds, ISO strings with embedded offsets, and naive timestamps localized with the chosen timezone are all converted to IST. Energy units (Wh/kWh) are summed within each series-hour and power units (W/kW) are averaged, then converted to Wh; the target is the mean across reporting series. Timestamps, values, negatives, series coverage, short-gap interpolation, excluded runs, and aggregation are all reported back. A failed parse returns the same validation report alongside the error.
- Evaluation reuses the Phase 3 functions: the first 60% of valid target hours trains, the test period starts at 00:00 IST of the day containing the 60th-percentile hour, the four baselines come from `evaluate.baseline_forecasts`, and all models share the actual + seasonal-naive evaluation mask, MASE scale, and paired Wilcoxon treatment. The fit-on-60% XGBoost uses the fixed 4.1 defaults, a train-only outlier cap, and only >=24 h lag/calendar features. The zero-shot model is a campus-trained XGBoost using only target lags and calendar features that exist on an upload (no campus reporting or weather), scale-aligned by the training-part mean only; no test-window label is read.
- Evidence on `smart_home_dataset.csv`: 48,972 rows used, 34,321 valid target hours (100% of span), split 20,587 train / 13,734 test hours over 573 days. Best baseline mean-of-4-weekdays MAE 35,984.8 Wh; zero-shot 37,104.1 Wh (worse, p=7.3e-10); fit-on-60% 27,587.3 Wh (better, p=0.0e+0); runtime 4.2 s. Multi-series check: 1,600 rows / 2 series aggregated to a 2,000 Wh target mean. Power check: 1,600 W readings averaged to 500 Wh with 799 same-hour readings merged. Row-accounting check: 60 raw rows → 58 used, 1 bad value + 1 negative, with the two-hour interior gap interpolated and reported.
- 6.3 states: uppercase `.CSV` accepted; client and backend both reject a 51 MB file (HTTP 400 / "File too large"); non-CSV rejected; upload progress and "Validating and evaluating…" shown; a 60-hour series returns a 200 partial report with "Evaluation not available — need at least 336 valid target hours, found 60"; results render from backend JSON with one daily chart, notes, and a 66.7 KB JSON download. The report itself carries the split, row counts, masks, and model fit metadata.
- Fixed during verification: pandas 3 string dtype made `dtype == object` false, so naive datetime columns were not suggested as timestamps (now uses `not is_numeric_dtype`); negative rows were double-counted in the dropped-row total (now computed as raw − used); a React warning from mixing `border` and `borderColor` was removed; zero-shot training was made feature-consistent with inference by dropping campus-only reporting features.
- Verification: live backend on 127.0.0.1:8014 and Vite on :3013; fresh page console showed only Vite/React dev messages; `npm run build` passed (48 modules, 398.66 kB); `python -m pytest backend/tests -q` → 17 passed.
- 6.1, 6.2, and 6.3 share one module, one endpoint set, and one page, so they are committed together; the split is recorded in DECISIONS.md.

## 2026-10-23 — 7.1 Test suite: metrics, split, leakage, gaps

- Added `test_metrics.py`: a fully hand-computed four-pair case (MAE 2.75, RMSE 2.8723, WAPE 11.0, sMAPE 12.1896, bias −0.25, R² 0.934, MASE 1.375), zero-denominator guards (WAPE/R² null, sMAPE 0), missing-pair handling, no-data responses, 24-h MASE scale, and day-ahead persistence/seasonal-naive lags.
- Added `test_gaps.py`: a 3-h interior gap interpolates to 12.5 / 15 / 17.5 and is flagged imputed; a 7-h gap and both edge runs stay missing and are flagged excluded; configured limits are honored; run lengths/edges/timestamps are reported.
- Added `test_split.py`: a 40-day synthetic series splits 576 train / 384 test hours at 2020-01-25 00:00 IST with a contiguous train end; an off-midnight start is floored to the day; a 10-day series returns the explicit 336-hour reason with no model or split payload. The upload model trainers are stubbed, so these tests exercise the split without training.
- Added `test_upload_validation.py`: regression for pandas 3 string dtypes (naive datetime columns must still be suggested as timestamps), epoch/unit suggestions, agreement of naive UTC / embedded-offset / epoch timestamps in IST, energy-sum vs power-mean aggregation with a per-hour series mean, and single counting of invalid vs negative rows.
- The new timestamp-offset regex has no capture group, removing the pandas match-group warning seen on every upload parse.
- Evidence: `python -m pytest backend/tests -q` → 34 passed in 15.19 s (was 17). The suite uses synthetic series only and does not read campus artifacts, the network, or the data directory.

## 2026-10-23 — 7.2 Makefile and Docker packaging

- Added a root `Makefile` with `help`, `install`, `test`, `eval`, `data-quality`, `build`, `dev-backend`, `dev-frontend`, `docker-build`, `docker-up`, and `docker-down`. GNU make is not installed on this Windows host, so the file's target/recipe structure was checked and every wrapped command was validated directly: `pytest` → 34 passed, `npm run build` → 48 modules / 398.66 kB, `npm ci --dry-run` → lockfile current, `pip install --dry-run -r backend/requirements.txt` → satisfied, and the Docker build/up/down sequence below.
- Backend image now copies the full runtime (`COPY . .`) instead of only `main.py` and `train_models.py`; `.dockerignore` excludes tests and caches. The built image contains `upload_validation.py`, `data_pipeline.py`, `evaluate.py`, `forecast_model.py`, `quantile_model.py`, `sarima_model.py`, `weather.py`, the weather cache, all three JSON artifacts, and `models/{train_stats,xgb_feature_cols,xgboost_model}.json`. Compose mounts `./data` read-only at `/data` (39 CSVs visible), which the zero-shot upload model needs, and drops the unused `smart-home-dataset-main` mount.
- Healthcheck now uses `python -c "urllib.request.urlopen(...)"` because the slim image has no curl; the frontend waits on `condition: service_healthy`. The runtime `VITE_API_URL=http://backend:8000` was removed because Vite reads env at build time; the built SPA uses same-origin `/api/`, which nginx proxies to `backend:8000`. Ports are overridable (`BACKEND_PORT`, `FRONTEND_PORT`) for isolated verification.
- Removed `torch==2.14.1`: its only import is inside the commented-out legacy LSTM. On Linux, `xgboost==3.4.1` hard-depends on a 305 MB `nvidia-nccl-cu13` wheel that this CPU-only service never uses and whose download timed out; requirements now install `xgboost-cpu==3.4.1` on Linux and keep the regular wheel elsewhere. Docker pip now uses a BuildKit cache mount with 10 retries and a 120 s timeout.
- Evidence: first build failed only at the NCCL download (`ReadTimeoutError` from files.pythonhosted.org, exit 2); after the dependency change, an isolated `docker compose -p electricast72` build succeeded with no errors. Backend image is 924 MB (down from 10.4 GB) and the container reports `xgboost 3.4.1` from `xgboost-cpu`, with no torch or nvidia packages. Both containers came up with backend `healthy`; `localhost:18000/` returned `data_loaded: true`, `localhost:13000/` served the SPA, and the same-origin `localhost:13000/api/backtest/summary` and `/api/upload/options` returned the same 4-fold payload and unit options as the backend. A 7,508-row `data/1.csv` upload through nginx returned HTTP 200 in 3.2 s with all four baselines, zero-shot status ok (MAE 131.6 Wh, scale 2.01), fit-on-60% ok (MAE 84.6 Wh), and best baseline persistence; the browser rendered the dashboard through port 13000 with no console messages. The isolated containers and images were removed afterward, and the pre-existing `electricast-*` stack was left untouched.

## 2026-10-23 — 7.3 Documentation refresh

- Rewrote `README.md` for the shipped product: four-fold day-ahead protocol, fold coverage, fold-mean metrics for all seven models, the three learned models (no LSTM), upload redesign, the eight dashboard pages, API groups, repository layout, test command, and limitations. It explicitly documents that the 2018 fold is 182 evaluable hours and that SARIMA has partial coverage there.
- Added `RESULTS.md` with per-fold tables for all seven models (MAE/RMSE/WAPE/MASE/bias/R²/sMAPE), paired Wilcoxon results against each fold's best baseline, interval coverage/pinball loss, fold means, data-quality inputs, reproduction commands, and caveats. Every value was read from `backend/results.json` (generated 2026-10-23T19:30:48+05:30) and `backend/data_quality.json`; nothing was retrained or hand-computed where an artifact value existed. The only derived numbers are the hour-weighted and fold-unweighted interval aggregates, computed from the four per-fold interval blocks.
- Rewrote `.github/copilot-instructions.md` around the current multi-module backend, JSON artifacts, Makefile/Docker commands, inline-style React conventions, the two data products, and the known pitfalls (no LSTM, SARIMA coverage, 2018 fold, pandas 3 dtypes, CPU-only XGBoost).
- Verification: `python -m pytest backend/tests -q` → 34 passed in 18.28 s; `npm run build` → 48 modules, 398.66 kB (gzip 127.20 kB); every existing relative link in `README.md` resolves (`FINAL_REPORT.md` is added at the end of the mission); a stale-term pass over all three docs leaves no claim of synthetic generation, a single-file backend, an LSTM result, a missing test suite, or an 80/20 split. `README.md` and `RESULTS.md` carry no metrics that are absent from the artifacts.

## 2026-10-23 — 8.1 "Ask ElectriCast" grounded chat panel

- Added `backend/config.py` (environment-driven provider settings) and three runtime modules: `chat_tools.py` (read-only tools over `results.json` and `data_quality.json`, a deterministic question router, the numeric-grounding packet, and the three suggested questions), `chat_service.py` (Ollama and Gemini clients, provider health/status with a 15 s cache, strict system prompt, numeric/date/fold/time guard with one retry), and `backend/scripts/check_chat.py` (stub or `--live` three-question verification).
- Tools cover model/fold metrics, error by month/year/hour, day predictions, data quality, limitations, and protocol. The router maps a date to day predictions, month/hour/error words to the error breakdown, quality words to data quality, limitation/why/fail/coverage words to limitations, and metric/model words to model metrics; unmatched questions are refused without calling a provider. The packet is built only from these tool return values.
- Added `GET /api/chat/status` (always HTTP 200; `available` plus provider/model/detail/limits) and `POST /api/chat` (400 invalid question, 503 unavailable, 502 grounding failure). Added the optional Compose environment (`CHAT_PROVIDER`, `OLLAMA_URL`, `OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`) with `host.docker.internal` and an `extra_hosts` entry; no new Python or npm dependency.
- Added `ChatDrawer.jsx` as a fixed launcher/drawer mounted in `App.jsx` on every page: provider/model badge, the three suggested questions, message history, a loading bubble, visible errors with retry, and an "Assistant unavailable" state with recheck. Input is capped at 500 characters; Enter sends and Escape closes. The drawer is hidden in print and becomes a full-width bottom sheet below 640 px.
- Verification: `python -m pytest backend/tests -q` → **52 passed in 20.02 s** (19 new chat tests: hand-computed tool outputs, planner routes, guard rounding/fabricated values/dates/times/list markers/fold ids, retry exhaustion, off-topic refusal, unavailable status, Gemini header/payload via mocked transport, configured question-length limit, and status-cache TTL). `python backend/scripts/check_chat.py` → **3/3 passed** in stub mode with 15 derived values independently re-derived from the raw artifacts; `--live` against Ollama → **3/3 passed**. Live answers included SARIMA 81.0733 Wh vs P50 99.9141 Wh (gap 18.8408 Wh) and December's 21 flagged / 13-of-24 stable apartments.
- Browser verification: fresh backend on `127.0.0.1:8001`, Vite on `:3002`, real Ollama. The drawer opened with the `ollama · llama3.2:3b` badge, the suggested question returned a grounded answer with tool/source labels, and the drawer stayed open while switching pages; Escape returned the launcher. A simulated network failure showed the loading bubble and disabled composer, then a visible error with Retry, which recovered. A second real backend (`OLLAMA_URL` pointed at a dead port) displayed "Assistant unavailable — Ollama is not reachable…" with no composer while the EDA page still loaded normally. Temporary servers were stopped and the verification `.env.local` removed.
- After the first live run misread the 100 Wh/h inactivity threshold as a weather threshold, added an explicit deterministic `summary` to `limitations().december_inactivity` and a prompt rule against moving numbers between categories; the rerun described "13 of 24 stable-panel apartments" and the load threshold correctly, and its test now asserts that summary.
- `docker compose config --quiet` → valid; the Docker image was not rebuilt in this item (the endpoint was exercised through uvicorn). Final `npm run build` → **49 modules, 407.11 kB (gzip 129.54 kB)** with no `VITE_API_URL`, i.e. same-origin as shipped.

## 2026-10-23 — 9.1 Use all valid history

- Added `backend/scripts/report_history.py`, a reproducible report over the raw CSVs. The panel starts at 2016-12-14 21:00 IST; the gap-handled target has 7,683 valid hours overall, and its first valid hour is 2016-12-14 21:00 IST.
- Early window 2016-12-14 00:00 → 2017-04-30 23:00: 2,143 valid target hours after gap filling (2,091 before) out of 3,291 clock hours. By month: Dec 317/432, Jan 412/744, Feb 393/672, Mar 636/744, Apr 385/720. Reporting stable apartments range 0–24 (median 21); 2,091 of the 3,291 hours have at least one reporter. The early target is therefore thin and variable but continuous enough to train on.
- December vacancy: Dec 2016 had 26 apartments with readings and 7 flagged suspect inactive (all 7 stay in the 24-apartment stable panel); Dec 2017 had 38 with readings and 21 flagged (13 stable). Only 6 apartments are flagged in both; 1 is 2016-only and 15 are 2017-only (Jaccard 0.27). So Dec 2016 does not show the same vacancy pattern as Dec 2017, and later pages must not generalize the Dec-2017 caveat backwards.
- All four folds now train from 2016-12-14 21:00 IST (`evaluate.history_start`). Valid training hours rose from 1,464 → 3,607 (Jul), 2,208 → 4,351 (Aug), 2,952 → 5,095 (Sep–Dec), and 5,326 → 7,469 (May–Jun 2018). Test windows, evaluation masks (744 / 744 / 2,243 / 182 hours; 3,913 total), baselines, and SARIMA are unchanged.
- Reran `python backend/evaluate.py` (5m14s). XGBoost MAE improved in all four folds: 69.31 → 66.28 Wh (Jul), 59.51 → 59.13 (Aug), 167.27 → 156.13 (Sep–Dec), 120.75 → 111.31 (May–Jun). Quantile P50 improved in all four: 65.99 → 64.56, 60.58 → 60.12, 159.14 → 149.60, 113.94 → 106.22. Baselines and SARIMA are identical to v1 because they do not use the pre-May-2017 history. The learned models still lose to the best baseline in Aug and Sep–Dec, so extra history narrows but does not remove that gap.
- Archived the old artifact as `backend/results_v1.json` (865 KB, generated 2026-10-23T19:30:48+05:30) and left `backend/results.json` as the live artifact. `backend/scripts/check_chat.py` re-read the new artifact: 15 derived values match and all 3 sample questions pass in stub mode; `python -m pytest backend/tests -q` → 52 passed.

## 2026-10-23 — 9.2 Five scoring folds plus a 2018 stress test

- Fold layout is now Jun, Jul, Aug, Sep–Nov, Dec 2017 plus a May–Jun 2018 stress test. Every fold still trains from 2016-12-14 21:00 IST. Evaluation coverage: 720 h / 30 d (Jun), 744 / 31 (Jul), 744 / 31 (Aug), 1,695 / 76 (Sep–Nov), 548 / 26 (Dec), 182 / 8 (stress); 4,633 hours and 202 days in total. Dec 2017 has only 548 evaluable hours because 593 target hours and 479 baseline hours survive the gaps.
- The 2018 fold is labeled "Stress test (May–Jun 2018)" and flagged `stress_test: true`. Its id remains `2018-05_06`. The API summary, assistant tools and Metrics page compute means, standard deviations, coverage and "beats" counts over the five scoring folds only; the stress test remains visible as a per-fold column. Fold means (MAE, Wh): seasonal naive 91.54, SARIMA 81.70, quantile P50 102.50, XGBoost 103.46, mean-of-4 116.67, same-hour-week 137.39, persistence 126.70. SARIMA is best; its gap to the seasonal-naive runner-up is 9.84 Wh.
- Per-fold MAE vs that fold's best baseline: Jun — XGBoost 73.09, SARIMA 76.64, P50 72.65 vs seasonal naive 83.06 (all learned models win); Jul — 66.28 / 65.66 / 64.56 vs 75.01 (all win); Aug — 59.13 / 60.60 / 60.12 vs mean-of-4 58.93 (none wins); Sep–Nov — 143.22 / 107.16 / 144.01 vs persistence 138.41 (only SARIMA wins); Dec — 175.59 / 98.43 / 171.16 vs persistence 47.55 (none wins). The Dec fold exposes the vacancy regime: XGBoost bias +136 Wh, SARIMA +40 Wh. In the excluded stress fold, XGBoost is 111.31 and SARIMA 93.80 (on 38 h) vs seasonal naive 132.87.
- Reran `python backend/evaluate.py` (6m25s) and rebuilt the summary. Month/seasonality features are still the full calendar set; 9.3 tests dropping month/day-of-year on shorter windows.
- Verification: `python -m pytest backend/tests -q` → 56 passed (new fold-layout, history-start and stress-exclusion tests for both the API summary and assistant tools). `python backend/scripts/check_chat.py` → 3/3 stub checks, 16 derived values matching the new artifact (SARIMA 81.6991 Wh, 3,762 of 4,451 scoring-fold hours; Dec XGBoost MAE 175.59). Browser on a fresh backend (:8002) and Vite (:3010): the Metrics page shows "5-fold day-ahead backtest + stress test", six fold chips with "excluded from means", per-fold means 5-fold counts, and a stress column; Predictions offers Jun 2017 dates and updated pooled subtitle; Limitations shows "5 scoring + 1 stress · 4,633 h" and the new caution. Console had only Vite/React dev messages; no mojibake.
- Fixed a real Windows-only bug found in that browser pass: `main.py` opened JSON artifacts with the locale codec, corrupting the stress label's en dash. All three artifact loads now use `encoding="utf-8"`; the API and UI render "Stress test (May–Jun 2018)" correctly. `npm run build` → 49 modules, 407.57 kB (gzip 129.72 kB).

## 2026-10-23 — 9.3 Feature ablation

- Added opt-in feature switches to `forecast_model.make_features`/`fit`/`predict`/`forecast`: drop `month`, `doy_sin`, `doy_cos`; omit weather; add target-hour temperature (oracle only). Defaults reproduce the production feature set exactly, and the chosen options are recorded in fit metadata.
- Added `backend/scripts/feature_ablation.py`, which runs the four variants on the same six folds with untuned `DEFAULT_PARAMS`, the same evaluation mask, train-only cap, and reporting features, and writes `backend/ablation_results.json`. The full model is the within-ablation baseline; the 2018 stress test is shown but excluded from the keep decision.
- MAE per fold (Wh), scoring folds first:

  | Fold | Eval h | Full | No annual calendar | No weather | ORACLE same-hour temp |
  |---|---:|---:|---:|---:|---:|
  | 2017-06 | 720 | 74.12 | 73.60 | 75.26 | 75.53 |
  | 2017-07 | 744 | 63.48 | 63.69 | 63.87 | 63.30 |
  | 2017-08 | 744 | 59.88 | 59.39 | 61.09 | 59.53 |
  | 2017-09_11 | 1,695 | 145.32 | 145.25 | 143.54 | 146.18 |
  | 2017-12 | 548 | 227.67 | 196.39 | 215.11 | 219.44 |
  | 2018-05_06 (stress, excluded) | 182 | 108.87 | 108.87 (not applied) | 99.75 | 104.84 |

- Summary vs full features: no annual calendar helped 4/5 scoring folds (mean 107.66 Wh, mean delta −6.43; Dec −31.28, other wins ≤0.53, Jul +0.21) and is kept. No weather helped 2/5 (mean 111.77, delta −2.32) and is discarded. ORACLE helped 3/5 (mean 112.80, delta −1.30) but is diagnostic only and can never be kept or become the winner. Winner: `no_annual_calendar` (`drop_annual_calendar: true`), which 9.4 will use as the comparison config.
- Verification: `python -m pytest backend/tests -q` → **60 passed in 16.76 s** (four new tests: calendar columns drop while hour/dow/holiday/semester stay; weather columns drop; oracle feature is opt-in and reads the target hour; feature options reach fit metadata). The default feature path is unchanged, so `results.json` was not regenerated for this item.

## 2026-10-23 — 9.4 Level-normalized target

- Added `backend/level_features.py` (`trailing_scale`, `near_zero_fraction`) and opt-in `target_scale` / `near_zero_fraction` parameters in `forecast_model`. The scale is the mean un-filled target over the 168 h ending one hour before each day's origin, with at least 24 observed hours; it is multiplied back after prediction. The near-zero feature is the share of reporting stable apartments below 100 Wh at t−24.
- Added `backend/scripts/level_experiment.py`, writing `backend/level_results.json`. It compares the 9.3 winner (raw target, no annual calendar when training <365 days) with normalized and normalized+near-zero variants on the same folds, untuned `DEFAULT_PARAMS`, and pairwise common hours.
- MAE per fold (Wh), with delta vs the 9.3 winner on the same common hours:

  | Fold | Eval h | Scale days | 9.3 winner | Normalized | Normalized + near-zero |
  |---|---:|---:|---:|---:|---:|
  | 2017-06 | 720 | 30/30 | 73.60 | 91.06 (+17.46) | 88.69 (+15.09) |
  | 2017-07 | 744 | 31/31 | 63.69 | 67.11 (+3.42) | 66.28 (+2.59) |
  | 2017-08 | 744 | 31/31 | 59.39 | 62.84 (+3.45) | 62.72 (+3.33) |
  | 2017-09_11 | 1,695 | 85/91 | 145.25 | 153.68 (+7.90) | 154.80 (+9.02) |
  | 2017-12 | 548 | 31/31 | 196.39 | 151.79 (−44.61) | 149.49 (−46.90) |
  | 2018-05_06 (stress, excluded) | 182 | 11/35 | 108.87 | 157.50 (+48.64) | 156.62 (+47.75) |

- Both normalized variants help only 1 of 5 scoring folds, so neither is adopted under the 9.3 rule. The normalized fold mean is lower (105.29 vs 107.66 Wh) and December bias falls from +161.35 to +37.37 Wh, but four folds regress; adding near-zero helps December by 2.30 Wh but does not change the 1/5 count. 9.5 therefore continues from the 9.3 winner. The scale is known at origin; days without a valid scale produce no forecast rather than a fallback.
- Verification: `python -m pytest backend/tests -q` → **67 passed in 16.94 s** (seven new tests: trailing scale no-leak/constant/minimum observations, hand-checked near-zero fraction, target-scale arithmetic, scale multiplication and missing-scale NaN, lag-24 feature and no-leak, and forecast scale-coverage metadata). Default feature behavior and `results.json` are unchanged.

## 2026-10-23 — 9.5 Frozen tuning parameters

- Added `backend/scripts/frozen_tuning.py`, writing `backend/tuning_results.json`. It runs one Optuna study on each of two windows, freezes the winning hyperparameters plus the median early-stopping iteration, and applies them unchanged to all six folds with untuned `DEFAULT_PARAMS` as the comparison. Both studies use the 9.3 feature set (the tuning windows are shorter than 12 months); folds apply the 9.3 rule per fold, so the 2018 stress fold keeps the full calendar.
- The literal largest window ends 2017-12-31 (6,994 rows, best CV MAE 150.6882, `n_estimators=72`) but contains every scoring test period. Its frozen parameters helped only 1/5 scoring folds (Dec −24.70 Wh; the other four were +0.006 to +1.20). This is not clean evidence, so the script also tunes on the largest window ending before June 2017 (2017-05-31, 2,543 rows, best CV MAE 75.9573, `n_estimators=150`).
- MAE per fold (Wh), common hours against untuned:

  | Fold | Eval h | Untuned | Largest-window frozen | Pre-scoring frozen | Best baseline |
  |---|---:|---:|---:|---:|---:|
  | 2017-06 | 720 | 73.60 | 74.33 (+0.73) | 73.13 (−0.47) | 83.06 |
  | 2017-07 | 744 | 63.69 | 63.69 (+0.01) | 64.98 (+1.30) | 75.01 |
  | 2017-08 | 744 | 59.39 | 60.58 (+1.20) | 60.24 (+0.86) | 58.93 |
  | 2017-09_11 | 1,695 | 145.25 | 145.64 (+0.39) | 144.01 (−1.24) | 138.41 |
  | 2017-12 | 548 | 196.39 | 171.69 (−24.70) | 168.08 (−28.31) | 47.55 |
  | 2018-05_06 (stress, excluded) | 182 | 108.87 | 110.01 (+1.13) | 112.70 (+3.83) | 132.87 |

- The pre-scoring frozen parameters helped 3/5 scoring folds and are adopted: mean MAE 102.09 vs 107.66 Wh (mean delta −5.57). Exact parameters: `max_depth=3`, `learning_rate=0.0189936`, `min_child_weight=17.4414`, `subsample=0.6232`, `colsample_bytree=0.7677`, `reg_lambda=8.2042`, `reg_alpha=0.7871`, `n_estimators=150`. The literal largest-window run remains in the artifact but is not adopted because its only meaningful gain is on test periods it trained through.
- This is still not a December solution: 168.08 Wh versus persistence at 47.55. The 5-fold mean beats untuned but the model still has lower MAE than the best baseline on only 2/5 folds (Jun, Jul). `CHECKPOINT_PHASE9.md` records the full v1 → current → 9.3 → 9.4 → 9.5 tables and the decisions.
- Verification: first run forced the no-calendar feature set onto the stress fold and produced 129.29 Wh there; corrected the script to apply the 9.3 rule per fold and reran, giving 112.70 Wh. Scoring folds are unchanged. `python -m pytest backend/tests -q` → **67 passed in 18.08 s**. No application code or live `results.json` changed in this item.

## 2026-10-23 — 9.6 Common-hours comparison

- Added `common_hours_block` to `evaluate.py` and a per-fold `common_hours` artifact block. The mask is the fold evaluation mask intersected with the forecasts of all four baselines and all three learned models; every model is scored on exactly those hours, with the common best baseline selected by the same MASE rule. Missing-model coverage and an explicit definition are stored with it; no fallback mask is used.
- The strict intersection keeps Jun–Aug complete but costs hours elsewhere: Sep–Nov 555/1,695 (32.7%) and Dec 387/548 (70.6%), limited by SARIMA/lagged-baseline gaps. The 2018 stress fold has 0 common hours because the four-weekday baseline has no valid history after the Jan–Apr 2018 gap, so its common table is N/A.
- MAE in Wh on common hours (common best baseline in bold):

  | Fold | n | Persistence | Seasonal naive | Same hour last week | Mean of 4 weekdays | XGBoost | SARIMA | P50 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Jun 2017 | 720 | 222.21 | **83.06** | 125.91 | 96.64 | 73.13 | 76.64 | 72.65 |
  | Jul 2017 | 744 | 119.56 | **75.01** | 95.00 | 86.94 | 64.98 | 65.66 | 64.56 |
  | Aug 2017 | 744 | 105.75 | 76.31 | 74.52 | **58.93** | 60.24 | 60.60 | 60.12 |
  | Sep–Nov 2017 | 555 | **130.92** | 180.08 | 205.79 | 158.13 | 160.99 | 112.57 | 157.94 |
  | Dec 2017 | 387 | **48.15** | 66.06 | 179.27 | 196.67 | 186.71 | 98.43 | 191.46 |
  | May–Jun 2018 (stress) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

- The common mask changes the Sep–Nov and Dec story relative to each model's own mask. XGBoost is 160.99 vs persistence 130.92 on the 555 common Sep–Nov hours (per-model table: 144.01 vs 138.41), and 186.71 vs 48.15 in Dec. XGBoost beats the common best baseline on 2/5 scoring folds (Jun, Jul), SARIMA on 3/5 (Jun, Jul, Sep–Nov), P50 on 2/5 (Jun, Jul); none wins Aug or Dec. The restricted comparison does not rescue the learned point models.
- Integrated the adopted 9.5 point configuration into the evaluator while regenerating: frozen pre-scoring parameters with `n_estimators=150`, no per-fold Optuna, and the 9.3 annual-calendar rule per fold. The regenerated point-XGBoost fold MAEs are 73.13 / 64.98 / 60.24 / 144.01 / 168.08 and the five-fold mean is 102.09 Wh (was 103.46). A unit test keeps `POINT_XGB_PARAMS` byte-equal to `tuning_results.json`; quantile, SARIMA and baselines are unchanged.
- API: each backtest-summary fold now exposes `common_hours` (`n`, `coverage`, `best_baseline`, `missing_models`) and each model's per-fold row carries a `common` metrics object; `common_hours_rule` explains the rule. Metrics page adds a "MAE on common hours" table under "MAE by fold" with the common n in each column header and N/A for the stress fold.
- Verification: `python backend/evaluate.py` regenerated `results.json` (1.1 MB) with all six folds and the common blocks. `python -m pytest backend/tests -q` → **73 passed in 20.23s** (new hand-checked intersection/empty-intersection tests, frozen-param sync test, feature-rule test, and summary mapping tests). `python backend/scripts/check_chat.py` → 3/3 in stub mode, 16 derived values still match the new artifact. Fresh backend on :8004 and Vite on :3011: the Metrics page rendered the new table with n=720/744/744/555/387/0, Sep–Nov XGBoost 161.0 and SARIMA 112.6, Dec 186.7/98.4, and the stress column N/A; console had only Vite/React dev messages. Servers stopped afterward. `npm run build` → 49 modules, 409.66 kB (gzip 130.07 kB).

## 2026-10-23 — 9.7 Day-level significance and block bootstrap

- Replaced the hourly Wilcoxon test in the backtest with a day-level test. `evaluate.daily_significance` aggregates absolute errors to one MAE per IST day, runs a two-sided paired Wilcoxon signed-rank test on the daily MAE differences against the fold's common-hours best baseline, and computes a 95% block-bootstrap CI of the hour-weighted MAE difference (2,000 resamples of contiguous 7-day blocks, seed 42). The artifact stores `mae_diff_wh`, the median/mean daily differences, `n_days`, `n_pairs`, the CI, `prob_model_better`, block count and a `<10 days` small-sample flag. The old top-level hourly `paired_vs_best` block is no longer written; the hourly helper remains only for upload validation.
- The API maps the day-level result to each model's `daily_vs_best` and exposes `significance_rule`. The Metrics page adds a "Day-level test vs best baseline (common hours)" table showing Δ = model MAE − best-baseline MAE (hour-weighted), its 95% CI, the Wilcoxon p-value, and the number of days. Δ was originally going to show the median daily difference, but that disagreed with the hour-weighted CI in Sep–Nov and Dec; the exact weighted `mae_diff_wh` is shown instead, with the page stating that Δ/CI are hour-weighted and the Wilcoxon test weights days equally.
- Rebuilt `RESULTS.md` from the current artifact because the existing copy still described the pre-9.1 four-fold protocol. It now covers the five scoring folds plus the 2018 stress test (context only), common-hours metrics, day-level p-values/CIs, current interval coverage, data-quality inputs and caveats.
- Findings on common hours: XGBoost is better than the best baseline in Jun (−9.93 Wh, p=0.0040) and Jul (−10.03, p=0.0044), not in Aug (+1.32, p=0.87), Sep–Nov (+30.07, p=0.78) or Dec (+138.55, p=0.0026); P50 mirrors it. SARIMA has lower MAE in Jun/Jul/Sep–Nov but only July is significant at 0.05 (Jun p=0.096, Sep–Nov p=0.070). No learned point model beats the common-hours best baseline on more than 2/5 scoring folds, and every learned model loses badly to persistence in December.
- Verification: `python backend/evaluate.py` regenerated `results.json` (1.1 MB, generated 2026-10-23T23:36:15+05:30); `python -m pytest backend/tests -q` → **75 passed in 19.70 s**; `python backend/scripts/check_chat.py` → 3/3 with 16 derived values matching the artifact; `npm run build` → 49 modules, 412.44 kB (gzip 130.66 kB). Browser check on backend :8004 / Vite :3011: the day-level table rendered Δ +30.1 Wh with CI [−10.8, +65.4] and p=0.779 for XGBoost Sep–Nov, Δ −9.9 with [−18.4, −4.2] and p=0.004 for Jun, Dec +138.6 with p=0.003, and N/A for the stress column; console had only Vite/React dev messages. Servers stopped and the orphaned Vite process on :3011 killed.

## 2026-10-23 — 9.8 Conformal interval calibration

- Added `conformal_calibrate_intervals` to `evaluate.py`: per IST test day, the interval is expanded by the 80% finite-sample quantile of CQR scores `max(P10 − actual, actual − P90)` from the prior 28 test days of the same fold. Warm-up is 7 days or fewer than 24 scores; q is floored at 0, so calibration never narrows an interval. Only strictly earlier days are read, so the scored day's actual cannot leak into its own bounds. The artifact's `intervals.conformal` block stores per-fold before/after metrics on calibrated and all days, width, q summary, daily q values and the exact rule; the hourly rows gain `p10_calibrated` and `p90_calibrated`.
- Regenerated `backend/results.json` (generated 2026-10-23T23:52:19+05:30). Point metrics and raw interval metrics are unchanged from 9.7; only conformal fields are new. Coverage before → after on calibrated days: Jun 71.01%→79.71%, Jul 80.90%→80.90%, Aug 81.60%→82.12%, Sep–Nov 55.29%→74.09%, Dec 35.00%→71.32%; on all days the scoring hour-weighted coverage rises 64.46%→75.29%. Mean width on scoring hours rises 236.65→320.45 Wh. The 2018 stress fold has 1 calibrated day (14 hours) and is context only.
- `RESULTS.md` adds the protocol rule, a before/after table per fold (calibrated and all days, width, median q), and a caveat that September–November and December remain below nominal because residuals are not exchangeable across the vacancy regime shift. `PLAN.md` 9.8 ticked.
- Fixed a day-count bug found by the new tests: `n_days_calibrated`/`warmup_days` were counting hourly rows instead of days. Tests cover prior-day-only scoring (including the first spike day), expansion-only behaviour, and the empty-evaluation case.
- Verification: `python -m pytest backend/tests -q` → **78 passed in 26.25 s**; `python backend/scripts/check_chat.py` → 3/3, 16 derived values still match. No frontend change in this item; the Metrics page still shows raw intervals until 9.9.

## 2026-10-24 — 9.9 Refresh docs, Limitations page and assistant

- Rewrote `RESULTS.md`, `README.md` and `FINAL_REPORT.md` for the current five-scoring-fold artifact: 4,633 evaluation hours, the 2018 stress test excluded from means, frozen point-XGBoost parameters, strict common-hours results, day-level p-values/block-bootstrap CIs and rolling conformal intervals. The common-hours bottom line is stated plainly: no learned model is reliably better than the best baseline (XGBoost/P50 2/5, SARIMA 3/5 with only July significant; all lose badly in December). README test count is 79 and the assistant section documents the deterministic period lead.
- `backend/main.py` builds the Limitations items from the artifact: each fold row now carries raw coverage, after-CQR coverage on all and calibrated days, calibrated/warm-up days and the 28-day window; the P10–P90 item shows before→after per fold and the scoring hour-weighted 64.46%→75.29%; a new learned-vs-baseline item is generated from the common-hours counts and day-level p-values; the tuning item now says "frozen once, pre-scoring" instead of claiming per-fold Optuna. Caveats use the dynamic after-coverage and counts, so no number is hardcoded in the backend text.
- `backend/chat_tools.py` exposes the same conformal fields and adds `december_context` (XGBoost MAE/bias/n plus the common-hours best baseline) so "Why is December different?" can be answered with error metrics rather than interval coverage. `frontend/src/pages/LimitationsPage.jsx` adds "Raw P10–P90", "After CQR" and "Cal. days" columns, a 7-day warm-up subtitle and the `interval_rule` footnote; every value still comes from `/api/limitations`.
- Live assistant check exposed a quality failure: `llama3.2:3b` answered the December question with 30.84% interval coverage only and ignored the prompt rule to lead with error metrics. Added a deterministic lead in `chat_service.ask`: for month-scoped `error_breakdown` plans, if the model's opening sentence lacks that month's MAE (or a negative bias is unsigned), prepend a grounded sentence built from `selected_month`; the numeric guard re-checks the combined text. Added `ANSWER FOCUS` to the tool prompt and a unit test (`test_service_prepends_grounded_period_lead`, 20 chat tests).
- Verification: `python -m pytest backend/tests -q` → **79 passed in 128.88 s**; `python backend/scripts/check_chat.py` → 3/3, **21 derived values** match the raw artifacts; `--live` against Ollama `llama3.2:3b` → 3/3 with the December answer opening "In Dec 2017, XGBoost has MAE 168.08 Wh and bias 133.22 Wh on 548 hours."; `npm run build` → 49 modules, 412.91 kB (gzip 130.79 kB). Browser check against backend :8004 through Vite :3011: Limitations showed Jun 72.6%→79.3%, Sep–Nov 56.9%→74.2%, Dec 30.8%→56.0%, cal. days 23/24/24/69/19/1, the learned-vs-baseline item and the interval-rule footnote; the assistant drawer returned the grounded December answer; console had only Vite/React dev messages. Vite was temporarily proxied to :8004 for the check and restored to :8000; both servers and the orphaned Vite process were stopped. `results.json` was not regenerated in 9.9 (still 2026-10-23T23:52:19+05:30).
