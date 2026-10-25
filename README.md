# ElectriCast — Day-Ahead Electricity Load Forecasting

A full-stack dashboard for day-ahead residential electricity load forecasting on the IIT Bombay
campus apartment dataset (Dec 2016 – Jun 2018, with the backtest concentrated in 2017–18). The
current evaluation product is a five-fold expanding-window backtest plus a separate 2018 stress
test, with four naive baselines and three learned models: frozen-parameter XGBoost, rolling
SARIMA, and quantile XGBoost (P10/P50/P90) with rolling conformal calibration. No LSTM is
shipped or displayed.

- Backend: FastAPI, pandas, NumPy, scikit-learn, statsmodels, XGBoost
- Frontend: React 18, Vite 5, Chart.js 4
- Tests: pytest (79 tests)
- Units: Wh internally; the UI labels Wh or divides by 1,000 for kWh
- Timezone: Asia/Kolkata (IST) throughout

## Quick start

Requirements: Python 3.12 and Node 20+.

```bash
make install          # pip install -r backend/requirements.txt && npm ci
make dev-backend      # http://localhost:8000  (docs at /docs)
make dev-frontend     # http://localhost:3000
```

Without GNU make:

```bash
python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)
(cd backend && python -m uvicorn main:app --reload --port 8000)   # terminal 1
(cd frontend && npm run dev)                                       # terminal 2
```

Docker:

```bash
docker compose up --build
```

The frontend is served at http://localhost:3000 and the backend at http://localhost:8000.
`BACKEND_PORT` and `FRONTEND_PORT` override the published ports. Compose mounts `./data`
read-only at `/data` (the upload zero-shot model reads the campus CSVs on first use) and
mounts `backend/model_results.json` read-only.

## Ask ElectriCast

A collapsible drawer on every page answers questions about the saved backtest. It is
read-only: the backend routes the question to small tools over `results.json` and
`data_quality.json`, sends only those tool results to the model, and rejects any answer whose
numbers/dates/times are not traceable to that packet (one retry, then HTTP 502). If the
artifacts do not contain the answer, the answer says so; off-topic questions are refused
without calling a model. For month-scoped questions, a grounded opening sentence with that
month's XGBoost MAE and bias is taken from the tool result, so the default small model cannot
omit the key error metrics. Conversation history is not sent, and the assistant cannot see
uploaded files or live data.

Configuration lives in `backend/config.py` and can be overridden by environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CHAT_PROVIDER` | `auto` | `auto`, `ollama`, or `gemini` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | local Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | model tag; must already be pulled |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | unset | optional Gemini fallback |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model when a key is set |
| `CHAT_TIMEOUT_SECONDS` | `60` | per-provider request timeout |

Local default:

```bash
ollama serve
ollama pull llama3.2:3b
```

With no reachable provider the drawer shows "Assistant unavailable" and the rest of the
dashboard works normally. Docker Compose defaults `OLLAMA_URL` to
`http://host.docker.internal:11434` and maps `host.docker.internal`; start the host Ollama with
`OLLAMA_HOST=0.0.0.0` so the container can reach it (its default loopback bind is not
reachable from another namespace). The Gemini path has mocked tests but was not live-tested on
this host, which has no API key.

Regenerate artifacts:

```bash
make data-quality     # backend/data_quality.json
make eval             # backend/results.json  (retrains learned models)
```

## Dataset

| Property | Value |
|---|---|
| Source | IIT Bombay residential campus apartments |
| Coverage | 2016-12-14 → 2018-06-04 IST (Jan–Apr 2018 has no readings) |
| Raw files | 39 apartment CSVs, 196,449 rows |
| Hourly panel | 12,891 h × 39 apartments |
| Stable panel | 24 apartments with >90% coverage |
| Target | mean hourly load per reporting stable apartment (`target_wh`, Wh) |
| Valid target hours | 7,683 |
| Short gaps | 83 h imputed (runs ≤ 6 h) |
| Excluded gaps | 5,208 h in 31 runs (excluded from train and evaluation) |
| Longest excluded run | 3,514 h |
| Suspect inactive | 21 apartments flagged in Dec 2017 (13 stable) |

The raw phase loads are resampled to hourly per apartment; the target is the mean across
apartments that report in that hour, so it is per-apartment load, not total building load.
The outlier cap is fitted on each fold's training window only.

## Evaluation protocol

- 24 hourly forecasts per test day, origin 00:00 IST; no test-day label is a forecast input.
- Information set: target observations strictly before the origin. Long-lag baselines use lags of
  at least 24 h; persistence holds the last observation; SARIMA advances its state with actuals up
  to each origin. Weather and reporting counts are lagged at least 24 h.
- Five scoring folds (June, July, August, September–November, December 2017) expand from the
  earliest valid target hour, 2016-12-14 21:00 IST. May–June 2018 is a stress test after a
  four-month outage and is excluded from every fold mean.
- Evaluation mask: actual present and seasonal naive available. Each model is scored on the hours
  where its own forecast also exists; `n` is reported. The common-hours table intersects the
  evaluation mask with **all seven** models, so its rows are strictly like-for-like.
- Metrics: MAE, RMSE, WAPE, MASE, bias, R², sMAPE; MASE scale is the mean absolute
  seasonal (24 h) difference over valid training pairs.
- Best baseline = lowest MASE on the all-four-baseline mask when it covers ≥50% of the
  evaluation hours, otherwise on the evaluation mask.
- Significance: absolute errors are aggregated to daily MAE per IST day; a paired Wilcoxon
  signed-rank test runs on daily MAE differences against the common-hours best baseline, with a
  95% block-bootstrap CI of the hour-weighted MAE difference (2,000 resamples of 7-day blocks).
  p-values are unadjusted and are not a model-selection procedure.
- Tuning: the point XGBoost parameters are frozen from one Optuna study on 2016-12-14 →
  2017-05-31, which ends before every scoring test month. There is no per-fold Optuna and no
  test-fold selection. Annual calendar features are dropped when a fold's training span is under
  365 days.
- Intervals: the raw P10–P90 band is recalibrated per test day by rolling conformalized quantile
  regression using the prior 28 test days in the same fold (7-day warm-up, expansion only).

| Fold | Train window | Test window | Eval hours | Eval days |
|---|---|---|---:|---:|
| 2017-06 | 2016-12-14 21:00 → 2017-05-31 | 2017-06-01 → 2017-06-30 | 720 | 30 |
| 2017-07 | 2016-12-14 21:00 → 2017-06-30 | 2017-07-01 → 2017-07-31 | 744 | 31 |
| 2017-08 | 2016-12-14 21:00 → 2017-07-31 | 2017-08-01 → 2017-08-31 | 744 | 31 |
| 2017-09_11 | 2016-12-14 21:00 → 2017-08-31 | 2017-09-01 → 2017-11-30 | 1,695 | 76 |
| 2017-12 | 2016-12-14 21:00 → 2017-11-30 | 2017-12-01 → 2017-12-31 | 548 | 26 |
| 2018-05_06 (stress) | 2016-12-14 21:00 → 2017-12-31 | 2018-05-01 → 2018-06-04 | 182 | 8 |

## Results

Fold means are unweighted across the five scoring folds; `n` is the total evaluable hours. The
2018 stress test is excluded from every aggregate. Full per-fold tables, common-hours results,
day-level p-values and interval coverage are in [RESULTS.md](RESULTS.md); the machine-readable
source is `backend/results.json`.

| Model | Folds | Eval hours | MAE (Wh) | RMSE (Wh) | WAPE % | MASE | R² | Beats seasonal naive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 4,451 | 126.70 ± 63.27 | 174.71 | 42.92 | 1.316 | −0.635 | 2/5 |
| Seasonal naive 24 h | 5 | 4,451 | 91.54 ± 33.96 | 137.46 | 38.29 | 0.958 | 0.057 | reference |
| Same hour last week | 5 | 4,135 | 137.39 ± 57.19 | 184.72 | 70.70 | 1.417 | −0.754 | 1/5 |
| Mean of 4 same weekdays | 5 | 3,498 | 116.67 ± 51.59 | 145.34 | 65.43 | 1.196 | −0.107 | 1/5 |
| XGBoost (frozen) | 5 | 4,451 | 102.09 ± 50.20 | 129.53 | 56.72 | 1.047 | 0.087 | 4/5 |
| SARIMA | 5 | 3,762 | 81.70 ± 20.35 | 110.10 | 41.83 | 0.843 | 0.399 | 4/5 |
| Quantile XGBoost (P50) | 5 | 4,451 | 102.50 ± 51.39 | 130.56 | 57.34 | 1.051 | 0.059 | 4/5 |

Reading the table:

- SARIMA has the lowest fold-mean MAE, but it is scored on 3,762 of 4,451 hours: the January–April
  2018 gap removes most of its 2018 forecasts, and its fold-mean advantage is not equally
  supported in every fold.
- On the **strict common-hours** mask (all seven models on exactly the same hours), XGBoost and
  P50 have lower MAE than the fold's best baseline only in June and July; SARIMA is lower in
  June, July and September–November, but only July reaches p < 0.05. Every learned model loses
  badly to persistence in December.
- **No learned model is reliably better than the best baseline on common hours.** The fold-mean
  ranking is descriptive, not a deployment recommendation.
- August favors the mean of four same weekdays; the learned models are not significantly
  different there. December is a vacancy regime where the target level collapses and every
  learned model carries a large positive bias.
- Raw P10–P90 coverage is 64.46% hour-weighted on the scoring folds (nominal 80%). Rolling
  conformal calibration raises it to 75.29%, but September–November and December remain
  undercovered because prior-day residuals cannot anticipate the vacancy shift.

## Models

**XGBoost (point forecast).** Direct day-ahead model with target lags 24–47, 168 and 336 h;
rolling statistics over hours 24–47; calendar, holiday and semester flags; `n_reporting` lagged
by 24 h; and Open-Meteo weather variables lagged by 24 h. Hyperparameters are frozen from one
Optuna study on 2016-12-14 → 2017-05-31 and reused in every fold; there is no per-fold Optuna.
Annual calendar features (`month`, day-of-year) are dropped when the training span is under
365 days. Labels are clipped with a train-only Q3 + 3·IQR cap. Feature gains are shown on the
Explain page.

**SARIMA.** Order (1,0,1)(0,1,1,24), refit weekly on the trailing 60 days of the gap-handled
target and advanced daily using only actuals available before each origin. A recovery refit
runs after coverage gaps. Prediction intervals are not reported for SARIMA.

**Quantile XGBoost.** Three models with `reg:quantileerror` at 0.1 / 0.5 / 0.9 on the same
features and cap as XGBoost; P50 is also scored as a point forecast. The raw P10–P90 interval is
recalibrated per test day by rolling conformalized quantile regression using only the prior 28
test days in the same fold (80% conformity quantile, 7-day warm-up, expansion only). Raw and
calibrated coverage and pinball loss are reported per fold in `RESULTS.md`.

**Phase 9 experiments (not all adopted).** Removing annual calendar features helped 4 of 5
scoring folds and is kept. Removing weather helped only 2 of 5 and is not adopted. A same-hour
temperature variant is labeled **ORACLE**, is diagnostic only, and is never presented as a real
result. Level-normalizing the target by its trailing 7-day mean helped only December and is not
adopted. Frozen pre-scoring tuning improved the five-fold point-XGBoost mean from 107.66 to
102.09 Wh and is adopted. None of these changes fixes the December vacancy regime.

**Baselines.** Persistence (last value before the day, held flat), seasonal naive (same hour
24 h earlier), same hour last week, and the mean of the last four same weekdays.

**Not shipped.** No LSTM/GRU. The legacy `model_results.json` still contains LSTM-era fields
that the EDA page does not surface; the LSTM metrics are marked disabled and no UI path renders
them.

## Upload & Validate

The Upload page accepts a CSV up to 50 MB (streamed; a 256 KB / 100-row preview is read
first), detects or lets the user map the timestamp, value and optional series columns, and
asks for unit (Wh/kWh energy summed per hour, W/kW power averaged per hour), naive timezone,
and timestamp format. Everything is converted to IST, then evaluated with the Phase 3
protocol on a chronological 60/40 split at an IST midnight:

- four Phase 3 baselines,
- a zero-shot campus-trained XGBoost using only upload-available features (target lags and
  calendar; no campus reporting or weather) and a scale factor from the training part only,
- an XGBoost fit on the first 60% with fixed defaults.

The report includes row accounting, gaps, masks, metrics, paired Wilcoxon results, and a JSON
download. Uploads with fewer than 336 valid target hours return a partial validation report
with an explicit reason instead of metrics.

## Dashboard pages

| Page | Content |
|---|---|
| EDA | 2017 aggregate hourly/daily/monthly/yearly profiles and hour×month heatmap |
| Predictions | Six-fold backtest day picker (five scoring + 2018 stress), XGBoost/SARIMA/P50 forecasts, P10–P90 band, error by hour/month, worst days |
| Weekly | Legacy 2017 weekday/weekend profile per model (from `model_results.json`) |
| Model Metrics | Fold mean ± std table and "beats seasonal naive?" column from `/api/backtest/summary` |
| Explain | XGBoost gain importance and per-fold residual plots |
| Data Quality | Raw-file coverage, gaps, stable panel, inactivity flags |
| Limitations | High-level caveats assembled from `data_quality.json` and `results.json` |
| Upload & Validate | CSV validation and fixed-protocol evaluation |

The Ask ElectriCast drawer is available on every page and is described above.

### Two data products

- `backend/results.json` (Phase 3/4 backtest) + `backend/data_quality.json` back the
  Predictions, Model Metrics, Explain, Data Quality, Limitations and Upload pages.
- `backend/model_results.json` is the legacy 2017 single-split aggregate. The EDA and Weekly
  pages still read its `/api/eda/*` and `/api/predictions/*` endpoints; its LSTM fields are
  not displayed anywhere. The two products are not directly comparable.

## API

All data endpoints are served under `/api` and return JSON. Interactive docs at `/docs`.

| Group | Endpoints |
|---|---|
| Backtest | `/api/backtest/summary`, `/days`, `/day?date=`, `/errors` |
| Explain | `/api/explain/importance`, `/residuals?fold=&model=` |
| Data quality | `/api/data_quality`, `/api/limitations` |
| EDA (legacy) | `/api/eda/hourly`, `/daily`, `/monthly`, `/yearly`, `/heatmap`, `/weekday_vs_weekend`, `/statistics` |
| Predictions (legacy) | `/api/predictions/monthly`, `/daily`, `/hourly`, `/weekly_pattern`, `/monthly_errors` |
| Upload | `/api/upload/options`, `POST /preview`, `POST /validate` |
| Assistant | `/api/chat/status`, `POST /api/chat` |

## Repository layout

```
backend/
  main.py                 FastAPI app and response shaping
  data_pipeline.py        raw CSVs → hourly panel → gap-handled target
  build_data_quality.py   data_quality.json
  evaluate.py             folds, baselines, metrics, significance, results.json
  forecast_model.py       XGBoost features/fit, Optuna tuning
  quantile_model.py       quantile XGBoost
  sarima_model.py         rolling weekly-refit SARIMA
  train_models.py         legacy 2017 single-split artifact (model_results.json)
  weather.py              Open-Meteo archive download/cache
  upload_validation.py    upload parse, split, evaluation
  config.py               assistant provider settings
  chat_tools.py           read-only artifact tools and deterministic router
  chat_service.py         Ollama/Gemini clients, prompt, numeric grounding guard
  scripts/check_chat.py   stub/live three-question grounding check
  tests/                  pytest suite
  models/                 committed XGBoost/train-stats artifacts
frontend/src/
  pages/                  eight page components
  components/             MetricsBar, ChatDrawer, shared inline UI
  hooks/useData.js        fetch hook (loading/error/abort)
  utils/api.js            API client
  utils/chartConfig.js    Chart.js defaults and model colors
data/                     raw apartment CSVs (read-only input, never deleted)
notebooks/                original exploration notebooks
```

## Tests

```bash
make test
# or: python -m pytest backend/tests -q
```

The 79 tests cover metric math and guards, day-level significance and bootstrap, conformal
calibration, the day-ahead split, lag/leakage rules, gap handling, upload parsing/aggregation,
and the assistant's tool outputs, planner, numeric guard, retries, and provider payloads. They
use synthetic series only and do not need the network or the campus CSVs.

The scripted assistant check can run without a provider (stub answers built from tool output)
or against the configured backend:

```bash
python backend/scripts/check_chat.py          # stub mode
python backend/scripts/check_chat.py --live   # requires a reachable provider
```

## Documentation

- [RESULTS.md](RESULTS.md) — protocol, per-fold metrics, common hours, significance, conformal intervals
- [CHECKPOINT_DATA.md](CHECKPOINT_DATA.md) — data-quality checkpoint
- [CHECKPOINT_EVAL.md](CHECKPOINT_EVAL.md) — baseline-only evaluation checkpoint
- [CHECKPOINT_MODELS.md](CHECKPOINT_MODELS.md) — learned-model checkpoint
- [CHECKPOINT_PHASE9.md](CHECKPOINT_PHASE9.md) — Phase 9 history/feature/tuning checkpoint
- [DECISIONS.md](DECISIONS.md) — design decisions and conservatism calls
- [PROGRESS.md](PROGRESS.md) — per-item work log and evidence
- [FINAL_REPORT.md](FINAL_REPORT.md) — summary of the improvement pass

## Limitations

- Five scoring folds only; the 2018 stress test has 182 evaluable hours and 38 SARIMA hours, and
  is excluded from every mean.
- The target is a per-reporting-apartment mean over 24 stable apartments; December 2017
  includes suspected vacancies.
- No test-fold tuning or test-window refitting is performed. The fold set is small, so fold-mean
  rankings are sensitive to the composition of the folds.
- **No learned model is reliably better than the best baseline on common hours.** XGBoost and P50
  win only June and July; SARIMA has lower MAE on 3 of 5 scoring folds but only July is
  significant at 0.05; every learned model loses badly in December.
- Raw P10–P90 coverage is 64.46% hour-weighted on the scoring folds; rolling conformal
  calibration raises it to 75.29%, but September–November and December remain below 80% because
  prior-day residuals cannot anticipate the vacancy regime shift. Calibration never narrows.
- Weather comes from one Open-Meteo grid point (Mumbai) and is not building-level.
- EDA and Weekly still show the legacy 2017 aggregate and are not the backtest numbers.
- The assistant is single-turn and artifacts-only. Its numeric guard checks provenance, not
  semantic truth, and a small local model can still phrase a grounded number poorly; the
  Gemini fallback has mocked coverage only.
