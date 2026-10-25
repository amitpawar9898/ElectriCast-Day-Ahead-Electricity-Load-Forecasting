# Project Guidelines — ElectriCast

## Architecture

Full-stack day-ahead electricity load forecasting dashboard on the IIT Bombay residential
apartment dataset (Dec 2016 – Jun 2018; Jan–Apr 2018 is a coverage gap). The backend serves
precomputed artifacts from disk; there is no database, no synthetic data generation, and no
LSTM in the product.

```
backend/main.py              FastAPI app, loads JSON artifacts at import, shapes API responses
backend/data_pipeline.py     raw CSVs → hourly per-apartment panel → gap-handled target
backend/build_data_quality.py  writes data_quality.json
backend/evaluate.py          four-fold backtest, baselines, learned models, results.json
backend/forecast_model.py    XGBoost features/fit, Optuna tuning
backend/quantile_model.py    P10/P50/P90 XGBoost
backend/sarima_model.py      rolling weekly-refit SARIMA
backend/weather.py           Open-Meteo archive cache (weather_mumbai.csv)
backend/upload_validation.py upload parsing, 60/40 split, fixed-protocol evaluation
backend/train_models.py      legacy 2017 single-split artifact (model_results.json)
backend/config.py            assistant provider settings (Ollama/Gemini)
backend/chat_tools.py        read-only artifact tools and deterministic question router
backend/chat_service.py      model clients, strict prompt, numeric grounding guard
backend/scripts/check_chat.py stub/live three-question grounding check

frontend/src/pages/         8 pages (EDA, Predictions, Weekly, Metrics, Explain,
                            Data Quality, Limitations, Upload)
frontend/src/components/    MetricsBar, ChatDrawer, DataError
frontend/src/hooks/useData.js
frontend/src/utils/api.js
frontend/src/utils/chartConfig.js
```

Data flow: Page → `useData()` → `api.js` → Vite proxy (`/api`) → FastAPI → JSON artifact →
Chart.js render. In Docker, nginx proxies `/api/` to the backend, so the SPA is same-origin.

The assistant is a separate path: `ChatDrawer` → `POST /api/chat` → deterministic route in
`chat_tools.plan()` → read-only tool results from the two current artifacts → provider
(Ollama by default, optional Gemini). The provider never sees the artifacts directly, and
`chat_service` rejects answers containing numbers, dates, fold ids, or times absent from the
tool packet (one retry, then 502). `GET /api/chat/status` always returns 200 so the dashboard
works when the assistant is unavailable.

Two data products, not interchangeable:

- `backend/results.json` (current four-fold backtest) and `backend/data_quality.json` back the
  Predictions, Metrics, Explain, Data Quality, Limitations and Upload pages.
- `backend/model_results.json` is the legacy 2017 single-split aggregate. The EDA and Weekly
  pages still read its `/api/eda/*` and `/api/predictions/*` endpoints. Its LSTM fields are
  disabled and no UI path renders them.

## Build and Test

```bash
make install          # backend requirements + npm ci
make dev-backend      # uvicorn on :8000, docs at /docs
make dev-frontend     # Vite on :3000
make test             # pytest backend/tests -q  (52 tests)
make build            # vite build
make eval             # rebuild results.json (retrains models)
make data-quality     # rebuild data_quality.json
docker compose up --build
```

Direct commands: `python -m uvicorn main:app --reload --port 8000` from `backend/`,
`npm run dev` from `frontend/`. The frontend has no test runner; validate UI changes with a
real browser against a live backend and `npm run build`. Tests use synthetic series and must
not depend on the network or the campus CSVs.

## Code Style

### Backend (Python / FastAPI)

- Keep `main.py` as the HTTP/response-shaping layer. Data preparation belongs in
  `data_pipeline.py`; metrics and folds in `evaluate.py`; upload logic in
  `upload_validation.py`.
- Endpoints are synchronous and return plain dicts/lists shaped for the frontend. Missing
  artifacts return 503 with an `error` key, not fabricated data.
- Metrics are rounded at the API/artifact boundary (`round(x, 4)`); hourly series and chart
  payloads may carry 1–2 decimals.
- All model evaluation goes through the shared protocol: 00:00 IST origins, the same evaluation
  mask, only pre-origin information (XGBoost lags ≥24 h; SARIMA state updates up to the
  origin), and no test-fold tuning. Add a new model by producing the same per-fold
  metrics/interval structure in `results.json`, not by special-casing a page.
- New numeric behavior needs a pytest case; new dependencies must be pinned in
  `requirements.txt` and recorded in `DECISIONS.md`.

### Frontend (React / Vite)

- Hooks-only state management; no Redux or Context. No router: page state lives in `App.jsx`.
- All styling is inline React styles; fonts are IBM Plex Mono (headings/metrics) and IBM Plex
  Sans (body). Each page registers the Chart.js scales/elements it uses.
- Use `lineDs()` / `barDs()` and `MODEL_META` / `COLORS` from `chartConfig.js`; do not invent
  per-page model colors.
- Units are Wh internally. Label Wh or divide by 1,000 and label kWh; never mix them.
- UI numbers must come from API JSON. Do not hardcode metrics, fold counts, or dates.

## Conventions

- Add API functions as named exports in `frontend/src/utils/api.js`. `BASE` is
  `import.meta.env.VITE_API_URL || ""`, so the default is same-origin.
- New pages follow the existing pattern: fetch with `useData()`, render `DataError` on failure,
  compose inline components, and add a `NAV_ITEMS` entry in `App.jsx`.
- Backtest payloads use lowercase model keys (`xgboost`, `sarima`, `xgboost_quantile`); display
  labels live in the backend or `MODEL_META`, not in ad-hoc JSX strings.
- The upload client uses XHR because fetch has no upload-progress event; keep the 50 MB limit
  in sync with the backend.
- Keep `data/` read-only. Never delete or rewrite raw CSVs or committed model artifacts.
- Assistant answers must be built from `ChatTools` output only. Do not pass raw artifacts or
  outside facts to a provider, do not bypass the numeric guard, and do not add a provider
  without updating `config.py`, `/api/chat/status`, and the tests.
- The assistant is single-turn; do not imply memory of prior questions. Its guard checks
  provenance, not semantic truth, so prefer explicit summaries/notes in tool output when a
  field could be misread.

## Pitfalls

- `results.json` is generated and large because it contains hourly backtest rows. Regenerate it
  with `evaluate.py`; do not hand-edit. The API derives backtest summaries/day views at startup.
- `evaluate.py` retrains models (Optuna + SARIMA refits) and is not run at API startup.
- SARIMA has partial coverage because it needs a contiguous pre-origin history; in the 2018
  fold it forecasts 96 hours and is scored on 38. Pages must use each model's own `n` and not
  assume all models cover the same hours.
- The 2018 fold is 182 evaluable hours over 8 days, and the Sep–Dec fold includes suspected
  December vacancies. Do not present fold means without this context.
- Quantile intervals undercover in Sep–Dec; they are diagnostics, not calibrated bounds.
- pandas 3 string dtype means `dtype == object` is false; use pandas type predicates
  (`is_numeric_dtype`, `is_datetime64_any_dtype`) when detecting columns.
- React StrictMode double-fires effects in dev; abortable fetches in `useData` are expected.
- Docker: the backend healthcheck uses the Python standard library (no curl in the slim
  image); Linux installs `xgboost-cpu`; there is no torch/LSTM dependency. The campus data
  directory is mounted read-only at `/data`.
- Assistant: `llama3.2:3b` is expected on a local Ollama at `127.0.0.1:11434`; no provider
  means “Assistant unavailable”, not an error page. Docker points at
  `host.docker.internal:11434`, which requires host Ollama to bind `0.0.0.0`. Gemini is
  optional and was only mock-tested.

See [README.md](../README.md) for the dataset and results overview and
[RESULTS.md](../RESULTS.md) for per-fold metrics and significance tests.
