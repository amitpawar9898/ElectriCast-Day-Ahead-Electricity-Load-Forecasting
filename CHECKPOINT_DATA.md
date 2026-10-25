# Phase 2 data checkpoint

Generated 2026-10-23 on branch `improve-electricast`. Every number below is
reproducible from `backend/data_pipeline.py` and `backend/build_data_quality.py`;
the JSON is committed as `backend/data_quality.json` and served at
`GET /api/data_quality`.

## What was built

- `backend/data_pipeline.py` — reads all `data/*.csv`, normalizes UTC to IST
  hours, builds the apartment panel, selects the stable panel, builds the
  target, interpolates short gaps, fits a train-only outlier cap, and flags
  inactive apartments.
- `backend/build_data_quality.py` — writes `backend/data_quality.json`.
- `GET /api/data_quality` — serves the committed report to the dashboard.
- `frontend/src/pages/DataQualityPage.jsx` — new **Data Quality** page with
  coverage, gap, inactivity tables and a monthly availability chart. All
  numbers come from the JSON; nothing is hardcoded in the UI.

## Raw data

| Fact | Value |
|---|---|
| Files found / read / failed | 39 / 39 / 0 |
| Raw rows | 196,449 |
| IST range | 2016-12-14 21:00 → 2018-06-04 23:30 |
| UTC range | 2016-12-14 15:30 → 2018-06-04 18:00 |
| Hourly panel | 12,891 hours × 39 apartments |
| Hours with no readings | 5,291 |
| Median / max apartments reporting | 23 / 39 |

Timestamp handling: 187,796 rows sit at :30 UTC (exactly on IST hours); 8,653
sit at :00 UTC and are floored into the containing IST hour. No apartment-hour
bucket is duplicated (0 collisions across all files).

## Stable panel and target

- Stable window: **2017-05-01 → 2017-11-30** (5,136 hours), the longest
  dense block.
- **24 of 39** apartments have ≥90% coverage there (90.1–91.5%); 15 are
  excluded (next best is 89.0%, then 83.9%, then 2.3%).
- `target_wh` is the mean load per reporting stable apartment; `n_reporting`
  counts them. Hours with zero reporters stay NaN.
- Before gap handling: 7,600 valid target hours, mean 335.54 Wh, max
  2,001.12 Wh.
- Gaps: 66 NaN runs. 83 hours in 35 interior runs of ≤6 h are linearly
  imputed; 5,208 hours in 31 longer/edge runs are excluded from training and
  evaluation. Longest run: 3,514 h (2017-12-28 14:00 → 2018-05-23 23:00 IST).
- After gap handling: **7,683 valid target hours**, mean 333 Wh.

## Window counts for Phase 3

| Window | Clock hours | Valid target hours | Imputed | Mean target |
|---|---:|---:|---:|---:|
| Stable window 2017-05→11 | 5,136 | 4,733 | 31 | 351.38 Wh |
| Fold: 2017-09→12 | 2,928 | 2,374 | 4 | 241.92 Wh |
| Fold: 2018-05→06-04 | 840 | 214 | 0 | 419.72 Wh |

The May–Jun 2018 fold is thin: 214 valid hours covering 2018-05-24→2018-06-01
plus a partial 2018-06-04 block, and `n_reporting` is 0 for most hours in the
clock window. Phase 3 must report fold-level sample counts next to every metric.

## Outliers and inactivity

- Outlier cap rule: Q3 + 3×IQR fitted on training hours only (minimum 100
  samples), upper tail clipped after fitting. With a provisional train slice
  before 2017-10-01 (5,543 h), the cap is 1,006.72 Wh; 4 hours exceed it
  (max 2,001.12 Wh). Full-data fit would have been 1,070.31 Wh, so fitting
  only on train matters.
- December 2017 activity: 21 of 38 reporting apartments average <100 Wh/h
  with ≥24 readings, including **13 of the 24 stable apartments**. This is
  flagged as suspect inactivity (likely semester vacancy), not confirmed
  meter failure. Flags are descriptive; no apartment is dropped by this rule.

## Commands

```bash
# Rebuild the data-quality report (writes backend/data_quality.json)
python backend/build_data_quality.py

# Run the API (restart after rebuilding the JSON; it loads at startup)
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Run the dashboard
cd frontend && npm run dev
```

## Remaining data risks

1. **No data at all for 2018-01→04**, and May–Jun 2018 is only ~13 days. Any
   model trained on 2017 and tested on 2018 sees a different season and a much
   smaller sample; this must be stated with the results.
2. **Target composition drifts.** `n_reporting` ranges 0–24 within the stable
   panel; a change in who reports moves the mean. Models must receive
   `n_reporting` as a feature and evaluation must report it.
3. **December vacancy is confounded with metering.** The inactivity flag cannot
   distinguish an empty apartment from a failing meter.
4. **Short-gap interpolation is not repair.** Only 83 h were imputed; long
   outages remain excluded and must never be filled or evaluated.
5. **Outlier cap is fold-dependent.** Phase 3/4 must refit it inside each
   training window; never use the provisional cap above for a test fold.
