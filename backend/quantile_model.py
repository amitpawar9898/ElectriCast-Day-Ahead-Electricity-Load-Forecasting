"""Quantile XGBoost day-ahead forecaster (P10/P50/P90).

Uses the same feature set, training window, and train-only outlier cap as the
point XGBoost model, with one boosted model per quantile. Predictions are
clipped at zero and rearranged row-wise so the quantiles never cross.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import data_pipeline as dp
import forecast_model as fm

QUANTILES = (0.1, 0.5, 0.9)
COLUMNS = {0.1: "p10", 0.5: "p50", 0.9: "p90"}


def fit(series, train_start, train_end, reporting=None, weather=None,
        params=None):
    """Fit one model per quantile on the training window."""
    hours = fm._hour_range(train_start, train_end)
    features = fm.make_features(
        series, hours, reporting=reporting, weather=weather
    )
    labels = series.reindex(hours).astype(float)
    keep = labels.notna() & features["lag_24"].notna()
    features = features.loc[keep]
    labels = labels.loc[keep]

    cap = None
    n_capped = 0
    if len(labels) >= dp.OUTLIER_MIN_SAMPLES:
        cap = dp.fit_outlier_cap(labels)
        labels, n_capped = dp.apply_outlier_cap(labels, cap)

    base = dict(fm.DEFAULT_PARAMS)
    if params:
        base.update(params)
    base.pop("objective", None)

    models = {}
    for quantile in QUANTILES:
        effective = dict(base)
        effective["objective"] = "reg:quantileerror"
        effective["quantile_alpha"] = quantile
        model = xgb.XGBRegressor(**effective)
        model.fit(features, labels)
        models[quantile] = model

    metadata = {
        "quantiles": list(QUANTILES),
        "n_train": int(len(labels)),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "cap_wh": None if cap is None else round(float(cap), 4),
        "n_capped": int(n_capped),
        "params": {k: v for k, v in base.items() if k != "n_jobs"},
        "features": list(features.columns),
    }
    return models, metadata


def predict(series, models, start, end, reporting=None, weather=None):
    """Return a P10/P50/P90 frame and the number of rows that crossed."""
    hours = fm._hour_range(start, end)
    features = fm.make_features(
        series, hours, reporting=reporting, weather=weather
    )
    raw = np.column_stack([
        models[quantile].predict(features) for quantile in QUANTILES
    ]).astype(float)
    clipped = np.maximum(raw, 0.0)
    crossings = int((np.diff(clipped, axis=1) < 0).any(axis=1).sum())
    ordered = np.sort(clipped, axis=1)
    frame = pd.DataFrame(
        ordered, index=hours, columns=[COLUMNS[q] for q in QUANTILES]
    )
    return frame, crossings


def forecast(series, train_start, train_end, test_start, test_end, reporting=None,
             weather=None, params=None):
    """Fit on the training window and forecast the test window."""
    models, metadata = fit(
        series, train_start, train_end, reporting=reporting, weather=weather,
        params=params,
    )
    frame, crossings = predict(
        series, models, test_start, test_end, reporting=reporting,
        weather=weather,
    )
    metadata["n_quantile_crossings"] = crossings
    return frame, metadata
