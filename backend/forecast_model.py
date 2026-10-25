"""Leakage-safe XGBoost day-ahead forecaster.

Every feature for a target hour t uses only observations at t - lag with
lag >= 24 h, so a forecast for any hour of day D depends only on values before
00:00 IST of D. The model is trained on labels from the fold's training window
and never sees evaluation-window targets during fitting.
"""
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

import data_pipeline as dp

optuna.logging.set_verbosity(optuna.logging.WARNING)

# The 24-47 h block is the 24-hour window ending 24 h before the target hour;
# the weekly lags provide the same clock hour one and two weeks earlier.
FEATURE_LAGS = tuple(range(24, 48)) + (168, 336)

# Hand-curated major national holidays and festivals inside the data period.
# This is a transparent approximation, not an authoritative government calendar.
HOLIDAYS = frozenset({
    "2016-12-25",
    "2017-01-01", "2017-01-26", "2017-03-13", "2017-04-14", "2017-06-26",
    "2017-08-07", "2017-08-14", "2017-08-15", "2017-09-02", "2017-09-30",
    "2017-10-02", "2017-10-19", "2017-10-20", "2017-11-04", "2017-12-25",
    "2018-01-01", "2018-01-26", "2018-03-02", "2018-03-30", "2018-04-30",
})

# Assumed academic terms: Jan-Apr and Jul-Nov in session, May-Jun and Dec breaks.
SEMESTER_MONTHS = frozenset({1, 2, 3, 4, 7, 8, 9, 10, 11})

FEATURE_DEFAULTS = {
    "drop_annual_calendar": False,
    "use_weather": True,
    "oracle_temperature": False,
}

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}

TUNE_TRIALS = 30
CV_SPLITS = 3
CV_VAL_DAYS = 14
EARLY_STOPPING_ROUNDS = 25
MAX_ESTIMATORS = 800


def make_features(series, hours, reporting=None, weather=None,
                  feature_options=None, near_zero_fraction=None):
    """Build lag, calendar and weather features for the given target hours.

    `hours` must be tz-aware IST timestamps. Reporting counts and weather, when
    supplied, are lagged by at least 24 h so they cannot reveal the forecast
    day. Missing history stays NaN; XGBoost handles it natively.

    `feature_options` is an ablation switch used by scripts/feature_ablation.py.
    The oracle option adds target-hour temperature, which is not known at the
    forecast origin and must never be treated as a deployable result.
    `near_zero_fraction` is an hourly series of stable-apartment vacancy share;
    it enters as a 24 h lag for the Phase 9 level experiment.
    """
    options = {**FEATURE_DEFAULTS, **(feature_options or {})}
    hours = pd.DatetimeIndex(hours)
    block = {}
    for lag in FEATURE_LAGS:
        block[f"lag_{lag}"] = series.reindex(
            hours - pd.Timedelta(hours=lag)
        ).to_numpy(dtype=float)

    daily = pd.DataFrame({lag: block[f"lag_{lag}"] for lag in range(24, 48)})
    block["lag24_47_mean"] = daily.mean(axis=1).to_numpy()
    block["lag24_47_std"] = daily.std(axis=1).to_numpy()
    block["lag24_47_min"] = daily.min(axis=1).to_numpy()
    block["lag24_47_max"] = daily.max(axis=1).to_numpy()

    local = hours.tz_convert(dp.TZ)
    day_of_year = local.day_of_year.to_numpy(dtype=float)
    block["hour"] = local.hour.to_numpy(dtype=int)
    block["dow"] = local.day_of_week.to_numpy(dtype=int)
    block["is_weekend"] = (local.day_of_week.to_numpy() >= 5).astype(int)
    if not options["drop_annual_calendar"]:
        block["month"] = local.month.to_numpy(dtype=int)
        block["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        block["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    block["is_holiday"] = np.array(
        [int(day in HOLIDAYS) for day in local.strftime("%Y-%m-%d")], dtype=int
    )
    block["semester_in_session"] = np.array(
        [int(month in SEMESTER_MONTHS) for month in local.month], dtype=int
    )

    if reporting is not None:
        block["n_reporting_lag_24"] = reporting.reindex(
            hours - pd.Timedelta(hours=24)
        ).to_numpy(dtype=float)
        counts = pd.DataFrame({
            lag: reporting.reindex(hours - pd.Timedelta(hours=lag)).to_numpy(dtype=float)
            for lag in range(24, 48)
        })
        block["n_reporting_lag24_47_mean"] = counts.mean(axis=1).to_numpy()

    if weather is not None and options["use_weather"]:
        for name in weather.columns:
            block[f"weather_{name}_lag_24"] = weather[name].reindex(
                hours - pd.Timedelta(hours=24)
            ).to_numpy(dtype=float)
        lagged = {
            name: pd.DataFrame({
                lag: weather[name].reindex(
                    hours - pd.Timedelta(hours=lag)
                ).to_numpy(dtype=float)
                for lag in range(24, 48)
            })
            for name in ("temperature_2m", "precipitation")
        }
        block["weather_temp_lag24_47_mean"] = lagged["temperature_2m"].mean(
            axis=1
        ).to_numpy()
        block["weather_precip_lag24_47_sum"] = lagged["precipitation"].sum(
            axis=1, min_count=1
        ).to_numpy()
    if options["oracle_temperature"]:
        if weather is None or "temperature_2m" not in weather:
            raise ValueError("oracle temperature needs the weather frame")
        block["oracle_temperature_same_hour"] = weather["temperature_2m"].reindex(
            hours
        ).to_numpy(dtype=float)
    if near_zero_fraction is not None:
        block["near_zero_fraction_lag_24"] = near_zero_fraction.reindex(
            hours - pd.Timedelta(hours=24)
        ).to_numpy(dtype=float)
    return pd.DataFrame(block, index=hours)


def _gain_importance(model, features):
    """XGBoost gain importance, normalized to shares, for trained features."""
    gains = model.get_booster().get_score(importance_type="gain")
    total = sum(float(gains.get(feature, 0.0)) for feature in features)
    rows = [
        {
            "feature": feature,
            "gain": round(float(gains.get(feature, 0.0)), 4),
            "share": round(float(gains.get(feature, 0.0)) / total, 4)
            if total > 0 else 0.0,
        }
        for feature in features
    ]
    rows.sort(key=lambda row: row["gain"], reverse=True)
    return {"method": "gain", "features": rows}


def _hour_range(start, end):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        return pd.date_range(start_ts, end_ts, freq="h")
    return pd.date_range(start_ts, end_ts, freq="h", tz=dp.TZ)


def _training_frame(series, hours, reporting=None, weather=None,
                    feature_options=None, near_zero_fraction=None):
    features = make_features(
        series, hours, reporting=reporting, weather=weather,
        feature_options=feature_options,
        near_zero_fraction=near_zero_fraction,
    )
    labels = series.reindex(hours).astype(float)
    keep = labels.notna() & features["lag_24"].notna()
    return features.loc[keep], labels.loc[keep]


def _apply_target_scale(features, labels, target_scale):
    """Divide labels by a known-at-origin scale and drop hours without one."""
    scale = target_scale.reindex(labels.index).astype(float)
    keep = scale.notna() & np.isfinite(scale) & (scale > 0)
    if not keep.any():
        return features.iloc[0:0], labels.iloc[0:0], scale.iloc[0:0]
    return (
        features.loc[keep],
        labels.loc[keep] / scale.loc[keep],
        scale.loc[keep],
    )


def _normalization_metadata(scales):
    if scales is None or len(scales) == 0:
        return None
    return {
        "method": "target divided by the trailing 7-day mean ending 1 h before each day's origin",
        "n_train_scaled": int(len(scales)),
        "scale_min_wh": round(float(scales.min()), 4),
        "scale_max_wh": round(float(scales.max()), 4),
        "scale_mean_wh": round(float(scales.mean()), 4),
    }


def _capped(labels):
    if len(labels) >= dp.OUTLIER_MIN_SAMPLES:
        cap = dp.fit_outlier_cap(labels)
        clipped, n_capped = dp.apply_outlier_cap(labels, cap)
        return clipped, cap, n_capped
    return labels, None, 0


def time_series_splits(train_start, train_end, n_splits=CV_SPLITS,
                       val_days=CV_VAL_DAYS):
    """Expanding-window CV splits entirely inside the training window.

    Validation blocks are contiguous, non-overlapping, and end at the last
    training day; each block's training data ends one hour before it starts.
    """
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    if start.tzinfo is None:
        start = start.tz_localize(dp.TZ)
    if end.tzinfo is None:
        end = end.tz_localize(dp.TZ)

    total_days = (end.normalize() - start.normalize()).days + 1
    effective_days = min(val_days, max(5, total_days // (n_splits + 2)))
    splits = []
    for i in range(n_splits):
        val_end_day = end.normalize() - pd.Timedelta(days=i * effective_days)
        val_start_day = val_end_day - pd.Timedelta(days=effective_days - 1)
        split_train_end = val_start_day - pd.Timedelta(hours=1)
        if split_train_end <= start:
            continue
        splits.append({
            "train_start": str(start),
            "train_end": str(split_train_end),
            "val_start": str(val_start_day),
            "val_end": str(val_end_day + pd.Timedelta(hours=23)),
        })
    return list(reversed(splits))


def _sample_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 20.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "n_estimators": MAX_ESTIMATORS,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }


def _prepare_cv(series, splits, reporting=None, weather=None,
                feature_options=None, target_scale=None,
                near_zero_fraction=None):
    prepared = []
    for split in splits:
        x_train, y_train = _training_frame(
            series,
            _hour_range(split["train_start"], split["train_end"]),
            reporting=reporting,
            weather=weather,
            feature_options=feature_options,
            near_zero_fraction=near_zero_fraction,
        )
        if target_scale is not None:
            x_train, y_train, _ = _apply_target_scale(
                x_train, y_train, target_scale
            )
        y_train, _, _ = _capped(y_train)
        x_val, y_val = _training_frame(
            series,
            _hour_range(split["val_start"], split["val_end"]),
            reporting=reporting,
            weather=weather,
            feature_options=feature_options,
            near_zero_fraction=near_zero_fraction,
        )
        if target_scale is not None:
            x_val, y_val, _ = _apply_target_scale(x_val, y_val, target_scale)
        if len(x_train) and len(x_val):
            prepared.append((x_train, y_train, x_val, y_val))
    return prepared


def _cv_mae(params, prepared):
    scores = []
    for x_train, y_train, x_val, y_val in prepared:
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        pred = np.maximum(model.predict(x_val).astype(float), 0.0)
        scores.append(float(np.mean(np.abs(pred - y_val.to_numpy(dtype=float)))))
    return float(np.mean(scores)) if scores else float("inf")


def tune(series, train_start, train_end, reporting=None, weather=None,
         feature_options=None, target_scale=None, near_zero_fraction=None,
         n_trials=TUNE_TRIALS, seed=42):
    """Tune hyperparameters on expanding-window CV inside the training window."""
    splits = time_series_splits(train_start, train_end)
    if not splits:
        raise ValueError("training window is too short for time-series CV")
    prepared = _prepare_cv(
        series, splits, reporting=reporting, weather=weather,
        feature_options=feature_options, target_scale=target_scale,
        near_zero_fraction=near_zero_fraction,
    )
    if not prepared:
        raise ValueError("no usable CV split inside the training window")
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda trial: _cv_mae(_sample_params(trial), prepared),
        n_trials=n_trials,
    )
    return dict(study.best_trial.params), float(study.best_value), splits


def _fit_tuned(series, train_start, train_end, reporting=None, weather=None,
               feature_options=None, target_scale=None,
               near_zero_fraction=None, n_trials=TUNE_TRIALS):
    best, cv_mae, splits = tune(
        series, train_start, train_end, reporting=reporting, weather=weather,
        feature_options=feature_options, target_scale=target_scale,
        near_zero_fraction=near_zero_fraction, n_trials=n_trials,
    )
    params = dict(best)
    params.update({
        "n_estimators": MAX_ESTIMATORS,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    })

    prepared = _prepare_cv(
        series, splits, reporting=reporting, weather=weather,
        feature_options=feature_options, target_scale=target_scale,
        near_zero_fraction=near_zero_fraction,
    )
    best_iterations = []
    for x_train, y_train, x_val, y_val in prepared:
        early = xgb.XGBRegressor(**params)
        early.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        best_iterations.append(int(early.best_iteration) + 1)
    best_iteration = int(np.median(best_iterations)) if best_iterations else 200
    best_iteration = max(1, min(best_iteration, MAX_ESTIMATORS))

    x_all, y_all = _training_frame(
        series, _hour_range(train_start, train_end),
        reporting=reporting, weather=weather,
        feature_options=feature_options,
        near_zero_fraction=near_zero_fraction,
    )
    scales = None
    if target_scale is not None:
        x_all, y_all, scales = _apply_target_scale(x_all, y_all, target_scale)
    y_all, cap, n_capped = _capped(y_all)
    final_params = {
        key: value for key, value in params.items()
        if key != "early_stopping_rounds"
    }
    final_params["n_estimators"] = best_iteration
    model = xgb.XGBRegressor(**final_params)
    model.fit(x_all, y_all)

    metadata = {
        "n_train": int(len(y_all)),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "cap_wh": None if cap is None else round(float(cap), 4),
        "n_capped": int(n_capped),
        "params": {
            key: value for key, value in final_params.items() if key != "n_jobs"
        },
        "features": list(x_all.columns),
        "feature_options": {**FEATURE_DEFAULTS, **(feature_options or {})},
        "normalization": _normalization_metadata(scales),
        "near_zero_fraction_feature": near_zero_fraction is not None,
        "importance": _gain_importance(model, x_all.columns),
        "tuning": {
            "n_trials": int(n_trials),
            "best_cv_mae": round(cv_mae, 4),
            "best_params": best,
            "cv_splits": splits,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iterations": best_iterations,
            "best_iteration": best_iteration,
            "selection": "median best_iteration across the CV validation slices",
        },
    }
    return model, metadata


def fit(series, train_start, train_end, params=None, reporting=None,
        weather=None, tune_model=False, n_trials=TUNE_TRIALS,
        feature_options=None, target_scale=None, near_zero_fraction=None):
    """Fit one pooled model on the training window.

    With `tune_model`, Optuna selects hyperparameters on expanding-window CV
    inside the training window, then early stopping picks the iteration count
    on a train-side validation slice. Without it, `params`/defaults are used.
    With `target_scale`, labels are divided by a known-at-origin level and
    predictions are multiplied back in `predict`.
    """
    if tune_model:
        return _fit_tuned(
            series, train_start, train_end, reporting=reporting,
            weather=weather, feature_options=feature_options,
            target_scale=target_scale,
            near_zero_fraction=near_zero_fraction, n_trials=n_trials,
        )

    hours = _hour_range(train_start, train_end)
    features, labels = _training_frame(
        series, hours, reporting=reporting, weather=weather,
        feature_options=feature_options,
        near_zero_fraction=near_zero_fraction,
    )
    scales = None
    if target_scale is not None:
        features, labels, scales = _apply_target_scale(
            features, labels, target_scale
        )
    labels, cap, n_capped = _capped(labels)

    effective = dict(DEFAULT_PARAMS)
    if params:
        effective.update(params)
    model = xgb.XGBRegressor(**effective)
    model.fit(features, labels)

    metadata = {
        "n_train": int(len(labels)),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "cap_wh": None if cap is None else round(float(cap), 4),
        "n_capped": int(n_capped),
        "params": {k: v for k, v in effective.items() if k != "n_jobs"},
        "features": list(features.columns),
        "feature_options": {**FEATURE_DEFAULTS, **(feature_options or {})},
        "normalization": _normalization_metadata(scales),
        "near_zero_fraction_feature": near_zero_fraction is not None,
        "importance": _gain_importance(model, features.columns),
    }
    return model, metadata


def predict(series, model, start, end, reporting=None, weather=None,
            feature_options=None, target_scale=None,
            near_zero_fraction=None):
    """Predict every hour in [start, end]; negative loads are clipped to zero.

    With `target_scale`, normalized predictions are multiplied by each day's
    known-at-origin level; hours without a valid scale stay NaN.
    """
    hours = _hour_range(start, end)
    features = make_features(
        series, hours, reporting=reporting, weather=weather,
        feature_options=feature_options,
        near_zero_fraction=near_zero_fraction,
    )
    values = np.maximum(model.predict(features).astype(float), 0.0)
    if target_scale is not None:
        scale = target_scale.reindex(hours).to_numpy(dtype=float)
        valid = np.isfinite(scale) & (scale > 0)
        values = np.where(valid, values * scale, np.nan)
    return pd.Series(values, index=hours, name="xgboost")


def forecast(series, train_start, train_end, test_start, test_end, params=None,
             reporting=None, weather=None, tune_model=False,
             n_trials=TUNE_TRIALS, feature_options=None, target_scale=None,
             near_zero_fraction=None):
    """Fit on the training window and forecast the test window."""
    model, metadata = fit(
        series, train_start, train_end, params=params, reporting=reporting,
        weather=weather, tune_model=tune_model, n_trials=n_trials,
        feature_options=feature_options, target_scale=target_scale,
        near_zero_fraction=near_zero_fraction,
    )
    prediction = predict(
        series, model, test_start, test_end, reporting=reporting, weather=weather,
        feature_options=feature_options, target_scale=target_scale,
        near_zero_fraction=near_zero_fraction,
    )
    if target_scale is not None and metadata.get("normalization"):
        metadata["normalization"]["predicted_hours"] = int(prediction.notna().sum())
        metadata["normalization"]["missing_scale_hours"] = int(prediction.isna().sum())
    return prediction, metadata
