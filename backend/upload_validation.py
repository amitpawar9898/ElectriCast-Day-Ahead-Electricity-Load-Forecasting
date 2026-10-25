"""Column-mapped CSV upload validation and day-ahead evaluation.

The upload path uses the same IST day boundaries and >=24 h information set as
the campus backtest. It never reports a metric without also reporting the rows,
hours, and coverage behind it.
"""
import io
import re
import threading
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import data_pipeline as dp

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PREVIEW_BYTES = 256 * 1024
PREVIEW_ROWS = 100

UNITS = {
    "Wh": {
        "label": "Wh per reading",
        "detail": "energy readings summed within each hour",
        "factor": 1.0,
        "aggregation": "sum",
    },
    "kWh": {
        "label": "kWh per reading",
        "detail": "energy readings summed within each hour, then ×1,000 to Wh",
        "factor": 1000.0,
        "aggregation": "sum",
    },
    "W": {
        "label": "W (average power)",
        "detail": "power readings averaged within each hour; mean W over 1 h = Wh",
        "factor": 1.0,
        "aggregation": "mean",
    },
    "kW": {
        "label": "kW (average power)",
        "detail": "power readings averaged within each hour, then ×1,000 to Wh",
        "factor": 1000.0,
        "aggregation": "mean",
    },
}

TIMEZONES = [
    "UTC",
    "Asia/Kolkata",
    "Asia/Dubai",
    "Asia/Singapore",
    "Europe/London",
    "Europe/Berlin",
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "Australia/Sydney",
]

TIMESTAMP_MODES = {
    "auto": "Auto-detect",
    "epoch_s": "Epoch seconds",
    "epoch_ms": "Epoch milliseconds",
    "datetime": "Date/time string",
}

ZERO_SHOT_DEFAULTS = "fixed 4.1 XGBoost defaults (400 trees, depth 4, learning rate 0.03)"

_ZERO_SHOT = None
_ZERO_SHOT_LOCK = threading.Lock()


class UploadError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def options_payload():
    return {
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "preview_bytes": PREVIEW_BYTES,
        "units": [
            {"key": key, **value}
            for key, value in UNITS.items()
        ],
        "timezones": TIMEZONES,
        "timestamp_modes": [
            {"key": key, "label": label}
            for key, label in TIMESTAMP_MODES.items()
        ],
    }


def _detect_delimiter(header):
    counts = {",": header.count(","), ";": header.count(";"),
              "\t": header.count("\t"), "|": header.count("|")}
    delimiter = max(counts, key=counts.get)
    return delimiter if counts[delimiter] > 0 else ","


def _read_csv(data, nrows=None):
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data[:65536].decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        header = text.splitlines()[0] if text.splitlines() else ""
        delimiter = _detect_delimiter(header)
        try:
            frame = pd.read_csv(
                io.BytesIO(data), sep=delimiter, encoding=encoding,
                nrows=nrows, on_bad_lines="skip" if nrows else "error",
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError, ValueError) as exc:
            last_error = exc
            continue
        if frame.empty:
            raise UploadError("The CSV contains a header but no data rows.")
        return frame, delimiter
    raise UploadError(f"Could not parse the CSV: {last_error}")


def _numeric_ratio(series):
    return float(pd.to_numeric(series, errors="coerce").notna().mean()) if len(series) else 0.0


def _timestamp_score(name):
    lower = name.lower()
    if "hour" in lower:
        return -6
    score = 0
    if "timestamp" in lower:
        score += 6
    if "datetime" in lower or "date time" in lower:
        score += 5
    if lower in ("ts", "time", "date"):
        score += 5
    elif "date" in lower:
        score += 3
    elif "time" in lower:
        score += 1
    if "transaction" in lower:
        score -= 4
    return score


def _value_score(name):
    lower = name.lower()
    score = 0
    for token, weight in (
        ("energy", 4), ("consumption", 3), ("power", 3), ("load", 3),
        ("kwh", 2), ("wh", 2), ("kw", 1),
    ):
        if token in lower:
            score += weight
    for token in ("transaction", "voltage", "current", "apparent", "hour", "month", "day"):
        if token in lower:
            score -= 4
    return score


def _series_score(name):
    lower = name.lower()
    score = 0
    for token in ("apartment", "building", "meter", "series", "home", "house"):
        if token in lower:
            score += 4
    if lower.endswith("id") or lower == "id":
        score += 1
    if "transaction" in lower:
        score -= 8
    return score


def _infer_unit(name):
    lower = name.lower()
    if "kwh" in lower:
        return "kWh"
    if "kw" in lower:
        return "kW"
    if "wh" in lower:
        return "Wh"
    if re.search(r"(^|[^a-z])w($|[^a-z])", lower) or "power" in lower:
        return "W"
    return "Wh"


def _json_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else ""
    return value


def _suggested_mapping(frame):
    columns = list(frame.columns)
    numeric = {column: _numeric_ratio(frame[column]) for column in columns}

    timestamp_candidates = [
        column for column in columns
        if _timestamp_score(column) > 0
        and (numeric[column] > 0.8
             or not pd.api.types.is_numeric_dtype(frame[column]))
    ]
    timestamp_col = max(
        timestamp_candidates,
        key=lambda column: (_timestamp_score(column), numeric[column]),
        default=None,
    )

    value_candidates = [
        column for column in columns
        if column != timestamp_col and numeric[column] >= 0.8
        and _value_score(column) > 0
    ]
    value_col = max(
        value_candidates,
        key=lambda column: (_value_score(column), numeric[column]),
        default=None,
    )
    if value_col is None:
        numeric_only = [
            column for column in columns
            if column != timestamp_col and numeric[column] >= 0.8
        ]
        value_col = numeric_only[0] if numeric_only else None

    series_candidates = [
        column for column in columns
        if column not in (timestamp_col, value_col)
        and _series_score(column) > 0
        and frame[column].nunique(dropna=True) > 1
    ]
    series_col = max(
        series_candidates, key=_series_score, default=None
    )

    timestamp_mode = "datetime"
    if timestamp_col is not None:
        numeric_ts = pd.to_numeric(frame[timestamp_col], errors="coerce")
        if numeric_ts.notna().mean() > 0.8:
            magnitude = float(numeric_ts.abs().median())
            timestamp_mode = "epoch_ms" if magnitude > 1e11 else "epoch_s"

    return {
        "timestamp_col": timestamp_col or "",
        "value_col": value_col or "",
        "series_col": series_col or "",
        "unit": _infer_unit(value_col) if value_col else "Wh",
        "timezone": "UTC",
        "timestamp_mode": timestamp_mode,
    }


def preview_csv(data):
    if not data:
        raise UploadError("The uploaded file is empty.")
    frame, delimiter = _read_csv(data, nrows=PREVIEW_ROWS)
    columns = [str(column) for column in frame.columns]
    sample = frame.head(5).to_dict(orient="records")
    sample_rows = [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in sample
    ]
    return {
        "columns": columns,
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "rows_previewed": int(len(frame)),
        "sample_rows": sample_rows,
        "suggested": _suggested_mapping(frame),
        "options": options_payload(),
    }


def _validate_timezone(timezone):
    if timezone not in TIMEZONES:
        raise UploadError(f"Unsupported timezone '{timezone}'.")
    try:
        ZoneInfo(timezone)
    except Exception as exc:
        raise UploadError(f"Invalid timezone '{timezone}': {exc}") from exc


def _parse_timestamps(values, mode, timezone):
    _validate_timezone(timezone)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())

    if mode == "auto":
        mode = "epoch" if numeric_ratio > 0.8 else "datetime"

    if mode in ("epoch", "epoch_s", "epoch_ms"):
        if numeric_ratio < 0.5:
            raise UploadError(
                "The timestamp column is not mostly numeric; choose date/time string mode."
            )
        if mode == "auto":
            magnitude = float(numeric.abs().median())
            unit = "ms" if magnitude > 1e11 else "s"
        else:
            unit = "ms" if mode == "epoch_ms" else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        resolved = f"epoch_{unit}"
        source_note = "epoch timestamp (absolute; analyzed in IST)"
        had_offset = False
    else:
        text = values.astype(str)
        has_offset = bool(
            text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False).mean() > 0.5
        )
        if has_offset:
            parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
            resolved = "datetime_with_offset"
            source_note = "embedded UTC offset honored; analyzed in IST"
            had_offset = True
        else:
            naive = pd.to_datetime(text, errors="coerce", format="mixed")
            if getattr(naive.dt, "tz", None) is not None:
                parsed = naive.dt.tz_convert("UTC")
                had_offset = True
                resolved = "datetime_with_offset"
                source_note = "embedded UTC offset honored; analyzed in IST"
            else:
                try:
                    parsed = naive.dt.tz_localize(
                        timezone, ambiguous="NaT", nonexistent="NaT"
                    ).dt.tz_convert("UTC")
                except Exception as exc:
                    raise UploadError(
                        f"Could not localize naive timestamps as {timezone}: {exc}"
                    ) from exc
                resolved = "datetime_naive"
                source_note = f"naive timestamps interpreted as {timezone}; analyzed in IST"
                had_offset = False

    ist = parsed.dt.tz_convert(dp.TZ)
    return ist, {
        "mode": resolved,
        "source_timezone": timezone,
        "had_offset": had_offset,
        "source_note": source_note,
    }


def _build_target(data, mapping, filename, size):
    if not data:
        raise UploadError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    frame, delimiter = _read_csv(data)
    columns = [str(column) for column in frame.columns]
    for key in ("timestamp_col", "value_col"):
        column = mapping.get(key)
        if not column or column not in frame.columns:
            raise UploadError(f"Column '{column}' is not present in the CSV.")
    series_col = mapping.get("series_col") or None
    if series_col and series_col not in frame.columns:
        raise UploadError(f"Series column '{series_col}' is not present in the CSV.")
    unit = mapping.get("unit", "Wh")
    if unit not in UNITS:
        raise UploadError(f"Unsupported unit '{unit}'.")

    timestamp_col = mapping["timestamp_col"]
    value_col = mapping["value_col"]
    mode = mapping.get("timestamp_mode", "auto")

    timestamps, timestamp_meta = _parse_timestamps(
        frame[timestamp_col], mode, mapping.get("timezone", "UTC")
    )
    values = pd.to_numeric(frame[value_col], errors="coerce")
    value_invalid = int(values.isna().sum())
    negative = values < 0
    values = values.mask(negative)

    raw_rows = int(len(frame))
    timestamp_invalid = int(timestamps.isna().sum())
    negative_rows = int(negative.sum())

    series = (
        frame[series_col].astype(str).str.strip().replace(
            {"": "(missing)", "nan": "(missing)", "None": "(missing)"}
        )
        if series_col else pd.Series("series", index=frame.index)
    )

    work = pd.DataFrame({
        "time": timestamps,
        "series": series,
        "value_wh": values * UNITS[unit]["factor"],
    }).dropna(subset=["time", "value_wh"])
    used_rows = int(len(work))
    if used_rows == 0:
        report = {
            "filename": filename,
            "bytes": int(size),
            "delimiter": "tab" if delimiter == "\t" else delimiter,
            "rows_raw": raw_rows,
            "rows_timestamp_invalid": timestamp_invalid,
            "rows_value_invalid": value_invalid,
            "rows_negative": negative_rows,
            "rows_used": 0,
            "series_count": 0,
            "unit": unit,
            "timestamp": timestamp_meta,
            "warnings": ["No row has both a valid timestamp and a numeric value."],
        }
        raise UploadError("No usable rows after timestamp and value parsing.", report)

    work["hour"] = work["time"].dt.floor("h")
    aggregation = UNITS[unit]["aggregation"]
    grouped = work.groupby(["series", "hour"])["value_wh"]
    hourly = grouped.sum(min_count=1) if aggregation == "sum" else grouped.mean()
    per_series = hourly.unstack("series").sort_index()
    reporting = per_series.notna().sum(axis=1).astype(int)
    target = per_series.mean(axis=1)

    full_index = pd.date_range(
        per_series.index.min(), per_series.index.max(), freq="h", tz=dp.TZ
    )
    target = target.reindex(full_index)
    filled, imputed, excluded, runs = dp.fill_short_gaps(target)

    observed = int(target.notna().sum())
    valid_hours = int(filled.notna().sum())
    span_hours = int(len(full_index))
    coverage = observed / span_hours if span_hours else 0.0
    longest_run = max(runs, key=lambda run: run["hours"]) if runs else None
    series_coverage = per_series.notna().mean()
    target_stats = filled.dropna()

    warnings = [
        f"All analysis uses IST day boundaries and reports Wh. Source: {timestamp_meta['source_note']}."
    ]
    if timestamp_invalid:
        warnings.append(
            f"{timestamp_invalid:,} rows had an unparseable timestamp and were dropped."
        )
    if value_invalid:
        warnings.append(
            f"{value_invalid:,} rows had a missing or non-numeric {value_col} and were dropped."
        )
    if negative_rows:
        warnings.append(
            f"{negative_rows:,} rows had a negative value and were excluded, because the target is non-negative load."
        )
    if len(work) > len(hourly):
        warnings.append(
            f"{len(work) - len(hourly):,} extra readings shared a series-hour and were aggregated by {aggregation}."
        )
    warnings.append(
        f"Unit {unit}: {UNITS[unit]['detail']}."
    )
    if series_col is None:
        warnings.append(
            "No series identifier was selected; all readings were treated as one series."
        )
    else:
        warnings.append(
            f"Target is the mean across {per_series.shape[1]:,} series at each hour (reporting series counted per hour)."
        )
    if coverage < 0.9:
        warnings.append(
            f"Only {coverage:.1%} of the spanned hours contain a target value; long gaps limit training and evaluation."
        )
    if int(imputed.sum()):
        warnings.append(
            f"{int(imputed.sum()):,} hours in interior gaps of at most {dp.MAX_GAP_HOURS} h were linearly interpolated."
        )
    if int(excluded.sum()):
        warnings.append(
            f"{int(excluded.sum()):,} hours in gaps longer than {dp.MAX_GAP_HOURS} h or at the edges were excluded."
        )
    if float(series_coverage.min()) < 0.5:
        warnings.append(
            f"At least one series covers only {float(series_coverage.min()):.1%} of the spanned hours."
        )
    if longest_run:
        warnings.append(
            f"Longest gap: {longest_run['hours']:,} h from "
            f"{longest_run['start'][:16]} to {longest_run['end'][:16]} IST."
        )

    report = {
        "filename": filename,
        "bytes": int(size),
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "rows_raw": raw_rows,
        "rows_timestamp_invalid": timestamp_invalid,
        "rows_value_invalid": value_invalid,
        "rows_negative": negative_rows,
        "rows_used": used_rows,
        "rows_aggregated": int(len(work) - len(hourly)),
        "series_count": int(per_series.shape[1]),
        "series_coverage_min": round(float(series_coverage.min()), 4),
        "series_coverage_median": round(float(series_coverage.median()), 4),
        "unit": {
            "key": unit,
            "label": UNITS[unit]["label"],
            "detail": UNITS[unit]["detail"],
            "factor_to_wh": UNITS[unit]["factor"],
            "aggregation": aggregation,
        },
        "timestamp": {
            **timestamp_meta,
            "column": timestamp_col,
            "first_ist": target.index[0].isoformat() if span_hours else None,
            "last_ist": target.index[-1].isoformat() if span_hours else None,
        },
        "mapping": {
            "timestamp_col": timestamp_col,
            "value_col": value_col,
            "series_col": series_col,
            "unit": unit,
            "timezone": mapping.get("timezone", "UTC"),
            "timestamp_mode": mode,
        },
        "hourly": {
            "span_hours": span_hours,
            "observed_hours": observed,
            "valid_hours": valid_hours,
            "coverage": round(coverage, 4),
            "reporting_median": int(reporting.median()) if len(reporting) else 0,
            "reporting_max": int(reporting.max()) if len(reporting) else 0,
            "imputed_hours": int(imputed.sum()),
            "excluded_hours": int(excluded.sum()),
            "gap_runs": len(runs),
            "max_gap_hours": dp.MAX_GAP_HOURS,
            "longest_gap_hours": int(longest_run["hours"]) if longest_run else 0,
            "zero_hours": int((filled == 0).sum()),
            "target_mean_wh": round(float(target_stats.mean()), 2),
            "target_median_wh": round(float(target_stats.median()), 2),
            "target_max_wh": round(float(target_stats.max()), 2),
        },
        "warnings": warnings,
    }
    return filled, report


def _campus_zero_shot():
    global _ZERO_SHOT
    if _ZERO_SHOT is not None:
        return _ZERO_SHOT
    with _ZERO_SHOT_LOCK:
        if _ZERO_SHOT is not None:
            return _ZERO_SHOT

        import forecast_model as fm
        import xgboost as xgb

        apartments, _ = dp.load_apartments()
        panel, _ = dp.build_panel(apartments)
        stable, _ = dp.stable_apartments(panel)
        target, _ = dp.build_target(panel, stable)
        filled, _, _, _ = dp.fill_short_gaps(target)
        valid = filled.dropna()
        features = fm.make_features(filled, valid.index)
        model = xgb.XGBRegressor(**fm.DEFAULT_PARAMS)
        model.fit(features, valid.to_numpy(dtype=float))
        _ZERO_SHOT = {
            "model": model,
            "features": list(features.columns),
            "train_mean_wh": float(valid.mean()),
            "n_train": int(len(valid)),
            "params": fm.DEFAULT_PARAMS,
        }
    return _ZERO_SHOT


def _zero_shot_predictions(filled, test_hours, train_end):
    import forecast_model as fm

    bundle = _campus_zero_shot()
    train_mean = float(filled.loc[:train_end].dropna().mean())
    if not np.isfinite(train_mean) or train_mean <= 0:
        raise ValueError("upload training mean is not positive")
    scale = bundle["train_mean_wh"] / train_mean
    scaled = filled * scale
    features = fm.make_features(scaled, test_hours).reindex(columns=bundle["features"])
    usable = features.notna().any(axis=1)
    predictions = pd.Series(np.nan, index=test_hours)
    if usable.any():
        predictions.loc[usable] = bundle["model"].predict(features.loc[usable])
    predictions = predictions.clip(lower=0) / scale
    return predictions, {
        "status": "ok",
        "n_train_campus": bundle["n_train"],
        "scale_factor_to_campus": round(float(scale), 6),
        "train_part_mean_wh": round(train_mean, 4),
        "campus_train_mean_wh": round(bundle["train_mean_wh"], 4),
        "features": len(bundle["features"]),
    }


def _fit_upload_model(filled, test_hours, train_end):
    import forecast_model as fm
    import xgboost as xgb

    train_hours = filled.loc[:train_end].index
    labels = filled.reindex(train_hours)
    features = fm.make_features(filled, train_hours)
    keep = labels.notna() & features.notna().any(axis=1)
    if int(keep.sum()) < 24 * 7:
        return pd.Series(np.nan, index=test_hours), {
            "status": "no_data",
            "reason": "fewer than 168 valid training hours",
            "n_train": int(keep.sum()),
        }

    y = labels.loc[keep].to_numpy(dtype=float)
    cap = None
    if len(y) >= dp.OUTLIER_MIN_SAMPLES:
        cap = dp.fit_outlier_cap(y)
        y = np.minimum(y, cap)
    model = xgb.XGBRegressor(**fm.DEFAULT_PARAMS)
    model.fit(features.loc[keep], y)

    test_features = fm.make_features(filled, test_hours)
    usable = test_features.notna().any(axis=1)
    predictions = pd.Series(np.nan, index=test_hours)
    if usable.any():
        predictions.loc[usable] = model.predict(test_features.loc[usable])
    predictions = predictions.clip(lower=0)
    return predictions, {
        "status": "ok",
        "n_train": int(keep.sum()),
        "cap_wh": round(float(cap), 4) if cap is not None else None,
        "n_capped": int((labels.loc[keep] > cap).sum()) if cap is not None else 0,
        "params": fm.DEFAULT_PARAMS,
        "features": int(features.shape[1]),
    }


def _series(values, hours):
    return pd.Series(values, index=hours, dtype=float)


def evaluate_upload(filled, train_fraction=0.6):
    from evaluate import (
        BASELINES, baseline_forecasts, compute_metrics, mase_scale,
        paired_significance, select_best_baseline,
    )

    valid = filled.dropna()
    if len(valid) < 24 * 14:
        return {
            "status": "insufficient_data",
            "reason": f"need at least 336 valid target hours, found {len(valid)}",
            "protocol": {},
            "models": {},
            "split": {},
            "daily": None,
            "notes": [],
        }

    split_position = int(round(len(valid) * train_fraction))
    split_position = min(max(split_position, 1), len(valid) - 1)
    test_start = valid.index[split_position].floor("D")
    test_end = valid.index[-1].floor("h")
    if test_end <= test_start:
        return {
            "status": "insufficient_data",
            "reason": "the 60/40 split leaves no test period",
            "protocol": {},
            "models": {},
            "split": {},
            "daily": None,
            "notes": [],
        }

    test_hours = pd.date_range(test_start, test_end, freq="h", tz=dp.TZ)
    if len(test_hours) < 24 * 7:
        return {
            "status": "insufficient_data",
            "reason": f"need at least 7 test days, found {len(test_hours) / 24:.1f}",
            "protocol": {},
            "models": {},
            "split": {},
            "daily": None,
            "notes": [],
        }

    train_end = test_start - pd.Timedelta(hours=1)
    train_hours = filled.loc[:train_end].dropna()
    if len(train_hours) < 24 * 14:
        return {
            "status": "insufficient_data",
            "reason": f"need at least 336 valid training hours, found {len(train_hours)}",
            "protocol": {},
            "models": {},
            "split": {},
            "daily": None,
            "notes": [],
        }

    actual = filled.reindex(test_hours)
    baseline_preds = baseline_forecasts(filled, test_start, test_end)
    eval_mask = actual.notna() & baseline_preds["seasonal_naive_24"].notna()
    if int(eval_mask.sum()) < 24:
        return {
            "status": "insufficient_data",
            "reason": "fewer than 24 evaluable test hours",
            "protocol": {},
            "models": {},
            "split": {},
            "daily": None,
            "notes": [],
        }

    scale = mase_scale(filled.loc[:train_end])
    best_baseline, mask_name, mask_hours, selection = select_best_baseline(
        actual, baseline_preds, eval_mask, scale
    )

    zero_pred = pd.Series(np.nan, index=test_hours)
    zero_meta = {"status": "not_run"}
    try:
        zero_pred, zero_meta = _zero_shot_predictions(filled, test_hours, train_end)
    except Exception as exc:
        zero_meta = {"status": "error", "reason": str(exc)}

    fit_pred = pd.Series(np.nan, index=test_hours)
    fit_meta = {"status": "not_run"}
    try:
        fit_pred, fit_meta = _fit_upload_model(filled, test_hours, train_end)
    except Exception as exc:
        fit_meta = {"status": "error", "reason": str(exc)}

    model_preds = {
        **baseline_preds,
        "zero_shot_xgboost": zero_pred,
        "fit_60_xgboost": fit_pred,
    }
    labels = {
        "persistence": "Persistence",
        "seasonal_naive_24": "Seasonal naive 24h",
        "same_hour_last_week": "Same hour last week",
        "mean_last_4_same_weekdays": "Mean of 4 weekdays",
        "zero_shot_xgboost": "Zero-shot XGBoost",
        "fit_60_xgboost": "Fit-on-60% XGBoost",
    }
    descriptions = {
        "persistence": "last valid value before the day's origin, held flat for 24 h",
        "seasonal_naive_24": "same hour on the previous day",
        "same_hour_last_week": "same hour one week earlier",
        "mean_last_4_same_weekdays": "mean of the same hour on the previous four same weekdays",
        "zero_shot_xgboost": "campus-trained XGBoost applied without fitting on the upload; target lags and calendar only, scale-aligned using the training-part mean",
        "fit_60_xgboost": "XGBoost fitted on the first 60% of uploaded target hours with the 4.1 fixed defaults and >=24 h lag features",
    }
    kinds = {name: "baseline" for name in BASELINES}
    kinds.update({"zero_shot_xgboost": "learned", "fit_60_xgboost": "learned"})

    models = {}
    for name, prediction in model_preds.items():
        metrics = compute_metrics(actual, prediction, scale)
        entry = {
            "label": labels[name],
            "kind": kinds[name],
            "description": descriptions[name],
            **metrics,
        }
        if name == "zero_shot_xgboost":
            entry["fit"] = zero_meta
        elif name == "fit_60_xgboost":
            entry["fit"] = fit_meta
        models[name] = entry

    significance = {}
    for name in ("zero_shot_xgboost", "fit_60_xgboost"):
        if best_baseline and models[name]["status"] == "ok":
            significance[name] = paired_significance(
                actual, model_preds[name], baseline_preds[best_baseline]
            )

    daily_frame = pd.DataFrame({
        "actual": actual,
        "seasonal_naive_24": baseline_preds["seasonal_naive_24"],
        "zero_shot_xgboost": zero_pred,
        "fit_60_xgboost": fit_pred,
    }, index=test_hours)
    daily = daily_frame.resample("D").mean()
    daily = daily[daily.index <= test_end.floor("D")]
    daily_payload = {
        "labels": [day.strftime("%Y-%m-%d") for day in daily.index],
        "actual": [round(float(value), 2) if np.isfinite(value) else None for value in daily["actual"]],
        "models": {
            key: [round(float(value), 2) if np.isfinite(value) else None
                  for value in daily[key]]
            for key in ("seasonal_naive_24", "zero_shot_xgboost", "fit_60_xgboost")
        },
    }

    notes = [
        "Evaluation uses IST day boundaries and day-ahead origins; all features are at least 24 h old at the target hour.",
        "The evaluation mask requires the actual target and a seasonal-naive forecast. Models are scored on their available hours, matching the campus backtest.",
        "No weather data is used for uploads; campus weather does not describe another building or period.",
        f"Fit-on-60% uses {ZERO_SHOT_DEFAULTS}; no tuning and no test-window labels enter training.",
        "Zero-shot scale alignment uses only the mean of the training part, never the test window.",
    ]
    if zero_meta.get("status") != "ok":
        notes.append(f"Zero-shot model unavailable: {zero_meta.get('reason', zero_meta.get('status'))}.")
    if fit_meta.get("status") != "ok":
        notes.append(f"Fit-on-60% unavailable: {fit_meta.get('reason', fit_meta.get('status'))}.")

    return {
        "status": "ok",
        "protocol": {
            "timezone": "Asia/Kolkata (IST)",
            "origin": "00:00 IST of each test day",
            "information_set": "target observations before the origin; every feature lag is at least 24 h",
            "evaluation_mask": "actual target and seasonal_naive_24 available",
            "train_fraction": train_fraction,
            "split": "first 60% of valid target hours; the test period starts at 00:00 IST of the day containing the 60th-percentile valid hour",
            "mase_scale": "mean absolute 24-h seasonal difference over the training part",
            "significance": "two-sided Wilcoxon signed-rank on paired absolute errors vs the best baseline; unadjusted p-values; small samples flagged",
            "zero_shot": descriptions["zero_shot_xgboost"],
            "fit_60": descriptions["fit_60_xgboost"],
        },
        "split": {
            "train_start": filled.loc[:train_end].dropna().index[0].isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_hours": int(len(train_hours)),
            "test_hours": int(len(test_hours)),
            "eval_hours": int(eval_mask.sum()),
            "test_days": int(len(daily)),
        },
        "models": models,
        "model_order": list(models.keys()),
        "best_baseline": {
            "name": best_baseline,
            "label": labels.get(best_baseline) if best_baseline else None,
            "selection_mask": mask_name,
            "selection_hours": int(mask_hours),
            "selection_metrics": selection,
        },
        "significance": significance,
        "daily": daily_payload,
        "notes": notes,
    }


def validate_and_evaluate(data, mapping, filename, size):
    filled, report = _build_target(data, mapping, filename, size)
    evaluation = evaluate_upload(filled)
    return {
        "status": "success" if evaluation["status"] == "ok" else "partial",
        "validation": report,
        "evaluation": evaluation,
    }
