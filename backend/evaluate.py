"""Day-ahead forecast evaluation for ElectriCast.

Baselines forecast every IST hour of a test day using observations strictly
before 00:00 IST of that day. Run from the repo root:

    python backend/evaluate.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import data_pipeline as dp
import forecast_model as fm
import quantile_model as qm
import sarima_model as sm
import weather

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

BASELINES = (
    "persistence",
    "seasonal_naive_24",
    "same_hour_last_week",
    "mean_last_4_same_weekdays",
)

LEARNED_MODELS = ("xgboost", "sarima", "xgboost_quantile")
ALL_MODELS = BASELINES + LEARNED_MODELS

DAY_BOOTSTRAP_BLOCK_DAYS = 7
DAY_BOOTSTRAP_REPS = 2000
DAY_BOOTSTRAP_SEED = 42

CONFORMAL_WINDOW_DAYS = 28
CONFORMAL_MIN_DAYS = 7
CONFORMAL_MIN_SCORES = 24
CONFORMAL_ALPHA = 0.2

# 9.5: one Optuna study on 2016-12-14..2017-05-31, before the first scoring
# test month, frozen across folds. Copied from tuning_results.json; a test
# keeps the two in sync.
POINT_XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.01899361327091004,
    "min_child_weight": 17.44142302645991,
    "subsample": 0.623231219926685,
    "colsample_bytree": 0.7677136574322302,
    "reg_lambda": 8.204192175585105,
    "reg_alpha": 0.7871371146811331,
    "n_estimators": 150,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
}

BASELINE_DESCRIPTIONS = {
    "persistence": "last observed value before the forecast day, held flat for 24 h",
    "seasonal_naive_24": "value at the same hour on the previous day",
    "same_hour_last_week": "value at the same hour one week earlier",
    "mean_last_4_same_weekdays": "mean of the same hour on the previous four same weekdays",
}

LEARNED_MODEL_DESCRIPTIONS = {
    "xgboost": "pooled direct day-ahead gradient-boosted trees on lags of 24-47, 168 and 336 h, calendar/holiday/semester flags, hour of day, lagged reporting counts, and lagged Mumbai weather; trained only on the fold's training labels, clipped at the train-only outlier cap",
    "sarima": "SARIMA(1,0,1)(0,1,1,24) on the gap-handled target, refit weekly on the trailing 60 days before each Monday origin and advanced daily with actuals up to each day's origin",
    "xgboost_quantile": "three XGBoost quantile models (P10/P50/P90) on the same features, training window, and train-only cap as the point XGBoost; fixed untuned hyperparameters; predictions clipped at zero and rearranged so quantiles do not cross",
}

FOLDS = (
    {
        "id": "2017-06",
        "train_end": "2017-05-31 23:00",
        "test_start": "2017-06-01 00:00",
        "test_end": "2017-06-30 23:00",
    },
    {
        "id": "2017-07",
        "train_end": "2017-06-30 23:00",
        "test_start": "2017-07-01 00:00",
        "test_end": "2017-07-31 23:00",
    },
    {
        "id": "2017-08",
        "train_end": "2017-07-31 23:00",
        "test_start": "2017-08-01 00:00",
        "test_end": "2017-08-31 23:00",
    },
    {
        "id": "2017-09_11",
        "train_end": "2017-08-31 23:00",
        "test_start": "2017-09-01 00:00",
        "test_end": "2017-11-30 23:00",
    },
    {
        "id": "2017-12",
        "train_end": "2017-11-30 23:00",
        "test_start": "2017-12-01 00:00",
        "test_end": "2017-12-31 23:00",
    },
    {
        "id": "2018-05_06",
        "train_end": "2017-12-31 23:00",
        "test_start": "2018-05-01 00:00",
        "test_end": "2018-06-04 23:00",
        "stress_test": True,
        "label": "Stress test (May\u2013Jun 2018)",
    },
)

PROTOCOL = {
    "timezone": dp.TZ,
    "target": "mean hourly load per reporting stable apartment (Wh)",
    "forecast": "24 hourly values for every test day",
    "origin": "00:00 IST of the test day",
    "information_set": "target observations strictly before the origin; lag baselines use >=24 h lags",
    "folds": "expanding window with train_start fixed at the earliest valid target hour (2016-12-14 21:00 IST); five scoring folds test Jun, Jul, Aug, Sep-Nov and Dec 2017; May-Jun 2018 is a stress test excluded from every fold mean",
    "evaluation_mask": "target present and seasonal_naive_24 available",
    "mase_scale": "mean absolute 24-h seasonal difference over valid training pairs",
    "best_baseline_selection": "lowest MASE on the all-four-baseline mask when it covers >=50% of evaluation hours (and >=24 h); otherwise on the evaluation mask",
    "significance": "two-sided Wilcoxon signed-rank on per-day MAE differences vs the fold's common-hours best baseline, plus a 95% block-bootstrap CI of the MAE difference (2,000 resamples of contiguous 7-day blocks, seed 42); p-values unadjusted; small_sample flags fewer than 10 days",
    "learned_models": "xgboost is fit on the fold's training window labels only; its features at target hour t use observations at t-lag with lag >= 24 h, so all features predate the day's 00:00 IST origin; reporting counts and weather are lagged >=24 h; calendar/holiday/semester flags are known in advance; labels clipped at a train-only Q3+3*IQR cap. sarima is refit weekly on the trailing 60 days before each Monday origin and advanced daily only with actuals before the day's origin",
    "sarima": "order (1,0,1)(0,1,1,24); weekly refit on the trailing 60 days of gap-handled target; daily state updates with actuals up to each origin; recovery refit after coverage gaps",
    "quantiles": "xgboost_quantile fits one model per quantile (0.1, 0.5, 0.9) with objective reg:quantileerror on the same features, labels, and cap as xgboost; P10-P90 coverage and pinball loss are reported per fold, and P50 is also scored as a point forecast. A rolling conformalized quantile regression step widens or keeps the interval using only the previous 28 test days' conformity scores (minimum 7 days and 24 hours), so no same-day or future actual enters the calibration",
    "conformal": "rolling CQR on P10-P90: for each IST test day, q is the finite-sample 80% empirical quantile of max(P10 - actual, actual - P90) over the prior 28 test days in the same fold; the interval becomes [P10 - q, P90 + q], q is floored at zero, and days with fewer than 7 prior days or 24 prior hours keep the raw interval",
    "tuning": "point xgboost uses hyperparameters frozen from one Optuna study on 2016-12-14..2017-05-31, a window that ends before the first scoring test month; the same parameters are used in every fold, with no per-fold test tuning. Annual calendar features are dropped when that fold's training span is under 365 days. The quantile models keep their fixed untuned defaults (4.6).",
    "common_hours": "evaluation mask intersected with the forecasts of all four baselines and all three learned models, so every row in the common-hours table is scored on exactly the same hours",
}


def load_target_with_reporting():
    """Return the gap-handled target and the stable-apartment reporting count."""
    apartments, _ = dp.load_apartments()
    panel, _ = dp.build_panel(apartments)
    stable, _ = dp.stable_apartments(panel)
    target, reporting = dp.build_target(panel, stable)
    filled, _, _, _ = dp.fill_short_gaps(target)
    return filled, reporting


def load_target():
    """Return the gap-handled target series (Wh per reporting apartment)."""
    filled, _ = load_target_with_reporting()
    return filled


def test_hours(start, end):
    return pd.date_range(start, end, freq="h", tz=dp.TZ)


def _lag_values(series, hours, lag):
    return series.reindex(hours - pd.Timedelta(hours=lag)).to_numpy(dtype=float)


def mase_scale(train_series):
    """Mean absolute 24-h seasonal difference over valid training pairs."""
    diffs = (train_series - train_series.shift(24)).abs().dropna()
    return float(diffs.mean()) if len(diffs) else None


def _clean(value, digits=6):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def compute_metrics(actual, pred, scale=None):
    """Metrics over the timestamps where both actual and prediction exist.

    sMAPE is guarded at a zero denominator (the pair contributes 0); WAPE,
    MASE and R² are None when their denominators are zero or missing.
    """
    frame = pd.concat(
        [pd.Series(actual).rename("actual"), pd.Series(pred).rename("pred")],
        axis=1,
    ).dropna()
    if frame.empty:
        return {
            "status": "no_data", "n": 0, "MAE": None, "RMSE": None,
            "WAPE": None, "MASE": None, "bias": None, "R2": None, "sMAPE": None,
        }

    a = frame["actual"].to_numpy(dtype=float)
    p = frame["pred"].to_numpy(dtype=float)
    err = p - a
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    abs_actual_sum = float(np.abs(a).sum())
    wape = float(abs_err.sum() / abs_actual_sum * 100) if abs_actual_sum > 0 else None

    denom = np.abs(a) + np.abs(p)
    smape_terms = np.divide(2 * abs_err, denom, out=np.zeros_like(abs_err),
                            where=denom > 0)
    smape = float(smape_terms.mean() * 100)

    sst = float(((a - a.mean()) ** 2).sum())
    r2 = float(1 - (err ** 2).sum() / sst) if sst > 0 else None
    mase = float(mae / scale) if scale is not None and scale > 0 else None

    return {
        "status": "ok",
        "n": int(len(a)),
        "MAE": _clean(mae, 4),
        "RMSE": _clean(rmse, 4),
        "WAPE": _clean(wape, 4),
        "MASE": _clean(mase, 4),
        "bias": _clean(float(err.mean()), 4),
        "R2": _clean(r2, 6),
        "sMAPE": _clean(smape, 4),
    }


def _pinball(actual, pred, quantile):
    diff = actual - pred
    return float(np.maximum(quantile * diff, (quantile - 1) * diff).mean())


def interval_metrics(actual, lower, median, upper):
    """P10-P90 coverage, mean interval width, and pinball loss."""
    frame = pd.concat(
        [
            pd.Series(actual).rename("actual"),
            pd.Series(lower).rename("lower"),
            pd.Series(median).rename("median"),
            pd.Series(upper).rename("upper"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return {
            "status": "no_data", "n": 0, "coverage": None, "mean_width_wh": None,
            "pinball_10": None, "pinball_50": None, "pinball_90": None,
            "pinball_mean": None,
        }

    a = frame["actual"].to_numpy(dtype=float)
    lo = frame["lower"].to_numpy(dtype=float)
    mid = frame["median"].to_numpy(dtype=float)
    up = frame["upper"].to_numpy(dtype=float)
    pinball_10 = _pinball(a, lo, 0.1)
    pinball_50 = _pinball(a, mid, 0.5)
    pinball_90 = _pinball(a, up, 0.9)
    return {
        "status": "ok",
        "n": int(len(a)),
        "coverage": _clean(float(((a >= lo) & (a <= up)).mean()), 4),
        "mean_width_wh": _clean(float((up - lo).mean()), 4),
        "pinball_10": _clean(pinball_10, 4),
        "pinball_50": _clean(pinball_50, 4),
        "pinball_90": _clean(pinball_90, 4),
        "pinball_mean": _clean((pinball_10 + pinball_50 + pinball_90) / 3, 4),
    }


def _conformal_empty(reason):
    return {
        "status": "no_data",
        "method": (
            "rolling conformalized quantile regression (CQR) on the P10-P90 "
            "interval using only prior test days' residuals"
        ),
        "score": "max(P10 - actual, actual - P90)",
        "window_days": CONFORMAL_WINDOW_DAYS,
        "min_days": CONFORMAL_MIN_DAYS,
        "min_scores": CONFORMAL_MIN_SCORES,
        "alpha": CONFORMAL_ALPHA,
        "nominal_coverage": 1 - CONFORMAL_ALPHA,
        "q_floor_wh": 0.0,
        "n_days": 0,
        "n_days_calibrated": 0,
        "n_hours": 0,
        "n_hours_calibrated": 0,
        "warmup_days": 0,
        "q_wh": {"min": None, "median": None, "max": None},
        "before_calibrated_days": interval_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float),
        ),
        "after_calibrated_days": interval_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float),
        ),
        "before_all_days": interval_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float),
        ),
        "after_all_days": interval_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float),
        ),
        "daily": [],
        "reason": reason,
    }


def conformal_calibrate_intervals(
    actual,
    quantile_frame,
    eval_mask,
    window_days=CONFORMAL_WINDOW_DAYS,
    min_days=CONFORMAL_MIN_DAYS,
    min_scores=CONFORMAL_MIN_SCORES,
    alpha=CONFORMAL_ALPHA,
):
    """Rolling CQR calibration of P10-P90 from prior test days only.

    For each IST day, the conformity score max(P10 - actual, actual - P90) is
    computed on the prior ``window_days`` days of the same fold. The
    finite-sample (1 - alpha) empirical quantile of those scores expands both
    bounds by the same amount for that day. Calibration starts only after
    ``min_days`` prior days and ``min_scores`` prior hours are available; before
    that the raw interval is kept. The score is floored at zero, so calibration
    can widen an interval but never narrow it.

    Returns the calibration block and a frame with raw and calibrated bounds.
    No actual from the day being calibrated (or later) enters its score.
    """
    frame = pd.concat(
        [
            pd.Series(actual).rename("actual"),
            quantile_frame["p10"].rename("p10"),
            quantile_frame["p50"].rename("p50"),
            quantile_frame["p90"].rename("p90"),
        ],
        axis=1,
    ).loc[eval_mask].dropna()
    if frame.empty:
        result = _conformal_empty("no evaluation hours with target and quantiles")
        empty = quantile_frame.copy()
        empty["p10_calibrated"] = empty["p10"]
        empty["p90_calibrated"] = empty["p90"]
        return result, empty

    frame = frame.copy()
    frame["day"] = frame.index.floor("D")
    days = sorted(frame["day"].unique())
    day_set = set(days)

    daily = []
    q_by_day = {}
    calibrated_by_day = {}
    for day in days:
        past = [
            candidate for candidate in days
            if day - pd.Timedelta(days=window_days) <= candidate < day
        ]
        past = [candidate for candidate in past if candidate in day_set]
        if len(past) < min_days:
            daily.append({
                "day": day.strftime("%Y-%m-%d"),
                "n_past_days": len(past),
                "n_scores": 0,
                "q_wh": None,
                "calibrated": False,
                "reason": "warm-up: fewer than min_days prior days",
            })
            calibrated_by_day[day] = False
            continue

        past_frame = frame[frame["day"].isin(past)]
        scores = np.maximum(
            past_frame["p10"].to_numpy(dtype=float)
            - past_frame["actual"].to_numpy(dtype=float),
            past_frame["actual"].to_numpy(dtype=float)
            - past_frame["p90"].to_numpy(dtype=float),
        )
        if len(scores) < min_scores:
            daily.append({
                "day": day.strftime("%Y-%m-%d"),
                "n_past_days": len(past),
                "n_scores": int(len(scores)),
                "q_wh": None,
                "calibrated": False,
                "reason": "fewer than min_scores prior hours",
            })
            calibrated_by_day[day] = False
            continue

        rank = min(
            len(scores),
            int(np.ceil((len(scores) + 1) * (1 - alpha))),
        )
        q = max(0.0, float(np.sort(scores)[rank - 1]))
        q_by_day[day] = q
        calibrated_by_day[day] = True
        daily.append({
            "day": day.strftime("%Y-%m-%d"),
            "n_past_days": len(past),
            "n_scores": int(len(scores)),
            "q_wh": _clean(q, 4),
            "calibrated": True,
        })

    calibrated = quantile_frame.copy()
    calibrated["p10_calibrated"] = calibrated["p10"]
    calibrated["p90_calibrated"] = calibrated["p90"]
    q_series = frame["day"].map(q_by_day)
    adjusted = calibrated.index.isin(frame.index)
    q_values = q_series.reindex(calibrated.index).fillna(0.0)
    calibrated.loc[adjusted, "p10_calibrated"] = np.maximum(
        0.0, calibrated.loc[adjusted, "p10"] - q_values.loc[adjusted]
    )
    calibrated.loc[adjusted, "p90_calibrated"] = (
        calibrated.loc[adjusted, "p90"] + q_values.loc[adjusted]
    )

    calibrated_mask = frame["day"].map(calibrated_by_day).fillna(False)
    calibrated_index = frame.index[calibrated_mask.to_numpy()]
    before_calibrated = interval_metrics(
        frame.loc[calibrated_mask, "actual"],
        frame.loc[calibrated_mask, "p10"],
        frame.loc[calibrated_mask, "p50"],
        frame.loc[calibrated_mask, "p90"],
    )
    after_calibrated = interval_metrics(
        frame.loc[calibrated_mask, "actual"],
        calibrated.loc[calibrated_index, "p10_calibrated"],
        calibrated.loc[calibrated_index, "p50"],
        calibrated.loc[calibrated_index, "p90_calibrated"],
    )
    before_all = interval_metrics(
        frame["actual"], frame["p10"], frame["p50"], frame["p90"]
    )
    after_all = interval_metrics(
        frame["actual"],
        calibrated.loc[frame.index, "p10_calibrated"],
        calibrated.loc[frame.index, "p50"],
        calibrated.loc[frame.index, "p90_calibrated"],
    )

    q_values_used = [q for q in q_by_day.values()]
    result = {
        "status": "ok" if calibrated_mask.any() else "no_calibrated_days",
        "method": (
            "rolling conformalized quantile regression (CQR) on the P10-P90 "
            "interval using only prior test days' residuals"
        ),
        "score": "max(P10 - actual, actual - P90)",
        "window_days": int(window_days),
        "min_days": int(min_days),
        "min_scores": int(min_scores),
        "alpha": float(alpha),
        "nominal_coverage": _clean(1 - alpha, 4),
        "q_floor_wh": 0.0,
        "n_days": len(days),
        "n_days_calibrated": int(len(q_by_day)),
        "n_hours": int(len(frame)),
        "n_hours_calibrated": int(frame["day"].isin(q_by_day.keys()).sum()),
        "warmup_days": int(len(days) - len(q_by_day)),
        "q_wh": {
            "min": _clean(min(q_values_used), 4) if q_values_used else None,
            "median": _clean(float(np.median(q_values_used)), 4)
            if q_values_used else None,
            "max": _clean(max(q_values_used), 4) if q_values_used else None,
        },
        "before_calibrated_days": before_calibrated,
        "after_calibrated_days": after_calibrated,
        "before_all_days": before_all,
        "after_all_days": after_all,
        "daily": daily,
        "reason": None,
    }
    return result, calibrated


def paired_significance(actual, pred, baseline_pred):
    """Two-sided Wilcoxon signed-rank test on paired absolute errors."""
    frame = pd.concat(
        [
            pd.Series(actual).rename("actual"),
            pd.Series(pred).rename("pred"),
            pd.Series(baseline_pred).rename("baseline"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return {
            "n_pairs": 0, "median_abs_error_diff_wh": None, "model_better": None,
            "small_sample": True, "statistic": None, "p_value": None,
            "significant_at_0.05": None, "reason": "no paired hours",
        }

    model_ae = (frame["pred"] - frame["actual"]).abs()
    base_ae = (frame["baseline"] - frame["actual"]).abs()
    diff = model_ae - base_ae
    median_diff = float(diff.median())

    result = {
        "n_pairs": int(len(diff)),
        "median_abs_error_diff_wh": _clean(median_diff, 4),
        "model_better": bool(median_diff < 0) if median_diff != 0 else None,
        "small_sample": bool(len(diff) < 10),
        "statistic": None,
        "p_value": None,
        "significant_at_0.05": None,
        "reason": None,
    }
    if bool((diff == 0).all()):
        result["reason"] = "all paired differences are zero"
        return result

    try:
        statistic, p_value = wilcoxon(
            diff.to_numpy(dtype=float), zero_method="wilcox", alternative="two-sided"
        )
        result["statistic"] = _clean(statistic, 4)
        result["p_value"] = float(p_value) if np.isfinite(p_value) else None
        result["significant_at_0.05"] = (
            bool(result["p_value"] < 0.05) if result["p_value"] is not None else None
        )
    except ValueError as exc:
        result["reason"] = str(exc)
    return result


def _empty_daily_significance(reason):
    return {
        "status": "no_data",
        "method": (
            "paired Wilcoxon signed-rank on daily MAE with a 95% "
            "block-bootstrap CI of the MAE difference"
        ),
        "n_days": 0,
        "n_pairs": 0,
        "mae_diff_wh": None,
        "median_daily_mae_diff_wh": None,
        "mean_daily_mae_diff_wh": None,
        "model_better": None,
        "small_sample": True,
        "statistic": None,
        "p_value": None,
        "significant_at_0.05": None,
        "ci_level": 0.95,
        "ci_low_wh": None,
        "ci_high_wh": None,
        "bootstrap_reps": DAY_BOOTSTRAP_REPS,
        "block_days": DAY_BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_seed": DAY_BOOTSTRAP_SEED,
        "n_blocks": 0,
        "prob_model_better": None,
        "reason": reason,
    }


def daily_significance(actual, pred, baseline_pred):
    """Day-level paired Wilcoxon and a 7-day block-bootstrap CI.

    Hourly errors are aggregated to a mean absolute error per IST day, using
    only hours where the actual, model and baseline all exist. The Wilcoxon
    test is run on the per-day MAE differences. The CI resamples contiguous
    seven-day blocks with replacement and recomputes the hour-weighted MAE
    difference, so within-week correlation is preserved.
    """
    frame = pd.concat(
        [
            pd.Series(actual).rename("actual"),
            pd.Series(pred).rename("pred"),
            pd.Series(baseline_pred).rename("baseline"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return _empty_daily_significance("no paired hours")

    frame["model_ae"] = (frame["pred"] - frame["actual"]).abs()
    frame["baseline_ae"] = (frame["baseline"] - frame["actual"]).abs()
    days = frame.index.floor("D")
    daily = frame.groupby(days).agg(
        model_mae=("model_ae", "mean"),
        baseline_mae=("baseline_ae", "mean"),
        n=("model_ae", "size"),
    )
    diff = daily["model_mae"] - daily["baseline_mae"]
    n_days = int(len(daily))

    result = _empty_daily_significance(None)
    result.update({
        "status": "ok",
        "n_days": n_days,
        "n_pairs": int(len(frame)),
        "mae_diff_wh": _clean(
            float(frame["model_ae"].mean() - frame["baseline_ae"].mean()), 4
        ),
        "median_daily_mae_diff_wh": _clean(float(diff.median()), 4),
        "mean_daily_mae_diff_wh": _clean(float(diff.mean()), 4),
        "model_better": bool(diff.median() < 0) if diff.median() != 0 else None,
        "small_sample": bool(n_days < 10),
    })

    if bool((diff == 0).all()):
        result["reason"] = "all daily differences are zero"
    else:
        try:
            statistic, p_value = wilcoxon(
                diff.to_numpy(dtype=float),
                zero_method="wilcox",
                alternative="two-sided",
            )
            result["statistic"] = _clean(statistic, 4)
            result["p_value"] = float(p_value) if np.isfinite(p_value) else None
            result["significant_at_0.05"] = (
                bool(result["p_value"] < 0.05)
                if result["p_value"] is not None else None
            )
        except ValueError as exc:
            result["reason"] = str(exc)

    first_day = daily.index.min()
    offsets = ((daily.index - first_day) / pd.Timedelta(days=1)).astype(int)
    block_ids = offsets // DAY_BOOTSTRAP_BLOCK_DAYS
    blocks = [
        np.flatnonzero(block_ids == block_id)
        for block_id in np.unique(block_ids)
    ]
    result["n_blocks"] = len(blocks)
    if len(blocks) < 2:
        result["reason"] = (
            result["reason"] or "fewer than two day blocks for a bootstrap CI"
        )
        return result

    model_mae = daily["model_mae"].to_numpy(dtype=float)
    baseline_mae = daily["baseline_mae"].to_numpy(dtype=float)
    weights = daily["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(DAY_BOOTSTRAP_SEED)
    boot = np.empty(DAY_BOOTSTRAP_REPS, dtype=float)
    for rep in range(DAY_BOOTSTRAP_REPS):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        index = np.concatenate([blocks[block] for block in pick])
        w = weights[index]
        boot[rep] = (
            np.average(model_mae[index], weights=w)
            - np.average(baseline_mae[index], weights=w)
        )
    low, high = np.percentile(boot, [2.5, 97.5])
    result["ci_low_wh"] = _clean(float(low), 4)
    result["ci_high_wh"] = _clean(float(high), 4)
    result["prob_model_better"] = _clean(float((boot < 0).mean()), 4)
    return result


def select_best_baseline(actual, preds, eval_mask, scale):
    """Pick the baseline with the lowest MASE on a common mask.

    The all-four-baseline mask is used when it covers at least half of the
    evaluation hours (and at least 24 h); otherwise the evaluation mask is used
    and baselines that lack the longer lags report fewer hours.
    """
    all_four = eval_mask.copy()
    for name in BASELINES:
        all_four &= preds[name].notna()

    if int(all_four.sum()) >= max(24, 0.5 * int(eval_mask.sum())):
        mask_name, mask = "all_four_baselines", all_four
    else:
        mask_name, mask = "evaluation_mask", eval_mask

    selection_metrics = {}
    scores = {}
    for name in BASELINES:
        metrics = compute_metrics(actual[mask], preds[name][mask], scale)
        selection_metrics[name] = metrics
        if metrics["status"] == "ok" and metrics["MASE"] is not None:
            scores[name] = (metrics["MASE"], metrics["MAE"], name)

    best = min(scores, key=lambda name: scores[name][:2]) if scores else None
    return best, mask_name, int(mask.sum()), selection_metrics


def point_feature_options(train_start, train_end):
    """9.3 rule: annual calendar is meaningful only with >=12 months of history."""
    start = pd.Timestamp(train_start)
    span = pd.Timestamp(train_end, tz=start.tz) - start
    return {"drop_annual_calendar": bool(span < pd.Timedelta(days=365))}


def common_hours_block(actual, preds, eval_mask, scale):
    """Score every model on the intersection of all models' forecast hours."""
    common = eval_mask.copy()
    for name in ALL_MODELS:
        common &= preds[name].notna()

    models = {
        name: compute_metrics(actual[common], preds[name][common], scale)
        for name in ALL_MODELS
    }
    best, selection_mask, selection_hours, selection_metrics = \
        select_best_baseline(actual, preds, common, scale)
    n = int(common.sum())
    eval_hours = int(eval_mask.sum())

    paired = {}
    for name in ALL_MODELS:
        if best is None:
            paired[name] = _empty_daily_significance(
                "no common-hours best baseline"
            )
        elif name == best:
            paired[name] = {
                "status": "reference",
                "reason": f"{name} is the common-hours best baseline",
            }
        else:
            paired[name] = daily_significance(
                actual[common], preds[name][common], preds[best][common]
            )

    return {
        "definition": (
            "evaluation mask intersected with the forecasts of all four "
            "baselines and all three learned models"
        ),
        "n": n,
        "eval_hours": eval_hours,
        "coverage": _clean(n / eval_hours, 4) if eval_hours else None,
        "models": models,
        "best_baseline": best,
        "selection": {
            "mask": selection_mask,
            "hours": selection_hours,
            "by": "MASE",
            "metrics": selection_metrics,
        },
        "paired_vs_best": paired,
        "hours_by_model": {
            name: int(preds[name][eval_mask].notna().sum())
            for name in ALL_MODELS
        },
        "missing_models": [
            name for name in ALL_MODELS
            if int(preds[name][eval_mask].notna().sum()) == 0
        ],
    }


def _hourly_records(actual, preds, quantile_frame, mask,
                    calibrated_quantile_frame=None):
    """Per-hour evaluation rows for the dashboard (Wh, local IST timestamps)."""
    sources = {
        "actual": actual,
        "seasonal_naive": preds["seasonal_naive_24"],
        "xgboost": preds["xgboost"],
        "sarima": preds["sarima"],
        "p10": quantile_frame["p10"] if quantile_frame is not None else None,
        "p50": quantile_frame["p50"] if quantile_frame is not None else None,
        "p90": quantile_frame["p90"] if quantile_frame is not None else None,
        "p10_calibrated": (
            calibrated_quantile_frame["p10_calibrated"]
            if calibrated_quantile_frame is not None else None
        ),
        "p90_calibrated": (
            calibrated_quantile_frame["p90_calibrated"]
            if calibrated_quantile_frame is not None else None
        ),
    }
    columns = ["time"] + list(sources)
    rows = []
    for timestamp in actual.index[mask]:
        row = [timestamp.strftime("%Y-%m-%dT%H:%M")]
        for series in sources.values():
            value = None if series is None else series.get(timestamp)
            row.append(None if value is None or not np.isfinite(value)
                       else round(float(value), 2))
        rows.append(row)
    return {"columns": columns, "rows": rows}


def evaluate_fold(series, fold, reporting=None, weather_frame=None):
    """Score baselines and learned models on one day-ahead fold.

    The evaluation mask is every hour with a target value and a seasonal-naive
    forecast. Models that cannot produce a value for an hour (SARIMA after a
    coverage gap) are scored on their available hours, as the lagged baselines
    already are; paired tests use the overlapping hours.
    """
    train = series.loc[fold["train_start"]:fold["train_end"]]
    scale = mase_scale(train)
    actual = series.reindex(test_hours(fold["test_start"], fold["test_end"]))
    preds = baseline_forecasts(series, fold["test_start"], fold["test_end"])
    model_fit = {}
    quantile_frame = None
    point_features = point_feature_options(
        fold["train_start"], fold["train_end"]
    )
    for name in LEARNED_MODELS:
        if name == "xgboost":
            preds[name], model_fit[name] = fm.forecast(
                series,
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
                params=POINT_XGB_PARAMS,
                reporting=reporting,
                weather=weather_frame,
                tune_model=False,
                feature_options=point_features,
            )
        elif name == "sarima":
            preds[name], model_fit[name] = sm.forecast(
                series, fold["test_start"], fold["test_end"]
            )
        elif name == "xgboost_quantile":
            quantile_frame, model_fit[name] = qm.forecast(
                series,
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
                reporting=reporting,
                weather=weather_frame,
            )
            preds[name] = quantile_frame["p50"]
    mask = actual.notna() & preds["seasonal_naive_24"].notna()

    models = {}
    for name in BASELINES + LEARNED_MODELS:
        models[name] = compute_metrics(actual[mask], preds[name][mask], scale)

    intervals = {}
    calibrated_quantiles = None
    if quantile_frame is not None:
        intervals["xgboost_quantile"] = interval_metrics(
            actual[mask],
            quantile_frame.loc[mask, "p10"],
            quantile_frame.loc[mask, "p50"],
            quantile_frame.loc[mask, "p90"],
        )
        intervals["conformal"], calibrated_quantiles = \
            conformal_calibrate_intervals(actual, quantile_frame, mask)

    best, selection_mask, selection_hours, selection_metrics = select_best_baseline(
        actual, preds, mask, scale
    )
    common = common_hours_block(actual, preds, mask, scale)

    return {
        "id": fold["id"],
        "label": fold.get("label"),
        "stress_test": bool(fold.get("stress_test")),
        "train": {
            "start": fold["train_start"],
            "end": fold["train_end"],
            "hours": int(len(train)),
            "valid_hours": int(train.notna().sum()),
        },
        "test": {
            "start": fold["test_start"],
            "end": fold["test_end"],
            "clock_hours": int(len(actual)),
            "target_valid_hours": int(actual.notna().sum()),
            "eval_hours": int(mask.sum()),
            "eval_days": int(mask.index[mask].floor("D").nunique()),
        },
        "mase_scale": _clean(scale, 4),
        "models": models,
        "common_hours": common,
        "intervals": intervals,
        "hourly": _hourly_records(
            actual, preds, quantile_frame, mask, calibrated_quantiles
        ),
        "model_fit": model_fit,
        "best_baseline": best,
        "selection": {
            "mask": selection_mask,
            "hours": selection_hours,
            "by": "MASE",
            "metrics": selection_metrics,
        },
    }


def history_start(series):
    """Earliest valid target hour, used as every fold's train_start."""
    first = series.first_valid_index()
    if first is None:
        raise ValueError("target series has no valid hours")
    return first.strftime("%Y-%m-%d %H:%M")


def run_backtest(series, reporting=None, weather_frame=None):
    start = history_start(series)
    return [
        evaluate_fold(
            series,
            {**fold, "train_start": start},
            reporting=reporting,
            weather_frame=weather_frame,
        )
        for fold in FOLDS
    ]


def build_results(series=None, reporting=None, weather_frame=None):
    if series is None:
        series, reporting = load_target_with_reporting()
    if weather_frame is None:
        weather_frame = weather.load()
    return {
        "generated_at": pd.Timestamp.now(tz=dp.TZ).isoformat(timespec="seconds"),
        "protocol": dict(PROTOCOL),
        "baselines": list(BASELINES),
        "baseline_descriptions": dict(BASELINE_DESCRIPTIONS),
        "learned_models": list(LEARNED_MODELS),
        "learned_model_descriptions": dict(LEARNED_MODEL_DESCRIPTIONS),
        "weather": weather.describe(weather_frame),
        "folds": run_backtest(series, reporting, weather_frame),
    }


def baseline_forecasts(series, start, end):
    """Forecast every hour in [start, end] with each baseline.

    The forecast origin for each hour is 00:00 IST of its calendar day. Every
    input timestamp is therefore at least 24 h before the target hour, except
    persistence, which uses the last observation before the origin.
    """
    hours = test_hours(start, end)
    preds = {}

    # Last valid observation before each day's origin, broadcast over the day.
    last_before_origin = series.ffill().shift(1)
    day_starts = pd.date_range(start, end, freq="D", tz=dp.TZ)
    daily_level = last_before_origin.reindex(day_starts)
    preds["persistence"] = pd.Series(
        daily_level.reindex(hours.floor("D")).to_numpy(dtype=float), index=hours
    )

    preds["seasonal_naive_24"] = pd.Series(_lag_values(series, hours, 24), index=hours)
    preds["same_hour_last_week"] = pd.Series(_lag_values(series, hours, 168), index=hours)

    weekly = np.column_stack([_lag_values(series, hours, lag)
                              for lag in (168, 336, 504, 672)])
    complete = ~np.isnan(weekly).any(axis=1)
    mean_4 = np.full(len(hours), np.nan)
    mean_4[complete] = weekly[complete].mean(axis=1)
    preds["mean_last_4_same_weekdays"] = pd.Series(mean_4, index=hours)

    return preds


def _show(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def main():
    results = build_results()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Wrote {RESULTS_PATH}")
    state = results["weather"]
    print(f"weather: {state['cache']}" if state["available"] else "weather: unavailable")
    print(f"Day-ahead backtest, {len(FOLDS)} expanding folds, origin 00:00 IST")
    for fold in results["folds"]:
        test = fold["test"]
        print(f"\n== {fold['id']}: train {fold['train']['hours']} h "
              f"({fold['train']['valid_hours']} valid), test {test['clock_hours']} h "
              f"({test['target_valid_hours']} target), eval {test['eval_hours']} h / "
              f"{test['eval_days']} d, MASE scale {_show(fold['mase_scale'])}")
        print(f"{'model':28s} {'n':>5s} {'MAE':>8s} {'RMSE':>8s} {'WAPE%':>8s} "
              f"{'MASE':>7s} {'bias':>8s} {'R2':>8s} {'sMAPE%':>8s}")
        for name in results["baselines"] + results["learned_models"]:
            m = fold["models"][name]
            print(f"{name:28s} {m['n']:5d} {_show(m['MAE']):>8s} {_show(m['RMSE']):>8s} "
                  f"{_show(m['WAPE']):>8s} {_show(m['MASE'], 3):>7s} "
                  f"{_show(m['bias']):>8s} {_show(m['R2'], 4):>8s} "
                  f"{_show(m['sMAPE']):>8s}")
        common = fold["common_hours"]
        print(f"common hours: {common['n']} of {common['eval_hours']} "
              f"({_show(common['coverage'], 3)}), best baseline "
              f"{common['best_baseline']}, missing models "
              f"{', '.join(common['missing_models']) or 'none'}")
        for name in results["baselines"] + results["learned_models"]:
            m = common["models"][name]
            print(f"  common {name:28s} {m['n']:5d} {_show(m['MAE']):>8s}")
        for name, meta in fold["model_fit"].items():
            if "n_train" in meta:
                print(f"fit {name}: n={meta['n_train']}, "
                      f"cap={_show(meta['cap_wh'])} Wh, capped={meta['n_capped']}")
            else:
                print(f"fit {name}: fits={meta.get('n_fits')}, "
                      f"hours={meta.get('predicted_hours')}, "
                      f"failures={meta.get('fit_failures')}")
        for name, im in fold.get("intervals", {}).items():
            if name == "conformal":
                print(f"interval conformal: days={im['n_days']} "
                      f"calibrated={im['n_days_calibrated']} "
                      f"warmup={im['warmup_days']} "
                      f"coverage before={_show(im['before_calibrated_days']['coverage'], 3)} "
                      f"after={_show(im['after_calibrated_days']['coverage'], 3)}")
                continue
            print(f"interval {name}: n={im['n']} coverage={_show(im['coverage'], 3)} "
                  f"width={_show(im['mean_width_wh'])} Wh "
                  f"pinball={_show(im['pinball_mean'], 3)}")
        selection = fold["selection"]
        print(f"best baseline: {fold['best_baseline']} "
              f"({selection['mask']}, {selection['hours']} h, by {selection['by']})")
        for name, comp in common["paired_vs_best"].items():
            if comp["status"] != "ok":
                print(f"  {name:28s} {comp['status']}: "
                      f"{comp['reason'] or 'reference'}")
                continue
            p_value = comp["p_value"] if comp["p_value"] is not None else comp["reason"]
            print(f"  {name:28s} days={comp['n_days']:3d} "
                  f"mae_diff={_show(comp['mae_diff_wh'])} Wh "
                  f"median_daily_diff={_show(comp['median_daily_mae_diff_wh'])} Wh "
                  f"CI95=[{_show(comp['ci_low_wh'])}, {_show(comp['ci_high_wh'])}] "
                  f"p={p_value}")


if __name__ == "__main__":
    main()
