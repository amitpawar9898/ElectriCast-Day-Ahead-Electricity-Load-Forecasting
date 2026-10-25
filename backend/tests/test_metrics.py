"""Hand-checked metric and baseline tests.

Run from the repo root:

    python -m pytest backend/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import evaluate as ev  # noqa: E402


def test_compute_metrics_hand_checked():
    metrics = ev.compute_metrics(
        pd.Series([10.0, 20.0, 30.0, 40.0]),
        pd.Series([12.0, 18.0, 33.0, 36.0]),
        scale=2.0,
    )
    assert metrics["status"] == "ok"
    assert metrics["n"] == 4
    assert metrics["MAE"] == 2.75
    assert metrics["RMSE"] == 2.8723
    assert metrics["WAPE"] == 11.0
    assert metrics["sMAPE"] == 12.1896
    assert metrics["bias"] == -0.25
    assert metrics["R2"] == 0.934
    assert metrics["MASE"] == 1.375


def test_compute_metrics_guards_zero_denominators():
    metrics = ev.compute_metrics(pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]), scale=1.0)
    assert metrics["status"] == "ok"
    assert metrics["MAE"] == 0.0
    assert metrics["WAPE"] is None
    assert metrics["R2"] is None
    assert metrics["sMAPE"] == 0.0
    assert metrics["MASE"] == 0.0


def test_compute_metrics_drops_missing_pairs_and_no_data():
    partial = ev.compute_metrics(
        pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 2.0, np.nan])
    )
    assert partial["status"] == "ok"
    assert partial["n"] == 1
    assert partial["MAE"] == 0.0

    empty = ev.compute_metrics(pd.Series([np.nan]), pd.Series([1.0]))
    assert empty["status"] == "no_data"
    assert empty["n"] == 0
    assert all(empty[key] is None for key in ("MAE", "RMSE", "WAPE", "MASE", "bias", "R2", "sMAPE"))


def test_mase_scale_uses_24_hour_seasonal_differences():
    index = pd.date_range("2021-01-01", periods=48, freq="h", tz="Asia/Kolkata")
    series = pd.Series(np.arange(48, dtype=float), index=index)
    assert ev.mase_scale(series) == 24.0
    assert ev.mase_scale(series.iloc[:24]) is None


def test_baselines_follow_day_ahead_lags():
    index = pd.date_range("2021-01-01", periods=8 * 24, freq="h", tz="Asia/Kolkata")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    hours = pd.date_range("2021-01-08", periods=24, freq="h", tz="Asia/Kolkata")

    preds = ev.baseline_forecasts(series, hours[0], hours[-1])

    assert preds["persistence"].nunique() == 1
    assert preds["persistence"].iloc[0] == series.loc["2021-01-07 23:00"].item()
    np.testing.assert_allclose(
        preds["seasonal_naive_24"].to_numpy(),
        series.reindex(hours - pd.Timedelta(hours=24)).to_numpy(),
    )


def test_phase9_fold_layout_and_stress_test_flag():
    ids = [fold["id"] for fold in ev.FOLDS]
    assert ids == [
        "2017-06", "2017-07", "2017-08", "2017-09_11", "2017-12",
        "2018-05_06",
    ]
    stress = [fold for fold in ev.FOLDS if fold.get("stress_test")]
    assert [fold["id"] for fold in stress] == ["2018-05_06"]
    assert stress[0]["label"].lower().startswith("stress test")
    assert all(not fold.get("stress_test") for fold in ev.FOLDS[:-1])
    assert ev.FOLDS[3]["test_end"] == "2017-11-30 23:00"
    assert ev.FOLDS[4]["test_start"] == "2017-12-01 00:00"


def test_history_start_is_the_first_valid_target_hour():
    index = pd.date_range("2020-01-01", periods=4, freq="h", tz="Asia/Kolkata")
    series = pd.Series([np.nan, 1.0, 2.0, 3.0], index=index)
    assert ev.history_start(series) == "2020-01-01 01:00"
    with np.testing.assert_raises(ValueError):
        ev.history_start(pd.Series([np.nan], index=index))


def test_frozen_point_params_match_the_95_artifact():
    import json

    artifact = json.loads(
        (BACKEND / "tuning_results.json").read_text(encoding="utf-8")
    )
    frozen = artifact["tuning"]["pre_scoring"]["params"]
    assert ev.POINT_XGB_PARAMS == frozen


def test_point_feature_options_follow_the_twelve_month_rule():
    short = ev.point_feature_options(
        pd.Timestamp("2016-12-14 21:00", tz=ev.dp.TZ),
        "2017-05-31 23:00",
    )
    long = ev.point_feature_options(
        pd.Timestamp("2016-12-14 21:00", tz=ev.dp.TZ),
        "2017-12-31 23:00",
    )
    assert short == {"drop_annual_calendar": True}
    assert long == {"drop_annual_calendar": False}


def _all_model_preds(index, values):
    return {
        name: pd.Series(values, index=index, dtype=float)
        for name in ev.ALL_MODELS
    }


def test_common_hours_block_intersects_every_model():
    index = pd.date_range("2021-01-01", periods=6, freq="h", tz=ev.dp.TZ)
    actual = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=index)
    preds = _all_model_preds(index, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0])
    preds["xgboost"].iloc[0] = np.nan
    preds["sarima"].iloc[1] = np.nan
    preds["persistence"].iloc[2] = np.nan
    mask = pd.Series(True, index=index)

    block = ev.common_hours_block(actual, preds, mask, scale=2.0)

    assert block["n"] == 3
    assert block["eval_hours"] == 6
    assert block["coverage"] == 0.5
    assert block["hours_by_model"]["xgboost"] == 5
    assert block["hours_by_model"]["sarima"] == 5
    assert block["hours_by_model"]["persistence"] == 5
    assert block["hours_by_model"]["seasonal_naive_24"] == 6
    assert block["missing_models"] == []
    for name in ev.ALL_MODELS:
        assert block["models"][name]["n"] == 3
        assert block["models"][name]["MAE"] == 5.0
    assert block["best_baseline"] in ev.BASELINES


def test_common_hours_block_reports_empty_intersection():
    index = pd.date_range("2021-01-01", periods=4, freq="h", tz=ev.dp.TZ)
    actual = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    preds = _all_model_preds(index, [11.0, 22.0, 33.0, 44.0])
    preds["mean_last_4_same_weekdays"] = pd.Series(np.nan, index=index)
    mask = pd.Series(True, index=index)

    block = ev.common_hours_block(actual, preds, mask, scale=2.0)

    assert block["n"] == 0
    assert block["coverage"] == 0.0
    assert block["best_baseline"] is None
    assert block["missing_models"] == ["mean_last_4_same_weekdays"]
    assert block["models"]["xgboost"]["status"] == "no_data"


def test_daily_significance_uses_days_and_block_bootstrap():
    index = pd.date_range("2021-01-01", periods=10 * 24, freq="h", tz=ev.dp.TZ)
    actual = pd.Series(0.0, index=index)
    pred = pd.Series(1.0, index=index)
    baseline = pd.Series(2.0, index=index)

    result = ev.daily_significance(actual, pred, baseline)

    assert result["status"] == "ok"
    assert result["n_days"] == 10
    assert result["n_pairs"] == 240
    assert result["mae_diff_wh"] == -1.0
    assert result["median_daily_mae_diff_wh"] == -1.0
    assert result["mean_daily_mae_diff_wh"] == -1.0
    assert result["model_better"] is True
    assert result["ci_low_wh"] == -1.0
    assert result["ci_high_wh"] == -1.0
    assert result["prob_model_better"] == 1.0
    assert result["n_blocks"] == 2
    assert result["p_value"] < 0.05
    assert result["small_sample"] is False


def test_daily_significance_reports_no_data_without_overlap():
    index = pd.date_range("2021-01-01", periods=24, freq="h", tz=ev.dp.TZ)
    result = ev.daily_significance(
        pd.Series(1.0, index=index),
        pd.Series(np.nan, index=index),
        pd.Series(2.0, index=index),
    )
    assert result["status"] == "no_data"
    assert result["n_days"] == 0
    assert result["mae_diff_wh"] is None
    assert result["p_value"] is None
    assert result["reason"] == "no paired hours"


def _conformal_frame(days=10):
    index = pd.date_range(
        "2021-01-01", periods=days * 24, freq="h", tz=ev.dp.TZ
    )
    actual = np.full(len(index), 10.0)
    actual[index >= index[0] + pd.Timedelta(days=5)] = 30.0
    actual = pd.Series(actual, index=index)
    quantiles = pd.DataFrame(
        {
            "p10": np.full(len(index), 9.0),
            "p50": np.full(len(index), 10.0),
            "p90": np.full(len(index), 11.0),
        },
        index=index,
    )
    return actual, quantiles, index


def test_conformal_calibration_uses_only_prior_days():
    actual, quantiles, index = _conformal_frame()
    mask = pd.Series(True, index=index)

    result, calibrated = ev.conformal_calibrate_intervals(
        actual, quantiles, mask, window_days=5, min_days=3, min_scores=24
    )

    assert result["status"] == "ok"
    assert result["n_days"] == 10
    assert result["n_days_calibrated"] == 7
    assert result["warmup_days"] == 3
    assert result["n_hours_calibrated"] == 7 * 24
    q_by_day = {row["day"]: row["q_wh"] for row in result["daily"]}
    # Day 4 sees only the normal regime; the first spike day (Jan 5) cannot
    # leak its own score into its own interval.
    assert q_by_day["2021-01-04"] == 0.0
    assert q_by_day["2021-01-05"] == 0.0
    # Day 8 sees three spiked days (Jan 5-7) in its five-day window, so q is 19.
    assert q_by_day["2021-01-08"] == 19.0
    # Day 9 sees 60% spiked scores in its five-day window, so q expands by 19 Wh.
    assert q_by_day["2021-01-09"] == 19.0
    assert q_by_day["2021-01-10"] == 19.0

    assert result["before_calibrated_days"]["coverage"] < \
        result["after_calibrated_days"]["coverage"]
    assert result["after_all_days"]["coverage"] == 0.9
    assert result["after_all_days"]["mean_width_wh"] > \
        result["before_all_days"]["mean_width_wh"]
    # Warm-up days keep the raw interval exactly.
    warm = index.floor("D") < index[0] + pd.Timedelta(days=3)
    assert np.array_equal(
        calibrated.loc[warm, "p10_calibrated"].to_numpy(),
        quantiles.loc[warm, "p10"].to_numpy(),
    )


def test_conformal_calibration_never_narrows_an_overwide_interval():
    index = pd.date_range(
        "2021-01-01", periods=8 * 24, freq="h", tz=ev.dp.TZ
    )
    actual = pd.Series(50.0, index=index)
    quantiles = pd.DataFrame(
        {
            "p10": np.zeros(len(index)),
            "p50": np.full(len(index), 50.0),
            "p90": np.full(len(index), 100.0),
        },
        index=index,
    )
    result, calibrated = ev.conformal_calibrate_intervals(
        actual, quantiles, pd.Series(True, index=index),
        window_days=5, min_days=3, min_scores=24,
    )
    assert result["q_wh"]["max"] == 0.0
    assert np.array_equal(
        calibrated["p10_calibrated"].to_numpy(),
        quantiles["p10"].to_numpy(),
    )
    assert np.array_equal(
        calibrated["p90_calibrated"].to_numpy(),
        quantiles["p90"].to_numpy(),
    )
    assert result["before_calibrated_days"]["coverage"] == 1.0
    assert result["after_calibrated_days"]["coverage"] == 1.0


def test_conformal_calibration_reports_no_data_without_evaluation_hours():
    actual, quantiles, index = _conformal_frame(3)
    result, calibrated = ev.conformal_calibrate_intervals(
        actual, quantiles, pd.Series(False, index=index)
    )
    assert result["status"] == "no_data"
    assert result["n_days"] == 0
    assert result["reason"] == "no evaluation hours with target and quantiles"
    assert "p10_calibrated" in calibrated.columns
