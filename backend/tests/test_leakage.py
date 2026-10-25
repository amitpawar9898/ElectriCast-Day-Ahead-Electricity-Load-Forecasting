"""Leakage tests for the day-ahead model and baselines.

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

import data_pipeline as dp  # noqa: E402
import evaluate as ev  # noqa: E402
import forecast_model as fm  # noqa: E402
import quantile_model as qm  # noqa: E402
import sarima_model as sm  # noqa: E402

TZ = dp.TZ
TEST_DAY = pd.Timestamp("2020-01-22", tz=TZ)
TEST_HOURS = pd.date_range(TEST_DAY, periods=24, freq="h", tz=TZ)


def synthetic_series(days=25):
    index = pd.date_range("2020-01-01", periods=days * 24, freq="h", tz=TZ)
    values = (
        500.0
        + 120.0 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
        + 0.2 * np.arange(len(index))
    )
    return pd.Series(values, index=index, name="target_wh")


def synthetic_reporting(series):
    return pd.Series(24, index=series.index, name="n_reporting")


def synthetic_weather(series):
    index = series.index
    return pd.DataFrame({
        "temperature_2m": 25.0 + 5.0 * np.sin(2 * np.pi * index.hour / 24),
        "relative_humidity_2m": 60.0 + 10.0 * np.cos(2 * np.pi * index.hour / 24),
        "apparent_temperature": 27.0 + 5.0 * np.sin(2 * np.pi * index.hour / 24),
        "precipitation": 0.0,
        "cloud_cover": 40.0,
        "wind_speed_10m": 12.0,
    }, index=index)


def with_future_replaced(series, start):
    mutated = series.copy()
    mutated.loc[start:] = 1e9
    return mutated


def test_all_feature_lags_are_at_least_24_hours():
    assert min(fm.FEATURE_LAGS) >= 24


def test_features_do_not_read_forecast_day_values():
    series = synthetic_series()
    before = fm.make_features(series, TEST_HOURS)
    after = fm.make_features(with_future_replaced(series, TEST_DAY), TEST_HOURS)
    pd.testing.assert_frame_equal(before, after)
    assert all(np.isfinite(before["lag_24"]))


def test_baselines_do_not_read_forecast_day_values():
    series = synthetic_series()
    start, end = TEST_DAY, TEST_HOURS[-1]
    before = ev.baseline_forecasts(series, start, end)
    after = ev.baseline_forecasts(with_future_replaced(series, TEST_DAY), start, end)
    for name in ev.BASELINES:
        np.testing.assert_allclose(
            before[name].to_numpy(), after[name].to_numpy(), equal_nan=True
        )


def test_xgboost_does_not_read_forecast_day_values():
    series = synthetic_series()
    params = {
        "n_estimators": 30,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
    }
    before, _ = fm.forecast(
        series, series.index[0], series.index[20 * 24 - 1],
        TEST_DAY, TEST_HOURS[-1], params=params,
    )
    after, _ = fm.forecast(
        with_future_replaced(series, TEST_DAY),
        series.index[0], series.index[20 * 24 - 1],
        TEST_DAY, TEST_HOURS[-1], params=params,
    )
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())
    assert before.notna().all()


def test_xgboost_does_not_read_future_reporting_counts():
    series = synthetic_series()
    reporting = synthetic_reporting(series)
    params = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.1, "n_jobs": 1}
    args = (series, series.index[0], series.index[20 * 24 - 1],
            TEST_DAY, TEST_HOURS[-1])
    before, _ = fm.forecast(*args, params=params, reporting=reporting)
    mutated = reporting.copy()
    mutated.loc[TEST_DAY:] = 0
    after, _ = fm.forecast(*args, params=params, reporting=mutated)
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())


def test_weather_features_do_not_read_forecast_day_values():
    series = synthetic_series()
    weather = synthetic_weather(series)
    before = fm.make_features(series, TEST_HOURS, weather=weather)
    mutated = weather.copy()
    mutated.loc[TEST_DAY:] = 999.0
    after = fm.make_features(series, TEST_HOURS, weather=mutated)
    pd.testing.assert_frame_equal(before, after)
    assert before["weather_temperature_2m_lag_24"].notna().all()


def test_xgboost_does_not_read_future_weather():
    series = synthetic_series()
    weather = synthetic_weather(series)
    params = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.1, "n_jobs": 1}
    args = (series, series.index[0], series.index[20 * 24 - 1],
            TEST_DAY, TEST_HOURS[-1])
    before, _ = fm.forecast(*args, params=params, weather=weather)
    mutated = weather.copy()
    mutated.loc[TEST_DAY:] = 999.0
    after, _ = fm.forecast(*args, params=params, weather=mutated)
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())


def test_reporting_features_do_not_read_forecast_day_values():
    series = synthetic_series()
    hours = TEST_HOURS
    before = fm.make_features(series, hours, reporting=synthetic_reporting(series))
    mutated = synthetic_reporting(series)
    mutated.loc[TEST_DAY:] = 999
    after = fm.make_features(series, hours, reporting=mutated)
    pd.testing.assert_frame_equal(before, after)
    assert (before["n_reporting_lag_24"] == 24).all()


def test_calendar_holiday_and_semester_flags():
    hours = pd.date_range("2017-10-16", "2017-10-23 23:00", freq="h", tz=TZ)
    features = fm.make_features(pd.Series(500.0, index=hours), hours)
    holiday = features.index.strftime("%Y-%m-%d").isin(["2017-10-19", "2017-10-20"])
    assert features.loc[holiday, "is_holiday"].eq(1).all()
    assert features.loc[~holiday, "is_holiday"].eq(0).all()
    assert features.loc["2017-10-21 00:00":"2017-10-22 23:00", "is_weekend"].eq(1).all()
    assert features.loc["2017-10-18", "is_weekend"].eq(0).all()
    assert features.loc["2017-10-18", "semester_in_session"].eq(1).all()

    may = pd.date_range("2017-05-01", periods=24, freq="h", tz=TZ)
    may_features = fm.make_features(pd.Series(500.0, index=may), may)
    assert may_features["semester_in_session"].eq(0).all()


def test_time_series_splits_stay_inside_training_window():
    splits = fm.time_series_splits("2017-05-01 00:00", "2017-06-30 23:00")
    assert len(splits) == 3
    window_start = pd.Timestamp("2017-05-01 00:00", tz=TZ)
    window_end = pd.Timestamp("2017-06-30 23:00", tz=TZ)
    previous_train_end = None
    for split in splits:
        train_start = pd.Timestamp(split["train_start"])
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])
        assert train_start == window_start
        assert train_end < val_start <= val_end <= window_end
        if previous_train_end is not None:
            assert train_end > previous_train_end
        previous_train_end = train_end


def test_tuned_fit_does_not_use_test_values():
    series = synthetic_series()
    args = (series, series.index[0], series.index[20 * 24 - 1])
    before, metadata = fm.forecast(
        *args, TEST_DAY, TEST_HOURS[-1], tune_model=True, n_trials=2
    )
    after, _ = fm.forecast(
        with_future_replaced(series, TEST_DAY), *args[1:],
        TEST_DAY, TEST_HOURS[-1], tune_model=True, n_trials=2,
    )
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())
    assert metadata["tuning"]["cv_splits"]
    assert metadata["tuning"]["best_iteration"] >= 1


def test_xgboost_training_labels_end_before_test_start():
    series = synthetic_series()
    _, metadata = fm.fit(series, series.index[0], series.index[20 * 24 - 1])
    assert pd.Timestamp(metadata["train_end"], tz=TZ) < TEST_DAY
    assert metadata["n_train"] > 0


def test_interval_metrics_hand_check():
    metrics = ev.interval_metrics(
        pd.Series([10.0, 20.0]),
        pd.Series([8.0, 15.0]),
        pd.Series([9.0, 22.0]),
        pd.Series([12.0, 25.0]),
    )
    assert metrics["n"] == 2
    assert metrics["coverage"] == 1.0
    assert metrics["mean_width_wh"] == 7.0
    assert abs(metrics["pinball_10"] - 0.35) < 1e-9
    assert abs(metrics["pinball_50"] - 0.75) < 1e-9
    assert abs(metrics["pinball_90"] - 0.35) < 1e-9
    assert abs(metrics["pinball_mean"] - 0.4833) < 1e-9

    missed = ev.interval_metrics(
        pd.Series([30.0]), pd.Series([8.0]), pd.Series([9.0]), pd.Series([12.0])
    )
    assert missed["coverage"] == 0.0


def test_quantile_forecast_does_not_read_forecast_day_values():
    series = synthetic_series()
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1, "n_jobs": 1}
    args = (series, series.index[0], series.index[20 * 24 - 1],
            TEST_DAY, TEST_HOURS[-1])
    before, metadata = qm.forecast(*args, params=params)
    after, _ = qm.forecast(with_future_replaced(series, TEST_DAY), *args[1:],
                           params=params)
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())
    assert list(before.columns) == ["p10", "p50", "p90"]
    assert before.notna().all().all()
    assert (before["p10"] <= before["p50"]).all()
    assert (before["p50"] <= before["p90"]).all()
    assert metadata["quantiles"] == [0.1, 0.5, 0.9]
    assert metadata["n_quantile_crossings"] >= 0


def test_sarima_history_stops_at_last_gap_and_enforces_minimum():
    index = pd.date_range("2020-01-01", periods=25 * 24, freq="h", tz=TZ)
    series = pd.Series(500.0, index=index)
    series.loc["2020-01-10":"2020-01-11 23:00"] = np.nan

    tail = sm.contiguous_tail(series, TEST_DAY - pd.Timedelta(hours=1))
    assert tail is not None
    assert tail.index[0] == pd.Timestamp("2020-01-12 00:00", tz=TZ)
    assert tail.index[-1] == TEST_DAY - pd.Timedelta(hours=1)
    assert tail.notna().all()

    short = series.loc[pd.Timestamp("2020-01-18", tz=TZ):]
    assert sm.contiguous_tail(short, TEST_DAY - pd.Timedelta(hours=1)) is None


def test_sarima_does_not_read_forecast_day_values():
    series = synthetic_series()
    args = (series, TEST_DAY, TEST_HOURS[-1])
    before, metadata = sm.forecast(*args)
    after, _ = sm.forecast(with_future_replaced(series, TEST_DAY), *args[1:])
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), equal_nan=True)
    assert metadata["n_fits"] == 1
    assert metadata["predicted_hours"] == 24


def test_sarima_skips_state_older_than_staleness_limit():
    series = synthetic_series()
    stale = series.loc[:pd.Timestamp("2020-01-15 23:00", tz=TZ)]
    pred, metadata = sm.forecast(stale, TEST_DAY, TEST_HOURS[-1])
    assert pred.isna().all()
    assert metadata["n_fits"] == 1
    assert metadata["predicted_hours"] == 0


def test_annual_calendar_features_can_be_dropped():
    series = synthetic_series()
    full = fm.make_features(series, TEST_HOURS)
    assert {"month", "doy_sin", "doy_cos"} <= set(full.columns)

    trimmed = fm.make_features(
        series, TEST_HOURS,
        feature_options={"drop_annual_calendar": True},
    )
    assert not {"month", "doy_sin", "doy_cos"} & set(trimmed.columns)
    assert {"hour", "dow", "is_weekend", "is_holiday",
            "semester_in_session"} <= set(trimmed.columns)


def test_weather_features_can_be_removed():
    series = synthetic_series()
    weather = synthetic_weather(series)
    full = fm.make_features(series, TEST_HOURS, weather=weather)
    no_weather = fm.make_features(
        series, TEST_HOURS, weather=weather,
        feature_options={"use_weather": False},
    )
    assert any(column.startswith("weather_") for column in full.columns)
    assert not any(column.startswith("weather_") for column in no_weather.columns)


def test_oracle_temperature_is_opt_in_and_reads_the_target_hour():
    series = synthetic_series()
    weather = synthetic_weather(series)
    default = fm.make_features(series, TEST_HOURS, weather=weather)
    assert "oracle_temperature_same_hour" not in default.columns

    options = {"oracle_temperature": True}
    oracle = fm.make_features(series, TEST_HOURS, weather=weather,
                              feature_options=options)
    np.testing.assert_allclose(
        oracle["oracle_temperature_same_hour"].to_numpy(),
        weather["temperature_2m"].reindex(TEST_HOURS).to_numpy(),
    )

    mutated = weather.copy()
    mutated.loc[TEST_DAY:] = 999.0
    oracle_mutated = fm.make_features(series, TEST_HOURS, weather=mutated,
                                      feature_options=options)
    assert oracle_mutated.loc[
        TEST_DAY:, "oracle_temperature_same_hour"
    ].eq(999.0).all()


def test_feature_options_reach_fit_metadata():
    series = synthetic_series()
    params = {"n_estimators": 10, "max_depth": 2, "n_jobs": 1}
    _, metadata = fm.fit(
        series, series.index[0], series.index[20 * 24 - 1], params=params,
        feature_options={"drop_annual_calendar": True, "use_weather": False},
    )
    assert metadata["feature_options"]["drop_annual_calendar"] is True
    assert metadata["feature_options"]["use_weather"] is False
    assert "month" not in metadata["features"]
    assert not any(column.startswith("weather_")
                   for column in metadata["features"])
