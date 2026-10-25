"""Tests for Phase 9.4 level normalization and the near-zero feature.

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
import forecast_model as fm  # noqa: E402
import level_features as lf  # noqa: E402

TZ = dp.TZ
TEST_DAY = pd.Timestamp("2021-01-08", tz=TZ)
TEST_HOURS = pd.date_range(TEST_DAY, periods=24, freq="h", tz=TZ)


def synthetic_series(days=25):
    index = pd.date_range("2021-01-01", periods=days * 24, freq="h", tz=TZ)
    values = (
        500.0
        + 120.0 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
        + 0.2 * np.arange(len(index))
    )
    return pd.Series(values, index=index, name="target_wh")


def test_trailing_scale_is_known_at_origin_and_constant_within_day():
    series = synthetic_series()
    scale = lf.trailing_scale(series, TEST_HOURS)
    assert scale.nunique() == 1
    expected = series.loc[
        TEST_DAY - pd.Timedelta(days=7):TEST_DAY - pd.Timedelta(hours=1)
    ].mean()
    assert abs(float(scale.iloc[0]) - float(expected)) < 1e-9

    mutated = series.copy()
    mutated.loc[TEST_DAY:] = 1e9
    after = lf.trailing_scale(mutated, TEST_HOURS)
    np.testing.assert_allclose(scale.to_numpy(), after.to_numpy())


def test_trailing_scale_requires_enough_observations():
    series = synthetic_series().iloc[:12]
    scale = lf.trailing_scale(series, TEST_HOURS)
    assert scale.isna().all()


def test_near_zero_fraction_hand_checked():
    index = pd.date_range("2021-01-01", periods=4, freq="h", tz=TZ)
    panel = pd.DataFrame({
        "a": [50.0, 150.0, np.nan, 0.0],
        "b": [99.0, 100.0, 200.0, np.nan],
        "c": [np.nan, 50.0, 50.0, 50.0],
    }, index=index)
    fraction = lf.near_zero_fraction(panel, ["a", "b", "c"])
    np.testing.assert_allclose(fraction.to_numpy(), [1.0, 1 / 3, 0.5, 1.0])
    assert fraction.name == "near_zero_fraction"


def test_apply_target_scale_hand_checked():
    index = pd.date_range("2021-01-01", periods=3, freq="h", tz=TZ)
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)
    labels = pd.Series([10.0, 20.0, 30.0], index=index)
    scale = pd.Series([2.0, np.nan, 0.0], index=index)

    kept_features, kept_labels, kept_scale = fm._apply_target_scale(
        features, labels, scale
    )
    assert list(kept_features.index) == [index[0]]
    assert list(kept_labels) == [5.0]
    assert list(kept_scale) == [2.0]


def test_predict_multiplies_by_scale_and_leaves_missing_scale_nan():
    series = synthetic_series()

    class FixedModel:
        def predict(self, features):
            return np.full(len(features), 0.5)

    scale = pd.Series(100.0, index=TEST_HOURS)
    scale.iloc[5] = np.nan
    prediction = fm.predict(
        series, FixedModel(), TEST_DAY, TEST_HOURS[-1], target_scale=scale
    )
    assert prediction.iloc[:5].eq(50.0).all()
    assert prediction.iloc[5:6].isna().all()
    assert prediction.iloc[6:].eq(50.0).all()


def test_near_zero_feature_is_lagged_24_hours():
    series = synthetic_series()
    fraction = pd.Series(
        np.linspace(0.0, 1.0, len(series)), index=series.index
    )
    features = fm.make_features(
        series, TEST_HOURS, near_zero_fraction=fraction
    )
    np.testing.assert_allclose(
        features["near_zero_fraction_lag_24"].to_numpy(),
        fraction.reindex(TEST_HOURS - pd.Timedelta(hours=24)).to_numpy(),
    )

    mutated = fraction.copy()
    mutated.loc[TEST_DAY:] = 1.0
    after = fm.make_features(
        series, TEST_HOURS, near_zero_fraction=mutated
    )
    pd.testing.assert_frame_equal(features, after)


def test_forecast_with_target_scale_reports_scale_coverage():
    series = synthetic_series()
    params = {"n_estimators": 20, "max_depth": 2, "n_jobs": 1}
    all_hours = pd.date_range(
        series.index[0], TEST_HOURS[-1], freq="h", tz=TZ
    )
    scale = pd.Series(100.0, index=all_hours)

    prediction, metadata = fm.forecast(
        series, series.index[0], series.index[20 * 24 - 1],
        TEST_DAY, TEST_HOURS[-1], params=params, target_scale=scale,
    )
    assert prediction.notna().all()
    assert metadata["normalization"]["n_train_scaled"] > 0
    assert metadata["normalization"]["predicted_hours"] == 24
    assert metadata["normalization"]["missing_scale_hours"] == 0
    assert (prediction >= 0).all()

    missing = scale.copy()
    missing.loc[TEST_DAY:] = np.nan
    prediction, metadata = fm.forecast(
        series, series.index[0], series.index[20 * 24 - 1],
        TEST_DAY, TEST_HOURS[-1], params=params, target_scale=missing,
    )
    assert prediction.isna().all()
    assert metadata["normalization"]["predicted_hours"] == 0
    assert metadata["normalization"]["missing_scale_hours"] == 24
