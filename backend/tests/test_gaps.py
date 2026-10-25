"""Gap handling tests for the hourly target.

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


def make_series(values):
    index = pd.date_range("2021-01-01", periods=len(values), freq="h", tz=dp.TZ)
    return pd.Series(values, index=index, dtype=float)


def test_short_interior_gap_is_interpolated_and_flagged():
    values = [
        np.nan, np.nan,
        10.0, 10.0, 10.0,
        np.nan, np.nan, np.nan,
        20.0,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        30.0,
        np.nan, np.nan, np.nan,
    ]
    filled, imputed, excluded, runs = dp.fill_short_gaps(make_series(values))

    np.testing.assert_allclose(filled.iloc[5:8].to_numpy(), [12.5, 15.0, 17.5])
    assert imputed.iloc[5:8].all()
    assert imputed.sum() == 3

    assert np.isnan(filled.iloc[0:2]).all()
    assert np.isnan(filled.iloc[9:16]).all()
    assert np.isnan(filled.iloc[17:20]).all()
    assert excluded.iloc[0:2].all()
    assert excluded.iloc[9:16].all()
    assert excluded.iloc[17:20].all()

    assert [(run["hours"], run["edge"]) for run in runs] == [
        (2, True), (3, False), (7, False), (3, True)
    ]
    assert runs[1]["start"].endswith("05:00:00+05:30")
    assert runs[1]["end"].endswith("07:00:00+05:30")


def test_long_gap_is_left_missing_and_excluded():
    values = [1.0] * 5 + [np.nan] * 7 + [2.0] * 5
    filled, imputed, excluded, runs = dp.fill_short_gaps(make_series(values))

    assert np.isnan(filled.iloc[5:12]).all()
    assert not imputed.any()
    assert excluded.iloc[5:12].all()
    assert len(runs) == 1
    assert runs[0]["hours"] == dp.MAX_GAP_HOURS + 1
    assert runs[0]["edge"] is False


def test_gap_limit_is_configurable_and_edges_are_never_filled():
    values = [np.nan, 1.0, np.nan, np.nan, 2.0, np.nan]
    filled, imputed, excluded, runs = dp.fill_short_gaps(
        make_series(values), max_gap_hours=2
    )

    np.testing.assert_allclose(
        filled.iloc[2:4].to_numpy(), [1.0 + 1.0 / 3.0, 1.0 + 2.0 / 3.0]
    )
    assert imputed.sum() == 2
    assert excluded.iloc[0] and excluded.iloc[-1]
    assert [(run["hours"], run["edge"]) for run in runs] == [
        (1, True), (2, False), (1, True)
    ]
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[-1])
