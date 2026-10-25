"""Chronological split tests for the upload evaluation.

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
import upload_validation as uv  # noqa: E402


def synthetic_target(start, days=40):
    index = pd.date_range(start, periods=days * 24, freq="h", tz=dp.TZ)
    values = (
        1000.0
        + 100.0 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
        + 0.1 * np.arange(len(index))
    )
    return pd.Series(values, index=index, name="target_wh")


def stub_model(filled, test_hours, train_end):
    return pd.Series(np.nan, index=test_hours), {"status": "stub"}


def test_split_is_chronological_and_starts_at_midnight(monkeypatch):
    filled = synthetic_target("2020-01-01")
    monkeypatch.setattr(uv, "_zero_shot_predictions", stub_model)
    monkeypatch.setattr(uv, "_fit_upload_model", stub_model)

    result = uv.evaluate_upload(filled)

    assert result["status"] == "ok"
    split = result["split"]
    assert split["train_hours"] == 576
    assert split["test_hours"] == 384
    assert split["eval_hours"] == 384
    assert split["test_days"] == 16
    assert split["test_start"] == "2020-01-25T00:00:00+05:30"
    assert split["train_end"] == "2020-01-24T23:00:00+05:30"
    assert result["models"]["fit_60_xgboost"]["fit"] == {"status": "stub"}


def test_split_floors_an_off_midnight_series_to_the_day(monkeypatch):
    filled = synthetic_target("2020-01-01 05:00")
    monkeypatch.setattr(uv, "_zero_shot_predictions", stub_model)
    monkeypatch.setattr(uv, "_fit_upload_model", stub_model)

    split = uv.evaluate_upload(filled)["split"]

    assert split["test_start"].endswith("T00:00:00+05:30")
    assert pd.Timestamp(split["train_end"]) + pd.Timedelta(hours=1) == pd.Timestamp(
        split["test_start"]
    )
    assert 336 < split["train_hours"] < 960


def test_short_series_returns_a_reason_instead_of_metrics():
    filled = synthetic_target("2020-01-01", days=10)
    result = uv.evaluate_upload(filled)

    assert result["status"] == "insufficient_data"
    assert result["reason"] == "need at least 336 valid target hours, found 240"
    assert result["models"] == {}
    assert result["split"] == {}
    assert result["daily"] is None
