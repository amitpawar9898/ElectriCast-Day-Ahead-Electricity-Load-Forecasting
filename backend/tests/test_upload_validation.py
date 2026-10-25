"""Column mapping, timestamp, unit, and row-accounting tests for uploads.

Run from the repo root:

    python -m pytest backend/tests -q
"""
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import upload_validation as uv  # noqa: E402


def mapping(**overrides):
    base = {
        "timestamp_col": "time",
        "value_col": "value",
        "series_col": "",
        "unit": "Wh",
        "timezone": "UTC",
        "timestamp_mode": "datetime",
    }
    base.update(overrides)
    return base


def test_suggested_mapping_finds_naive_datetime_columns():
    frame = pd.DataFrame({
        "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "load": [1000.0, 1010.0],
    })
    suggested = uv._suggested_mapping(frame)

    assert suggested["timestamp_col"] == "timestamp"
    assert suggested["value_col"] == "load"
    assert suggested["timestamp_mode"] == "datetime"


def test_suggested_mapping_finds_epoch_and_unit():
    frame = pd.DataFrame({
        "Unix Timestamp": [1672531200, 1672534800],
        "Energy Consumption (kWh)": [1.0, 2.0],
    })
    suggested = uv._suggested_mapping(frame)

    assert suggested["timestamp_col"] == "Unix Timestamp"
    assert suggested["timestamp_mode"] == "epoch_s"
    assert suggested["unit"] == "kWh"


def test_timestamp_parsing_agrees_across_naive_utc_offset_and_epoch():
    naive, naive_meta = uv._parse_timestamps(
        pd.Series(["2023-01-01 00:00:00"]), "datetime", "UTC"
    )
    offset, offset_meta = uv._parse_timestamps(
        pd.Series(["2023-01-01T00:00:00Z"]), "datetime", "UTC"
    )
    epoch, epoch_meta = uv._parse_timestamps(
        pd.Series([1672531200]), "epoch_s", "UTC"
    )

    assert naive[0].isoformat() == "2023-01-01T05:30:00+05:30"
    assert offset[0] == naive[0]
    assert epoch[0] == naive[0]
    assert naive_meta["mode"] == "datetime_naive"
    assert offset_meta["mode"] == "datetime_with_offset"
    assert epoch_meta["mode"] == "epoch_s"


def test_energy_readings_are_summed_and_series_are_averaged():
    text = "\n".join([
        "time,apt,value",
        "2023-01-01 00:00:00,A,0.5",
        "2023-01-01 00:10:00,A,0.5",
        "2023-01-01 00:00:00,B,2.0",
        "2023-01-01 00:30:00,B,2.0",
        "2023-01-01 01:00:00,A,1.0",
        "2023-01-01 01:30:00,A,1.0",
        "2023-01-01 01:00:00,B,3.0",
        "2023-01-01 01:30:00,B,3.0",
    ])
    filled, report = uv._build_target(
        text.encode(),
        mapping(series_col="apt", unit="kWh", timezone="Asia/Kolkata"),
        "energy.csv",
        len(text),
    )

    assert report["rows_used"] == 8
    assert report["series_count"] == 2
    assert report["hourly"]["target_mean_wh"] == 3250.0
    assert float(filled.iloc[0]) == 2500.0
    assert float(filled.iloc[1]) == 4000.0
    assert any("aggregated by sum" in warning for warning in report["warnings"])


def test_power_readings_are_averaged_within_the_hour():
    text = "\n".join([
        "time,apt,value",
        "2023-01-01 00:00:00,A,400",
        "2023-01-01 00:30:00,A,600",
        "2023-01-01 00:00:00,B,1000",
        "2023-01-01 00:30:00,B,2000",
    ])
    filled, report = uv._build_target(
        text.encode(),
        mapping(series_col="apt", unit="W", timezone="Asia/Kolkata"),
        "power.csv",
        len(text),
    )

    assert report["rows_used"] == 4
    assert report["hourly"]["target_mean_wh"] == 1000.0
    assert float(filled.iloc[0]) == 1000.0
    assert any("aggregated by mean" in warning for warning in report["warnings"])


def test_invalid_and_negative_rows_are_counted_once():
    text = "\n".join([
        "time,value",
        "2023-01-01 00:00:00,10",
        "2023-01-01 01:00:00,oops",
        "2023-01-01 02:00:00,-5",
        "2023-01-01 03:00:00,20",
    ])
    filled, report = uv._build_target(
        text.encode(), mapping(), "bad_rows.csv", len(text)
    )

    assert report["rows_raw"] == 4
    assert report["rows_used"] == 2
    assert report["rows_timestamp_invalid"] == 0
    assert report["rows_value_invalid"] == 1
    assert report["rows_negative"] == 1
    assert report["rows_raw"] - report["rows_used"] == 2
    assert report["hourly"]["valid_hours"] == 4
    assert float(filled.iloc[0]) == 10.0
    assert abs(float(filled.iloc[1]) - (10.0 + 10.0 / 3.0)) < 1e-9
