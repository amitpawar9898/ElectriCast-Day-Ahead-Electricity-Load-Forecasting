"""Open-Meteo archive weather for the IIT Bombay campus, cached to disk.

The cache is fetched once and committed, so backtests run offline and are
reproducible. Weather at the target hour is never used directly: the model
only receives weather lags of at least 24 h, matching the day-ahead origin.
"""
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

import data_pipeline as dp

LATITUDE = 19.1334
LONGITUDE = 72.9133
START_DATE = "2017-03-01"
END_DATE = "2018-06-30"
HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
)
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weather_mumbai.csv"
)
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch():
    """Download the archive window and write the CSV cache."""
    query = urllib.parse.urlencode({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    })
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=120) as response:
        payload = json.load(response)

    hourly = payload["hourly"]
    frame = pd.DataFrame({
        "timestamp_ist": pd.to_datetime(hourly["time"]).tz_localize(dp.TZ),
        **{name: hourly[name] for name in HOURLY_VARS},
    }).set_index("timestamp_ist")

    expected = pd.date_range(
        START_DATE, f"{END_DATE} 23:00", freq="h", tz=dp.TZ
    )
    frame = frame.reindex(expected)
    frame.index.name = "timestamp_ist"
    frame.to_csv(CACHE_PATH, float_format="%.2f")
    return frame


def load():
    """Load the cached weather frame, or None when the cache is absent."""
    if not os.path.exists(CACHE_PATH):
        return None
    frame = pd.read_csv(CACHE_PATH)
    index = pd.to_datetime(frame.pop("timestamp_ist"), utc=True)
    frame.index = index.dt.tz_convert(dp.TZ)
    frame.index.name = "timestamp_ist"
    return frame[list(HOURLY_VARS)].astype(float)


def describe(frame):
    """Small JSON-serializable description for the results file."""
    if frame is None:
        return {
            "available": False,
            "cache": os.path.basename(CACHE_PATH),
            "note": "weather cache missing; model runs without weather features",
        }
    missing = {
        name: int(frame[name].isna().sum()) for name in HOURLY_VARS
    }
    return {
        "available": True,
        "cache": os.path.basename(CACHE_PATH),
        "source": API_URL,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "hours": int(len(frame)),
        "variables": list(HOURLY_VARS),
        "missing_values": missing,
        "information_set": "weather lags >= 24 h only; no target-hour weather",
    }


def main():
    frame = fetch()
    print(f"Wrote {CACHE_PATH}")
    print(f"{len(frame)} hours, {frame.index.min()} to {frame.index.max()}")
    print(frame.isna().sum().to_string())


if __name__ == "__main__":
    main()
