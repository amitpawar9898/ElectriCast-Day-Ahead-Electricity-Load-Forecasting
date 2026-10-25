"""Data pipeline for the IIT Bombay apartment meter CSVs.

Raw schema: TS (Unix seconds, UTC), V1-V3 (volts), W1-W3 (phase energy, Wh).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TZ = "Asia/Kolkata"
REQUIRED_COLUMNS = ["TS", "W1", "W2", "W3"]
MAX_GAP_HOURS = 6

LOAD_ERRORS = (
    FileNotFoundError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    UnicodeDecodeError,
    ValueError,
)


def apartment_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_apartment_file(path):
    """Read one apartment CSV, validate its schema and parse timestamps."""
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("file has no data rows")
    df = df.sort_values("TS").reset_index(drop=True)
    df["timestamp_utc"] = pd.to_datetime(df["TS"], unit="s", utc=True)
    return df


def ist_hour_index(df):
    """Convert UTC stamps to IST and floor each to its containing hour."""
    return df["timestamp_utc"].dt.tz_convert(TZ).dt.floor("h")


def apartment_hourly(df):
    """Sum the three phases and resample one apartment to IST hours."""
    timestamps = ist_hour_index(df)
    phase_sum = df[["W1", "W2", "W3"]].sum(axis=1, min_count=1)
    series = pd.Series(phase_sum.to_numpy(), index=timestamps)
    series = series.sort_index()
    return series.resample("h").sum(min_count=1)


def build_panel(apartments):
    """Combine per-apartment hourly series and count reporting apartments."""
    hourly = {name: apartment_hourly(df) for name, df in apartments.items()}
    panel = pd.concat(hourly, axis=1).sort_index()
    panel.index.name = "timestamp_ist"
    n_reporting = panel.notna().sum(axis=1).astype(int)
    n_reporting.name = "n_reporting"
    return panel, n_reporting


STABLE_START = "2017-05-01"
STABLE_END = "2017-11-30 23:00"
STABLE_MIN_COVERAGE = 0.90


def apartment_coverage(panel, start=STABLE_START, end=STABLE_END):
    """Fraction of expected IST hours observed per apartment in a window."""
    index = pd.date_range(start, end, freq="h", tz=TZ)
    return panel.reindex(index).notna().mean().sort_values(ascending=False)


def stable_apartments(panel, start=STABLE_START, end=STABLE_END,
                      min_coverage=STABLE_MIN_COVERAGE):
    coverage = apartment_coverage(panel, start, end)
    return coverage[coverage >= min_coverage].index.tolist(), coverage


def build_target(panel, stable):
    """Mean load per reporting stable apartment and the reporting count."""
    subset = panel[stable]
    n_reporting = subset.notna().sum(axis=1).astype(int)
    n_reporting.name = "n_reporting"
    target = subset.mean(axis=1)
    target.name = "target_wh"
    return target, n_reporting


def fill_short_gaps(series, max_gap_hours=MAX_GAP_HOURS):
    """Linearly interpolate interior NaN runs up to max_gap_hours.

    Returns (filled, imputed, excluded, runs). Runs longer than the limit and
    runs touching either end stay NaN and are marked excluded.
    """
    filled = series.astype(float).copy()
    imputed = pd.Series(False, index=series.index)
    excluded = pd.Series(False, index=series.index)
    runs = []
    isna = series.isna()
    groups = (isna != isna.shift()).cumsum()

    for _, block in series.groupby(groups):
        if not block.isna().all():
            continue
        hours = len(block)
        start, end = block.index[0], block.index[-1]
        edge = start == series.index[0] or end == series.index[-1]
        runs.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "hours": hours,
            "edge": bool(edge),
        })
        if hours <= max_gap_hours and not edge:
            left = series.loc[:start].dropna().iloc[-1]
            right = series.loc[end:].dropna().iloc[0]
            filled.loc[block.index] = np.linspace(left, right, hours + 2)[1:-1]
            imputed.loc[block.index] = True
        else:
            excluded.loc[block.index] = True

    return filled, imputed, excluded, runs


OUTLIER_IQR_MULTIPLIER = 3.0
OUTLIER_MIN_SAMPLES = 100


def fit_outlier_cap(train_values, k=OUTLIER_IQR_MULTIPLIER,
                    min_samples=OUTLIER_MIN_SAMPLES):
    """Upper cap = Q3 + k*IQR over training values only.

    Callers must pass only training hours (no validation/test hours), so the
    cap cannot leak information from the evaluation window.
    """
    values = pd.Series(train_values, dtype=float).dropna()
    if len(values) < min_samples:
        raise ValueError(
            f"need at least {min_samples} training values, got {len(values)}"
        )
    q1, q3 = values.quantile([0.25, 0.75])
    return float(q3 + k * (q3 - q1))


def apply_outlier_cap(series, cap):
    """Clip values above a train-fitted cap; return (capped, n_capped)."""
    capped = series.clip(upper=cap)
    n_capped = int((series > cap).sum())
    return capped, n_capped


INACTIVE_MONTH = "2017-12"
INACTIVE_MAX_MEAN_WH = 100.0
INACTIVE_MIN_HOURS = 24


def apartment_activity(panel, month=INACTIVE_MONTH,
                       threshold=INACTIVE_MAX_MEAN_WH,
                       min_hours=INACTIVE_MIN_HOURS):
    """Per-apartment mean load and inactivity flag for one IST month.

    Apartments with fewer than min_hours readings in the month are not flagged
    (insufficient evidence); apartments with no readings get NaN mean.
    """
    start = pd.Timestamp(f"{month}-01", tz=TZ)
    end = start + pd.DateOffset(months=1)
    index = pd.date_range(start, end, freq="h", tz=TZ, inclusive="left")
    sub = panel.reindex(index)
    stats = pd.DataFrame({
        "mean_wh": sub.mean(),
        "hours": sub.notna().sum().astype(int),
    })
    stats.index.name = "apartment"
    stats["suspect_inactive"] = (
        (stats["hours"] >= min_hours)
        & stats["mean_wh"].notna()
        & (stats["mean_wh"] < threshold)
    )
    return stats.sort_values("mean_wh")



def load_apartments(data_dir=DATA_DIR):
    """Load every CSV in data_dir and return (apartments, report).

    apartments maps apartment id -> DataFrame with timestamp_utc.
    report records files found/read/failed and the overall timestamp range.
    """
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.csv")),
        key=lambda p: apartment_id(p),
    )
    apartments = {}
    report = {
        "data_dir": os.path.abspath(data_dir),
        "files_found": len(paths),
        "read_ok": 0,
        "failed": 0,
        "rows_total": 0,
        "first_utc": None,
        "last_utc": None,
        "first_ist": None,
        "last_ist": None,
        "files": [],
        "errors": [],
    }

    for path in paths:
        name = apartment_id(path)
        try:
            df = read_apartment_file(path)
        except LOAD_ERRORS as exc:
            report["failed"] += 1
            report["errors"].append({
                "file": os.path.basename(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        apartments[name] = df
        report["read_ok"] += 1
        report["rows_total"] += len(df)
        first_utc = df["timestamp_utc"].iloc[0]
        last_utc = df["timestamp_utc"].iloc[-1]
        report["files"].append({
            "file": os.path.basename(path),
            "rows": int(len(df)),
            "first_utc": first_utc.isoformat(),
            "last_utc": last_utc.isoformat(),
            "first_ist": first_utc.tz_convert(TZ).isoformat(),
            "last_ist": last_utc.tz_convert(TZ).isoformat(),
        })

    if report["files"]:
        report["first_utc"] = min(f["first_utc"] for f in report["files"])
        report["last_utc"] = max(f["last_utc"] for f in report["files"])
        report["first_ist"] = min(f["first_ist"] for f in report["files"])
        report["last_ist"] = max(f["last_ist"] for f in report["files"])

    return apartments, report


if __name__ == "__main__":
    _, summary = load_apartments()
    print(json.dumps(summary, indent=2))
