"""Level-normalization helpers for the Phase 9 experiments.

The target scale is the mean of observed targets over the seven days ending
one hour before each day's 00:00 IST origin, so it is known at forecast time.
Pass the un-filled target so interpolated values cannot leak across the origin.

The near-zero fraction is the share of stable-panel apartments with an hourly
reading below the existing 100 Wh inactivity threshold. It reuses that
threshold as an hourly proxy; it is a feature, not an inactivity finding.
"""

import numpy as np
import pandas as pd

import data_pipeline as dp

TRAILING_DAYS = 7
MIN_TRAILING_HOURS = 24
NEAR_ZERO_WH = dp.INACTIVE_MAX_MEAN_WH


def trailing_scale(series, hours, days=TRAILING_DAYS,
                   min_periods=MIN_TRAILING_HOURS):
    """Per-target-hour level, constant within each IST day.

    `series` should contain observed targets only (NaNs at gaps). A day gets a
    scale only when at least `min_periods` observations fall in its trailing
    window; otherwise the scale is NaN and normalized forecasts are not made.
    """
    hours = pd.DatetimeIndex(hours)
    origins = pd.DatetimeIndex(hours.floor("D").unique())
    ends = origins - pd.Timedelta(hours=1)
    rolling = series.rolling(
        pd.Timedelta(hours=days * 24), min_periods=min_periods
    ).mean()
    by_day = pd.Series(rolling.reindex(ends).to_numpy(dtype=float), index=origins)
    values = by_day.reindex(hours.floor("D")).to_numpy(dtype=float)
    return pd.Series(values, index=hours, name="target_scale")


def near_zero_fraction(panel, stable, threshold=NEAR_ZERO_WH):
    """Hourly share of reporting stable apartments below `threshold` Wh."""
    subset = panel[stable]
    reporting = subset.notna().sum(axis=1)
    near_zero = (subset < threshold) & subset.notna()
    fraction = near_zero.sum(axis=1) / reporting.where(reporting > 0)
    fraction.name = "near_zero_fraction"
    return fraction
