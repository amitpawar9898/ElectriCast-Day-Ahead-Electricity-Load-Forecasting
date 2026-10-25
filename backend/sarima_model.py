"""SARIMA rolling day-ahead forecaster with weekly refits.

The model is refit on Mondays (and on the first test day) using the trailing
60 days of gap-handled target ending one hour before the origin. Within a week
the fitted state is advanced with actuals up to each day's origin, so no value
from the forecast day is ever read. A coverage gap breaks that state chain; when
a new contiguous block appears the model is refit on it.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import data_pipeline as dp

ORDER = (1, 0, 1)
SEASONAL_ORDER = (0, 1, 1, 24)
MAX_HISTORY = 60 * 24
MIN_HISTORY = 7 * 24
REFIT_DAYS = 7
MAX_STALENESS_HOURS = 48


def _hour_range(start, end):
    return pd.date_range(start, end, freq="h", tz=dp.TZ)


def contiguous_tail(series, origin, max_history=MAX_HISTORY,
                    min_history=MIN_HISTORY):
    """Last contiguous valid run ending at or before `origin`, or None.

    The run is capped to `max_history` hours. Runs shorter than `min_history`
    are rejected because the seasonal state would be too poorly identified.
    """
    history = series.loc[:origin]
    last_valid = history.last_valid_index()
    if last_valid is None:
        return None
    before = history.loc[:last_valid]
    missing = before[before.isna()]
    if missing.empty:
        start = before.index[0]
    else:
        start = missing.index[-1] + pd.Timedelta(hours=1)
    tail = before.loc[start:]
    if len(tail) < min_history:
        return None
    return tail.iloc[-max_history:]


def _fit(tail):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            tail,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def forecast(series, test_start, test_end, max_history=MAX_HISTORY,
             min_history=MIN_HISTORY):
    """Forecast every hour in [test_start, test_end].

    Returns a prediction series (NaN where no model state exists) and metadata.
    """
    hours = _hour_range(test_start, test_end)
    pred = pd.Series(np.nan, index=hours, name="sarima")
    result = None
    state_end = None
    last_refit = None
    refit_origins = []
    fit_failures = 0

    for day in pd.date_range(test_start, test_end, freq="D", tz=dp.TZ):
        origin = day
        cutoff = origin - pd.Timedelta(hours=1)
        tail = contiguous_tail(
            series, cutoff, max_history=max_history, min_history=min_history
        )

        need_refit = result is None
        if result is not None and tail is not None:
            if tail.index[-1] > state_end:
                if tail.index[-1] == cutoff:
                    if origin.dayofweek == 0 or (
                        last_refit is not None
                        and origin - last_refit >= pd.Timedelta(days=REFIT_DAYS)
                    ):
                        need_refit = True
                    else:
                        update = series.loc[
                            state_end + pd.Timedelta(hours=1):cutoff
                        ]
                        expected = int((cutoff - state_end) / pd.Timedelta(hours=1))
                        if len(update) == expected and update.notna().all():
                            try:
                                result = result.append(update, refit=False)
                                state_end = cutoff
                            except Exception:
                                need_refit = True
                        elif last_refit is not None and (
                            origin - last_refit >= pd.Timedelta(days=1)
                        ):
                            need_refit = True
                elif last_refit is not None and (
                    origin - last_refit >= pd.Timedelta(days=1)
                ):
                    need_refit = True

        if need_refit and tail is not None:
            try:
                result = _fit(tail)
                state_end = tail.index[-1]
                last_refit = origin
                refit_origins.append(str(origin))
            except Exception:
                fit_failures += 1

        if result is None or state_end is None or state_end >= origin:
            continue
        if origin - state_end > pd.Timedelta(hours=MAX_STALENESS_HOURS):
            continue
        day_end = day + pd.Timedelta(hours=23)
        steps = int((day_end - state_end) / pd.Timedelta(hours=1))
        if steps < 24:
            continue
        try:
            values = result.get_forecast(steps=steps).predicted_mean.iloc[-24:]
            pred.loc[day:day_end] = np.maximum(
                values.to_numpy(dtype=float), 0.0
            )
        except Exception:
            fit_failures += 1

    metadata = {
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "max_history_hours": int(max_history),
        "min_history_hours": int(min_history),
        "max_staleness_hours": int(MAX_STALENESS_HOURS),
        "n_fits": len(refit_origins),
        "fit_failures": fit_failures,
        "refit_origins": refit_origins,
        "predicted_hours": int(pred.notna().sum()),
        "nan_hours": int(pred.isna().sum()),
        "refit_policy": "Mondays and first test day; recovery refit after coverage gaps",
    }
    return pred, metadata
