"""Read-only tools over the ElectriCast backtest artifacts.

Everything the chat assistant can state must arrive here first. The tools
never write to disk, never call a model, and never invent a fallback value:
missing artifacts or hours come back as explicit ``None``/``found: false``.

The deterministic planner maps a question to one or more tool calls. That
keeps the 3B local model out of the tool-selection loop and makes the
behavior testable with hand-built artifacts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


MODEL_LABELS = {
    "persistence": "Persistence",
    "seasonal_naive_24": "Seasonal naive 24h",
    "same_hour_last_week": "Same hour last week",
    "mean_last_4_same_weekdays": "Mean of 4 same weekdays",
    "xgboost": "XGBoost",
    "sarima": "SARIMA",
    "xgboost_quantile": "Quantile XGBoost (P50)",
}

SUMMARY_METRICS = ("MAE", "RMSE", "WAPE", "MASE", "R2", "sMAPE", "bias")

# (hourly column, canonical metrics key, label)
HOURLY_MODELS = (
    ("seasonal_naive", "seasonal_naive_24", "Seasonal naive 24h"),
    ("xgboost", "xgboost", "XGBoost"),
    ("sarima", "sarima", "SARIMA"),
    ("p50", "xgboost_quantile", "Quantile XGBoost (P50)"),
)
HOURLY_FOR_MODEL = {key: column for column, key, _ in HOURLY_MODELS}

MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

SUGGESTED_QUESTIONS = (
    "Which model is best and by how much?",
    "Why is December different?",
    "Where does XGBoost fail?",
)

SOURCES = ("backend/results.json", "backend/data_quality.json")

OFF_TOPIC_ANSWER = (
    "I can only answer questions about the ElectriCast backtest artifacts. "
    "Ask about model metrics, errors by month or hour, an evaluated day, "
    "data quality, or limitations."
)

_ALIASES = (
    ("mean of 4 same weekdays", "mean_last_4_same_weekdays"),
    ("mean of four same weekdays", "mean_last_4_same_weekdays"),
    ("mean of 4 weekdays", "mean_last_4_same_weekdays"),
    ("four same weekdays", "mean_last_4_same_weekdays"),
    ("seasonal naive 24", "seasonal_naive_24"),
    ("seasonal naive", "seasonal_naive_24"),
    ("seasonal-naive", "seasonal_naive_24"),
    ("same hour last week", "same_hour_last_week"),
    ("same hour weekly", "same_hour_last_week"),
    ("last week", "same_hour_last_week"),
    ("quantile xgboost", "xgboost_quantile"),
    ("quantile", "xgboost_quantile"),
    ("p50", "xgboost_quantile"),
    ("xgboost", "xgboost"),
    ("xgb", "xgboost"),
    ("sarima", "sarima"),
    ("persistence", "persistence"),
    ("naive", "seasonal_naive_24"),
)

_MONTH_LOOKUP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_QUALITY_TERMS = (
    "data quality", "missing", "gap", "coverage", "stable panel", "stable",
    "apartment", "inactive", "vacanc", "raw file", "files", "rows",
    "weather", "report", "dataset",
)
_LIMIT_TERMS = (
    "limitation", "caveat", "why", "fail", "weak", "worst", "trust",
    "confiden", "uncertain", "interval", "undercover", "calibrat",
    "different", "differ", "problem", "bias", "coverage",
)
_METRIC_TERMS = (
    "metric", "mae", "rmse", "wape", "mase", "smape", "r2", "r\u00b2",
    "score", "accuracy", "accurate", "best", "compare", "comparison",
    "rank", "performance", "beat", "better", "good", "bad", "model",
    "fold", "backtest", "result", "evaluation", "improve",
)
_ERROR_TERMS = ("error", "fail", "weak", "worst", "wrong", "off")
_PROTOCOL_PATTERNS = (
    r"\bprotocol\b", r"\btarget\b", r"\bunits?\b", r"\bwh\b", r"\bkwh\b",
    r"\btimezone\b", r"\borigin\b", r"\bleak", r"\btrain", r"\bhorizon\b",
    r"\bmethod\b",
)

_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_FOLD_RE = re.compile(r"\b(20\d{2}-\d{2}(?:_\d{2})?)\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_CLOCK_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_AMPM_RE = re.compile(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", re.IGNORECASE)
_AT_HOUR_RE = re.compile(r"\b(?:hour|hr|at)\s+([01]?\d|2[0-3])\b", re.IGNORECASE)


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _fold_display(fold) -> str:
    return (fold or {}).get("label") or _fold_label((fold or {}).get("test"))


def _fold_label(test) -> str:
    start = str((test or {}).get("start", ""))[:10]
    end = str((test or {}).get("end", ""))[:10]
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        return start[:7] or "unknown fold"
    if (start_dt.year, start_dt.month) == (end_dt.year, end_dt.month):
        return f"{MONTH_NAMES[start_dt.month - 1]} {start_dt.year}"
    if start_dt.year == end_dt.year:
        return (f"{MONTH_NAMES[start_dt.month - 1]}\u2013"
                f"{MONTH_NAMES[end_dt.month - 1]} {start_dt.year}")
    return (f"{MONTH_NAMES[start_dt.month - 1]} {start_dt.year}\u2013"
            f"{MONTH_NAMES[end_dt.month - 1]} {end_dt.year}")


def _period_label(period: str) -> str:
    try:
        year, month = period.split("-")
        return f"{MONTH_NAMES[int(month) - 1]} {year}"
    except (ValueError, IndexError):
        return period


def _round(value, digits=4):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _mean_std(values):
    numbers = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    if not numbers:
        return {"mean": None, "std": None, "n": 0}
    std = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0
    return {
        "mean": round(float(np.mean(numbers)), 4),
        "std": round(std, 4),
        "n": len(numbers),
    }


def _pair_stats(records, key):
    pairs = []
    for record in records:
        actual = record.get("actual")
        predicted = record.get(key)
        if actual is None or predicted is None:
            continue
        try:
            actual = float(actual)
            predicted = float(predicted)
        except (TypeError, ValueError):
            continue
        if math.isfinite(actual) and math.isfinite(predicted):
            pairs.append((actual, predicted))
    if not pairs:
        return {"n": 0, "mae_wh": None, "bias_wh": None}
    mae = sum(abs(pred - actual) for actual, pred in pairs) / len(pairs)
    bias = sum(pred - actual for actual, pred in pairs) / len(pairs)
    return {"n": len(pairs), "mae_wh": round(mae, 2), "bias_wh": round(bias, 2)}


def _mean_actual(records):
    values = []
    for record in records:
        actual = record.get("actual")
        if actual is None:
            continue
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            continue
        if math.isfinite(actual):
            values.append(actual)
    return round(float(np.mean(values)), 2) if values else None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


class ChatTools:
    """Deterministic, read-only queries over ``results.json`` and ``data_quality.json``."""

    def __init__(self, results, quality):
        self.results = results or {}
        self.quality = quality or {}
        self.folds = list(self.results.get("folds") or [])
        self.fold_ids = {fold.get("id") for fold in self.folds}
        self.fold_labels = {
            fold.get("id"): _fold_display(fold) for fold in self.folds
        }
        self.scoring_folds = [
            fold for fold in self.folds if not fold.get("stress_test")
        ]
        self.scoring_fold_ids = [fold.get("id") for fold in self.scoring_folds]
        self.stress_fold_ids = [
            fold.get("id") for fold in self.folds if fold.get("stress_test")
        ]
        self.eval_hours_total = sum(
            int((fold.get("test") or {}).get("eval_hours") or 0)
            for fold in self.scoring_folds
        )
        self._records_by_fold = {}
        self._all_records = []
        for fold in self.folds:
            hourly = fold.get("hourly") or {}
            columns = hourly.get("columns") or []
            records = [
                dict(zip(columns, row)) for row in (hourly.get("rows") or [])
            ]
            self._records_by_fold[fold.get("id")] = records
            self._all_records.extend(records)

        month_groups = {}
        hour_groups = {hour: [] for hour in range(24)}
        for record in self._all_records:
            timestamp = str(record.get("time", ""))
            if len(timestamp) >= 13:
                month_groups.setdefault(timestamp[:7], []).append(record)
                try:
                    hour_groups[int(timestamp[11:13])].append(record)
                except ValueError:
                    pass
        self._month_groups = month_groups
        self._hour_groups = hour_groups
        self._summary = self._build_summary()

    # ── summary helpers ───────────────────────────────────────────

    def _build_summary(self):
        descriptions = {}
        descriptions.update(self.results.get("baseline_descriptions") or {})
        descriptions.update(self.results.get("learned_model_descriptions") or {})
        baselines = set(self.results.get("baselines") or [])
        model_names = list((self.folds[0].get("models") or {}).keys()) \
            if self.folds else []

        summaries = []
        for name in model_names:
            per_fold = []
            for fold in self.folds:
                metrics = (fold.get("models") or {}).get(name) or {
                    "status": "no_data", "n": 0
                }
                seasonal = (fold.get("models") or {}).get("seasonal_naive_24") or {}
                seasonal_mae = (
                    seasonal.get("MAE")
                    if seasonal.get("status") == "ok" else None
                )
                ok = metrics.get("status") == "ok"
                beats = None
                if ok and name != "seasonal_naive_24" and metrics.get("MAE") is not None \
                        and seasonal_mae is not None:
                    beats = bool(metrics["MAE"] < seasonal_mae)
                per_fold.append({
                    "id": fold.get("id"),
                    "label": _fold_display(fold),
                    "stress_test": bool(fold.get("stress_test")),
                    "status": metrics.get("status", "no_data"),
                    "n": metrics.get("n", 0),
                    **{metric: metrics.get(metric) for metric in SUMMARY_METRICS},
                    "beats_seasonal_naive": beats,
                    "seasonal_naive_MAE": seasonal_mae,
                })

            scored = [row for row in per_fold if not row["stress_test"]]
            stats = {
                metric: _mean_std([
                    row[metric] for row in scored if row["status"] == "ok"
                ])
                for metric in SUMMARY_METRICS
            }
            comparable = [
                row for row in scored
                if row["status"] == "ok" and row["seasonal_naive_MAE"] is not None
            ]
            folds_beaten = sum(
                1 for row in comparable if row["beats_seasonal_naive"] is True
            )
            n_total = sum(
                int(row["n"] or 0) for row in scored if row["status"] == "ok"
            )
            summaries.append({
                "key": name,
                "label": MODEL_LABELS.get(name, name),
                "kind": "baseline" if name in baselines else "learned",
                "description": descriptions.get(name, ""),
                "mean": {metric: stats[metric]["mean"] for metric in SUMMARY_METRICS},
                "std": {metric: stats[metric]["std"] for metric in SUMMARY_METRICS},
                "n_folds": stats["MAE"]["n"],
                "n_total_hours": n_total,
                "eval_hours": self.eval_hours_total,
                "coverage_percent": (
                    round(100.0 * n_total / self.eval_hours_total, 2)
                    if self.eval_hours_total else None
                ),
                "folds_beaten": folds_beaten,
                "comparable_folds": len(comparable),
                "beats_seasonal_naive": (
                    None if name == "seasonal_naive_24"
                    else (folds_beaten > len(comparable) / 2 if comparable else None)
                ),
                "per_fold": per_fold,
            })
        return summaries

    def _model_summary(self, key):
        return next((row for row in self._summary if row["key"] == key), None)

    def _best_block(self):
        ranked = sorted(
            [
                row for row in self._summary
                if row["mean"]["MAE"] is not None
            ],
            key=lambda row: row["mean"]["MAE"],
        )
        if not ranked:
            return None
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        seasonal = self._model_summary("seasonal_naive_24")
        seasonal_mae = seasonal["mean"]["MAE"] if seasonal else None
        block = {
            "key": best["key"],
            "label": best["label"],
            "metric": "unweighted mean of per-fold MAE over the scoring folds, in Wh",
            "fold_mean_MAE_wh": best["mean"]["MAE"],
            "n_total_hours": best["n_total_hours"],
            "eval_hours": best["eval_hours"],
            "coverage_percent": best["coverage_percent"],
            "folds_beaten": best["folds_beaten"],
            "comparable_folds": best["comparable_folds"],
            "runner_up": None,
            "gap_to_runner_up_wh": None,
            "seasonal_naive_MAE_wh": seasonal_mae,
            "gap_to_seasonal_naive_wh": None,
            "note": (
                f"{best['label']} has the lowest unweighted fold-mean MAE "
                f"over the scoring folds. It is scored on "
                f"{best['n_total_hours']} of {best['eval_hours']} scoring-fold "
                f"evaluation hours, so models with different coverage are not "
                f"directly interchangeable. The 2018 stress test is excluded "
                f"from the mean and this coverage count."
            ),
        }
        if runner is not None:
            block["runner_up"] = {
                "key": runner["key"],
                "label": runner["label"],
                "fold_mean_MAE_wh": runner["mean"]["MAE"],
                "n_total_hours": runner["n_total_hours"],
                "coverage_percent": runner["coverage_percent"],
            }
            block["gap_to_runner_up_wh"] = round(
                runner["mean"]["MAE"] - best["mean"]["MAE"], 4
            )
        if seasonal_mae is not None and best["key"] != "seasonal_naive_24":
            block["gap_to_seasonal_naive_wh"] = round(
                seasonal_mae - best["mean"]["MAE"], 4
            )
        return block

    def _summary_row(self, row, include_per_fold):
        result = {
            "key": row["key"],
            "label": row["label"],
            "kind": row["kind"],
            "description": row["description"],
            "mean": row["mean"],
            "std": row["std"],
            "n_folds": row["n_folds"],
            "n_total_hours": row["n_total_hours"],
            "eval_hours": row["eval_hours"],
            "coverage_percent": row["coverage_percent"],
            "folds_beaten": row["folds_beaten"],
            "comparable_folds": row["comparable_folds"],
            "beats_seasonal_naive": row["beats_seasonal_naive"],
        }
        if include_per_fold:
            result["per_fold"] = [dict(item) for item in row["per_fold"]]
        return result

    # ── public tools ──────────────────────────────────────────────

    def model_metrics(self, model=None, fold=None):
        if not self.folds:
            return {"scope": "model_metrics", "available": False,
                    "reason": "results.json contains no folds."}

        if fold:
            match = self._resolve_fold(fold)
            if match is None:
                return {
                    "scope": "model_metrics",
                    "available": False,
                    "requested_fold": fold,
                    "reason": "That fold is not in results.json.",
                    "fold_ids": sorted(self.fold_ids),
                }
            fold_id, fold_row = match
            models = []
            for name, metrics in (fold_row.get("models") or {}).items():
                if model and name != model:
                    continue
                seasonal = (fold_row.get("models") or {}).get("seasonal_naive_24") or {}
                seasonal_mae = (
                    seasonal.get("MAE") if seasonal.get("status") == "ok" else None
                )
                beats = None
                if name != "seasonal_naive_24" and metrics.get("status") == "ok" \
                        and metrics.get("MAE") is not None and seasonal_mae is not None:
                    beats = bool(metrics["MAE"] < seasonal_mae)
                models.append({
                    "key": name,
                    "label": MODEL_LABELS.get(name, name),
                    "status": metrics.get("status", "no_data"),
                    "n": metrics.get("n", 0),
                    **{metric: metrics.get(metric) for metric in SUMMARY_METRICS},
                    "beats_seasonal_naive": beats,
                })
            models.sort(key=lambda row: (
                row["MAE"] is None, row["MAE"] if row["MAE"] is not None else 0
            ))
            test = fold_row.get("test") or {}
            return {
                "scope": "fold",
                "available": True,
                "fold": {
                    "id": fold_id,
                    "label": self.fold_labels.get(fold_id),
                    "stress_test": bool(fold_row.get("stress_test")),
                    "test_start": test.get("start"),
                    "test_end": test.get("end"),
                    "eval_hours": test.get("eval_hours"),
                    "eval_days": test.get("eval_days"),
                },
                "models": models,
                "best_by_MAE": models[0] if models else None,
                "note": (
                    "Models are scored only on the hours where each produced a "
                    "forecast; missing hours are not imputed."
                ),
            }

        if model:
            row = self._model_summary(model)
            if row is None:
                return {
                    "scope": "model_metrics",
                    "available": False,
                    "requested_model": model,
                    "reason": f"Unknown model key '{model}'.",
                    "model_keys": [item["key"] for item in self._summary],
                }
            return {
                "scope": "model",
                "available": True,
                "model": self._summary_row(row, include_per_fold=True),
                "best_model_overall": self._best_block(),
                "units": "MAE, RMSE and bias in Wh; WAPE, MASE and sMAPE in percent; R2 unitless.",
            }

        return {
            "scope": "overview",
            "available": True,
            "models": [self._summary_row(row, include_per_fold=False)
                       for row in self._summary],
            "best_model": self._best_block(),
            "scoring_fold_ids": list(self.scoring_fold_ids),
            "stress_test_fold_ids": list(self.stress_fold_ids),
            "coverage_note": (
                f"Each model is scored only on its own forecast hours. "
                f"Total evaluation hours across the scoring folds: "
                f"{self.eval_hours_total}. The 2018 stress test is excluded "
                f"from the mean and from this total."
            ),
            "units": "MAE, RMSE and bias in Wh; WAPE, MASE and sMAPE in percent; R2 unitless.",
        }

    def error_breakdown(self, model=None, month=None, year=None, hour=None):
        if not self._all_records:
            return {"scope": "error_breakdown", "available": False,
                    "reason": "results.json contains no hourly backtest rows."}

        requested = MODEL_LABELS.get(model) if model else None
        if model and model not in HOURLY_FOR_MODEL:
            return {
                "scope": "error_breakdown",
                "available": False,
                "requested_model": model,
                "reason": (
                    "Hourly rows exist only for seasonal_naive_24, xgboost, "
                    "sarima and xgboost_quantile. This model has fold-level "
                    "metrics only."
                ),
                "available_models": sorted(HOURLY_FOR_MODEL),
            }

        if model:
            hourly_key = HOURLY_FOR_MODEL[model]
            meta = next(item for item in HOURLY_MODELS if item[0] == hourly_key)
            selected_meta = [meta]
        else:
            selected_meta = list(HOURLY_MODELS)

        def model_cell(records, column):
            stats = _pair_stats(records, column)
            return {
                "label": next(item[2] for item in HOURLY_MODELS if item[0] == column),
                **stats,
            }

        by_month = []
        for period in sorted(self._month_groups):
            records = self._month_groups[period]
            by_month.append({
                "period": period,
                "label": _period_label(period),
                "n_actual": sum(1 for record in records
                                if record.get("actual") is not None),
                "actual_mean_wh": _mean_actual(records),
                "models": {
                    column: model_cell(records, column)
                    for column, _, _ in selected_meta
                },
            })

        by_hour = []
        for clock_hour in range(24):
            records = self._hour_groups[clock_hour]
            by_hour.append({
                "hour": clock_hour,
                "label": f"{clock_hour:02d}:00",
                "n_actual": sum(1 for record in records
                                if record.get("actual") is not None),
                "models": {
                    column: model_cell(records, column)
                    for column, _, _ in selected_meta
                },
            })

        selected_month = None
        if month is not None:
            period = self._resolve_period(month, year)
            if period is None:
                selected_month = {
                    "requested_month": month,
                    "requested_year": year,
                    "found": False,
                    "reason": "That month is not covered by the evaluated hourly rows.",
                }
            else:
                entry = next(item for item in by_month if item["period"] == period)
                selected_month = {"requested_month": month,
                                  "requested_year": year,
                                  "found": True, **entry}

        selected_hour = None
        if hour is not None:
            selected_hour = next(
                (item for item in by_hour if item["hour"] == hour), None
            )

        result = {
            "scope": "error_breakdown",
            "available": True,
            "model": (
                {
                    "key": model,
                    "label": requested,
                    "hourly_column": HOURLY_FOR_MODEL[model],
                } if model else None
            ),
            "units": "mae_wh and bias_wh in Wh; bias = forecast minus actual.",
            "selected_month": selected_month,
            "selected_hour": selected_hour,
            "by_month": by_month,
            "by_hour": by_hour,
            "min_bucket_hours_for_worst": 24,
        }

        if model:
            hourly_key = HOURLY_FOR_MODEL[model]
            result["worst_month"] = self._worst_bucket(
                by_month, hourly_key, "period"
            )
            result["worst_hour"] = self._worst_bucket(
                by_hour, hourly_key, "hour"
            )
        else:
            result["worst_by_model"] = {
                column: {
                    "label": label,
                    "worst_month": self._worst_bucket(by_month, column, "period"),
                    "worst_hour": self._worst_bucket(by_hour, column, "hour"),
                }
                for column, _, label in HOURLY_MODELS
            }
        return result

    def day_predictions(self, date):
        if not _DATE_RE.fullmatch(str(date or "")):
            return {"scope": "day_predictions", "available": False,
                    "requested_date": date,
                    "reason": "Dates must use YYYY-MM-DD."}
        try:
            weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        except ValueError:
            return {"scope": "day_predictions", "available": False,
                    "requested_date": date, "reason": "That is not a valid date."}

        match = None
        for fold in self.folds:
            records = self._records_by_fold.get(fold.get("id")) or []
            day_records = [record for record in records
                           if str(record.get("time", "")).startswith(date)]
            if day_records:
                match = (fold, day_records)
                break

        if match is None:
            return {
                "scope": "day_predictions",
                "available": False,
                "found": False,
                "requested_date": date,
                "reason": "That date has no evaluated hourly rows in results.json.",
                "fold_windows": [
                    {
                        "id": fold.get("id"),
                        "label": _fold_label(fold.get("test")),
                        "test_start": (fold.get("test") or {}).get("start"),
                        "test_end": (fold.get("test") or {}).get("end"),
                        "eval_hours": (fold.get("test") or {}).get("eval_hours"),
                    }
                    for fold in self.folds
                ],
            }

        fold, day_records = match
        by_hour = {}
        for record in day_records:
            timestamp = str(record.get("time", ""))
            try:
                clock_hour = int(timestamp[11:13])
            except ValueError:
                continue
            by_hour[clock_hour] = record

        hours = []
        for clock_hour in range(24):
            record = by_hour.get(clock_hour)
            if record is None:
                hours.append({"hour": clock_hour, "label": f"{clock_hour:02d}:00"})
                continue
            hours.append({
                "hour": clock_hour,
                "label": f"{clock_hour:02d}:00",
                "actual": _round(record.get("actual"), 2),
                "seasonal_naive": _round(record.get("seasonal_naive"), 2),
                "xgboost": _round(record.get("xgboost"), 2),
                "sarima": _round(record.get("sarima"), 2),
                "p10": _round(record.get("p10"), 2),
                "p50": _round(record.get("p50"), 2),
                "p90": _round(record.get("p90"), 2),
            })

        actuals = [record["actual"] for record in day_records
                   if record.get("actual") is not None]
        return {
            "scope": "day_predictions",
            "available": True,
            "found": True,
            "date": date,
            "weekday": weekday,
            "fold": {
                "id": fold.get("id"),
                "label": _fold_label(fold.get("test")),
            },
            "n_eval_hours": len(day_records),
            "n_actual": len(actuals),
            "actual_peak_wh": round(max(actuals), 2) if actuals else None,
            "actual_mean_wh": _mean_actual(day_records),
            "model_errors": {
                column: {
                    "label": label,
                    **_pair_stats(day_records, column),
                }
                for column, _, label in HOURLY_MODELS
            },
            "hours": hours,
            "units": "Wh; forecasts are the evaluated day-ahead values.",
        }

    def data_quality(self, month=None, year=None):
        quality = self.quality
        if not quality:
            return {"scope": "data_quality", "available": False,
                    "reason": "data_quality.json is not loaded."}
        source = quality.get("source") or {}
        panel = quality.get("panel") or {}
        stable = quality.get("stable_panel") or {}
        target = quality.get("target") or {}
        inactive = quality.get("inactive") or {}
        monthly = quality.get("monthly_coverage") or []
        selected = None
        if month is not None:
            period = self._resolve_period(month, year, rows=monthly)
            selected = next(
                (dict(row) for row in monthly if row.get("month") == period),
                {
                    "requested_month": month,
                    "requested_year": year,
                    "found": False,
                },
            )
            if "found" not in selected:
                selected["found"] = True
        weather = self.results.get("weather") or {}
        longest_runs = target.get("longest_runs") or []
        return {
            "scope": "data_quality",
            "available": True,
            "source": {
                "files_found": source.get("files_found"),
                "files_read": source.get("files_read"),
                "files_failed": source.get("files_failed"),
                "rows_total": source.get("rows_total"),
                "first_ist": source.get("first_ist"),
                "last_ist": source.get("last_ist"),
            },
            "panel": {
                "hours": panel.get("hours"),
                "apartments": panel.get("apartments"),
                "hours_zero_reporting": panel.get("hours_zero_reporting"),
                "reporting_median": panel.get("reporting_median"),
                "reporting_max": panel.get("reporting_max"),
            },
            "stable_panel": {
                "count": stable.get("count"),
                "min_coverage_percent": _round(
                    float(stable.get("min_coverage") or 0) * 100, 2
                ) if stable.get("min_coverage") is not None else None,
                "window_start": stable.get("window_start"),
                "window_end": stable.get("window_end"),
                "apartment_ids": [
                    str(item.get("apartment"))
                    for item in (stable.get("apartments") or [])
                ],
            },
            "target": {
                "valid_hours_before_gaps": target.get("valid_hours_before_gaps"),
                "valid_hours": target.get("valid_hours"),
                "imputed_hours": target.get("imputed_hours"),
                "excluded_hours": target.get("excluded_hours"),
                "excluded_runs": target.get("excluded_runs"),
                "max_gap_hours": target.get("max_gap_hours"),
                "longest_gap_hours": target.get("longest_gap_hours"),
                "mean_wh": target.get("mean_wh"),
                "longest_runs": longest_runs[:3],
            },
            "monthly_coverage": monthly,
            "selected_month": selected,
            "inactive": {
                "month": inactive.get("month"),
                "threshold_wh": inactive.get("threshold_wh"),
                "min_hours": inactive.get("min_hours"),
                "flagged_count": len(inactive.get("flagged") or []),
                "flagged": inactive.get("flagged") or [],
                "flagged_stable_count": len(inactive.get("flagged_stable") or []),
                "flagged_stable": inactive.get("flagged_stable") or [],
            },
            "weather": {
                "available": weather.get("available"),
                "hours": weather.get("hours"),
                "latitude": weather.get("latitude"),
                "longitude": weather.get("longitude"),
                "variables": len(weather.get("variables") or []),
                "missing_values_total": sum(
                    int(value or 0)
                    for value in (weather.get("missing_values") or {}).values()
                ),
                "information_set": weather.get("information_set"),
            },
            "notes": quality.get("notes") or [],
        }

    def limitations(self):
        if not self.folds:
            return {"scope": "limitations", "available": False,
                    "reason": "results.json contains no folds."}
        quality = self.quality
        target = quality.get("target") or {}
        inactive = quality.get("inactive") or {}
        stable = quality.get("stable_panel") or {}
        monthly = quality.get("monthly_coverage") or []
        weather = self.results.get("weather") or {}
        protocol = self.results.get("protocol") or {}

        fold_rows = []
        for fold in self.folds:
            test = fold.get("test") or {}
            sarima = (fold.get("models") or {}).get("sarima") or {}
            interval = (fold.get("intervals") or {}).get("xgboost_quantile") or {}
            conformal = (fold.get("intervals") or {}).get("conformal") or {}
            coverage = interval.get("coverage")
            after_all = (conformal.get("after_all_days") or {}).get("coverage")
            after_calibrated = (
                conformal.get("after_calibrated_days") or {}
            ).get("coverage")
            fold_rows.append({
                "id": fold.get("id"),
                "label": _fold_label(test),
                "test_start": test.get("start"),
                "test_end": test.get("end"),
                "eval_hours": test.get("eval_hours"),
                "eval_days": test.get("eval_days"),
                "stress_test": bool(fold.get("stress_test")),
                "sarima_hours": sarima.get("n") if sarima.get("status") == "ok" else 0,
                "interval_n": interval.get("n"),
                "interval_coverage_percent": _round(
                    float(coverage) * 100, 2
                ) if coverage is not None else None,
                "interval_after_percent": _round(
                    float(after_all) * 100, 2
                ) if after_all is not None else None,
                "interval_after_calibrated_percent": _round(
                    float(after_calibrated) * 100, 2
                ) if after_calibrated is not None else None,
                "calibrated_days": conformal.get("n_days_calibrated"),
                "warmup_days": conformal.get("warmup_days"),
                "conformal_window_days": conformal.get("window_days"),
            })
        total_eval = sum(int(row["eval_hours"] or 0) for row in fold_rows)
        total_sarima = sum(int(row["sarima_hours"] or 0) for row in fold_rows)
        coverage_rows = [row for row in fold_rows
                         if row["interval_coverage_percent"] is not None]
        worst_coverage = min(
            coverage_rows,
            key=lambda row: row["interval_coverage_percent"],
        ) if coverage_rows else None
        after_rows = [
            row for row in fold_rows if row["interval_after_percent"] is not None
        ]

        def _weighted_interval(rows, key):
            pairs = [
                (row[key], row["interval_n"]) for row in rows
                if row.get(key) is not None and row.get("interval_n")
            ]
            total = sum(n for _, n in pairs)
            if not total:
                return None
            return _round(sum(value * n for value, n in pairs) / total, 2)

        scoring_rows = [row for row in fold_rows if not row["stress_test"]]
        weighted_before = _weighted_interval(
            scoring_rows, "interval_coverage_percent"
        )
        weighted_after = _weighted_interval(
            scoring_rows, "interval_after_percent"
        )
        conformal_window = next(
            (row["conformal_window_days"] for row in fold_rows
             if row["conformal_window_days"] is not None),
            None,
        )
        has_conformal = bool(after_rows)
        thin_2018 = next(
            (row for row in fold_rows
             if str(row["test_start"] or "").startswith("2018")),
            None,
        )
        longest_gap = target.get("longest_gap_hours")
        december_coverage = next(
            (row for row in monthly if row.get("month") == "2017-12"), None
        )
        december_fold = next(
            (fold for fold in self.folds if fold.get("id") == "2017-12"), None
        )
        december_models = (december_fold or {}).get("models") or {}
        december_xgb = december_models.get("xgboost") or {}
        december_common = (december_fold or {}).get("common_hours") or {}
        december_best = december_common.get("best_baseline")
        december_best_metrics = (
            (december_common.get("models") or {}).get(december_best) or {}
        )
        return {
            "scope": "limitations",
            "available": True,
            "folds": fold_rows,
            "totals": {
                "eval_hours": total_eval,
                "eval_days": sum(int(row["eval_days"] or 0) for row in fold_rows),
                "sarima_hours": total_sarima,
            },
            "sarima_coverage": {
                "by_fold_hours": [
                    {"id": row["id"], "label": row["label"],
                     "sarima_hours": row["sarima_hours"],
                     "eval_hours": row["eval_hours"]}
                    for row in fold_rows
                ],
                "total_sarima_hours": total_sarima,
                "total_eval_hours": total_eval,
                "note": "Missing SARIMA hours are not imputed or backfilled.",
            },
            "prediction_intervals": {
                "nominal_percent": 80.0,
                "by_fold": [
                    {"id": row["id"], "label": row["label"],
                     "n": row["interval_n"],
                     "coverage_percent": row["interval_coverage_percent"]}
                    for row in coverage_rows
                ],
                "worst_fold": (
                    {
                        "id": worst_coverage["id"],
                        "label": worst_coverage["label"],
                        "coverage_percent": worst_coverage["interval_coverage_percent"],
                    } if worst_coverage else None
                ),
                "conformal": (
                    {
                        "method": (
                            "rolling conformalized quantile regression on P10-P90 "
                            "using only prior test days in the same fold"
                        ),
                        "window_days": conformal_window,
                        "by_fold": [
                            {
                                "id": row["id"],
                                "label": row["label"],
                                "before_percent": row["interval_coverage_percent"],
                                "after_percent": row["interval_after_percent"],
                                "after_calibrated_percent":
                                    row["interval_after_calibrated_percent"],
                                "calibrated_days": row["calibrated_days"],
                                "warmup_days": row["warmup_days"],
                            }
                            for row in after_rows
                        ],
                        "scoring_hour_weighted": {
                            "before_percent": weighted_before,
                            "after_percent": weighted_after,
                        },
                        "note": (
                            "Calibration expands intervals only, never narrows them; "
                            "the first 7 days of each fold keep the raw bounds."
                        ),
                    } if has_conformal else None
                ),
                "note": "Intervals are calibration diagnostics, not guaranteed bounds.",
            },
            "thin_2018_fold": (
                {
                    "id": thin_2018["id"],
                    "label": thin_2018["label"],
                    "eval_hours": thin_2018["eval_hours"],
                    "eval_days": thin_2018["eval_days"],
                    "sarima_hours": thin_2018["sarima_hours"],
                } if thin_2018 else None
            ),
            "target_gaps": {
                "valid_hours": target.get("valid_hours"),
                "imputed_hours": target.get("imputed_hours"),
                "excluded_hours": target.get("excluded_hours"),
                "excluded_runs": target.get("excluded_runs"),
                "max_gap_hours": target.get("max_gap_hours"),
                "longest_gap_hours": longest_gap,
                "longest_run": (target.get("longest_runs") or [None])[0],
            },
            "december_inactivity": {
                "month": inactive.get("month") or "2017-12",
                "flagged_count": len(inactive.get("flagged") or []),
                "flagged_stable_count": len(inactive.get("flagged_stable") or []),
                "stable_panel_count": stable.get("count"),
                "threshold_wh": inactive.get("threshold_wh"),
                "min_hours": inactive.get("min_hours"),
                "clock_hours": december_coverage.get("hours") if december_coverage else None,
                "target_hours": (december_coverage or {}).get("target_hours"),
                "summary": (
                    f"{len(inactive.get('flagged') or [])} apartments were flagged "
                    f"suspect inactive, including "
                    f"{len(inactive.get('flagged_stable') or [])} of "
                    f"{stable.get('count')} stable-panel apartments. "
                    + (
                        f"An apartment is flagged when its mean load is below "
                        f"{inactive.get('threshold_wh')} Wh/h for at least "
                        f"{inactive.get('min_hours')} hours. "
                        if inactive.get("threshold_wh") is not None
                        and inactive.get("min_hours") is not None else ""
                    )
                    + "The threshold is a mean-load rule, not a weather "
                    "threshold; flags are descriptive, not confirmed vacancy."
                ),
                "note": (
                    "Flags are descriptive and those apartments stay in the target; "
                    "low December readings are consistent with vacancy but are not "
                    "confirmed meter failures."
                ),
            },
            "december_context": (
                {
                    "model": "xgboost",
                    "mae_wh": december_xgb.get("MAE"),
                    "bias_wh": december_xgb.get("bias"),
                    "n": december_xgb.get("n"),
                    "common_hours_best_baseline": december_best,
                    "common_hours_best_mae_wh": december_best_metrics.get("MAE"),
                    "summary": (
                        "In Dec 2017, XGBoost has MAE "
                        f"{december_xgb.get('MAE')} Wh and bias "
                        f"{december_xgb.get('bias')} Wh on {december_xgb.get('n')} hours; "
                        f"the common-hours best baseline ({december_best}) has MAE "
                        f"{december_best_metrics.get('MAE')} Wh."
                    ),
                } if december_xgb.get("MAE") is not None else None
            ),
            "weather": {
                "available": weather.get("available"),
                "hours": weather.get("hours"),
                "latitude": weather.get("latitude"),
                "longitude": weather.get("longitude"),
                "variables": len(weather.get("variables") or []),
                "information_set": weather.get("information_set"),
            },
            "protocol_note": protocol.get("information_set"),
            "single_site": {
                "stable_apartments": stable.get("count"),
                "apartments": (quality.get("panel") or {}).get("apartments"),
            },
            "caveats": [
                f"This is a retrospective {len(self.scoring_folds)}-fold backtest "
                "plus a 2018 stress test, not a live deployment.",
                "No model estimates causal effects of policy, tariffs, or appliance changes.",
                "The target is one IIT Bombay residential site; results are not evidence for other sites.",
                "Models are scored only on hours with a forecast; missing hours are not imputed.",
                "Fold-mean differences smaller than the fold-to-fold spread are not a stable ranking.",
                "December inactivity is a descriptive flag, not a confirmed vacancy cause.",
            ],
        }

    def protocol(self):
        protocol = self.results.get("protocol") or {}
        weather = self.results.get("weather") or {}
        return {
            "scope": "protocol",
            "available": bool(protocol),
            "generated_at": self.results.get("generated_at"),
            "timezone": protocol.get("timezone"),
            "target": protocol.get("target"),
            "forecast": protocol.get("forecast"),
            "origin": protocol.get("origin"),
            "information_set": protocol.get("information_set"),
            "evaluation_mask": protocol.get("evaluation_mask"),
            "fold_design": protocol.get("folds"),
            "mase_scale": protocol.get("mase_scale"),
            "best_baseline_selection": protocol.get("best_baseline_selection"),
            "significance": protocol.get("significance"),
            "learned_models": protocol.get("learned_models"),
            "sarima": protocol.get("sarima"),
            "quantiles": protocol.get("quantiles"),
            "weather": {
                "available": weather.get("available"),
                "hours": weather.get("hours"),
                "information_set": weather.get("information_set"),
            },
            "sources": list(SOURCES),
        }

    # ── planner ──────────────────────────────────────────────────

    def plan(self, question):
        text = (question or "").lower()
        if not text.strip():
            return []

        date = _first_date(text)
        months = _find_months(text)
        month = months[0] if len(months) == 1 else None
        year = _find_year(text) if month is not None else None
        hour = _find_hour(text)
        models = _find_models(text)
        model = models[0] if len(models) == 1 else None

        quality_intent = _contains_any(text, _QUALITY_TERMS)
        limit_intent = _contains_any(text, _LIMIT_TERMS)
        metric_intent = _contains_any(text, _METRIC_TERMS)
        protocol_intent = _matches_any(text, _PROTOCOL_PATTERNS)
        error_intent = (
            month is not None
            or hour is not None
            or _contains_any(text, _ERROR_TERMS)
            or "hour" in text
        )

        calls = []
        if date:
            calls.append(ToolCall("day_predictions", {"date": date}))
        if error_intent:
            calls.append(ToolCall("error_breakdown", {
                "model": model, "month": month, "year": year, "hour": hour,
            }))
        if quality_intent:
            calls.append(ToolCall("data_quality", {"month": month, "year": year}))
        if limit_intent:
            calls.append(ToolCall("limitations", {}))
        if metric_intent:
            calls.append(ToolCall("model_metrics", {"model": model}))
        if protocol_intent:
            calls.append(ToolCall("protocol", {}))
        if not calls and models:
            calls.append(ToolCall("model_metrics", {"model": model}))

        deduped = []
        seen = set()
        for call in calls:
            signature = (call.name, tuple(sorted(call.arguments.items())))
            if signature not in seen:
                seen.add(signature)
                deduped.append(call)
        return deduped

    def call(self, tool_call):
        methods = {
            "model_metrics": self.model_metrics,
            "error_breakdown": self.error_breakdown,
            "day_predictions": self.day_predictions,
            "data_quality": self.data_quality,
            "limitations": self.limitations,
            "protocol": self.protocol,
        }
        method = methods.get(tool_call.name)
        if method is None:
            raise ValueError(f"Unknown tool '{tool_call.name}'.")
        return method(**tool_call.arguments)

    # ── internal selectors ───────────────────────────────────────

    def _resolve_fold(self, value):
        key = str(value or "").strip()
        if key in self.fold_ids:
            return key, next(fold for fold in self.folds if fold.get("id") == key)
        lowered = key.lower()
        for fold in self.folds:
            if _fold_label(fold.get("test")).lower() == lowered:
                return fold.get("id"), fold
        return None

    def _resolve_period(self, month, year=None, rows=None):
        try:
            month_number = int(month)
        except (TypeError, ValueError):
            return None
        if not 1 <= month_number <= 12:
            return None
        available = (
            [row.get("month") for row in rows]
            if rows is not None else sorted(self._month_groups)
        )
        if year is not None:
            candidate = f"{int(year):04d}-{month_number:02d}"
            return candidate if candidate in available else None
        matches = [item for item in available
                   if str(item).endswith(f"-{month_number:02d}")]
        return matches[-1] if matches else None

    @staticmethod
    def _worst_bucket(buckets, column, key):
        candidates = []
        for bucket in buckets:
            cell = (bucket.get("models") or {}).get(column) or {}
            if cell.get("mae_wh") is None or not cell.get("n"):
                continue
            candidates.append((bucket, cell))
        if not candidates:
            return None
        robust = [item for item in candidates if item[1]["n"] >= 24]
        chosen_bucket, chosen_cell = max(
            robust or candidates,
            key=lambda item: item[1]["mae_wh"],
        )
        return {
            key: chosen_bucket.get(key),
            "label": chosen_bucket.get("label"),
            "mae_wh": chosen_cell.get("mae_wh"),
            "bias_wh": chosen_cell.get("bias_wh"),
            "n": chosen_cell.get("n"),
        }


def _first_date(text):
    match = _DATE_RE.search(text)
    return match.group(1) if match else None


def _find_year(text):
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def _find_months(text):
    found = []
    for match in _MONTH_RE.finditer(text):
        month = _MONTH_LOOKUP[match.group(1).lower()]
        if month not in found:
            found.append(month)
    return found


def _find_hour(text):
    match = _CLOCK_RE.search(text)
    if match:
        return int(match.group(0).split(":")[0])
    match = _AMPM_RE.search(text)
    if match:
        hour = int(match.group(1)) % 12
        if match.group(2).lower() == "pm":
            hour += 12
        return hour
    match = _AT_HOUR_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _find_models(text):
    found = []
    for alias, key in _ALIASES:
        if re.search(r"\b" + re.escape(alias) + r"\b", text) and key not in found:
            found.append(key)
    return found
