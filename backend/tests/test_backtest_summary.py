"""Fold-mean tests for the API backtest summary."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402


def _metrics(mae, status="ok", n=24):
    return {
        "status": status,
        "n": n if status == "ok" else 0,
        "MAE": mae,
        "RMSE": mae + 1,
        "WAPE": 10.0,
        "MASE": 1.0,
        "R2": 0.5,
        "sMAPE": 9.0,
        "bias": mae,
    }


def _fold(fold_id, start, end, mae, stress=False, label=None):
    models = {
        "seasonal_naive_24": _metrics(mae + 1),
        "xgboost": _metrics(mae),
    }
    return {
        "id": fold_id,
        "label": label,
        "stress_test": stress,
        "test": {
            "start": start, "end": end, "eval_hours": 24, "eval_days": 1,
        },
        "models": models,
        "common_hours": {
            "n": 12,
            "eval_hours": 24,
            "coverage": 0.5,
            "best_baseline": "seasonal_naive_24",
            "missing_models": [],
            "models": {
                "seasonal_naive_24": _metrics(mae + 2, n=12),
                "xgboost": _metrics(mae + 1, n=12),
            },
            "paired_vs_best": {
                "seasonal_naive_24": {"status": "reference"},
                "xgboost": {
                    "status": "ok",
                    "n_days": 12,
                    "n_pairs": 288,
                    "median_daily_mae_diff_wh": -1.0,
                    "ci_low_wh": -2.0,
                    "ci_high_wh": 0.1,
                    "p_value": 0.04,
                    "significant_at_0.05": True,
                    "small_sample": False,
                },
            },
        },
    }


def test_summary_means_exclude_stress_fold():
    results = {
        "folds": [
            _fold("2017-07", "2017-07-01 00:00", "2017-07-01 23:00", 10.0),
            _fold("2017-08", "2017-08-01 00:00", "2017-08-01 23:00", 20.0),
            _fold(
                "2018-05_06", "2018-05-01 00:00", "2018-05-01 23:00", 500.0,
                stress=True, label="Stress test (May\u2013Jun 2018)",
            ),
        ],
    }
    summary = main._build_backtest_summary(results)
    xgb = next(row for row in summary["models"] if row["key"] == "xgboost")

    assert xgb["mean"]["MAE"] == 15.0
    assert xgb["n_folds"] == 2
    assert xgb["folds_beaten"] == 2
    assert xgb["comparable_folds"] == 2
    assert summary["scoring_fold_ids"] == ["2017-07", "2017-08"]
    assert summary["stress_test_fold_ids"] == ["2018-05_06"]
    assert summary["fold_order"] == ["2017-07", "2017-08", "2018-05_06"]
    assert summary["folds"][-1]["stress_test"] is True
    assert summary["folds"][-1]["label"] == "Stress test (May\u2013Jun 2018)"
    assert xgb["per_fold"][-1]["stress_test"] is True


def test_summary_exposes_common_hours():
    results = {
        "folds": [
            _fold("2017-07", "2017-07-01 00:00", "2017-07-01 23:00", 10.0),
            _fold(
                "2018-05_06", "2018-05-01 00:00", "2018-05-01 23:00", 500.0,
                stress=True, label="Stress test (May\u2013Jun 2018)",
            ),
        ],
    }
    summary = main._build_backtest_summary(results)
    xgb = next(row for row in summary["models"] if row["key"] == "xgboost")

    first_fold = summary["folds"][0]
    assert first_fold["common_hours"] == {
        "n": 12,
        "eval_hours": 24,
        "coverage": 0.5,
        "best_baseline": "seasonal_naive_24",
        "missing_models": [],
    }
    assert xgb["per_fold"][0]["common"]["n"] == 12
    assert xgb["per_fold"][0]["common"]["MAE"] == 11.0
    assert xgb["per_fold"][0]["daily_vs_best"]["n_days"] == 12
    assert xgb["per_fold"][0]["daily_vs_best"]["ci_low_wh"] == -2.0
    assert xgb["per_fold"][1]["common"]["status"] == "ok"
    assert "common-hours" in summary["common_hours_rule"]


def test_summary_common_hours_default_to_no_data():
    fold = _fold("2017-07", "2017-07-01 00:00", "2017-07-01 23:00", 10.0)
    fold.pop("common_hours")
    summary = main._build_backtest_summary({"folds": [fold]})
    xgb = next(row for row in summary["models"] if row["key"] == "xgboost")

    assert summary["folds"][0]["common_hours"]["n"] == 0
    assert xgb["per_fold"][0]["common"]["status"] == "no_data"
    assert xgb["per_fold"][0]["common"]["MAE"] is None
    assert xgb["per_fold"][0]["daily_vs_best"]["status"] == "no_data"
