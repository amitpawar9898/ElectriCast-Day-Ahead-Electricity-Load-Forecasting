"""Level-normalized target experiment for Phase 9.4.

Run from the repository root:

    python backend/scripts/level_experiment.py

Compares the 9.3 winner (raw target, no annual calendar) with:
  * the same features on target / trailing 7-day mean, multiplied back; and
  * that normalized model plus a lag-24 near-zero stable-apartment share.

The scale at each day is the mean observed target over the 168 h ending one
hour before that day's 00:00 IST origin, so it is known at forecast time. The
un-filled target is used for the scale so interpolation cannot cross the origin.
The 2018 stress test is reported but excluded from the keep decision.

Decision rule, matching 9.3: a normalized variant is adopted only if it has a
lower common-hours MAE than the 9.3 winner on at least 3 of the 5 scoring folds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import data_pipeline as dp  # noqa: E402
import evaluate as ev  # noqa: E402
import forecast_model as fm  # noqa: E402
import level_features as lf  # noqa: E402
import weather  # noqa: E402

OUTPUT_PATH = BACKEND / "level_results.json"
WINNER_FEATURES = {"drop_annual_calendar": True}
HELP_FOLDS_REQUIRED = 3

VARIANTS = {
    "raw_9_3_winner": {
        "normalized": False,
        "near_zero": False,
    },
    "normalized": {
        "normalized": True,
        "near_zero": False,
    },
    "normalized_near_zero": {
        "normalized": True,
        "near_zero": True,
    },
}


def _metrics(actual, pred, scale):
    metrics = ev.compute_metrics(actual, pred, scale)
    return {
        "n": metrics["n"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "bias": metrics["bias"],
    }


def main():
    apartments, _ = dp.load_apartments()
    panel, _ = dp.build_panel(apartments)
    stable, _ = dp.stable_apartments(panel)
    raw_target, reporting = dp.build_target(panel, stable)
    series, _, _, _ = dp.fill_short_gaps(raw_target)
    weather_frame = weather.load()
    near_zero = lf.near_zero_fraction(panel, stable)
    train_start = ev.history_start(series)

    fold_rows = []
    for fold in ev.FOLDS:
        fold = {**fold, "train_start": train_start}
        train = series.loc[fold["train_start"]:fold["train_end"]]
        scale = ev.mase_scale(train)
        actual = series.reindex(ev.test_hours(fold["test_start"], fold["test_end"]))
        baseline = ev.baseline_forecasts(
            series, fold["test_start"], fold["test_end"]
        )["seasonal_naive_24"]
        mask = actual.notna() & baseline.notna()

        train_hours = ev.test_hours(fold["train_start"], fold["train_end"])
        test_hours = ev.test_hours(fold["test_start"], fold["test_end"])
        all_hours = train_hours.append(test_hours)
        target_scale = lf.trailing_scale(raw_target, all_hours)
        test_scale = target_scale.reindex(test_hours)
        scale_days = test_scale.groupby(test_scale.index.floor("D")).first()
        scale_day_count = int(scale_days.notna().sum())
        test_day_count = int(len(scale_days))
        less_than_year = (
            pd.Timestamp(fold["train_end"]) - pd.Timestamp(fold["train_start"])
        ) < pd.Timedelta(days=365)
        feature_options = dict(WINNER_FEATURES) if less_than_year else {}

        predictions = {}
        metadata = {}
        for key, config in VARIANTS.items():
            pred, meta = fm.forecast(
                series,
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
                reporting=reporting,
                weather=weather_frame,
                tune_model=False,
                feature_options=feature_options,
                target_scale=target_scale if config["normalized"] else None,
                near_zero_fraction=near_zero if config["near_zero"] else None,
            )
            predictions[key] = pred
            metadata[key] = meta

        reference = predictions["raw_9_3_winner"]
        variant_rows = {}
        for key, pred in predictions.items():
            common = mask & reference.notna() & pred.notna()
            row = _metrics(actual[common], pred[common], scale)
            reference_common = _metrics(
                actual[common], reference[common], scale
            )
            row["common_hours_with_reference"] = int(common.sum())
            row["reference_MAE_on_pair_common_wh"] = reference_common["MAE"]
            row["delta_MAE_vs_winner_wh"] = (
                None if key == "raw_9_3_winner"
                else round(float(row["MAE"] - reference_common["MAE"]), 4)
            )
            row["normalized"] = VARIANTS[key]["normalized"]
            row["near_zero"] = VARIANTS[key]["near_zero"]
            row["n_train"] = metadata[key]["n_train"]
            if key != "raw_9_3_winner":
                row["normalization"] = metadata[key].get("normalization")
            variant_rows[key] = row

        fold_rows.append({
            "id": fold["id"],
            "label": fold.get("label"),
            "stress_test": bool(fold.get("stress_test")),
            "eval_hours": int(mask.sum()),
            "drop_annual_calendar": less_than_year,
            "scale_days": scale_day_count,
            "test_days": test_day_count,
            "scale_min_wh": None if scale_days.isna().all() else round(
                float(scale_days.min()), 4
            ),
            "scale_max_wh": None if scale_days.isna().all() else round(
                float(scale_days.max()), 4
            ),
            "variants": variant_rows,
        })

    scoring = [row for row in fold_rows if not row["stress_test"]]
    summary_variants = {}
    for key in VARIANTS:
        if key == "raw_9_3_winner":
            continue
        helped = [
            row for row in scoring
            if row["variants"][key]["delta_MAE_vs_winner_wh"] < 0
        ]
        deltas = {
            row["id"]: row["variants"][key]["delta_MAE_vs_winner_wh"]
            for row in scoring
        }
        mae_values = [row["variants"][key]["MAE"] for row in scoring]
        summary_variants[key] = {
            "folds_helped": len(helped),
            "helped_fold_ids": [row["id"] for row in helped],
            "mean_MAE_wh": round(float(np.mean(mae_values)), 4),
            "mean_delta_MAE_vs_winner_wh": round(
                float(np.mean(list(deltas.values()))), 4
            ),
            "delta_MAE_vs_winner_wh": deltas,
            "adopted": len(helped) >= HELP_FOLDS_REQUIRED,
        }

    normalized = summary_variants["normalized"]
    with_near_zero = summary_variants["normalized_near_zero"]
    candidates = []
    if normalized["adopted"]:
        candidates.append("normalized")
    if with_near_zero["adopted"]:
        candidates.append("normalized_near_zero")
    if candidates:
        winner = min(candidates, key=lambda key: summary_variants[key]["mean_MAE_wh"])
    else:
        winner = "raw_9_3_winner"

    results = {
        "generated_at": pd.Timestamp.now(tz=dp.TZ).isoformat(timespec="seconds"),
        "protocol": {
            "folds": ev.PROTOCOL["folds"],
            "evaluation_mask": ev.PROTOCOL["evaluation_mask"],
            "config": "untuned forecast_model.DEFAULT_PARAMS, identical across variants",
            "feature_options": (
                "drop month and day-of-year only when the training window is "
                "shorter than 365 days (the 9.3 rule); the 2018 stress test "
                "therefore keeps the full calendar"
            ),
            "target_scale": (
                "mean of the un-filled target over the 168 h ending 1 h before "
                "each day's 00:00 IST origin; constant within the day; days with "
                "fewer than 24 observed hours get no scale and no forecast"
            ),
            "near_zero_fraction": (
                "share of reporting stable-panel apartments below 100 Wh in an "
                "hour, used at a 24 h lag"
            ),
            "comparison_mask": (
                "evaluation mask intersected with both models' forecasts so MAE "
                "differences are not caused by different coverage"
            ),
            "decision_rule": (
                "adopt a normalized variant only if it lowers common-hours MAE "
                f"on at least {HELP_FOLDS_REQUIRED} of {len(scoring)} scoring folds"
            ),
        },
        "config": {
            key: value for key, value in fm.DEFAULT_PARAMS.items()
            if key != "n_jobs"
        },
        "folds": fold_rows,
        "summary": {
            "scoring_fold_ids": [row["id"] for row in scoring],
            "stress_test_fold_ids": [
                row["id"] for row in fold_rows if row["stress_test"]
            ],
            "variants": summary_variants,
            "winner": winner,
            "winner_config": {
                **VARIANTS[winner],
                "feature_options": "drop_annual_calendar when training <365 days",
            },
            "near_zero_note": (
                "The 100 Wh threshold is the existing monthly inactivity "
                "threshold reused as an hourly proxy; the fraction is a feature, "
                "not evidence that an apartment is vacant."
            ),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH}")

    print(f"\nLevel-normalized target, {len(fold_rows)} folds, untuned fixed config")
    print(f"{'fold':<10} {'eval h':>7} {'scale d':>8} "
          f"{'9.3 win':>9} {'normal':>9} {'+near0':>9}")
    for row in fold_rows:
        v = row["variants"]
        print(f"{row['id']:<10} {row['eval_hours']:>7} "
              f"{row['scale_days']:>3}/{row['test_days']:<4} "
              f"{v['raw_9_3_winner']['MAE']:>9.2f} "
              f"{v['normalized']['MAE']:>9.2f} "
              f"{v['normalized_near_zero']['MAE']:>9.2f}"
              f"{'  stress' if row['stress_test'] else ''}")

    print("\nScoring-fold summary on common hours (negative delta = better)")
    for key, stats in summary_variants.items():
        print(f"{key:<25} helped {stats['folds_helped']}/{len(scoring)} scoring folds, "
              f"mean MAE {stats['mean_MAE_wh']:.2f} Wh, "
              f"mean delta {stats['mean_delta_MAE_vs_winner_wh']:+.2f} Wh, "
              f"adopted={stats['adopted']}")
    print(f"\nwinner: {winner} {VARIANTS[winner]}")


if __name__ == "__main__":
    main()
