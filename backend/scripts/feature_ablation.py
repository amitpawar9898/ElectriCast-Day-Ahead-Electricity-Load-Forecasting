"""Feature ablation for the day-ahead XGBoost model on the Phase 9 folds.

Run from the repository root:

    python backend/scripts/feature_ablation.py

Every variant uses the same untuned DEFAULT_PARAMS on the same folds, so the
MAE differences isolate the feature change instead of re-tuning noise. The
2018 stress test is reported for context but excluded from the keep decision.

The oracle variant adds temperature at the target hour. That value is not
known at the 00:00 IST origin, so it is diagnostic only: it is labelled ORACLE
and is never eligible as the kept variant.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import evaluate as ev  # noqa: E402
import forecast_model as fm  # noqa: E402
import weather  # noqa: E402

OUTPUT_PATH = BACKEND / "ablation_results.json"

VARIANTS = {
    "full": {},
    "no_annual_calendar": {"drop_annual_calendar": True},
    "no_weather": {"use_weather": False},
    "oracle_same_hour_temperature": {"oracle_temperature": True},
}
ORACLE_VARIANTS = {"oracle_same_hour_temperature"}
HELP_FOLDS_REQUIRED = 3


def _less_than_12_months(train_start, train_end):
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    # A train window ending on day D covers D-start+1 days; use the full span.
    return (end - start) < pd.Timedelta(days=365)


def _variant_available(key, less_than_year, weather_available):
    if key == "no_annual_calendar" and not less_than_year:
        return False, "training window spans at least 12 months; full calendar retained"
    if key == "no_weather" and not weather_available:
        return False, "weather cache unavailable; full model has no weather either"
    if key in ORACLE_VARIANTS and not weather_available:
        return False, "weather cache unavailable; oracle temperature cannot be built"
    return True, None


def _summary_row(metrics):
    return {
        "n": metrics["n"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "bias": metrics["bias"],
    }


def main():
    series, reporting = ev.load_target_with_reporting()
    weather_frame = weather.load()
    weather_available = weather_frame is not None
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
        less_than_year = _less_than_12_months(fold["train_start"], fold["train_end"])

        variant_rows = {}
        for key, feature_options in VARIANTS.items():
            available, note = _variant_available(
                key, less_than_year, weather_available
            )
            if not available:
                variant_rows[key] = {
                    "applied": False,
                    "oracle": key in ORACLE_VARIANTS,
                    "note": note,
                    **{field: variant_rows["full"][field]
                       for field in ("n", "MAE", "RMSE", "bias")},
                }
                continue
            pred, metadata = fm.forecast(
                series,
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
                reporting=reporting,
                weather=weather_frame,
                tune_model=False,
                feature_options=feature_options,
            )
            metrics = ev.compute_metrics(actual[mask], pred[mask], scale)
            variant_rows[key] = {
                "applied": True,
                "oracle": key in ORACLE_VARIANTS,
                "n": metrics["n"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "bias": metrics["bias"],
                "n_train": metadata["n_train"],
                "n_features": len(metadata["features"]),
            }

        for key, row in variant_rows.items():
            if row["applied"] and row["MAE"] is not None:
                row["delta_MAE_vs_full_wh"] = round(
                    float(row["MAE"] - variant_rows["full"]["MAE"]), 4
                )
            else:
                row["delta_MAE_vs_full_wh"] = None

        fold_rows.append({
            "id": fold["id"],
            "label": fold.get("label"),
            "stress_test": bool(fold.get("stress_test")),
            "train": {
                "start": fold["train_start"],
                "end": fold["train_end"],
                "valid_hours": int(train.notna().sum()),
                "less_than_12_months": less_than_year,
            },
            "eval_hours": int(mask.sum()),
            "variants": variant_rows,
        })

    scoring = [row for row in fold_rows if not row["stress_test"]]
    summary_variants = {}
    for key in VARIANTS:
        if key == "full":
            continue
        applied_folds = [
            row for row in scoring if row["variants"][key]["applied"]
        ]
        helped = [
            row for row in applied_folds
            if row["variants"][key]["MAE"] < row["variants"]["full"]["MAE"]
        ]
        deltas = {
            row["id"]: row["variants"][key]["delta_MAE_vs_full_wh"]
            for row in applied_folds
        }
        mae_values = [row["variants"][key]["MAE"] for row in applied_folds]
        is_oracle = key in ORACLE_VARIANTS
        summary_variants[key] = {
            "applied_scoring_folds": len(applied_folds),
            "folds_helped": len(helped),
            "helped_fold_ids": [row["id"] for row in helped],
            "delta_MAE_vs_full_wh": deltas,
            "mean_MAE_wh": round(float(np.mean(mae_values)), 4)
            if mae_values else None,
            "mean_delta_MAE_vs_full_wh": round(
                float(np.mean([deltas[row["id"]] for row in applied_folds])), 4
            ) if applied_folds else None,
            "kept": (
                not is_oracle
                and len(helped) >= HELP_FOLDS_REQUIRED
                and len(applied_folds) == len(scoring)
            ),
            "diagnostic_only": is_oracle,
        }

    full_mae = np.mean(
        [row["variants"]["full"]["MAE"] for row in scoring]
    )
    eligible = [
        key for key, stats in summary_variants.items()
        if stats["kept"] and stats["mean_MAE_wh"] is not None
    ]
    if eligible:
        winner = min(eligible, key=lambda key: summary_variants[key]["mean_MAE_wh"])
    else:
        winner = "full"
    winner_options = dict(VARIANTS[winner])

    results = {
        "generated_at": pd.Timestamp.now(tz=ev.dp.TZ).isoformat(timespec="seconds"),
        "protocol": {
            "folds": ev.PROTOCOL["folds"],
            "evaluation_mask": ev.PROTOCOL["evaluation_mask"],
            "config": "untuned forecast_model.DEFAULT_PARAMS, identical across variants",
            "calendar_rule": (
                "drop month, doy_sin and doy_cos only when the training window "
                "spans fewer than 365 days"
            ),
            "weather": (
                "lagged >=24 h in full and no_annual_calendar; omitted in "
                "no_weather; target-hour temperature in ORACLE only"
            ),
            "help_rule": (
                "strictly lower MAE than the full-feature model on a scoring "
                "fold; keep a real variant only if it helps on at least "
                f"{HELP_FOLDS_REQUIRED} of {len(scoring)} scoring folds"
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
            "full_mean_MAE_wh": round(float(full_mae), 4),
            "variants": summary_variants,
            "kept_variants": [
                key for key, stats in summary_variants.items() if stats["kept"]
            ],
            "winner": winner,
            "winner_feature_options": winner_options,
            "oracle_warning": (
                "oracle_same_hour_temperature uses target-hour temperature, "
                "which is not known at the 00:00 IST origin. It bounds the "
                "value of same-day weather and is never a deployable result: "
                "it cannot be kept or become the winner."
            ),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH}")

    print(f"\nFeature ablation, {len(fold_rows)} folds, untuned fixed config")
    print(f"{'fold':<10} {'eval h':>7} "
          f"{'full':>9} {'no-cal':>9} {'no-wx':>9} {'ORACLE':>9}")
    for row in fold_rows:
        v = row["variants"]
        print(f"{row['id']:<10} {row['eval_hours']:>7} "
              f"{v['full']['MAE']:>9.2f} "
              f"{v['no_annual_calendar']['MAE']:>9.2f} "
              f"{v['no_weather']['MAE']:>9.2f} "
              f"{v['oracle_same_hour_temperature']['MAE']:>9.2f}"
              f"{'  stress' if row['stress_test'] else ''}")

    print("\nScoring-fold summary vs full features (negative delta = better)")
    for key, stats in summary_variants.items():
        label = key.upper() if stats["diagnostic_only"] else key
        print(f"{label:<30} helped {stats['folds_helped']}/"
              f"{stats['applied_scoring_folds']} scoring folds, "
              f"mean MAE {stats['mean_MAE_wh']:.2f} Wh, "
              f"mean delta {stats['mean_delta_MAE_vs_full_wh']:+.2f} Wh, "
              f"kept={stats['kept']}")
    print(f"\nwinner: {winner} {winner_options or '(default features)'}")


if __name__ == "__main__":
    main()
