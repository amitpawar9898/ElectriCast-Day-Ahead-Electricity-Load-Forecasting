"""One-shot Optuna tuning with parameters frozen across Phase 9 folds.

Run from the repository root:

    python backend/scripts/frozen_tuning.py

Two tuning windows are compared:
  * largest: history through 2017-12-31, the 2018 fold's training window. This
    is what 9.5 asks for, but it contains the scoring folds' test periods, so a
    scoring-fold gain is not clean out-of-sample evidence for tuning.
  * pre_scoring: history through 2017-05-31, before the first scoring test
    month. This is the clean check but a shorter tuning window.

Both tuning windows are shorter than 12 months, so both use the 9.3 winning
feature set (no annual calendar) and 30 TPE trials on three expanding-window CV
splits. The best hyperparameters plus the median early-stopping iteration are
frozen and applied unchanged to every fold. The 9.3 feature rule is applied per
fold: annual calendar features are dropped only when that fold's training span
is shorter than 365 days, which excludes the 2018 stress fold. All variants
within a fold share the same features, so each comparison is apples-to-apples.
The stress test is not counted in the decision.

A frozen variant is adopted only if it lowers common-hours MAE on at least 3 of
the 5 scoring folds. Because the largest window contains the scoring test
periods, the clean pre-scoring window is preferred when it helps and the
largest-window parameters do not; the artifact records both and the reason.
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

OUTPUT_PATH = BACKEND / "tuning_results.json"
FEATURE_OPTIONS = {"drop_annual_calendar": True}
N_TRIALS = 30
HELP_FOLDS_REQUIRED = 3


def _feature_options(train_start, train_end):
    """Apply the 9.3 rule: drop annual calendar only below 12 months."""
    start = pd.Timestamp(train_start)
    span = pd.Timestamp(train_end, tz=start.tz) - start
    if span < pd.Timedelta(days=365):
        return dict(FEATURE_OPTIONS)
    return dict(fm.FEATURE_DEFAULTS)

TUNING_WINDOWS = {
    "largest": {
        "train_end": "2017-12-31 23:00",
        "contains_scoring_test_periods": True,
    },
    "pre_scoring": {
        "train_end": "2017-05-31 23:00",
        "contains_scoring_test_periods": False,
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
    series, reporting = ev.load_target_with_reporting()
    weather_frame = weather.load()
    train_start = ev.history_start(series)

    tuning = {}
    for key, spec in TUNING_WINDOWS.items():
        _, metadata = fm.fit(
            series,
            train_start,
            spec["train_end"],
            reporting=reporting,
            weather=weather_frame,
            tune_model=True,
            n_trials=N_TRIALS,
            feature_options=FEATURE_OPTIONS,
        )
        tuning[key] = {
            "train_start": train_start,
            "train_end": spec["train_end"],
            "contains_scoring_test_periods": spec[
                "contains_scoring_test_periods"
            ],
            "n_train": metadata["n_train"],
            "params": metadata["params"],
            "tuning": metadata["tuning"],
        }

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

        fold_features = _feature_options(
            fold["train_start"], fold["train_end"]
        )
        predictions = {}
        metadata = {}
        variants = {
            "untuned_default": None,
            "frozen_largest": tuning["largest"]["params"],
            "frozen_pre_scoring": tuning["pre_scoring"]["params"],
        }
        for key, params in variants.items():
            pred, meta = fm.forecast(
                series,
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
                params=params,
                reporting=reporting,
                weather=weather_frame,
                tune_model=False,
                feature_options=fold_features,
            )
            predictions[key] = pred
            metadata[key] = meta

        reference = predictions["untuned_default"]
        variant_rows = {}
        for key, pred in predictions.items():
            common = mask & reference.notna() & pred.notna()
            row = _metrics(actual[common], pred[common], scale)
            reference_common = _metrics(
                actual[common], reference[common], scale
            )
            row["common_hours_with_untuned"] = int(common.sum())
            row["untuned_MAE_on_pair_common_wh"] = reference_common["MAE"]
            row["delta_MAE_vs_untuned_wh"] = (
                None if key == "untuned_default"
                else round(float(row["MAE"] - reference_common["MAE"]), 4)
            )
            row["n_train"] = metadata[key]["n_train"]
            row["n_estimators"] = metadata[key]["params"]["n_estimators"]
            variant_rows[key] = row

        fold_rows.append({
            "id": fold["id"],
            "label": fold.get("label"),
            "stress_test": bool(fold.get("stress_test")),
            "eval_hours": int(mask.sum()),
            "feature_options": fold_features,
            "variants": variant_rows,
        })

    scoring = [row for row in fold_rows if not row["stress_test"]]
    summary_variants = {}
    for key in ("frozen_largest", "frozen_pre_scoring"):
        helped = [
            row for row in scoring
            if row["variants"][key]["delta_MAE_vs_untuned_wh"] < 0
        ]
        deltas = {
            row["id"]: row["variants"][key]["delta_MAE_vs_untuned_wh"]
            for row in scoring
        }
        mae_values = [row["variants"][key]["MAE"] for row in scoring]
        summary_variants[key] = {
            "folds_helped": len(helped),
            "helped_fold_ids": [row["id"] for row in helped],
            "mean_MAE_wh": round(float(np.mean(mae_values)), 4),
            "mean_delta_MAE_vs_untuned_wh": round(
                float(np.mean(list(deltas.values()))), 4
            ),
            "delta_MAE_vs_untuned_wh": deltas,
            "helps_at_all": len(helped) >= HELP_FOLDS_REQUIRED,
        }

    largest = summary_variants["frozen_largest"]
    clean = summary_variants["frozen_pre_scoring"]
    if clean["helps_at_all"] and largest["helps_at_all"]:
        winner = "frozen_pre_scoring"
        decision = (
            "tuning helps on both windows; use the clean pre-scoring frozen "
            "parameters rather than the largest-window parameters, because the "
            "largest window contains the scoring test periods"
        )
    elif clean["helps_at_all"]:
        winner = "frozen_pre_scoring"
        decision = (
            "the requested largest-window parameters help on only "
            f"{largest['folds_helped']}/5 scoring folds, and that window "
            "contains the scoring test periods; the clean pre-scoring tuning "
            f"helps on {clean['folds_helped']}/5, so use its frozen parameters"
        )
    elif largest["helps_at_all"]:
        winner = "untuned_default"
        decision = (
            "the largest-window frozen parameters help but the clean "
            "pre-scoring tuning does not; treat the gain as likely test-period "
            "contamination and keep the untuned defaults"
        )
    else:
        winner = "untuned_default"
        decision = "tuning does not help on at least 3 scoring folds; keep the untuned defaults"

    results = {
        "generated_at": pd.Timestamp.now(tz=ev.dp.TZ).isoformat(timespec="seconds"),
        "protocol": {
            "folds": ev.PROTOCOL["folds"],
            "evaluation_mask": ev.PROTOCOL["evaluation_mask"],
            "feature_options": (
                "9.3 rule applied per fold: drop month/day-of-year when the "
                "fold's training span is shorter than 365 days; the 2018 stress "
                "fold therefore keeps the full calendar"
            ),
            "tuning": (
                "Optuna TPE, seed 42, 30 trials, three expanding-window CV "
                "splits entirely inside each tuning window; the frozen "
                "iteration count is the median best iteration across those "
                "splits; parameters are then applied unchanged to every fold"
            ),
            "information_set": (
                "the largest window ends 2017-12-31 and therefore contains "
                "the Jun-Dec 2017 scoring test periods; the pre-scoring window "
                "ends 2017-05-31 and contains none of them"
            ),
            "comparison_mask": (
                "evaluation mask intersected with both models' forecasts so MAE "
                "differences are not caused by different coverage"
            ),
            "decision_rule": (
                "tuning counts as helping only if it lowers common-hours MAE "
                f"on at least {HELP_FOLDS_REQUIRED} of {len(scoring)} scoring "
                "folds; the 2018 stress test is excluded"
            ),
        },
        "tuning": tuning,
        "folds": fold_rows,
        "summary": {
            "scoring_fold_ids": [row["id"] for row in scoring],
            "stress_test_fold_ids": [
                row["id"] for row in fold_rows if row["stress_test"]
            ],
            "variants": summary_variants,
            "winner": winner,
            "decision": decision,
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH}")

    for key, run in tuning.items():
        best = run["params"]
        print(f"\n{key}: {run['n_train']} train rows, "
              f"best CV MAE {run['tuning']['best_cv_mae']:.4f}, "
              f"n_estimators {best['n_estimators']}")
        print("  " + ", ".join(
            f"{name}={best[name]}" for name in (
                "max_depth", "learning_rate", "min_child_weight",
                "subsample", "colsample_bytree", "reg_lambda", "reg_alpha",
            )
        ))

    print(f"\nFrozen tuning, {len(fold_rows)} folds, per-fold 9.3 feature rule")
    print(f"{'fold':<10} {'eval h':>7} {'untuned':>9} "
          f"{'largest':>9} {'clean':>9}")
    for row in fold_rows:
        v = row["variants"]
        print(f"{row['id']:<10} {row['eval_hours']:>7} "
              f"{v['untuned_default']['MAE']:>9.2f} "
              f"{v['frozen_largest']['MAE']:>9.2f} "
              f"{v['frozen_pre_scoring']['MAE']:>9.2f}"
              f"{'  stress' if row['stress_test'] else ''}")

    print("\nScoring-fold summary on common hours (negative delta = better)")
    for key, stats in summary_variants.items():
        print(f"{key:<20} helped {stats['folds_helped']}/{len(scoring)} scoring folds, "
              f"mean MAE {stats['mean_MAE_wh']:.2f} Wh, "
              f"mean delta {stats['mean_delta_MAE_vs_untuned_wh']:+.2f} Wh, "
              f"helps={stats['helps_at_all']}")
    print(f"\nwinner: {winner}")
    print(decision)


if __name__ == "__main__":
    main()
