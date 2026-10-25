"""Build backend/data_quality.json from the raw meter CSVs.

Run from the repo root:  python backend/build_data_quality.py
"""
import json
import os

import pandas as pd

import data_pipeline as dp

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_quality.json")


def _num(value, digits=4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def build_report():
    apartments, report = dp.load_apartments()
    panel, _ = dp.build_panel(apartments)
    stable, coverage = dp.stable_apartments(panel)
    target, _ = dp.build_target(panel, stable)
    filled, imputed, excluded, runs = dp.fill_short_gaps(target)
    activity = dp.apartment_activity(panel)

    reporting = panel.notna().sum(axis=1)
    valid = filled.notna()

    period = panel.index.tz_localize(None).to_period("M")
    monthly = []
    for month in period.unique():
        mask = period == month
        monthly.append({
            "month": str(month),
            "hours": int(mask.sum()),
            "apartments_reporting": int(panel.loc[mask].notna().any().sum()),
            "target_hours": int(valid[mask].sum()),
        })

    longest_runs = sorted(runs, key=lambda run: run["hours"], reverse=True)[:10]
    short_runs = [r for r in runs if r["hours"] <= dp.MAX_GAP_HOURS and not r["edge"]]
    excluded_runs = [r for r in runs if r["hours"] > dp.MAX_GAP_HOURS or r["edge"]]
    flagged = activity[activity.suspect_inactive]
    flagged_stable = sorted(set(stable) & set(flagged.index), key=int)

    summary = {
        "generated_at": pd.Timestamp.now(tz=dp.TZ).isoformat(timespec="seconds"),
        "source": {
            "data_dir": "data",
            "files_found": report["files_found"],
            "files_read": report["read_ok"],
            "files_failed": report["failed"],
            "rows_total": report["rows_total"],
            "first_utc": report["first_utc"],
            "last_utc": report["last_utc"],
            "first_ist": report["first_ist"],
            "last_ist": report["last_ist"],
            "errors": report["errors"],
            "files": report["files"],
        },
        "panel": {
            "timezone": dp.TZ,
            "hours": int(len(panel)),
            "apartments": int(panel.shape[1]),
            "first_ist": panel.index[0].isoformat(),
            "last_ist": panel.index[-1].isoformat(),
            "hours_any_data": int(panel.notna().any(axis=1).sum()),
            "hours_zero_reporting": int((reporting == 0).sum()),
            "reporting_median": int(reporting.median()),
            "reporting_max": int(reporting.max()),
        },
        "stable_panel": {
            "window_start": dp.STABLE_START,
            "window_end": dp.STABLE_END,
            "min_coverage": dp.STABLE_MIN_COVERAGE,
            "count": len(stable),
            "apartments": [
                {"apartment": str(name), "coverage": _num(coverage[name], 6)}
                for name in coverage.index
            ],
            "excluded": [str(name) for name in coverage.index if name not in stable],
        },
        "target": {
            "valid_hours_before_gaps": int(target.notna().sum()),
            "valid_hours": int(valid.sum()),
            "imputed_hours": int(imputed.sum()),
            "excluded_hours": int(excluded.sum()),
            "gap_runs": len(runs),
            "short_runs": len(short_runs),
            "excluded_runs": len(excluded_runs),
            "max_gap_hours": dp.MAX_GAP_HOURS,
            "longest_gap_hours": int(longest_runs[0]["hours"]) if longest_runs else 0,
            "mean_wh": _num(filled[valid].mean()),
            "min_wh": _num(filled[valid].min()),
            "max_wh": _num(filled[valid].max()),
            "longest_runs": longest_runs,
        },
        "monthly_coverage": monthly,
        "inactive": {
            "month": dp.INACTIVE_MONTH,
            "threshold_wh": dp.INACTIVE_MAX_MEAN_WH,
            "min_hours": dp.INACTIVE_MIN_HOURS,
            "flagged": [str(name) for name in flagged.index],
            "flagged_stable": flagged_stable,
            "apartments": [
                {
                    "apartment": str(name),
                    "mean_wh": _num(row.mean_wh),
                    "hours": int(row.hours),
                    "suspect_inactive": bool(row.suspect_inactive),
                }
                for name, row in activity.iterrows()
            ],
        },
    }

    summary["notes"] = [
        f"Raw files: {report['read_ok']} read, {report['failed']} failed; "
        f"{report['rows_total']:,} rows from {report['first_ist']} to {report['last_ist']} IST.",
        f"The hourly panel covers {len(panel):,} IST hours and {panel.shape[1]} apartments; "
        f"{summary['panel']['hours_zero_reporting']:,} hours have no readings.",
        f"Stable window {dp.STABLE_START} to {dp.STABLE_END}: "
        f"{len(stable)} of {panel.shape[1]} apartments have at least "
        f"{dp.STABLE_MIN_COVERAGE:.0%} hourly coverage.",
        f"Gaps: {int(imputed.sum())} hours in runs of at most {dp.MAX_GAP_HOURS} h were "
        f"interpolated; {int(excluded.sum()):,} hours in "
        f"{len(excluded_runs)} longer or edge runs are excluded from training and evaluation.",
        f"{pd.Timestamp(dp.INACTIVE_MONTH + '-01').strftime('%B %Y')}: "
        f"{len(flagged)} apartments average below {dp.INACTIVE_MAX_MEAN_WH:.0f} Wh/h "
        f"(including {len(flagged_stable)} of the {len(stable)} stable apartments); "
        "this is flagged as suspect inactivity, likely vacancy, not confirmed meter failure.",
        "Rows stamped at :00 UTC, including the May-2018 block, are converted to IST and "
        "floored to the containing hour; no apartment-hour bucket is duplicated.",
    ]
    return summary


def main():
    summary = build_report()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
