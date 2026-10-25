"""Report how much usable history precedes the original 2017-05-01 train start.

Run from the repository root:

    python backend/scripts/report_history.py

The report covers:
1. valid target hours in 2016-12-14 .. 2017-04-30 and their monthly spread;
2. the earliest valid target hour, which Phase 9 uses as every fold's
   train_start;
3. whether December 2016 shows the same apartment-vacancy pattern as
   December 2017 (apartment_activity flags).
"""

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import data_pipeline as dp  # noqa: E402

EARLY_START = "2016-12-14 00:00"
EARLY_END = "2017-04-30 23:00"


def _month_stats(series, start, end):
    index = pd.date_range(start, end, freq="h", tz=dp.TZ)
    values = series.reindex(index)
    rows = []
    for month, block in values.groupby(values.index.strftime("%Y-%m")):
        rows.append({
            "month": str(month),
            "hours_in_range": int(len(block)),
            "valid_hours": int(block.notna().sum()),
        })
    return rows


def _activity(panel, month, stable):
    all_stats = dp.apartment_activity(panel, month=month)
    stable_stats = all_stats.reindex(stable)
    return {
        "month": month,
        "apartments_with_readings": int(all_stats["hours"].gt(0).sum()),
        "flagged_all": sorted(all_stats.index[all_stats["suspect_inactive"]]),
        "flagged_stable": sorted(
            stable_stats.index[stable_stats["suspect_inactive"]]
        ),
        "stable_with_readings": int(stable_stats["hours"].gt(0).sum()),
    }


def main():
    apartments, load_report = dp.load_apartments()
    panel, panel_reporting = dp.build_panel(apartments)
    stable, coverage = dp.stable_apartments(panel)
    target, target_reporting = dp.build_target(panel, stable)
    filled, imputed, excluded, runs = dp.fill_short_gaps(target)

    early_mask = (filled.index >= pd.Timestamp(EARLY_START, tz=dp.TZ)) & (
        filled.index <= pd.Timestamp(EARLY_END, tz=dp.TZ)
    )
    early = filled[early_mask]
    valid_all = filled.dropna()
    earliest = valid_all.index.min()
    early_valid = early.dropna()

    reporting_early = target_reporting.reindex(early.index)
    dec16 = _activity(panel, "2016-12", stable)
    dec17 = _activity(panel, "2017-12", stable)
    flagged16 = set(dec16["flagged_all"])
    flagged17 = set(dec17["flagged_all"])
    union = flagged16 | flagged17

    report = {
        "raw_range_ist": [load_report["first_ist"], load_report["last_ist"]],
        "files_read": load_report["read_ok"],
        "panel_hours": int(len(panel)),
        "stable_apartments": len(stable),
        "stable_coverage_min": round(float(coverage[stable].min()), 4),
        "target": {
            "first_timestamp": str(target.index.min()),
            "last_timestamp": str(target.index.max()),
            "valid_hours_raw": int(target.notna().sum()),
            "valid_hours_after_fill": int(valid_all.shape[0]),
            "earliest_valid_hour": str(earliest),
            "imputed_hours": int(imputed.sum()),
            "excluded_hours": int(excluded.sum()),
            "gap_runs": len(runs),
        },
        "early_window": {
            "start": EARLY_START,
            "end": EARLY_END,
            "hours_in_range": int(early.shape[0]),
            "valid_hours_after_fill": int(early_valid.shape[0]),
            "valid_hours_raw": int(target.reindex(early.index).notna().sum()),
            "first_valid": str(early_valid.index.min()),
            "last_valid": str(early_valid.index.max()),
            "reporting_apartments": {
                "min": int(reporting_early.min()),
                "median": float(reporting_early.median()),
                "max": int(reporting_early.max()),
                "hours_with_reporting": int((reporting_early > 0).sum()),
            },
            "by_month": _month_stats(filled, EARLY_START, EARLY_END),
        },
        "december_activity": {
            "threshold_wh": dp.INACTIVE_MAX_MEAN_WH,
            "min_hours": dp.INACTIVE_MIN_HOURS,
            "dec_2016": dec16,
            "dec_2017": dec17,
            "overlap": {
                "union": len(union),
                "both": len(flagged16 & flagged17),
                "only_2016": len(flagged16 - flagged17),
                "only_2017": len(flagged17 - flagged16),
                "jaccard": round(
                    len(flagged16 & flagged17) / len(union), 4
                ) if union else None,
            },
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
