"""Scripted check for the Ask ElectriCast pipeline.

Default mode uses a deterministic stub provider that writes answers from the
tool outputs, so the check runs offline. ``--live`` sends the same three
questions to the configured Ollama/Gemini backend.

Both modes enforce:
1. every number in an answer exists in results.json, data_quality.json, or a
   selected read-only tool result (all tool results are also cross-checked
   against the raw artifacts below);
2. each suggested question routes to the expected tools and states an
   artifact-backed value.

Run from the repository root:

    python backend/scripts/check_chat.py
    python backend/scripts/check_chat.py --live
"""

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from chat_service import (  # noqa: E402
    ChatService,
    ProviderUnavailable,
    find_unsupported_numbers,
)
from chat_tools import ChatTools, SUGGESTED_QUESTIONS  # noqa: E402

RESULTS_PATH = BACKEND / "results.json"
QUALITY_PATH = BACKEND / "data_quality.json"

EXPECTED_TOOLS = {
    "Which model is best and by how much?": ["model_metrics"],
    "Why is December different?": ["error_breakdown", "limitations"],
    "Where does XGBoost fail?": ["error_breakdown", "limitations"],
}


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value, digits=2):
    return f"{value:.{digits}f}"


def _rounded_forms(value):
    forms = set()
    for digits in range(0, 5):
        text = f"{value:.{digits}f}"
        forms.add(text)
        if "." in text:
            forms.add(text.rstrip("0").rstrip("."))
        integer, _, decimals = text.partition(".")
        if integer.lstrip("-").isdigit() and len(integer.lstrip("-")) > 3:
            forms.add(f"{int(integer):,}" + (f".{decimals}" if decimals else ""))
    return forms


def _answer_has_any(answer, values):
    return any(form in answer for value in values for form in _rounded_forms(value))


def _hourly_records(results):
    for fold in results["folds"]:
        hourly = fold.get("hourly") or {}
        columns = hourly.get("columns") or []
        for row in hourly.get("rows") or []:
            yield dict(zip(columns, row))


def _error_stats(records, key):
    pairs = [
        (record["actual"], record[key])
        for record in records
        if record.get("actual") is not None and record.get(key) is not None
    ]
    if not pairs:
        return None, None, 0
    mae = sum(abs(pred - actual) for actual, pred in pairs) / len(pairs)
    bias = sum(pred - actual for actual, pred in pairs) / len(pairs)
    return round(mae, 2), round(bias, 2), len(pairs)


def _worst_bucket(records, group_key, label_fn, key="xgboost"):
    groups = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)
    candidates = []
    for group, rows in groups.items():
        mae, _, n = _error_stats(rows, key)
        if mae is not None:
            candidates.append((label_fn(group), mae, n))
    robust = [item for item in candidates if item[2] >= 24]
    return max(robust or candidates, key=lambda item: item[1])


def _verify_tool_sources(tools, results, quality):
    """Cross-check the assistant's derived aggregates against the raw JSON."""
    failures = []
    derived = {}

    def check(label, actual, expected, tolerance=1e-9):
        if actual is None or abs(float(actual) - float(expected)) > tolerance:
            failures.append(f"{label}: tool={actual} raw={expected}")
        else:
            derived[label] = expected

    folds = [fold for fold in results["folds"] if not fold.get("stress_test")]

    def _fold_mae(key):
        values = [
            (fold.get("models") or {}).get(key, {}).get("MAE")
            for fold in folds
        ]
        values = [value for value in values if value is not None]
        return round(sum(values) / len(values), 4) if values else None

    def _fold_hours(key):
        return sum(
            (fold.get("models") or {}).get(key, {}).get("n", 0)
            for fold in folds
            if (fold.get("models") or {}).get(key, {}).get("status") == "ok"
        )

    eval_hours = sum(fold["test"]["eval_hours"] for fold in folds)

    overview = tools.model_metrics()
    best = overview["best_model"]
    runner = best["runner_up"]
    check("best_fold_mean_mae", best["fold_mean_MAE_wh"], _fold_mae(best["key"]))
    check("best_model_hours", best["n_total_hours"], _fold_hours(best["key"]))
    check("best_eval_hours", best["eval_hours"], eval_hours)
    check("runner_fold_mean_mae", runner["fold_mean_MAE_wh"],
          _fold_mae(runner["key"]))
    check("best_gap", best["gap_to_runner_up_wh"],
          round(runner["fold_mean_MAE_wh"] - best["fold_mean_MAE_wh"], 4),
          tolerance=1e-4)
    check("scoring_fold_count", len(overview["scoring_fold_ids"]), len(folds))

    records = list(_hourly_records(results))
    december = [record for record in records
                if str(record.get("time", "")).startswith("2017-12")]
    dec_mae, dec_bias, dec_n = _error_stats(december, "xgboost")
    month = tools.error_breakdown(month=12)["selected_month"]
    check("december_xgb_mae", month["models"]["xgboost"]["mae_wh"], dec_mae)
    check("december_xgb_bias", month["models"]["xgboost"]["bias_wh"], dec_bias)
    check("december_xgb_n", month["models"]["xgboost"]["n"], dec_n)

    limits = tools.limitations()
    inactive = quality["inactive"]
    check("december_flagged", limits["december_inactivity"]["flagged_count"],
          len(inactive.get("flagged") or []))
    check("december_flagged_stable",
          limits["december_inactivity"]["flagged_stable_count"],
          len(inactive.get("flagged_stable") or []))
    check("december_stable_count",
          limits["december_inactivity"]["stable_panel_count"],
          quality["stable_panel"]["count"])
    december_context = limits.get("december_context") or {}
    check("december_context_mae", december_context.get("mae_wh"), dec_mae,
          tolerance=1e-2)
    check("december_context_bias", december_context.get("bias_wh"), dec_bias,
          tolerance=1e-2)

    conformal = limits["prediction_intervals"].get("conformal")
    if conformal:
        pairs = []
        for fold in folds:
            interval = fold["intervals"]["xgboost_quantile"]
            after = fold["intervals"]["conformal"]["after_all_days"]["coverage"]
            pairs.append((interval["coverage"], after, interval["n"]))
        total = sum(n for _, _, n in pairs)
        weighted_before = round(
            100.0 * sum(before * n for before, _, n in pairs) / total, 2
        )
        weighted_after = round(
            100.0 * sum(after * n for _, after, n in pairs) / total, 2
        )
        check(
            "conformal_before_scoring",
            conformal["scoring_hour_weighted"]["before_percent"],
            weighted_before, tolerance=1e-2,
        )
        check(
            "conformal_after_scoring",
            conformal["scoring_hour_weighted"]["after_percent"],
            weighted_after, tolerance=1e-2,
        )
        check(
            "conformal_window_days",
            conformal["window_days"],
            folds[0]["intervals"]["conformal"]["window_days"],
        )

    errors = tools.error_breakdown(model="xgboost")
    worst_month_label, worst_month_mae, worst_month_n = _worst_bucket(
        records, lambda record: str(record.get("time", ""))[:7],
        lambda period: period,
    )
    worst_hour_label, worst_hour_mae, worst_hour_n = _worst_bucket(
        records, lambda record: int(str(record.get("time", ""))[11:13]),
        lambda hour: f"{hour:02d}:00",
    )
    check("worst_month_mae", errors["worst_month"]["mae_wh"], worst_month_mae)
    check("worst_month_n", errors["worst_month"]["n"], worst_month_n)
    check("worst_hour_mae", errors["worst_hour"]["mae_wh"], worst_hour_mae)
    check("worst_hour_n", errors["worst_hour"]["n"], worst_hour_n)

    if errors["worst_month"]["period"] != worst_month_label:
        failures.append(
            f"worst month: tool={errors['worst_month']['period']} "
            f"raw={worst_month_label}"
        )
    if errors["worst_hour"]["label"] != worst_hour_label:
        failures.append(
            f"worst hour: tool={errors['worst_hour']['label']} "
            f"raw={worst_hour_label}"
        )
    return failures, derived


def _stub_answers(tools):
    overview = tools.model_metrics()
    best = overview["best_model"]
    runner = best["runner_up"]

    december = tools.error_breakdown(month=12)["selected_month"]
    xgb_dec = december["models"]["xgboost"]
    dec_flags = tools.limitations()["december_inactivity"]

    xgb_errors = tools.error_breakdown(model="xgboost")
    worst_month = xgb_errors["worst_month"]
    worst_hour = xgb_errors["worst_hour"]

    return {
        "Which model is best and by how much?": (
            f"{best['label']} has the lowest fold-mean MAE at "
            f"{_fmt(best['fold_mean_MAE_wh'], 4)} Wh. "
            f"{runner['label']} is next at {_fmt(runner['fold_mean_MAE_wh'], 4)} Wh, "
            f"a gap of {_fmt(best['gap_to_runner_up_wh'], 4)} Wh. "
            f"{best['label']} is scored on {best['n_total_hours']} of "
            f"{best['eval_hours']} evaluation hours, so coverage differs."
        ),
        "Why is December different?": (
            f"In {december['label']}, XGBoost has MAE {_fmt(xgb_dec['mae_wh'])} Wh "
            f"and bias {_fmt(xgb_dec['bias_wh'])} Wh. "
            f"{dec_flags['flagged_count']} apartments were flagged suspect inactive, "
            f"including {dec_flags['flagged_stable_count']} of "
            f"{dec_flags['stable_panel_count']} stable apartments. "
            "The flag is descriptive, not confirmed vacancy."
        ),
        "Where does XGBoost fail?": (
            f"XGBoost's worst month is {worst_month['label']} with MAE "
            f"{_fmt(worst_month['mae_wh'])} Wh and bias "
            f"{_fmt(worst_month['bias_wh'])} Wh. Its worst hour is "
            f"{worst_hour['label']} with MAE {_fmt(worst_hour['mae_wh'])} Wh."
        ),
    }


def _expected_numbers(question, tools):
    if question == SUGGESTED_QUESTIONS[0]:
        best = tools.model_metrics()["best_model"]
        return [best["fold_mean_MAE_wh"], best["runner_up"]["fold_mean_MAE_wh"],
                best["gap_to_runner_up_wh"]]
    if question == SUGGESTED_QUESTIONS[1]:
        month = tools.error_breakdown(month=12)["selected_month"]
        flags = tools.limitations()["december_inactivity"]
        return [month["models"]["xgboost"]["mae_wh"],
                month["models"]["xgboost"]["bias_wh"],
                float(flags["flagged_count"])]
    errors = tools.error_breakdown(model="xgboost")
    return [errors["worst_month"]["mae_wh"], errors["worst_hour"]["mae_wh"],
            errors["worst_month"]["bias_wh"]]


def _check_answer(question, answer, tool_names, tools, grounding, live):
    failures = []
    if tool_names != EXPECTED_TOOLS[question]:
        failures.append(f"tools {tool_names} != {EXPECTED_TOOLS[question]}")
    if not answer or len(answer.strip()) < 20:
        failures.append("answer is empty or too short")

    unsupported = find_unsupported_numbers(answer, grounding)
    if unsupported:
        failures.append(f"numbers not in artifacts or tool results: {unsupported}")

    expected = _expected_numbers(question, tools)
    if not _answer_has_any(answer, expected):
        failures.append(f"answer does not state an expected artifact value {expected}")
    if live and not any(char.isdigit() for char in answer):
        failures.append("live answer contains no numbers")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="ask the configured Ollama/Gemini backend instead of the stub",
    )
    args = parser.parse_args()

    results = _load(RESULTS_PATH)
    quality = _load(QUALITY_PATH)
    tools = ChatTools(results, quality)

    source_failures, derived = _verify_tool_sources(tools, results, quality)
    if source_failures:
        print("FAIL tool outputs differ from the raw artifacts:")
        for failure in source_failures:
            print(f"  - {failure}")
        return 1
    print(f"source check: {len(derived)} derived values match the raw artifacts")

    service = ChatService(tools)
    status = service.status(force=True)
    if args.live and not status.get("available"):
        print(f"FAIL assistant unavailable: {status.get('detail')}")
        return 2
    print(
        f"mode={'live' if args.live else 'stub'} "
        f"provider={status.get('provider')} model={status.get('model')} "
        f"generated_at={results.get('generated_at')}"
    )

    current = {"question": SUGGESTED_QUESTIONS[0]}
    if args.live:
        provider = None
    else:
        answers = _stub_answers(tools)

        def provider(system, user):
            return answers[current["question"]]

    total_failures = 0
    for question in SUGGESTED_QUESTIONS:
        current["question"] = question
        calls = tools.plan(question)
        grounding = {
            "results": results,
            "data_quality": quality,
            "derived": derived,
            "selected_tool_results": [
                {"name": call.name, "result": tools.call(call)}
                for call in calls
            ],
        }
        try:
            result = service.ask(question, provider=provider)
        except ProviderUnavailable as exc:
            print(f"FAIL [{question}] assistant unavailable: {exc}")
            total_failures += 1
            continue
        except Exception as exc:
            print(f"FAIL [{question}] {type(exc).__name__}: {exc}")
            total_failures += 1
            continue

        tool_names = [tool["name"] for tool in result["tools"]]
        failures = _check_answer(
            question, result["answer"], tool_names, tools, grounding, args.live
        )
        if failures:
            total_failures += 1
            print(f"FAIL [{question}] " + "; ".join(failures))
        else:
            print(f"PASS [{question}] tools={tool_names}")
        print(f"  {result['answer']}")

    if total_failures:
        print(f"\n{total_failures} of {len(SUGGESTED_QUESTIONS)} checks failed.")
        return 1
    print(f"\nAll {len(SUGGESTED_QUESTIONS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
