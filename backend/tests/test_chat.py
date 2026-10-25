"""Tests for the grounded assistant tools, guard, planner, and providers.

The fixtures are hand-built artifacts with exact expected aggregates, so the
suite does not read ``results.json`` and never touches the network.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import chat_service  # noqa: E402
from chat_service import (  # noqa: E402
    ChatService,
    GroundingError,
    ProviderError,
    ProviderUnavailable,
    find_unsupported_numbers,
)
from chat_tools import ChatTools, SUGGESTED_QUESTIONS  # noqa: E402


COLUMNS = ["time", "actual", "seasonal_naive", "xgboost", "sarima",
           "p10", "p50", "p90"]


def _hourly_rows(month, actual, errors, sarima=True):
    rows = []
    for day in (1, 2):
        for hour in range(24):
            special = day == 1 and hour == 9 and month == 7
            xgb = errors["xgboost"] + (20 if special else 0)
            seasonal = errors["seasonal_naive"] + (20 if special else 0)
            p50 = errors["p50"] + (10 if special else 0)
            sarima_error = None
            if sarima and day == 1:
                sarima_error = 6 + (10 if special else 0)
            rows.append([
                f"{2017}-{'07' if month == 7 else '08'}-"
                f"{day:02d}T{hour:02d}:00",
                actual,
                actual + seasonal,
                actual + xgb,
                actual + sarima_error if sarima_error is not None else None,
                actual + p50 - 10,
                actual + p50,
                actual + p50 + 10,
            ])
    return rows


def _model(n, mae, status="ok"):
    if status != "ok":
        return {
            "status": status, "n": 0, "MAE": None, "RMSE": None,
            "WAPE": None, "MASE": None, "R2": None, "sMAPE": None,
            "bias": None,
        }
    return {
        "status": "ok", "n": n, "MAE": mae, "RMSE": mae + 1,
        "WAPE": 10.0, "MASE": 1.0, "R2": 0.5, "sMAPE": 9.0,
        "bias": mae,
    }


def build_artifacts():
    fold1 = {
        "id": "2017-07",
        "test": {
            "start": "2017-07-01 00:00", "end": "2017-07-02 23:00",
            "eval_hours": 48, "eval_days": 2,
        },
        "models": {
            "seasonal_naive_24": _model(48, 10.4167),
            "xgboost": _model(48, 8.4167),
            "sarima": _model(24, 6.4167),
            "xgboost_quantile": _model(48, 7.2083),
        },
        "intervals": {"xgboost_quantile": {"status": "ok", "n": 48,
                                           "coverage": 0.75},
                      "conformal": {
                          "status": "ok", "window_days": 28,
                          "n_days_calibrated": 1, "warmup_days": 1,
                          "after_all_days": {"coverage": 0.85},
                          "after_calibrated_days": {"coverage": 0.9},
                      }},
        "hourly": {
            "columns": COLUMNS,
            "rows": _hourly_rows(7, 100.0, {
                "seasonal_naive": 10, "xgboost": 8, "p50": 7,
            }),
        },
    }
    fold2 = {
        "id": "2017-08",
        "test": {
            "start": "2017-08-01 00:00", "end": "2017-08-02 23:00",
            "eval_hours": 48, "eval_days": 2,
        },
        "models": {
            "seasonal_naive_24": _model(48, 20.0),
            "xgboost": _model(48, 12.0),
            "sarima": _model(0, None, status="no_data"),
            "xgboost_quantile": _model(48, 11.0),
        },
        "intervals": {"xgboost_quantile": {"status": "ok", "n": 48,
                                           "coverage": 0.5},
                      "conformal": {
                          "status": "ok", "window_days": 28,
                          "n_days_calibrated": 1, "warmup_days": 1,
                          "after_all_days": {"coverage": 0.6},
                          "after_calibrated_days": {"coverage": 0.65},
                      }},
        "hourly": {
            "columns": COLUMNS,
            "rows": _hourly_rows(8, 200.0, {
                "seasonal_naive": 20, "xgboost": 12, "p50": 11,
            }, sarima=False),
        },
    }
    results = {
        "generated_at": "2026-01-02T03:04:05+05:30",
        "protocol": {
            "timezone": "Asia/Kolkata",
            "target": "mean hourly load per reporting stable apartment (Wh)",
            "origin": "00:00 IST of the test day",
            "information_set": "target observations strictly before the origin",
            "evaluation_mask": "target present and seasonal naive available",
            "folds": "expanding window",
        },
        "baselines": ["seasonal_naive_24"],
        "learned_models": ["xgboost", "sarima", "xgboost_quantile"],
        "weather": {
            "available": True, "hours": 48, "latitude": 19.0,
            "longitude": 72.0, "variables": ["temperature_2m"],
            "missing_values": {"temperature_2m": 0},
            "information_set": "weather lags >= 24 h only",
        },
        "folds": [fold1, fold2],
    }
    quality = {
        "generated_at": "2026-01-02T03:00:00+05:30",
        "source": {
            "files_found": 2, "files_read": 2, "files_failed": 0,
            "rows_total": 100, "first_ist": "2017-01-01T00:00:00+05:30",
            "last_ist": "2017-12-31T23:00:00+05:30",
        },
        "panel": {
            "hours": 96, "apartments": 2, "hours_zero_reporting": 1,
            "reporting_median": 2, "reporting_max": 2,
        },
        "stable_panel": {
            "count": 2, "min_coverage": 0.9,
            "window_start": "2017-01-01", "window_end": "2017-12-31 23:00",
            "apartments": [
                {"apartment": "1", "coverage": 0.95},
                {"apartment": "2", "coverage": 0.91},
            ],
        },
        "target": {
            "valid_hours": 8, "imputed_hours": 1, "excluded_hours": 2,
            "excluded_runs": 1, "max_gap_hours": 1, "longest_gap_hours": 2,
            "mean_wh": 150.0,
            "longest_runs": [
                {"start": "2017-12-01T00:00:00+05:30",
                 "end": "2017-12-01T02:00:00+05:30", "hours": 2,
                 "edge": False},
            ],
        },
        "monthly_coverage": [
            {"month": "2017-12", "hours": 744, "apartments_reporting": 2,
             "target_hours": 500},
        ],
        "inactive": {
            "month": "2017-12", "threshold_wh": 100.0, "min_hours": 24,
            "flagged": ["1", "2"], "flagged_stable": ["1"],
        },
        "notes": ["synthetic fixture"],
    }
    return results, quality


@pytest.fixture()
def tools():
    results, quality = build_artifacts()
    return ChatTools(results, quality)


def make_settings(**overrides):
    values = {
        "CHAT_PROVIDER": "ollama",
        "OLLAMA_URL": "http://127.0.0.1:1",
        "OLLAMA_MODEL": "test:1b",
        "GEMINI_API_KEY": "",
        "GEMINI_MODEL": "gemini-test",
        "GEMINI_API_BASE": "http://127.0.0.1:1",
        "CHAT_TIMEOUT_SECONDS": 0.1,
        "CHAT_HEALTH_TIMEOUT_SECONDS": 0.1,
        "CHAT_MAX_QUESTION_CHARS": 500,
        "CHAT_MAX_ANSWER_CHARS": 4000,
        "CHAT_STATUS_TTL_SECONDS": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ── tools ───────────────────────────────────────────────────────


def test_model_metrics_summary_is_hand_checked(tools):
    overview = tools.model_metrics()
    by_key = {row["key"]: row for row in overview["models"]}

    assert by_key["xgboost"]["mean"]["MAE"] == 10.2083
    assert by_key["xgboost"]["std"]["MAE"] == 2.5338
    assert by_key["seasonal_naive_24"]["mean"]["MAE"] == 15.2083
    assert by_key["xgboost_quantile"]["mean"]["MAE"] == 9.1042
    assert by_key["sarima"]["mean"]["MAE"] == 6.4167
    assert by_key["sarima"]["n_folds"] == 1
    assert by_key["sarima"]["n_total_hours"] == 24
    assert by_key["sarima"]["coverage_percent"] == 25.0
    assert by_key["xgboost"]["folds_beaten"] == 2
    assert by_key["xgboost"]["beats_seasonal_naive"] is True
    assert by_key["seasonal_naive_24"]["beats_seasonal_naive"] is None

    best = overview["best_model"]
    assert best["key"] == "sarima"
    assert best["fold_mean_MAE_wh"] == 6.4167
    assert best["runner_up"]["key"] == "xgboost_quantile"
    assert best["gap_to_runner_up_wh"] == 2.6875
    assert best["gap_to_seasonal_naive_wh"] == 8.7916
    assert overview["units"].startswith("MAE, RMSE and bias in Wh")


def test_model_metrics_fold_and_model_views(tools):
    fold = tools.model_metrics(fold="2017-08")
    assert fold["fold"]["label"] == "Aug 2017"
    assert fold["models"][0]["key"] == "xgboost_quantile"
    assert fold["models"][0]["MAE"] == 11.0
    sarima = next(row for row in fold["models"] if row["key"] == "sarima")
    assert sarima["status"] == "no_data"
    assert sarima["MAE"] is None

    single = tools.model_metrics(model="sarima")
    assert single["model"]["n_folds"] == 1
    assert single["model"]["per_fold"][0]["id"] == "2017-07"
    assert single["model"]["per_fold"][1]["status"] == "no_data"
    assert single["best_model_overall"]["key"] == "sarima"

    missing = tools.model_metrics(model="lstm")
    assert missing["available"] is False
    assert "xgboost" in missing["model_keys"]


def test_error_breakdown_by_month_and_hour(tools):
    xgb = tools.error_breakdown(model="xgboost")
    july = next(row for row in xgb["by_month"] if row["period"] == "2017-07")
    august = next(row for row in xgb["by_month"] if row["period"] == "2017-08")
    assert july["models"]["xgboost"] == {
        "label": "XGBoost", "n": 48, "mae_wh": 8.42, "bias_wh": 8.42,
    }
    assert august["models"]["xgboost"]["mae_wh"] == 12.0
    assert xgb["worst_month"]["period"] == "2017-08"
    assert xgb["worst_month"]["mae_wh"] == 12.0
    assert xgb["worst_hour"]["hour"] == 9
    assert xgb["worst_hour"]["mae_wh"] == 15.0

    hour_nine = tools.error_breakdown(model="xgboost", hour=9)["selected_hour"]
    assert hour_nine["hour"] == 9
    assert hour_nine["models"]["xgboost"]["n"] == 4
    assert hour_nine["models"]["xgboost"]["mae_wh"] == 15.0

    missing_month = tools.error_breakdown(month=12)["selected_month"]
    assert missing_month["found"] is False

    all_models = tools.error_breakdown()
    assert set(all_models["worst_by_model"]) == {
        "seasonal_naive", "xgboost", "sarima", "p50",
    }
    assert all_models["by_month"][0]["models"]["sarima"]["n"] == 24

    persistence = tools.error_breakdown(model="persistence")
    assert persistence["available"] is False
    assert "seasonal_naive_24" in persistence["available_models"]


def test_day_predictions_known_day_and_missing_day(tools):
    day = tools.day_predictions("2017-07-01")
    assert day["found"] is True
    assert day["weekday"] == "Saturday"
    assert day["fold"]["label"] == "Jul 2017"
    assert day["n_eval_hours"] == 24
    assert day["actual_peak_wh"] == 100.0
    assert day["actual_mean_wh"] == 100.0
    assert day["model_errors"]["xgboost"]["mae_wh"] == 8.83
    assert day["model_errors"]["sarima"]["mae_wh"] == 6.42
    assert day["hours"][9]["actual"] == 100.0
    assert day["hours"][9]["xgboost"] == 128.0

    second = tools.day_predictions("2017-07-02")
    assert second["model_errors"]["sarima"]["n"] == 0
    assert second["model_errors"]["sarima"]["mae_wh"] is None

    missing = tools.day_predictions("2018-03-01")
    assert missing["found"] is False
    assert "no evaluated hourly rows" in missing["reason"]
    assert len(missing["fold_windows"]) == 2

    assert tools.day_predictions("07/01/2017")["available"] is False
    assert tools.day_predictions("2017-02-30")["available"] is False


def test_data_quality_and_limitations(tools):
    quality = tools.data_quality(month=12)
    assert quality["source"]["rows_total"] == 100
    assert quality["stable_panel"]["count"] == 2
    assert quality["stable_panel"]["min_coverage_percent"] == 90.0
    assert quality["selected_month"]["found"] is True
    assert quality["selected_month"]["target_hours"] == 500
    assert quality["inactive"]["flagged_count"] == 2
    assert quality["inactive"]["flagged_stable_count"] == 1

    limits = tools.limitations()
    assert limits["totals"]["eval_hours"] == 96
    assert limits["totals"]["sarima_hours"] == 24
    assert limits["sarima_coverage"]["total_sarima_hours"] == 24
    assert limits["prediction_intervals"]["nominal_percent"] == 80.0
    assert limits["prediction_intervals"]["worst_fold"]["label"] == "Aug 2017"
    assert limits["prediction_intervals"]["worst_fold"]["coverage_percent"] == 50.0
    conformal = limits["prediction_intervals"]["conformal"]
    assert conformal["window_days"] == 28
    assert conformal["scoring_hour_weighted"] == {
        "before_percent": 62.5, "after_percent": 72.5,
    }
    assert conformal["by_fold"][0]["after_percent"] == 85.0
    assert conformal["by_fold"][0]["after_calibrated_percent"] == 90.0
    assert conformal["by_fold"][0]["calibrated_days"] == 1
    assert limits["december_inactivity"]["flagged_count"] == 2
    assert limits["december_inactivity"]["target_hours"] == 500
    summary = limits["december_inactivity"]["summary"]
    assert "2 apartments were flagged suspect inactive" in summary
    assert "1 of 2 stable-panel apartments" in summary
    assert "100.0 Wh/h for at least 24 hours" in summary
    assert "not a weather threshold" in summary
    assert limits["target_gaps"]["longest_gap_hours"] == 2
    assert limits["thin_2018_fold"] is None

    protocol = tools.protocol()
    assert protocol["timezone"] == "Asia/Kolkata"
    assert protocol["weather"]["hours"] == 48


def test_planner_routes_sample_questions(tools):
    plans = {
        question: tools.plan(question)
        for question in SUGGESTED_QUESTIONS
    }

    best = plans["Which model is best and by how much?"]
    assert [call.name for call in best] == ["model_metrics"]
    assert best[0].arguments["model"] is None

    december = plans["Why is December different?"]
    assert [call.name for call in december] == ["error_breakdown", "limitations"]
    assert december[0].arguments["month"] == 12
    assert december[0].arguments["model"] is None

    fail = plans["Where does XGBoost fail?"]
    assert [call.name for call in fail] == ["error_breakdown", "limitations"]
    assert fail[0].arguments["model"] == "xgboost"

    assert [call.name for call in tools.plan("2017-07-01 forecast")] == [
        "day_predictions"
    ]
    assert [call.name for call in tools.plan("What are the limitations?")] == [
        "limitations"
    ]
    assert [call.name for call in tools.plan("What data is missing?")] == [
        "data_quality"
    ]
    assert tools.plan("Tell me a joke") == []


# ── grounding guard ────────────────────────────────────────────


def test_numeric_guard_allows_rounding_and_rejects_fabrications():
    packet = {"tool_results": [{"result": {
        "mae": 81.0733,
        "negative": -0.9165,
        "date": "2017-12-01",
        "time": "09:00",
        "folds": 4,
        "beaten": 3,
        "label": "P50",
    }}]}

    assert find_unsupported_numbers(
        "SARIMA MAE is 81.07 Wh (about 81.1), with R2 of -0.9165.", packet
    ) == []
    assert find_unsupported_numbers(
        "It beat the baseline in 3 of 4 folds on 2017-12-01 at 09:00.", packet
    ) == []
    assert find_unsupported_numbers("P50 is not a new number.", packet) == []
    assert find_unsupported_numbers(
        "1. SARIMA is best.\n2. It uses 4 folds.", packet
    ) == []

    assert find_unsupported_numbers("The MAE is 999 Wh.", packet) == ["999"]
    assert find_unsupported_numbers(
        "It happened on 2017-12-02.", packet
    ) == ["2017-12-02"]
    assert find_unsupported_numbers("At 10:00 the MAE was 81.", packet) == ["10:00"]


def test_numeric_guard_checks_fold_ids_and_missing_packet():
    packet = {"tool_results": [{"result": {"id": "2017-09_12"}}]}
    assert find_unsupported_numbers("Fold 2017-09_12.", packet) == []
    assert find_unsupported_numbers("Fold 2017-07.", packet) == ["2017-07"]
    assert find_unsupported_numbers("Anything 5.", {"tool_results": []}) == ["5"]


# ── service ────────────────────────────────────────────────────


def test_service_retries_once_then_accepts_grounded_answer(tools):
    service = ChatService(tools, settings=make_settings())
    prompts = []

    def fake_provider(system, user):
        prompts.append(user)
        if len(prompts) == 1:
            return "SARIMA has a fold-mean MAE of 999 Wh."
        return "SARIMA has a fold-mean MAE of 6.4167 Wh."

    result = service.ask("Which model is best and by how much?",
                         provider=fake_provider)
    assert len(prompts) == 2
    assert result["grounded"] is True
    assert result["provider"] == "stub"
    assert result["tools"] == [{"name": "model_metrics", "arguments": {"model": None}}]
    assert "6.4167" in result["answer"]
    assert "PREVIOUS ANSWER (rejected)" in prompts[1]
    assert "999" in prompts[1]
    json.dumps(result)


def test_service_rejects_answer_that_fails_twice(tools):
    service = ChatService(tools, settings=make_settings())

    with pytest.raises(GroundingError) as excinfo:
        service.ask("Which model is best?", provider=lambda system, user: "It is 999.")
    assert excinfo.value.unsupported == ["999"]
    assert excinfo.value.attempts == 2


def test_service_prepends_grounded_period_lead(tools):
    results, quality = build_artifacts()
    rows = []
    for day in (1, 2):
        for hour in range(24):
            rows.append([f"2017-12-{day:02d}T{hour:02d}:00", 100.0, 110.0,
                         120.0, None, 90.0, 100.0, 110.0])
    results["folds"].append({
        "id": "2017-12",
        "test": {"start": "2017-12-01 00:00", "end": "2017-12-02 23:00",
                 "eval_hours": 48, "eval_days": 2},
        "models": {
            "seasonal_naive_24": _model(48, 10.0),
            "xgboost": _model(48, 20.0),
            "sarima": _model(0, None, status="no_data"),
            "xgboost_quantile": _model(48, 20.0),
        },
        "hourly": {"columns": COLUMNS, "rows": rows},
    })
    tools = ChatTools(results, quality)
    service = ChatService(tools, settings=make_settings())
    prompts = []

    def provider(system, user):
        prompts.append(user)
        return (
            "The low load is consistent with vacancy. "
            "XGBoost has MAE 20.00 Wh in December."
        )

    result = service.ask("Why is December different?", provider=provider)

    assert "ANSWER FOCUS: the question is about month 12" in prompts[0]
    assert result["answer"] == (
        "In Dec 2017, XGBoost has MAE 20.00 Wh and bias 20.00 Wh on 48 hours. "
        "The low load is consistent with vacancy. "
        "XGBoost has MAE 20.00 Wh in December."
    )


def test_service_off_topic_answer_needs_no_provider(tools):
    service = ChatService(tools, settings=make_settings())
    result = service.ask("Tell me a joke")
    assert result["provider"] == "none"
    assert result["tools"] == []
    assert result["answer"].startswith("I can only answer questions")


def test_service_validates_question(tools):
    service = ChatService(tools, settings=make_settings())
    with pytest.raises(ValueError):
        service.ask("   ")
    with pytest.raises(ValueError):
        service.ask("x" * 501)

    short = ChatService(
        tools, settings=make_settings(CHAT_MAX_QUESTION_CHARS=10)
    )
    with pytest.raises(ValueError):
        short.ask("x" * 11)


def test_status_cache_honors_configured_ttl(tools):
    now = [0.0]
    probes = []
    service = ChatService(
        tools,
        settings=make_settings(
            CHAT_PROVIDER="ollama", CHAT_STATUS_TTL_SECONDS=10.0
        ),
        clock=lambda: now[0],
    )
    service._probe_ollama = lambda: probes.append(1) or (True, "ok")

    assert service.status()["available"] is True
    assert service.status()["available"] is True
    assert len(probes) == 1

    now[0] = 11.0
    service.status()
    assert len(probes) == 2


def test_service_reports_unavailable_without_provider(tools):
    service = ChatService(
        tools,
        settings=make_settings(CHAT_PROVIDER="auto",
                               OLLAMA_URL="http://127.0.0.1:1",
                               GEMINI_API_KEY=""),
    )
    status = service.status(force=True)
    assert status["available"] is False
    assert status["provider"] is None
    assert "No GEMINI_API_KEY" in status["detail"]
    with pytest.raises(ProviderUnavailable):
        service.ask("Which model is best?")


def test_auto_status_falls_back_to_gemini(tools):
    service = ChatService(
        tools,
        settings=make_settings(CHAT_PROVIDER="auto", GEMINI_API_KEY="secret"),
    )
    service._probe_ollama = lambda: (False, "Ollama is not reachable.")
    status = service.status(force=True)
    assert status["available"] is True
    assert status["provider"] == "gemini"
    assert status["model"] == "gemini-test"
    assert "Falling back" in status["detail"]


def test_forced_gemini_status_does_not_probe_ollama(tools):
    service = ChatService(
        tools,
        settings=make_settings(CHAT_PROVIDER="gemini", GEMINI_API_KEY="secret"),
    )

    def fail_probe():
        raise AssertionError("Ollama should not be probed")

    service._probe_ollama = fail_probe
    status = service.status(force=True)
    assert status["available"] is True
    assert status["provider"] == "gemini"

    no_key = ChatService(
        tools, settings=make_settings(CHAT_PROVIDER="gemini", GEMINI_API_KEY="")
    )
    no_key._probe_ollama = fail_probe
    assert no_key.status(force=True)["available"] is False


def test_gemini_request_uses_header_and_parses_answer(tools, monkeypatch):
    calls = {}

    def fake_http(url, payload=None, headers=None, timeout=10.0):
        calls.update({"url": url, "payload": payload, "headers": headers or {}})
        return {"candidates": [{"content": {"parts": [{"text": " A grounded answer. "}]}}]}

    monkeypatch.setattr(chat_service, "_http_json", fake_http)
    service = ChatService(
        tools,
        settings=make_settings(CHAT_PROVIDER="gemini", GEMINI_API_KEY="secret-key"),
    )
    answer = service._gemini_chat("system prompt", "user prompt", 5.0)
    assert answer == "A grounded answer."
    assert calls["url"].endswith("/models/gemini-test:generateContent")
    assert "secret-key" not in calls["url"]
    assert calls["headers"]["x-goog-api-key"] == "secret-key"
    assert calls["payload"]["system_instruction"]["parts"][0]["text"] == "system prompt"
    assert calls["payload"]["generationConfig"]["temperature"] == 0


def test_provider_error_on_empty_gemini_answer(tools, monkeypatch):
    monkeypatch.setattr(
        chat_service, "_http_json",
        lambda *args, **kwargs: {"candidates": []},
    )
    service = ChatService(
        tools, settings=make_settings(CHAT_PROVIDER="gemini", GEMINI_API_KEY="k")
    )
    with pytest.raises(ProviderError):
        service._gemini_chat("system", "user", 1.0)


def _stress_fold():
    return {
        "id": "2018-05_06",
        "label": "Stress test (May\u2013Jun 2018)",
        "stress_test": True,
        "test": {
            "start": "2018-05-01 00:00", "end": "2018-05-02 23:00",
            "eval_hours": 48, "eval_days": 2,
        },
        "models": {
            "seasonal_naive_24": _model(48, 100.0),
            "xgboost": _model(48, 500.0),
            "sarima": _model(48, 400.0),
            "xgboost_quantile": _model(48, 450.0),
        },
        "hourly": {
            "columns": COLUMNS,
            "rows": _hourly_rows(5, 500.0, {
                "seasonal_naive": 100, "xgboost": 500, "p50": 450,
            }, sarima=False),
        },
    }


def test_fold_means_exclude_stress_test():
    results, quality = build_artifacts()
    results["folds"].append(_stress_fold())
    tools = ChatTools(results, quality)

    overview = tools.model_metrics()
    by_key = {row["key"]: row for row in overview["models"]}
    xgb = by_key["xgboost"]

    assert xgb["mean"]["MAE"] == 10.2083
    assert xgb["n_folds"] == 2

    per_fold = tools.model_metrics(model="xgboost")["model"]["per_fold"]
    assert per_fold[-1]["stress_test"] is True
    assert per_fold[-1]["label"].startswith("Stress test")
    assert overview["scoring_fold_ids"] == ["2017-07", "2017-08"]
    assert overview["stress_test_fold_ids"] == ["2018-05_06"]
    assert overview["best_model"]["eval_hours"] == 96
    assert overview["best_model"]["key"] == "sarima"

    fold = tools.model_metrics(fold="2018-05_06")["fold"]
    assert fold["stress_test"] is True
    assert fold["label"].startswith("Stress test")
