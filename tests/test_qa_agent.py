"""
Tests for System1/QA/qa_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis, no LLM calls.
Integration : marked @pytest.mark.integration — require live credentials
              and Docker running.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    run_type: str = "full",
    countries: list[str] | None = None,
    profile: dict | None = None,
    records: list[dict] | None = None,
    anomalies: list[dict] | None = None,
    qa_severity: str | None = None,
) -> dict:
    """Return a minimal valid AgentState dict for QA tests."""
    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       countries or ["FR"],
        "date_from":       datetime(2024, 6, 1, 0,  tzinfo=timezone.utc),
        "date_to":         datetime(2024, 6, 1, 3,  tzinfo=timezone.utc),
        "run_type":        run_type,
        "messages":        [],
        "records":         records or [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],
        "cycle_count":     0,
        "profile":         profile or {},
        "profile_summary": "",
        "profiling_error": None,
        "anomalies":       anomalies or [],
        "qa_severity":     qa_severity,
        "qa_error":        None,
        "qa_summary":      "",
    }


def _make_records(
    country: str = "FR",
    variable: str = "generation_solar",
    source_api: str = "entsoe",
    n: int = 3,
    base_value: float = 1000.0,
) -> list[dict]:
    """Generate synthetic records compatible with energy_climate_records schema."""
    return [
        {
            "timestamp":  datetime(2024, 6, 1, i, tzinfo=timezone.utc),
            "source_api": source_api,
            "country":    country,
            "variable":   variable,
            "value":      base_value + i * 10.0,
            "unit":       "MW",
            "metadata":   {},
        }
        for i in range(n)
    ]


def _make_clean_profile(country: str = "FR") -> dict:
    """Return a profile with no drift and no missing variables."""
    return {
        country: {
            "n_records": 12,
            "schema": {"missing": [], "unexpected": []},
            "stats":  {"generation_solar": {"mean": 1000.0, "std": 50.0,
                                             "min": 900.0,  "max": 1100.0,
                                             "p25": 950.0,  "p50": 1000.0,
                                             "p75": 1050.0, "n": 3}},
            "drift":  {"generation_solar": {"kl": 0.02, "drift_detected": False,
                                             "skipped": False, "skip_reason": None,
                                             "n_current": 3, "n_historical": 20,
                                             "threshold_used": 0.1}},
        }
    }


def _make_drift_profile(country: str = "FR") -> dict:
    """Return a profile with drift detected."""
    return {
        country: {
            "n_records": 12,
            "schema": {"missing": [], "unexpected": []},
            "stats":  {"generation_solar": {"mean": 5000.0, "std": 200.0,
                                             "min": 4600.0, "max": 5400.0,
                                             "p25": 4800.0, "p50": 5000.0,
                                             "p75": 5200.0, "n": 12}},
            "drift":  {"generation_solar": {"kl": 0.35, "drift_detected": True,
                                             "skipped": False, "skip_reason": None,
                                             "n_current": 12, "n_historical": 50,
                                             "threshold_used": 0.1}},
        }
    }


_MINIMAL_RULES = {
    "non_negative_variables": ["generation_solar", "climate_solar_radiation"],
    "completeness": {
        "min_ratio_critical": 0.50,
        "min_ratio_medium":   0.80,
        "min_ratio_low":      0.95,
    },
    "drift":             {"severity": "MEDIUM"},
    "missing_variables": {"severity": "MEDIUM"},
}


# ===========================================================================
# TestValidateBusinessRules
# ===========================================================================

class TestValidateBusinessRules:
    def test_no_violations_for_positive_values(self):
        """Positive values on non-negative variables must produce no violations."""
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "generation_solar", base_value=500.0)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert result == []

    def test_negative_value_produces_critical_violation(self):
        """A negative value on a non-negative variable must produce a CRITICAL violation."""
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "generation_solar", base_value=-10.0)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert len(result) > 0
        assert all(v["severity"] == "CRITICAL" for v in result)
        assert all(v["rule"] == "non_negative" for v in result)

    def test_negative_value_on_allowed_variable_not_flagged(self):
        """
        A negative value on a variable not in non_negative_variables must
        not produce a violation (e.g. temperature can be negative).
        """
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "climate_temperature_2m", base_value=-5.0)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert result == []

    def test_violation_contains_required_keys(self):
        """Each violation dict must contain rule, country, variable, value, severity, detail."""
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "generation_solar", base_value=-1.0, n=1)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert len(result) == 1
        for key in ("rule", "country", "variable", "value", "severity", "detail"):
            assert key in result[0], f"Missing key: {key}"

    def test_zero_value_not_flagged(self):
        """Zero is not negative — must not produce a violation."""
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "generation_solar", base_value=0.0)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert result == []

    def test_empty_records_returns_empty(self):
        """Empty record list must produce no violations."""
        from System1.QA.qa_agent import validate_business_rules
        result = validate_business_rules([], _MINIMAL_RULES)
        assert result == []

    def test_multiple_negative_records_all_flagged(self):
        """All records with negative values must be flagged independently."""
        from System1.QA.qa_agent import validate_business_rules
        records = _make_records("FR", "generation_solar", base_value=-100.0, n=3)
        result = validate_business_rules(records, _MINIMAL_RULES)
        assert len(result) == 3


# ===========================================================================
# TestCheckCompleteness
# ===========================================================================

class TestCheckCompleteness:
    _DATE_FROM = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
    _DATE_TO   = datetime(2024, 6, 1, 3, tzinfo=timezone.utc)  # 3-hour window

    def test_full_completeness_no_violation(self):
        """
        3-hour window, 1 variable, 3 records → ratio = 1.0 → no violation.
        """
        from System1.QA.qa_agent import check_completeness
        records = _make_records("FR", "generation_solar", n=3)
        result = check_completeness(
            records, ["FR"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        assert result == []

    def test_low_completeness_produces_low_severity(self):
        """
        3-hour window, 1 variable, 2 records → ratio 67% → between 50-80% → MEDIUM.
        But 2/3 ≈ 67%, which is < 80% (medium threshold) and > 50% (critical) → MEDIUM.
        """
        from System1.QA.qa_agent import check_completeness
        records = _make_records("FR", "generation_solar", n=2)  # 2 of 3 expected
        result = check_completeness(
            records, ["FR"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        assert len(result) == 1
        assert result[0]["severity"] == "MEDIUM"

    def test_very_low_completeness_produces_critical(self):
        """
        3-hour window, 1 variable, 1 record → ratio 33% → < 50% → CRITICAL.
        """
        from System1.QA.qa_agent import check_completeness
        records = _make_records("FR", "generation_solar", n=1)
        result = check_completeness(
            records, ["FR"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        assert len(result) == 1
        assert result[0]["severity"] == "CRITICAL"

    def test_violation_contains_ratio_and_counts(self):
        """Violation dict must contain ratio, received, and expected keys."""
        from System1.QA.qa_agent import check_completeness
        records = _make_records("FR", "generation_solar", n=1)
        result = check_completeness(
            records, ["FR"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        assert len(result) == 1
        for key in ("ratio", "received", "expected"):
            assert key in result[0], f"Missing key: {key}"

    def test_empty_records_produces_no_violation(self):
        """
        If there are no records at all for a country, check_completeness
        has no pair to evaluate — must return empty list.
        (Missing variables are flagged by flag_anomalies via the profile.)
        """
        from System1.QA.qa_agent import check_completeness
        result = check_completeness(
            [], ["FR"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        assert result == []

    def test_multiple_countries_checked_independently(self):
        """Each country must be evaluated independently."""
        from System1.QA.qa_agent import check_completeness
        records = (
            _make_records("FR", "generation_solar", n=3)
            + _make_records("DE", "generation_solar", n=1)
        )
        result = check_completeness(
            records, ["FR", "DE"], self._DATE_FROM, self._DATE_TO, _MINIMAL_RULES
        )
        countries_with_violations = {v["country"] for v in result}
        assert "DE" in countries_with_violations
        assert "FR" not in countries_with_violations


# ===========================================================================
# TestFlagAnomalies
# ===========================================================================

class TestFlagAnomalies:

    def test_no_inputs_returns_empty_and_none_severity(self):
        """Empty violations + clean profile → empty anomaly list, None severity."""
        from System1.QA.qa_agent import flag_anomalies
        anomalies, severity = flag_anomalies([], _make_clean_profile("FR"), _MINIMAL_RULES)
        assert anomalies == []
        assert severity is None

    def test_drift_alert_produces_medium_anomaly(self):
        """drift_detected=True in profile must produce a MEDIUM anomaly."""
        from System1.QA.qa_agent import flag_anomalies
        anomalies, severity = flag_anomalies([], _make_drift_profile("FR"), _MINIMAL_RULES)
        drift_anomalies = [a for a in anomalies if a["rule"] == "drift"]
        assert len(drift_anomalies) == 1
        assert drift_anomalies[0]["severity"] == "MEDIUM"

    def test_missing_variable_produces_medium_anomaly(self):
        """A missing variable in the profile must produce a MEDIUM anomaly."""
        from System1.QA.qa_agent import flag_anomalies
        profile = {
            "FR": {
                "schema": {"missing": ["entsoe:load_actual_aggregated"], "unexpected": []},
                "drift":  {},
            }
        }
        anomalies, _ = flag_anomalies([], profile, _MINIMAL_RULES)
        missing_anomalies = [a for a in anomalies if a["rule"] == "missing_variable"]
        assert len(missing_anomalies) == 1

    def test_critical_rule_violation_dominates_severity(self):
        """
        When a CRITICAL rule violation and a MEDIUM drift alert coexist,
        max_severity must be CRITICAL.
        """
        from System1.QA.qa_agent import flag_anomalies
        critical_violation = [{
            "rule": "non_negative", "country": "FR",
            "variable": "generation_solar", "value": -1.0,
            "severity": "CRITICAL", "detail": "negative value",
        }]
        anomalies, severity = flag_anomalies(
            critical_violation, _make_drift_profile("FR"), _MINIMAL_RULES
        )
        assert severity == "CRITICAL"

    def test_all_anomaly_dicts_have_required_keys(self):
        """Every anomaly dict must contain rule, country, severity, detail."""
        from System1.QA.qa_agent import flag_anomalies
        violation = [{
            "rule": "non_negative", "country": "FR",
            "variable": "generation_solar", "value": -1.0,
            "severity": "CRITICAL", "detail": "negative",
        }]
        anomalies, _ = flag_anomalies(violation, _make_drift_profile("FR"), _MINIMAL_RULES)
        for a in anomalies:
            for key in ("rule", "country", "severity", "detail"):
                assert key in a, f"Missing key '{key}' in anomaly: {a}"

    def test_skipped_drift_not_flagged(self):
        """
        drift_detected=False (even if skipped=False) must not produce an anomaly.
        """
        from System1.QA.qa_agent import flag_anomalies
        anomalies, severity = flag_anomalies([], _make_clean_profile("FR"), _MINIMAL_RULES)
        drift_anomalies = [a for a in anomalies if a["rule"] == "drift"]
        assert drift_anomalies == []


# ===========================================================================
# TestQaNode
# ===========================================================================

class TestQaNode:
    def test_clean_run_produces_no_anomalies(self):
        """A clean batch with a clean profile must produce an empty anomaly list."""
        state = _make_state(
            records=_make_records("FR", "generation_solar", n=3, base_value=1000.0),
            profile=_make_clean_profile("FR"),
        )
        with patch("System1.QA.qa_agent._load_rules", return_value=_MINIMAL_RULES):
            from System1.QA.qa_agent import qa_node
            result = qa_node(state)

        assert result["anomalies"] == []
        assert result["qa_severity"] is None

    def test_negative_value_produces_critical(self):
        """Negative value in a non-negative variable must set qa_severity=CRITICAL."""
        state = _make_state(
            records=_make_records("FR", "generation_solar", n=3, base_value=-50.0),
            profile=_make_clean_profile("FR"),
        )
        with patch("System1.QA.qa_agent._load_rules", return_value=_MINIMAL_RULES):
            from System1.QA.qa_agent import qa_node
            result = qa_node(state)

        assert result["qa_severity"] == "CRITICAL"
        assert any(a["rule"] == "non_negative" for a in result["anomalies"])

    def test_drift_profile_produces_medium(self):
        """A profile with drift detected must produce qa_severity=MEDIUM."""
        state = _make_state(
            records=_make_records("FR", "generation_solar", n=3),
            profile=_make_drift_profile("FR"),
        )
        with patch("System1.QA.qa_agent._load_rules", return_value=_MINIMAL_RULES):
            from System1.QA.qa_agent import qa_node
            result = qa_node(state)

        assert result["qa_severity"] == "MEDIUM"
        assert any(a["rule"] == "drift" for a in result["anomalies"])

    def test_returns_anomalies_and_severity_keys(self):
        """qa_node must always return both anomalies and qa_severity keys."""
        state = _make_state(profile=_make_clean_profile("FR"))
        with patch("System1.QA.qa_agent._load_rules", return_value=_MINIMAL_RULES):
            from System1.QA.qa_agent import qa_node
            result = qa_node(state)

        assert "anomalies"   in result
        assert "qa_severity" in result


# ===========================================================================
# TestSummaryNode
# ===========================================================================

class TestSummaryNode:

    def test_no_anomalies_skips_llm(self):
        """When anomalies is empty, summary_node must not call the LLM."""
        state = _make_state(anomalies=[], qa_severity=None)
        with patch("System1.QA.qa_agent.chat_complete") as mock_llm:
            from System1.QA.qa_agent import summary_node
            result = summary_node(state)

        mock_llm.assert_not_called()
        assert "passed" in result["qa_summary"].lower()

    def test_anomalies_present_calls_llm_once(self):
        """When anomalies exist, summary_node must call chat_complete exactly once."""
        anomalies = [{"severity": "MEDIUM", "detail": "drift in solar", "rule": "drift",
                      "country": "FR"}]
        state = _make_state(anomalies=anomalies, qa_severity="MEDIUM")

        with patch("System1.QA.qa_agent.chat_complete",
                   return_value=("Data quality issues found.", "groq")) as mock_llm:
            from System1.QA.qa_agent import summary_node
            result = summary_node(state)

        mock_llm.assert_called_once()
        assert result["qa_summary"] == "Data quality issues found."

    def test_llm_failure_returns_fallback_summary(self):
        """If chat_complete raises, summary_node must not propagate the exception."""
        anomalies = [{"severity": "CRITICAL", "detail": "negative value", "rule": "non_negative",
                      "country": "FR"}]
        state = _make_state(anomalies=anomalies, qa_severity="CRITICAL")

        with patch("System1.QA.qa_agent.chat_complete",
                   side_effect=RuntimeError("Both providers failed")):
            from System1.QA.qa_agent import summary_node
            result = summary_node(state)

        assert "qa_summary" in result
        assert result["llm_provider"] is None

    def test_stores_llm_provider(self):
        """llm_provider must be set from chat_complete return value."""
        anomalies = [{"severity": "MEDIUM", "detail": "drift", "rule": "drift",
                      "country": "FR"}]
        state = _make_state(anomalies=anomalies, qa_severity="MEDIUM")

        with patch("System1.QA.qa_agent.chat_complete",
                   return_value=("Summary text.", "groq")):
            from System1.QA.qa_agent import summary_node
            result = summary_node(state)

        assert result["llm_provider"] == "groq"


# ===========================================================================
# TestSaveQaNode
# ===========================================================================

class TestSaveQaNode:

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def test_no_error_on_success(self):
        """qa_error must be None when the DB update succeeds."""
        anomalies = [{"severity": "MEDIUM", "rule": "drift", "detail": "drift in solar",
                      "country": "FR"}]
        state = _make_state(anomalies=anomalies, qa_severity="MEDIUM")
        state["qa_summary"] = "Some issues found."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            result = save_qa_node(state)

        assert result["qa_error"] is None

    def test_publishes_to_redis(self):
        """save_qa_node must publish exactly one message to Redis."""
        state = _make_state(anomalies=[], qa_severity=None)
        state["qa_summary"] = "All checks passed."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            save_qa_node(state)

        mock_redis.publish.assert_called_once()

    def test_redis_payload_event_is_qa_complete(self):
        """event field in Redis payload must be 'qa_complete'."""
        state = _make_state(anomalies=[], qa_severity=None)
        state["qa_summary"] = "Clean run."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            save_qa_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["event"] == "qa_complete"

    def test_redis_payload_contains_required_fields(self):
        """Redis payload must contain run_id, event, n_anomalies, qa_severity, countries, timestamp."""
        state = _make_state(anomalies=[], qa_severity=None)
        state["qa_summary"] = "Clean."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            save_qa_node(state)

        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "n_anomalies", "qa_severity", "countries", "timestamp"):
            assert field in parsed, f"Missing field: {field}"

    def test_error_captured_on_db_failure(self):
        """qa_error must be set (not raised) when the DB update fails."""
        state = _make_state(anomalies=[], qa_severity=None)
        state["qa_summary"] = "Clean."
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            result = save_qa_node(state)

        assert result["qa_error"] is not None
        assert "DB unavailable" in result["qa_error"]

    def test_redis_publishes_even_on_db_failure(self):
        """Redis publish must happen even when the DB update fails."""
        state = _make_state(anomalies=[], qa_severity=None)
        state["qa_summary"] = "Clean."
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System1.QA.qa_agent.engine",    mock_engine),
            patch("System1.QA.qa_agent.get_redis", return_value=mock_redis),
        ):
            from System1.QA.qa_agent import save_qa_node
            save_qa_node(state)

        mock_redis.publish.assert_called_once()


# ===========================================================================
# Integration tests — require live credentials + Docker running
# ===========================================================================

@pytest.mark.integration
class TestQaIntegration:
    """
    End-to-end run of the QA Agent graph.
    Records are seeded directly into AgentState to avoid depending on
    the Ingestion and Profiling Agents running first.
    """

    def test_clean_run_fr(self):
        """QA run with clean data must produce no anomalies and a clean summary."""
        from System1.QA.qa_agent import invoke_qa_graph
        state = _make_state(
            records=(
                _make_records("FR", "generation_solar",        n=3, base_value=1000.0)
                + _make_records("FR", "load_actual_aggregated", n=3, base_value=50000.0)
            ),
            profile=_make_clean_profile("FR"),
        )
        result = invoke_qa_graph(state)

        assert result["qa_error"] is None
        assert result["qa_severity"] is None
        assert isinstance(result["qa_summary"], str)

    def test_drift_run_produces_medium_severity(self):
        """QA run with drift in profile must produce qa_severity=MEDIUM."""
        from System1.QA.qa_agent import invoke_qa_graph
        state = _make_state(
            records=_make_records("FR", "generation_solar", n=3),
            profile=_make_drift_profile("FR"),
        )
        result = invoke_qa_graph(state)

        assert result["qa_error"] is None
        assert result["qa_severity"] == "MEDIUM"
        assert "FR" in result["qa_summary"] or len(result["qa_summary"]) > 0