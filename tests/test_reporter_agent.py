"""
Tests for System1/Reporter/reporter_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis, no LLM calls.
Integration : marked @pytest.mark.integration — require live credentials
              and Docker running.

Run unit tests only:
    python -m pytest tests/test_reporter_agent.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_reporter_agent.py -v -m integration
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
    with_anomalies: bool = True,
    with_rca: bool = True,
    profile: dict | None = None,
) -> dict:
    """Return a minimal valid AgentState for reporter tests."""
    base_profile = profile or {
        "FR": {
            "n_records": 12,
            "schema": {"missing": [], "unexpected": []},
            "stats":  {"generation_solar": {"mean": 1200.0, "std": 150.0,
                                             "min": 900.0, "max": 1500.0,
                                             "p25": 1100.0, "p50": 1200.0,
                                             "p75": 1350.0, "n": 12}},
            "drift":  {"generation_solar": {"kl": 0.05, "drift_detected": False,
                                             "skipped": False, "skip_reason": None,
                                             "n_current": 12, "n_historical": 50,
                                             "threshold_used": 0.1}},
        },
        "DE": {
            "n_records": 10,
            "schema": {"missing": [], "unexpected": []},
            "stats":  {"generation_solar": {"mean": 800.0, "std": 200.0,
                                             "min": 400.0, "max": 1200.0,
                                             "p25": 650.0, "p50": 800.0,
                                             "p75": 950.0, "n": 10}},
            "drift":  {"generation_solar": {"kl": 0.31, "drift_detected": True,
                                             "skipped": False, "skip_reason": None,
                                             "n_current": 10, "n_historical": 40,
                                             "threshold_used": 0.1}},
        },
    }

    anomalies = (
        [{"severity": "MEDIUM", "variable": "generation_solar",
          "country": "DE", "rule": "KL drift exceeds threshold"}]
        if with_anomalies else []
    )

    rca_result = (
        "Likely cause: sustained high-pressure system over Central Europe "
        "reduced wind generation. Supported by 2 RAG sources."
        if with_rca else None
    )

    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       ["FR", "DE"],
        "date_from":       datetime(2024, 6, 1, 0,  tzinfo=timezone.utc),
        "date_to":         datetime(2024, 6, 1, 3,  tzinfo=timezone.utc),
        "run_type":        "full",
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    "groq",
        "tool_results":    [],
        "cycle_count":     0,
        "profile":         base_profile,
        "profile_summary": "FR: schema complete, no drift. DE: drift in generation_solar.",
        "profiling_error": None,
        "anomalies":       anomalies,
        "qa_summary":      "",
        "qa_error":        None,
        "rca_result":      rca_result,
        "rca_error":       None,
        "run_report":      "",
        "reporter_error":  None,
    }


# ===========================================================================
# TestBuildPrompt
# ===========================================================================

class TestBuildPrompt:
    """
    Why test _build_prompt in isolation?
    It is pure Python — no LLM, no DB. Verifying its output ensures the
    LLM always receives the right signals without needing to make a real
    LLM call in these tests.
    """

    def test_contains_countries(self):
        """Prompt must mention all countries in the run."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state()
        prompt = _build_prompt(state)
        assert "FR" in prompt
        assert "DE" in prompt

    def test_contains_period(self):
        """Prompt must contain the date range."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state()
        prompt = _build_prompt(state)
        assert "2024-06-01" in prompt

    def test_contains_profile_summary(self):
        """Prompt must include the profile_summary from AgentState."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state()
        prompt = _build_prompt(state)
        assert state["profile_summary"] in prompt

    def test_contains_anomaly_severity(self):
        """Prompt must list anomaly severities when anomalies exist."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state(with_anomalies=True)
        prompt = _build_prompt(state)
        assert "MEDIUM" in prompt

    def test_no_anomalies_says_none_detected(self):
        """When no anomalies, prompt must say 'None detected'."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state(with_anomalies=False)
        prompt = _build_prompt(state)
        assert "None detected" in prompt

    def test_contains_rca_when_present(self):
        """Prompt must include the rca_result text when available."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state(with_rca=True)
        prompt = _build_prompt(state)
        assert "high-pressure system" in prompt

    def test_rca_not_triggered_when_absent(self):
        """When rca_result is None, prompt must say RCA was not triggered."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state(with_rca=False)
        prompt = _build_prompt(state)
        assert "Not triggered" in prompt

    def test_contains_drift_alert(self):
        """Prompt must mention drift alerts from the profile."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state()
        prompt = _build_prompt(state)
        # DE has drift_detected=True with KL=0.31
        assert "generation_solar" in prompt
        assert "DE" in prompt

    def test_total_records_is_sum_across_countries(self):
        """Total records in prompt must equal sum of n_records across countries."""
        from System1.Reporter.reporter_agent import _build_prompt
        state = _make_state()
        prompt = _build_prompt(state)
        # FR=12 + DE=10 = 22
        assert "22" in prompt


# ===========================================================================
# TestReporterNode
# ===========================================================================

class TestReporterNode:
    """
    Why mock chat_complete and not _call_groq directly?
    reporter_node calls chat_complete() — the public interface of llm_client.
    Mocking at that level verifies the node uses the shared client without
    coupling the test to provider internals.
    """

    def test_returns_run_report_string(self):
        """reporter_node must return a non-empty run_report string."""
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            return_value=("This run processed 22 records for FR and DE.", "groq"),
        ):
            from System1.Reporter.reporter_agent import reporter_node
            result = reporter_node(state)

        assert isinstance(result["run_report"], str)
        assert len(result["run_report"]) > 0

    def test_returns_llm_provider(self):
        """reporter_node must return the provider from chat_complete."""
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            return_value=("Report text.", "groq"),
        ):
            from System1.Reporter.reporter_agent import reporter_node
            result = reporter_node(state)

        assert result["llm_provider"] == "groq"

    def test_calls_chat_complete_exactly_once(self):
        """reporter_node must make exactly one LLM call."""
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            return_value=("Report.", "groq"),
        ) as mock_llm:
            from System1.Reporter.reporter_agent import reporter_node
            reporter_node(state)

        mock_llm.assert_called_once()

    def test_fallback_on_llm_failure(self):
        """
        When chat_complete raises, reporter_node must not propagate the
        exception — it must return a Python-built fallback report instead.
        """
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            side_effect=RuntimeError("Both providers failed"),
        ):
            from System1.Reporter.reporter_agent import reporter_node
            result = reporter_node(state)

        assert "run_report" in result
        assert isinstance(result["run_report"], str)
        assert len(result["run_report"]) > 0

    def test_fallback_provider_is_none(self):
        """On LLM failure, llm_provider must be None in the returned dict."""
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            side_effect=RuntimeError("Both providers failed"),
        ):
            from System1.Reporter.reporter_agent import reporter_node
            result = reporter_node(state)

        assert result["llm_provider"] is None

    def test_fallback_report_mentions_record_count(self):
        """
        Python fallback report must include the total record count so it
        is still informative even without LLM output.
        """
        state = _make_state()

        with patch(
            "System1.Reporter.reporter_agent.chat_complete",
            side_effect=RuntimeError("fail"),
        ):
            from System1.Reporter.reporter_agent import reporter_node
            result = reporter_node(state)

        # FR=12 + DE=10 = 22 total records
        assert "22" in result["run_report"]


# ===========================================================================
# TestSaveReporterNode
# ===========================================================================

class TestSaveReporterNode:
    """
    Why patch engine.connect and get_redis?
    save_reporter_node has two external dependencies. Patching at the engine
    level lets us verify the correct SQL UPDATE is attempted and the correct
    Redis event is published without requiring live services.
    """

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def test_no_error_on_success(self):
        """reporter_error must be None when the DB update succeeds."""
        state = _make_state()
        state["run_report"] = "Fluent executive report text."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            result = save_reporter_node(state)

        assert result["reporter_error"] is None

    def test_publishes_to_redis(self):
        """save_reporter_node must publish exactly one message to Redis."""
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        mock_redis.publish.assert_called_once()

    def test_redis_event_is_system1_complete(self):
        """Redis payload event must be 'system1_complete'."""
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["event"] == "system1_complete"

    def test_redis_payload_contains_required_fields(self):
        """Redis message must contain run_id, event, n_records, countries, run_report, timestamp."""
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "n_records", "countries", "run_report", "timestamp"):
            assert field in parsed, f"Missing field in Redis payload: {field}"

    def test_redis_payload_n_records_is_sum_across_countries(self):
        """n_records in Redis payload must equal sum of n_records across countries (22)."""
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["n_records"] == 22

    def test_error_captured_on_db_failure(self):
        """reporter_error must be set (not raised) when the DB update fails."""
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        mock_redis = MagicMock()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            result = save_reporter_node(state)

        assert result["reporter_error"] is not None
        assert "DB unavailable" in result["reporter_error"]

    def test_redis_still_publishes_on_db_failure(self):
        """
        Redis publish must happen even when the DB update fails —
        Sistema 2 must always be notified that Sistema 1 finished.
        """
        state = _make_state()
        state["run_report"] = "Report."
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        mock_redis = MagicMock()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        mock_redis.publish.assert_called_once()

    def test_run_report_is_included_in_redis_payload(self):
        """The full run_report text must be present in the Redis payload."""
        state = _make_state()
        state["run_report"] = "Unique report content for this test."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Reporter.reporter_agent.engine",    mock_engine),
            patch("System1.Reporter.reporter_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Reporter.reporter_agent import save_reporter_node
            save_reporter_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert "Unique report content for this test." in json.loads(payload)["run_report"]


# ===========================================================================
# Integration tests — require live credentials + Docker running
# ===========================================================================

@pytest.mark.integration
class TestReporterIntegration:
    """
    End-to-end run of the Reporter Agent graph.
    Requires: GROQ_API_KEY, PostgreSQL + Redis running, and a pre-existing
    data_quality_runs row for the run_id (normally created by save_profile_node).

    For standalone testing, the test inserts its own row first and cleans up
    afterward.
    """

    def test_full_reporter_run(self):
        """Reporter must produce a non-empty run_report and no reporter_error."""
        from shared.db import engine
        from sqlalchemy import text
        from System1.Reporter.reporter_agent import invoke_reporter_graph

        state = _make_state()
        run_id = state["run_id"]

        # Insert a minimal data_quality_runs row so UPDATE has a target
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO data_quality_runs
                        (run_id, started_at, source_api, n_records, status)
                    VALUES
                        (:run_id, NOW(), 'system1_reporter_test', 0, 'running')
                """),
                {"run_id": run_id},
            )
            conn.commit()

        try:
            result = invoke_reporter_graph(state)
            assert result["reporter_error"] is None
            assert isinstance(result["run_report"], str)
            assert len(result["run_report"]) > 50  # must be more than a stub
        finally:
            # Clean up test row
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM data_quality_runs WHERE run_id = :run_id"),
                    {"run_id": run_id},
                )
                conn.commit()

    def test_run_report_mentions_countries(self):
        """LLM report must mention at least one of the countries in the run."""
        from shared.db import engine
        from sqlalchemy import text
        from System1.Reporter.reporter_agent import invoke_reporter_graph

        state = _make_state()
        run_id = state["run_id"]

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO data_quality_runs
                        (run_id, started_at, source_api, n_records, status)
                    VALUES
                        (:run_id, NOW(), 'system1_reporter_test', 0, 'running')
                """),
                {"run_id": run_id},
            )
            conn.commit()

        try:
            result = invoke_reporter_graph(state)
            report = result["run_report"]
            countries = state["countries"]
            country_mentions = {"FR": "France", "DE": "Germany", "ES": "Spain"}
            assert any(
                c in report or country_mentions.get(c, c) in report
                for c in countries
            ), f"Report must mention at least one of {countries}"
        finally:
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM data_quality_runs WHERE run_id = :run_id"),
                    {"run_id": run_id},
                )
                conn.commit()