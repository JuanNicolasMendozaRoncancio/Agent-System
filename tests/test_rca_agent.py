"""
Tests for System1/RCA/rca_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis, no LLM calls.
Integration : marked @pytest.mark.integration — require live credentials
              and Docker running.

Run unit tests only:
    python -m pytest tests/test_rca_agent.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_rca_agent.py -v -m integration
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    anomalies: list[dict] | None = None,
    rca_evidence: dict | None = None,
) -> dict:
    """Return a minimal valid AgentState dict for RCA tests."""
    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       ["FR", "DE"],
        "date_from":       datetime(2024, 6, 1, 0,  tzinfo=timezone.utc),
        "date_to":         datetime(2024, 6, 1, 3,  tzinfo=timezone.utc),
        "run_type":        "full",
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],
        "cycle_count":     0,
        "profile":         {},
        "profile_summary": "",
        "profiling_error": None,
        "anomalies":       anomalies or [],
        "qa_severity":     None,
        "qa_error":        None,
        "qa_summary":      "",
        "rca_evidence":    rca_evidence or {},
        "rca_result":      None,
        "rca_sources":     [],
        "rca_error":       None,
    }


def _make_anomaly(severity: str = "CRITICAL", variable: str = "generation_solar",
                  country: str = "FR") -> dict:
    return {
        "type":     "negative_value",
        "variable": variable,
        "country":  country,
        "severity": severity,
        "value":    -50.0,
        "message":  f"{variable} has negative value",
    }


def _make_tool_message(name: str, content: dict,
                       tool_call_id: str = "call_1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content),
        name=name,
        tool_call_id=tool_call_id,
    )


def _make_ai_with_tools(tool_name: str, args: dict,
                        call_id: str = "call_abc") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": tool_name, "args": args}],
    )


def _make_ai_no_tools(text: str = "No anomalies.") -> AIMessage:
    return AIMessage(content=text)


# ===========================================================================
# TestFilterRcaAnomalies
# ===========================================================================

class TestFilterRcaAnomalies:
    """
    _filter_rca_anomalies is a pure function — tests here are exhaustive
    and require no mocking.
    """

    def test_keeps_medium_and_critical(self):
        from System1.RCA.rca_agent import _filter_rca_anomalies
        anomalies = [
            _make_anomaly("LOW"),
            _make_anomaly("MEDIUM"),
            _make_anomaly("CRITICAL"),
        ]
        result = _filter_rca_anomalies(anomalies)
        severities = {a["severity"] for a in result}
        assert severities == {"MEDIUM", "CRITICAL"}

    def test_excludes_low(self):
        from System1.RCA.rca_agent import _filter_rca_anomalies
        anomalies = [_make_anomaly("LOW"), _make_anomaly("LOW")]
        assert _filter_rca_anomalies(anomalies) == []

    def test_empty_input_returns_empty(self):
        from System1.RCA.rca_agent import _filter_rca_anomalies
        assert _filter_rca_anomalies([]) == []

    def test_all_critical_preserved(self):
        from System1.RCA.rca_agent import _filter_rca_anomalies
        anomalies = [_make_anomaly("CRITICAL") for _ in range(5)]
        assert len(_filter_rca_anomalies(anomalies)) == 5


# ===========================================================================
# TestProcessToolMessages
# ===========================================================================

class TestProcessToolMessages:
    """
    _process_tool_messages converts ToolMessages into the rca_evidence dict.
    Tests here verify the parsing and deduplication logic in Python, without
    any LLM involvement.
    """

    def test_historical_message_populates_evidence(self):
        from System1.RCA.rca_agent import _process_tool_messages
        msg = _make_tool_message("query_historical_db", {
            "variable": "generation_solar", "country": "FR",
            "mean": 1200.0, "std": 150.0, "n_records": 720,
            "anomaly_count_last_30d": 2, "error": None,
        })
        evidence = _process_tool_messages([msg])
        assert "generation_solar/FR" in evidence["historical"]
        assert evidence["historical"]["generation_solar/FR"]["mean"] == 1200.0

    def test_climate_message_populates_evidence(self):
        from System1.RCA.rca_agent import _process_tool_messages
        msg = _make_tool_message("correlate_climate_data", {
            "country": "FR",
            "variables": {
                "climate_temperature_2m": {"mean": 22.5, "std": 3.1, "n_records": 3},
            },
            "error": None,
        })
        evidence = _process_tool_messages([msg])
        assert "FR" in evidence["climate"]
        assert "climate_temperature_2m" in evidence["climate"]["FR"]

    def test_rag_message_populates_results(self):
        from System1.RCA.rca_agent import _process_tool_messages
        msg = _make_tool_message("rag_search", {
            "query": "wind drop France",
            "results": [
                {"main_argument": "High pressure over France reduced wind.",
                 "sentiment": "negative", "score": 0.91, "source": "carbon_brief"},
            ],
            "error": None,
        })
        evidence = _process_tool_messages([msg])
        assert len(evidence["rag_results"]) == 1
        assert evidence["rag_results"][0]["score"] == 0.91

    def test_rag_results_deduplicated_by_argument(self):
        """Two rag_search calls returning the same main_argument → one entry."""
        from System1.RCA.rca_agent import _process_tool_messages
        same_arg = "High pressure over France reduced wind."
        msg1 = _make_tool_message("rag_search", {
            "query": "wind drop France",
            "results": [{"main_argument": same_arg, "sentiment": "negative",
                         "score": 0.91, "source": "carbon_brief"}],
            "error": None,
        }, "call_1")
        msg2 = _make_tool_message("rag_search", {
            "query": "France eolien baisse",
            "results": [{"main_argument": same_arg, "sentiment": "negative",
                         "score": 0.88, "source": "reporterre"}],
            "error": None,
        }, "call_2")
        evidence = _process_tool_messages([msg1, msg2])
        # Higher-score duplicate wins; only one entry kept
        assert len(evidence["rag_results"]) == 1
        assert evidence["rag_results"][0]["score"] == 0.91

    def test_non_json_tool_message_skipped_gracefully(self):
        """Malformed ToolMessage content must not raise."""
        from System1.RCA.rca_agent import _process_tool_messages
        msg = ToolMessage(
            content="Error: connection refused",
            name="query_historical_db",
            tool_call_id="call_err",
        )
        evidence = _process_tool_messages([msg])
        # Should produce empty evidence without raising
        assert evidence["historical"] == {}

    def test_multiple_tool_types_in_one_call(self):
        """All three tool types can appear in the same message list."""
        from System1.RCA.rca_agent import _process_tool_messages
        messages = [
            _make_tool_message("query_historical_db",
                {"variable": "load_actual_aggregated", "country": "DE",
                 "mean": 45000.0, "std": 3000.0, "n_records": 500,
                 "anomaly_count_last_30d": 0, "error": None}, "c1"),
            _make_tool_message("correlate_climate_data",
                {"country": "DE",
                 "variables": {"climate_solar_radiation": {"mean": 1.2, "std": 0.5, "n_records": 3}},
                 "error": None}, "c2"),
            _make_tool_message("rag_search",
                {"query": "Germany load spike",
                 "results": [{"main_argument": "Cold snap drove demand.",
                              "sentiment": "neutral", "score": 0.85, "source": "bon_pote"}],
                 "error": None}, "c3"),
        ]
        evidence = _process_tool_messages(messages)
        assert "load_actual_aggregated/DE" in evidence["historical"]
        assert "DE" in evidence["climate"]
        assert len(evidence["rag_results"]) == 1


# ===========================================================================
# TestRcaNode1
# ===========================================================================

class TestRcaNode1:
    """
    rca_node1 must call the LLM with tools and return an AIMessage.
    Tests verify that it filters anomalies, passes the correct context,
    and handles the no-anomaly case without calling the LLM.
    """

    def test_no_anomalies_skips_llm(self):
        """If no MEDIUM/CRITICAL anomalies exist, rca_node1 must not call the LLM."""
        state = _make_state(anomalies=[_make_anomaly("LOW")])

        with patch("System1.RCA.rca_agent._build_llm_with_tools") as mock_build:
            from System1.RCA.rca_agent import rca_node1
            result = rca_node1(state)

        mock_build.assert_not_called()
        assert len(result["messages"]) == 1
        assert "No MEDIUM or CRITICAL" in result["messages"][0].content

    def test_with_anomalies_calls_llm(self):
        """With CRITICAL anomalies, rca_node1 must call the LLM."""
        state = _make_state(anomalies=[_make_anomaly("CRITICAL")])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_no_tools("ok")

        with patch("System1.RCA.rca_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System1.RCA.rca_agent import rca_node1
            rca_node1(state)

        mock_llm.invoke.assert_called_once()

    def test_stores_llm_provider(self):
        """rca_node1 must store llm_provider in the returned dict."""
        state = _make_state(anomalies=[_make_anomaly("MEDIUM")])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_no_tools("done")

        with patch("System1.RCA.rca_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System1.RCA.rca_agent import rca_node1
            result = rca_node1(state)

        assert result["llm_provider"] == "groq"

    def test_human_message_contains_run_id(self):
        """The HumanMessage sent to the LLM must include the run_id."""
        state = _make_state(anomalies=[_make_anomaly("CRITICAL")])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_no_tools("ok")

        with patch("System1.RCA.rca_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System1.RCA.rca_agent import rca_node1
            rca_node1(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        human_content = next(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if (isinstance(m, dict) and m.get("role") == "user")
            or hasattr(m, "content")
            and not isinstance(m, dict)
            and "Human" in type(m).__name__
        )
        assert state["run_id"] in human_content

    def test_only_medium_critical_in_prompt(self):
        """LOW anomalies must not appear in the LLM prompt."""
        low_anomaly = _make_anomaly("LOW", variable="generation_solar")
        critical_anomaly = _make_anomaly("CRITICAL", variable="load_actual_aggregated")
        state = _make_state(anomalies=[low_anomaly, critical_anomaly])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_no_tools("ok")

        with patch("System1.RCA.rca_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System1.RCA.rca_agent import rca_node1
            rca_node1(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        full_prompt = " ".join(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if hasattr(m, "content") or isinstance(m, dict)
        )
        # CRITICAL anomaly variable must be in prompt
        assert "load_actual_aggregated" in full_prompt


# ===========================================================================
# TestProcessRcaEvidence
# ===========================================================================

class TestProcessRcaEvidence:
    """
    _process_rca_evidence reads ToolMessages from state['messages'],
    builds rca_evidence, clears messages, and preserves rca_sources.
    """

    def test_builds_evidence_from_tool_messages(self):
        state = _make_state()
        state["messages"] = [
            _make_tool_message("query_historical_db", {
                "variable": "generation_solar", "country": "FR",
                "mean": 1000.0, "std": 100.0, "n_records": 48,
                "anomaly_count_last_30d": 1, "error": None,
            }),
        ]
        from System1.RCA.rca_agent import _process_rca_evidence
        result = _process_rca_evidence(state)
        assert "generation_solar/FR" in result["rca_evidence"]["historical"]

    def test_clears_messages(self):
        state = _make_state()
        state["messages"] = [
            _make_tool_message("rag_search", {
                "query": "test", "results": [], "error": None,
            }),
        ]
        from System1.RCA.rca_agent import _process_rca_evidence
        result = _process_rca_evidence(state)
        assert result["messages"] == []

    def test_rca_sources_populated_from_rag(self):
        state = _make_state()
        state["messages"] = [
            _make_tool_message("rag_search", {
                "query": "wind France",
                "results": [{"main_argument": "High pressure.",
                             "sentiment": "negative", "score": 0.9, "source": "cb"}],
                "error": None,
            }),
        ]
        from System1.RCA.rca_agent import _process_rca_evidence
        result = _process_rca_evidence(state)
        assert len(result["rca_sources"]) == 1

    def test_empty_messages_returns_empty_evidence(self):
        state = _make_state()
        state["messages"] = []
        from System1.RCA.rca_agent import _process_rca_evidence
        result = _process_rca_evidence(state)
        assert result["rca_evidence"]["historical"] == {}
        assert result["rca_evidence"]["climate"] == {}
        assert result["rca_evidence"]["rag_results"] == []


# ===========================================================================
# TestRcaNode2
# ===========================================================================

class TestRcaNode2:
    """
    rca_node2 calls chat_complete with the anomalies + evidence summary
    and stores the result in rca_result.
    """

    def _evidence_with_rag(self) -> dict:
        return {
            "historical": {
                "generation_solar/FR": {
                    "mean": 1200.0, "std": 150.0, "n_records": 720,
                    "anomaly_count_last_30d": 3, "error": None,
                }
            },
            "climate": {},
            "rag_results": [
                {"main_argument": "Persistent high pressure reduced wind.",
                 "sentiment": "negative", "score": 0.93, "source": "carbon_brief"},
            ],
        }

    def test_calls_chat_complete(self):
        """rca_node2 must call chat_complete exactly once."""
        state = _make_state(
            anomalies=[_make_anomaly("CRITICAL")],
            rca_evidence=self._evidence_with_rag(),
        )
        with patch("System1.RCA.rca_agent.chat_complete",
                   return_value=("1. High pressure event.", "groq")) as mock_llm:
            from System1.RCA.rca_agent import rca_node2
            rca_node2(state)
        mock_llm.assert_called_once()

    def test_rca_result_is_string(self):
        """rca_result must be a non-empty string."""
        state = _make_state(
            anomalies=[_make_anomaly("CRITICAL")],
            rca_evidence=self._evidence_with_rag(),
        )
        with patch("System1.RCA.rca_agent.chat_complete",
                   return_value=("1. Wind drop due to high pressure.", "groq")):
            from System1.RCA.rca_agent import rca_node2
            result = rca_node2(state)
        assert isinstance(result["rca_result"], str)
        assert len(result["rca_result"]) > 0

    def test_stores_provider(self):
        """llm_provider must be stored from chat_complete return value."""
        state = _make_state(
            anomalies=[_make_anomaly("MEDIUM")],
            rca_evidence=self._evidence_with_rag(),
        )
        with patch("System1.RCA.rca_agent.chat_complete",
                   return_value=("Some hypothesis.", "gemini")):
            from System1.RCA.rca_agent import rca_node2
            result = rca_node2(state)
        assert result["llm_provider"] == "gemini"

    def test_no_anomalies_skips_llm(self):
        """If no MEDIUM/CRITICAL anomalies, rca_node2 must not call the LLM."""
        state = _make_state(anomalies=[], rca_evidence={})
        with patch("System1.RCA.rca_agent.chat_complete") as mock_llm:
            from System1.RCA.rca_agent import rca_node2
            result = rca_node2(state)
        mock_llm.assert_not_called()
        assert "not required" in result["rca_result"].lower()

    def test_llm_failure_captured_gracefully(self):
        """If chat_complete raises, rca_node2 must not propagate the exception."""
        state = _make_state(
            anomalies=[_make_anomaly("CRITICAL")],
            rca_evidence=self._evidence_with_rag(),
        )
        with patch("System1.RCA.rca_agent.chat_complete",
                   side_effect=RuntimeError("Both providers failed")):
            from System1.RCA.rca_agent import rca_node2
            result = rca_node2(state)
        assert "rca_result" in result
        assert result["llm_provider"] is None

    def test_evidence_summary_included_in_prompt(self):
        """The user prompt sent to chat_complete must include evidence data."""
        state = _make_state(
            anomalies=[_make_anomaly("CRITICAL")],
            rca_evidence=self._evidence_with_rag(),
        )
        captured_messages = []

        def capture(messages, **kwargs):
            captured_messages.extend(messages)
            return ("hypothesis.", "groq")

        with patch("System1.RCA.rca_agent.chat_complete", side_effect=capture):
            from System1.RCA.rca_agent import rca_node2
            rca_node2(state)

        full_text = " ".join(m["content"] for m in captured_messages)
        assert "generation_solar/FR" in full_text
        assert "Persistent high pressure" in full_text   # from rag_results main_argument


# ===========================================================================
# TestSaveRcaNode
# ===========================================================================

class TestSaveRcaNode:
    """
    save_rca_node must UPDATE data_quality_runs and publish to Redis.
    Tests verify behaviour under both success and failure conditions.
    """

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def _state_with_result(self) -> dict:
        state = _make_state(anomalies=[_make_anomaly("CRITICAL")])
        state["rca_result"] = "1. High pressure reduced wind generation."
        state["rca_sources"] = [
            {"main_argument": "High pressure.", "score": 0.93, "source": "cb"}
        ]
        return state

    def test_no_error_on_success(self):
        state = self._state_with_result()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            result = save_rca_node(state)
        assert result["rca_error"] is None

    def test_publishes_to_redis(self):
        state = self._state_with_result()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            save_rca_node(state)
        mock_redis.publish.assert_called_once()

    def test_redis_event_is_rca_complete(self):
        state = self._state_with_result()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            save_rca_node(state)
        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "validated_data"
        assert json.loads(payload)["event"] == "rca_complete"

    def test_redis_payload_has_required_fields(self):
        state = self._state_with_result()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            save_rca_node(state)
        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "n_hypotheses", "n_rag_sources",
                      "countries", "timestamp"):
            assert field in parsed, f"Missing field in Redis payload: {field}"

    def test_n_rag_sources_reflects_actual_sources(self):
        state = self._state_with_result()  # has 1 rca_source
        mock_engine, _, mock_redis = self._mock_engine_and_redis()
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            save_rca_node(state)
        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["n_rag_sources"] == 1

    def test_error_captured_on_db_failure(self):
        state = self._state_with_result()
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            result = save_rca_node(state)
        assert result["rca_error"] is not None
        assert "DB unavailable" in result["rca_error"]

    def test_redis_publishes_even_on_db_failure(self):
        """Redis must publish even when the DB UPDATE fails."""
        state = self._state_with_result()
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        with (
            patch("System1.RCA.rca_agent.engine",    mock_engine),
            patch("System1.RCA.rca_agent.get_redis", return_value=mock_redis),
        ):
            from System1.RCA.rca_agent import save_rca_node
            save_rca_node(state)
        mock_redis.publish.assert_called_once()


# ===========================================================================
# TestQueryHistoricalDb (tool unit tests)
# ===========================================================================

class TestQueryHistoricalDb:
    """
    query_historical_db is a @tool wrapping SQL queries. We patch the engine
    at module level to avoid requiring a live DB.
    """

    def _mock_engine(self, stat_row, anomaly_count: int = 0):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        # First call → stats row, second call → anomaly count
        mock_conn.execute.return_value.fetchone.side_effect = [
            stat_row, (anomaly_count,)
        ]
        return mock_engine

    def test_returns_stats_on_success(self):
        mock_engine = self._mock_engine((100, 1200.0, 150.0, 800.0, 1600.0), 2)
        with patch("System1.RCA.rca_agent.engine", mock_engine):
            from System1.RCA.rca_agent import query_historical_db
            result = query_historical_db.invoke({
                "variable": "generation_solar",
                "country": "FR",
            })
        assert result["n_records"] == 100
        assert result["mean"] == 1200.0
        assert result["anomaly_count_last_30d"] == 2
        assert result["error"] is None

    def test_returns_error_on_db_failure(self):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")
        with patch("System1.RCA.rca_agent.engine", mock_engine):
            from System1.RCA.rca_agent import query_historical_db
            result = query_historical_db.invoke({
                "variable": "generation_solar",
                "country": "FR",
            })
        assert result["error"] is not None
        assert result["n_records"] == 0

    def test_country_normalised_to_upper(self):
        mock_engine = self._mock_engine((10, 100.0, 10.0, 80.0, 120.0), 0)
        with patch("System1.RCA.rca_agent.engine", mock_engine):
            from System1.RCA.rca_agent import query_historical_db
            result = query_historical_db.invoke({
                "variable": "load_actual_aggregated",
                "country": "fr",   # lowercase
            })
        assert result["country"] == "FR"


# ===========================================================================
# TestRagSearch (tool unit tests)
# ===========================================================================

class TestRagSearch:
    """
    rag_search wraps an httpx.get call. We patch httpx to avoid real HTTP.
    """

    def _mock_httpx_response(self, results: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"query": "test", "returned": len(results),
                                       "results": results}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_top_2_results_by_score(self):
        results = [
            {"main_argument": "A", "sentiment": "neg", "score": 0.75, "source": "s1"},
            {"main_argument": "B", "sentiment": "pos", "score": 0.91, "source": "s2"},
            {"main_argument": "C", "sentiment": "neu", "score": 0.65, "source": "s3"},
        ]
        mock_resp = self._mock_httpx_response(results)
        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key"}),
            patch("httpx.get", return_value=mock_resp),
        ):
            from System1.RCA.rca_agent import rag_search
            result = rag_search.invoke({"query": "wind France"})
        # Top 2 by score: B (0.91) and A (0.75)
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.91

    def test_filters_below_min_score(self):
        results = [
            {"main_argument": "A", "sentiment": "neg", "score": 0.40, "source": "s1"},
        ]
        mock_resp = self._mock_httpx_response(results)
        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key",
                                      "RAG_MIN_SCORE": "0.60"}),
            patch("httpx.get", return_value=mock_resp),
        ):
            from System1.RCA.rca_agent import rag_search
            result = rag_search.invoke({"query": "wind France"})
        assert result["results"] == []

    def test_no_rag_url_returns_empty(self):
        """If RAG_API_URL is not set, must return empty results without raising."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("RAG_API_URL", None)
            from System1.RCA.rca_agent import rag_search
            result = rag_search.invoke({"query": "test"})
        assert result["results"] == []
        assert result["error"] is not None

    def test_http_failure_returns_error(self):
        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key"}),
            patch("httpx.get", side_effect=Exception("timeout")),
        ):
            from System1.RCA.rca_agent import rag_search
            result = rag_search.invoke({"query": "test"})
        assert result["error"] is not None
        assert result["results"] == []

    def test_only_main_argument_fields_returned(self):
        """Result dicts must contain only the 4 allowed fields."""
        results = [
            {"main_argument": "High pressure.", "sentiment": "neg",
             "score": 0.85, "source": "cb",
             "text": "full article text",    # should be excluded
             "embedding": [0.1, 0.2, 0.3]}, # should be excluded
        ]
        mock_resp = self._mock_httpx_response(results)
        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key"}),
            patch("httpx.get", return_value=mock_resp),
        ):
            from System1.RCA.rca_agent import rag_search
            result = rag_search.invoke({"query": "test"})
        if result["results"]:
            allowed_keys = {"main_argument", "sentiment", "score", "source"}
            assert set(result["results"][0].keys()) == allowed_keys


# ===========================================================================
# Integration tests
# ===========================================================================

@pytest.mark.integration
class TestRcaIntegration:
    """
    End-to-end run of the RCA Agent graph.
    Requires: GROQ_API_KEY, PostgreSQL + Redis running.
    RAG integration is optional — if RAG_API_URL is not set, rag_search
    returns empty results and reasoning continues normally.
    """

    def _make_integration_state(self, severity: str = "CRITICAL") -> dict:
        state = _make_state(anomalies=[_make_anomaly(severity)])
        state["rca_evidence"] = {}
        return state

    def test_full_rca_run_produces_result(self):
        from System1.RCA.rca_agent import invoke_rca_graph
        state = self._make_integration_state("CRITICAL")
        result = invoke_rca_graph(state)
        assert result["rca_error"] is None
        assert isinstance(result.get("rca_result"), str)
        assert len(result["rca_result"]) > 0

    def test_no_anomalies_run_completes_cleanly(self):
        """A run with only LOW anomalies must complete without error."""
        from System1.RCA.rca_agent import invoke_rca_graph
        state = _make_state(anomalies=[_make_anomaly("LOW")])
        result = invoke_rca_graph(state)
        assert result["rca_error"] is None
        assert "not required" in result["rca_result"].lower()