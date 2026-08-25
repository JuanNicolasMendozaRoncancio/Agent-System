"""
Tests for System1/Ingestion/ingestion_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis.
Integration : marked @pytest.mark.integration — require live credentials.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_state(run_type: str = "full", countries: list[str] | None = None) -> dict:
    """
    Return a minimal valid AgentState dict for testing.
    """
    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       countries or ["FR", "DE"],
        "date_from":       datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
        "date_to":         datetime(2024, 6, 1, 3, tzinfo=timezone.utc),
        "run_type":        run_type,
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],    # NEW: required by summarize_node and ingestion_node
        "cycle_count":     0,     # NEW: required by _should_continue and summarize_node
    }


def _make_tool_message(name: str, content: dict, tool_call_id: str = "call_1") -> ToolMessage:
    """
    Return a ToolMessage whose content is a JSON-serialised summary dict.
    """
    return ToolMessage(
        content=json.dumps(content),
        name=name,
        tool_call_id=tool_call_id,
    )


def _sample_energy_record(country: str = "FR") -> dict:
    return {
        "timestamp":  "2024-06-01T00:00:00+00:00",
        "source_api": "entsoe",
        "country":    country,
        "variable":   "generation_solar",
        "value":      1234.5,
        "unit":       "MW",
        "metadata":   {"psr_type": "B16"},
    }


def _sample_climate_record(country: str = "FR") -> dict:
    return {
        "timestamp":  "2024-06-01T00:00:00+00:00",
        "source_api": "copernicus",
        "country":    country,
        "variable":   "climate_temperature_2m",
        "value":      18.3,
        "unit":       "°C",
        "metadata":   None,
    }


# ===========================================================================
# TestIngestionTools — each @tool wrapper behaves correctly
# ===========================================================================

class TestIngestionTools:

    def test_fetch_generation_calls_client(self):
        """fetch_generation parses date strings and delegates to entsoe_client module."""
        mock_records = [_sample_energy_record()]
        mock_module = MagicMock()
        mock_module.fetch_generation.return_value = mock_records

        with patch(
            "System1.Ingestion.ingestion_agent._get_entsoe_client",
            return_value=mock_module,
        ):
            from System1.Ingestion.ingestion_agent import fetch_generation
            result = fetch_generation.invoke({
                "run_id":       "test-run-1",
                "country_code": "fr",
                "date_from":    "2024-06-01T00:00:00",
                "date_to":      "2024-06-02T00:00:00",
            })

        mock_module.fetch_generation.assert_called_once()
        assert mock_module.fetch_generation.call_args[0][0] == "FR"
        assert result["fetched"] == len(mock_records)
        assert result["source"] == "entsoe"

    def test_fetch_load_calls_client(self):
        """fetch_load parses date strings and delegates to entsoe_client module."""
        mock_records = [_sample_energy_record()]
        mock_module = MagicMock()
        mock_module.fetch_load.return_value = mock_records

        with patch(
            "System1.Ingestion.ingestion_agent._get_entsoe_client",
            return_value=mock_module,
        ):
            from System1.Ingestion.ingestion_agent import fetch_load
            result = fetch_load.invoke({
                "run_id":       "test-run-2",
                "country_code": "DE",
                "date_from":    "2024-06-01T00:00:00",
                "date_to":      "2024-06-02T00:00:00",
            })

        mock_module.fetch_load.assert_called_once()
        assert result["fetched"] == len(mock_records)
        assert result["source"] == "entsoe"

    def test_fetch_temperature_calls_client(self):
        """fetch_temperature delegates to copernicus_client module."""
        mock_records = [_sample_climate_record()]
        mock_module = MagicMock()
        mock_module.fetch_temperature.return_value = mock_records

        with patch(
            "System1.Ingestion.ingestion_agent._get_copernicus_client",
            return_value=mock_module,
        ):
            from System1.Ingestion.ingestion_agent import fetch_temperature
            result = fetch_temperature.invoke({
                "run_id":       "test-run-3",
                "country_code": "FR",
                "date_from":    "2024-06-01T00:00:00",
                "date_to":      "2024-06-02T00:00:00",
            })

        mock_module.fetch_temperature.assert_called_once()
        assert result["fetched"] == len(mock_records)
        assert result["source"] == "copernicus"

    def test_fetch_solar_radiation_calls_client(self):
        """fetch_solar_radiation delegates to copernicus_client module."""
        mock_records = [_sample_climate_record()]
        mock_module = MagicMock()
        mock_module.fetch_solar_radiation.return_value = mock_records

        with patch(
            "System1.Ingestion.ingestion_agent._get_copernicus_client",
            return_value=mock_module,
        ):
            from System1.Ingestion.ingestion_agent import fetch_solar_radiation
            result = fetch_solar_radiation.invoke({
                "run_id":       "test-run-4",
                "country_code": "ES",
                "date_from":    "2024-06-01T00:00:00",
                "date_to":      "2024-06-02T00:00:00",
            })

        mock_module.fetch_solar_radiation.assert_called_once()
        assert result["fetched"] == len(mock_records)
        assert result["source"] == "copernicus"


# ===========================================================================
# TestIngestionNode — the primary LangGraph node
# ===========================================================================

class TestIngestionNode:

    def _make_ai_with_tools(self, tool_name: str, args: dict) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"id": "call_abc", "name": tool_name, "args": args}],
        )

    def _make_ai_final(self) -> AIMessage:
        return AIMessage(content="All data fetched successfully.")

    def test_ingestion_node_returns_llm_provider(self):
        """ingestion_node stores the provider name in the returned dict."""
        state = _make_state()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_ai_final()

        with patch(
            "System1.Ingestion.ingestion_agent._build_llm_with_tools",
            return_value=(mock_llm, "groq"),
        ):
            from System1.Ingestion.ingestion_agent import ingestion_node
            result = ingestion_node(state)

        assert result["llm_provider"] == "groq"

    def test_ingestion_node_appends_ai_message(self):
        """ingestion_node appends the LLM response to messages."""
        state = _make_state()
        mock_llm = MagicMock()
        ai_msg = self._make_ai_final()
        mock_llm.invoke.return_value = ai_msg

        with patch(
            "System1.Ingestion.ingestion_agent._build_llm_with_tools",
            return_value=(mock_llm, "groq"),
        ):
            from System1.Ingestion.ingestion_agent import ingestion_node
            result = ingestion_node(state)

        assert ai_msg in result["messages"]

    def test_ingestion_node_first_cycle_sends_run_context(self):
        """
        On cycle_count == 0, ingestion_node must send the run context
        (countries, date range, run_type) in the HumanMessage — not a
        tool_results summary.
        """
        state = _make_state()
        state["cycle_count"] = 0
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_ai_final()

        with patch(
            "System1.Ingestion.ingestion_agent._build_llm_with_tools",
            return_value=(mock_llm, "groq"),
        ):
            from System1.Ingestion.ingestion_agent import ingestion_node
            ingestion_node(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        human_content = next(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if (isinstance(m, dict) and m.get("role") == "user")
            or isinstance(m, HumanMessage)
        )
        assert "Run ID" in human_content
        assert "Countries" in human_content

    def test_ingestion_node_retry_cycle_sends_tool_results_summary(self):
        """
        On cycle_count > 0, ingestion_node must build the HumanMessage from
        state['tool_results'].
        """
        state = _make_state()
        state["cycle_count"] = 1
        state["tool_results"] = [
            {"tool": "fetch_generation", "country": "FR", "status": "ok",    "n_records": 48, "error": None},
            {"tool": "fetch_load",       "country": "FR", "status": "error", "n_records": 0,  "error": "timeout"},
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_ai_final()

        with patch(
            "System1.Ingestion.ingestion_agent._build_llm_with_tools",
            return_value=(mock_llm, "groq"),
        ):
            from System1.Ingestion.ingestion_agent import ingestion_node
            ingestion_node(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        human_content = next(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if (isinstance(m, dict) and m.get("role") == "user")
            or isinstance(m, HumanMessage)
        )
        assert "fetch_load" in human_content
        assert "FAILED" in human_content or "error" in human_content.lower()

    def test_should_continue_routes_to_tool_node_when_tool_calls_and_under_cap(self):
        """Routes to tool_node when tool_calls present and cycle_count < MAX_CYCLES."""
        state = _make_state()
        state["cycle_count"] = 0
        state["messages"] = [
            self._make_ai_with_tools(
                "fetch_generation",
                {"run_id": "x", "country_code": "FR",
                 "date_from": "2024-06-01T00:00:00", "date_to": "2024-06-02T00:00:00"},
            )
        ]

        from System1.Ingestion.ingestion_agent import _should_continue
        assert _should_continue(state) == "tool_node"

    def test_should_continue_routes_to_save_node_when_no_tool_calls(self):
        """Routes to save_node when the LLM emits no tool_calls."""
        state = _make_state()
        state["cycle_count"] = 0
        state["messages"] = [self._make_ai_final()]

        from System1.Ingestion.ingestion_agent import _should_continue
        assert _should_continue(state) == "save_node"

    def test_should_continue_routes_to_save_node_when_cycle_cap_reached(self):
        """
        Routes to save_node when cycle_count == MAX_CYCLES, even if the LLM
        still wants to call tools. This guarantees the graph always terminates.
        """
        from System1.Ingestion.ingestion_agent import _should_continue, MAX_CYCLES
        state = _make_state()
        state["cycle_count"] = MAX_CYCLES
        state["messages"] = [
            self._make_ai_with_tools(
                "fetch_load",
                {"run_id": "x", "country_code": "DE",
                 "date_from": "2024-06-01T00:00:00", "date_to": "2024-06-02T00:00:00"},
            )
        ]

        assert _should_continue(state) == "save_node"


# ===========================================================================
# TestSummarizeNode — the new intermediate node
# ===========================================================================

class TestSummarizeNode:


    def test_builds_tool_results_from_tool_messages(self):
        """summarize_node converts ToolMessages into structured tool_results."""
        state = _make_state()
        state["messages"] = [
            _make_tool_message(
                "fetch_generation",
                {"fetched": 48, "source": "entsoe", "dataset": "generation", "country": "FR"},
                "call_1",
            ),
            _make_tool_message(
                "fetch_load",
                {"fetched": 24, "source": "entsoe", "dataset": "load", "country": "FR"},
                "call_2",
            ),
        ]

        from System1.Ingestion.ingestion_agent import summarize_node
        result = summarize_node(state)

        assert len(result["tool_results"]) == 2
        gen = next(tr for tr in result["tool_results"] if tr["tool"] == "fetch_generation")
        assert gen["status"] == "ok"
        assert gen["n_records"] == 48
        assert gen["country"] == "FR"

    def test_increments_cycle_count(self):
        """summarize_node must increment cycle_count by exactly 1."""
        state = _make_state()
        state["cycle_count"] = 0
        state["messages"] = [
            _make_tool_message(
                "fetch_generation",
                {"fetched": 10, "source": "entsoe", "dataset": "generation", "country": "FR"},
            )
        ]

        from System1.Ingestion.ingestion_agent import summarize_node
        result = summarize_node(state)

        assert result["cycle_count"] == 1

    def test_clears_messages(self):
        """
        summarize_node must return an empty messages list so the next
        ingestion_node call starts with a clean slate.
        """
        state = _make_state()
        state["messages"] = [
            _make_tool_message(
                "fetch_generation",
                {"fetched": 10, "source": "entsoe", "dataset": "generation", "country": "FR"},
            )
        ]

        from System1.Ingestion.ingestion_agent import summarize_node
        result = summarize_node(state)

        assert result["messages"] == []

    def test_merges_results_on_retry(self):
        state = _make_state()
        # Previous cycle had fetch_generation OK and fetch_load failed
        state["tool_results"] = [
            {"tool": "fetch_generation", "country": "FR", "status": "ok",    "n_records": 48, "error": None},
            {"tool": "fetch_load",       "country": "FR", "status": "error", "n_records": 0,  "error": "timeout"},
        ]
        # This cycle retried only fetch_load and it succeeded
        state["messages"] = [
            _make_tool_message(
                "fetch_load",
                {"fetched": 24, "source": "entsoe", "dataset": "load", "country": "FR"},
                "call_retry",
            )
        ]

        from System1.Ingestion.ingestion_agent import summarize_node
        result = summarize_node(state)

        # Still 2 entries total — not 3
        assert len(result["tool_results"]) == 2
        load = next(tr for tr in result["tool_results"] if tr["tool"] == "fetch_load")
        assert load["status"] == "ok"
        assert load["n_records"] == 24

        # fetch_generation entry must be preserved unchanged
        gen = next(tr for tr in result["tool_results"] if tr["tool"] == "fetch_generation")
        assert gen["status"] == "ok"
        assert gen["n_records"] == 48

    def test_captures_tool_error_in_tool_results(self):
        state = _make_state()
        state["messages"] = [
            ToolMessage(
                content="Error: connection timeout after 30s",
                name="fetch_load",
                tool_call_id="call_err",
            )
        ]

        from System1.Ingestion.ingestion_agent import summarize_node
        result = summarize_node(state)

        assert len(result["tool_results"]) == 1
        tr = result["tool_results"][0]
        assert tr["status"] == "error"
        assert tr["tool"] == "fetch_load"
        assert "timeout" in tr["error"]


# ===========================================================================
# TestSaveNode — PostgreSQL + Redis persistence
# ===========================================================================

class TestSaveNode:

    def _state_with_records(self) -> dict:
        """Return a state whose run_id has records pre-seeded in _record_store."""
        from System1.Ingestion.ingestion_agent import _store_records
        state = _make_state()
        _store_records(state["run_id"], [_sample_energy_record("FR"), _sample_energy_record("DE")])
        _store_records(state["run_id"], [_sample_climate_record("FR")])
        return state

    def test_save_node_returns_records(self):
        """save_node must populate state['records'] from the in-process store."""
        state = self._state_with_records()
        mock_conn = MagicMock()
        mock_redis = MagicMock()

        with (
            patch("System1.Ingestion.ingestion_agent.engine") as mock_engine,
            patch("System1.Ingestion.ingestion_agent.get_redis", return_value=mock_redis),
        ):
            mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            from System1.Ingestion.ingestion_agent import save_node
            result = save_node(state)

        assert len(result["records"]) == 3

    def test_save_node_publishes_to_redis(self):
        """save_node must publish exactly one message to CHANNEL_VALIDATED_DATA."""
        state = self._state_with_records()
        mock_conn = MagicMock()
        mock_redis = MagicMock()

        with (
            patch("System1.Ingestion.ingestion_agent.engine") as mock_engine,
            patch("System1.Ingestion.ingestion_agent.get_redis", return_value=mock_redis),
        ):
            mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            from System1.Ingestion.ingestion_agent import save_node
            save_node(state)

        mock_redis.publish.assert_called_once()
        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "validated_data"
        parsed = json.loads(payload)
        assert parsed["n_records"] == 3
        assert "run_id" in parsed

    def test_save_node_sets_no_error_on_success(self):
        """ingestion_error must be None when everything succeeds."""
        state = self._state_with_records()
        mock_conn = MagicMock()
        mock_redis = MagicMock()

        with (
            patch("System1.Ingestion.ingestion_agent.engine") as mock_engine,
            patch("System1.Ingestion.ingestion_agent.get_redis", return_value=mock_redis),
        ):
            mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            from System1.Ingestion.ingestion_agent import save_node
            result = save_node(state)

        assert result["ingestion_error"] is None

    def test_save_node_sets_error_on_db_failure(self):
        """ingestion_error must be set (not raised) when the DB insert fails."""
        state = self._state_with_records()
        mock_redis = MagicMock()

        with (
            patch("System1.Ingestion.ingestion_agent.engine") as mock_engine,
            patch("System1.Ingestion.ingestion_agent.get_redis", return_value=mock_redis),
        ):
            mock_engine.connect.side_effect = Exception("DB connection refused")

            from System1.Ingestion.ingestion_agent import save_node
            result = save_node(state)

        assert result["ingestion_error"] is not None
        assert "DB connection refused" in result["ingestion_error"]


# ===========================================================================
# Integration tests — require live credentials + Docker running
# ===========================================================================

@pytest.mark.integration
class TestIngestionIntegration:
    """
    End-to-end run of the full Ingestion Agent graph against real APIs.
    Requires: ENTSOE_API_KEY, COPERNICUS_URL, COPERNICUS_API_KEY,
              GROQ_API_KEY, PostgreSQL + Redis running.
    """

    def test_full_run_fr(self):
        """Full run for France — all four data sources must return records."""
        from System1.Ingestion.ingestion_agent import invoke_ingestion_graph
        state = _make_state(run_type="full", countries=["FR"])
        result = invoke_ingestion_graph(state)

        assert len(result["records"]) > 0
        assert result["ingestion_error"] is None
        sources = {r["source_api"] for r in result["records"]}
        assert "entsoe" in sources
        assert "copernicus" in sources

    def test_incremental_run_de(self):
        """Incremental run for Germany — only ENTSO-E sources expected."""
        from System1.Ingestion.ingestion_agent import invoke_ingestion_graph
        state = _make_state(run_type="incremental", countries=["DE"])
        result = invoke_ingestion_graph(state)

        assert len(result["records"]) > 0
        assert result["ingestion_error"] is None
        variables = {r["variable"] for r in result["records"]}
        assert not any("climate" in v for v in variables), (
            "Incremental run must not fetch climate data"
        )