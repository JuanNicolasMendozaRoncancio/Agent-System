"""
Tests for System2/analysis_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis, no LLM calls.
Integration : marked @pytest.mark.integration — require live credentials
              and Docker running.

Run unit tests only:
    python -m pytest tests/test_analysis_agent.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_analysis_agent.py -v -m integration
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
    countries: list[str] | None = None,
    analysis_results: dict | None = None,
    rag_topics: list[dict] | None = None,
) -> dict:
    """Return a minimal valid AgentState dict for analysis tests."""
    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       countries or ["FR", "DE"],
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
        "anomalies":       [],
        "qa_severity":     None,
        "qa_error":        None,
        "qa_summary":      "",
        "rca_evidence":    {},
        "rca_result":      None,
        "rca_sources":     [],
        "rca_error":       None,
        "run_report":      "",
        "reporter_error":  None,
        "analysis_results": analysis_results or {},
        "rag_topics":       rag_topics or [],
        "analysis_error":   None,
    }


def _make_tool_message(name: str, content: dict,
                       tool_call_id: str = "call_1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content),
        name=name,
        tool_call_id=tool_call_id,
    )


def _make_patterns_result(country: str = "FR", fallback: bool = False) -> dict:
    return {
        "country":              country,
        "run_id":               "test-run",
        "window_days":          30,
        "actual_window_days":   1 if fallback else 30,
        "requested_window_days": 30,
        "fallback_used":        fallback,
        "variables": {
            "generation_solar": {"slope": 0.5, "mean": 1200.0,
                                  "min": 900.0, "max": 1500.0, "n": 6},
            "load_actual_aggregated": {"slope": -0.1, "mean": 34000.0,
                                       "min": 33000.0, "max": 35000.0, "n": 6},
        },
    }


def _make_risk_result(country: str = "FR", has_temp: bool = True) -> dict:
    return {
        "country":              country,
        "run_id":               "test-run",
        "score":                35.0,
        "has_temperature_data": has_temp,
        "components": {
            "demand_coverage":         5.0,
            "renewable_intermittency": 45.0,
            "hydraulic_buffer":        10.0,
            "temperature_demand":      0.0 if not has_temp else 20.0,
        },
        "weights_used": {
            "demand_coverage":         0.375 if not has_temp else 0.30,
            "renewable_intermittency": 0.3125 if not has_temp else 0.25,
            "hydraulic_buffer":        0.3125 if not has_temp else 0.25,
            "temperature_demand":      0.0 if not has_temp else 0.20,
        },
        "auxiliary": {
            "total_generation_mw": 50000.0,
            "load_mw":             34000.0,
            "renewables_mw":       22000.0,
            "hydro_buffer_mw":     2000.0,
            "temperature_c":       None if not has_temp else 22.5,
        },
        "error": None,
    }


def _make_rag_result(n_topics: int = 2) -> dict:
    return {
        "topics": [
            {"id": f"topic_{i}", "title": f"Topic {i}", "relevance": 0.9 - i * 0.1}
            for i in range(n_topics)
        ],
        "n_topics": n_topics,
        "error":    None,
    }


# ===========================================================================
# TestComputeTrend — pure function tests
# ===========================================================================

class TestComputeTrend:
    """
    _compute_trend is a pure function — no mocking needed.
    Tests verify the slope and stats arithmetic.
    """

    def test_empty_list_returns_zeros(self):
        from System2.Analysis.analysis_agent import _compute_trend
        result = _compute_trend([])
        assert result["n"] == 0
        assert result["slope"] == 0.0

    def test_single_value_no_slope(self):
        from System2.Analysis.analysis_agent import _compute_trend
        result = _compute_trend([100.0])
        assert result["n"] == 1
        assert result["slope"] == 0.0
        assert result["mean"] == pytest.approx(100.0)

    def test_two_values_no_slope(self):
        """Two values is insufficient for thirds method — slope returns 0."""
        from System2.Analysis.analysis_agent import _compute_trend
        result = _compute_trend([100.0, 200.0])
        assert result["slope"] == 0.0
        assert result["mean"] == pytest.approx(150.0)

    def test_rising_trend_positive_slope(self):
        """Monotonically increasing series must produce positive slope."""
        from System2.Analysis.analysis_agent import _compute_trend
        values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        result = _compute_trend(values)
        assert result["slope"] > 0.0

    def test_falling_trend_negative_slope(self):
        """Monotonically decreasing series must produce negative slope."""
        from System2.Analysis.analysis_agent import _compute_trend
        values = [600.0, 500.0, 400.0, 300.0, 200.0, 100.0]
        result = _compute_trend(values)
        assert result["slope"] < 0.0

    def test_flat_trend_zero_slope(self):
        """Constant series must produce slope = 0."""
        from System2.Analysis.analysis_agent import _compute_trend
        values = [500.0] * 9
        result = _compute_trend(values)
        assert result["slope"] == pytest.approx(0.0)

    def test_mean_min_max_correct(self):
        from System2.Analysis.analysis_agent import _compute_trend
        values = [100.0, 200.0, 300.0]
        result = _compute_trend(values)
        assert result["mean"] == pytest.approx(200.0)
        assert result["min"]  == pytest.approx(100.0)
        assert result["max"]  == pytest.approx(300.0)

    def test_n_equals_input_length(self):
        from System2.Analysis.analysis_agent import _compute_trend
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _compute_trend(values)["n"] == 5


# ===========================================================================
# TestComputeRiskForCountry — component arithmetic
# ===========================================================================

class TestComputeRiskForCountry:
    """
    Tests for _compute_risk_for_country via patched DB records.

    Why patch _fetch_records_for_window and not engine directly:
        _fetch_records_for_window is the seam between the risk function and
        PostgreSQL. Patching it lets us inject controlled record sets without
        mocking SQL internals.
    """

    def _make_records(
        self,
        total_gen: float = 50000.0,
        load: float = 40000.0,
        wind: float = 20000.0,
        solar: float = 5000.0,
        hydro_reservoir: float = 2000.0,
        pumped_storage: float = 1000.0,
        temperature: float | None = 22.0,
    ) -> list[dict]:
        """Build a synthetic record set for risk computation tests."""
        records = [
            {"variable": "generation_fossil_gas",      "value": total_gen - wind - solar
                                                                 - hydro_reservoir - pumped_storage,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
            {"variable": "generation_wind_onshore",    "value": wind,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
            {"variable": "generation_solar",           "value": solar,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
            {"variable": "generation_hydro_water_reservoir", "value": hydro_reservoir,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
            {"variable": "generation_hydro_pumped_storage",  "value": pumped_storage,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
            {"variable": "load_actual_aggregated",     "value": load,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc), "source_api": "entsoe"},
        ]
        if temperature is not None:
            records.append({
                "variable": "climate_temperature_2m", "value": temperature,
                "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
                "source_api": "copernicus",
            })
        return records

    def test_score_in_valid_range(self):
        """Risk score must always be between 0 and 100."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records()
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        assert 0.0 <= result["score"] <= 100.0

    def test_has_temperature_data_true_when_copernicus_present(self):
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records(temperature=22.0)
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        assert result["has_temperature_data"] is True

    def test_has_temperature_data_false_when_no_copernicus(self):
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records(temperature=None)
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("DE", "run-1")
        assert result["has_temperature_data"] is False

    def test_weights_sum_to_one_with_temperature(self):
        """Full weights (with temperature data) must sum to exactly 1.0."""
        from System2.Analysis.analysis_agent import _WEIGHTS_FULL
        assert sum(_WEIGHTS_FULL.values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_without_temperature(self):
        """Redistributed weights (no temperature data) must sum to exactly 1.0."""
        from System2.Analysis.analysis_agent import _WEIGHTS_NO_TEMP
        assert sum(_WEIGHTS_NO_TEMP.values()) == pytest.approx(1.0)

    def test_temperature_weight_zero_when_no_copernicus(self):
        """temperature_demand weight must be 0.0 when Copernicus data is absent."""
        from System2.Analysis.analysis_agent import _WEIGHTS_NO_TEMP
        assert _WEIGHTS_NO_TEMP["temperature_demand"] == 0.0

    def test_c1_risk_zero_when_generation_exceeds_load(self):
        """When total generation > load, demand_coverage component must be 0."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        # Generation (50000) >> load (30000) → C1 = 0
        records = self._make_records(total_gen=50000.0, load=30000.0, temperature=None)
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("DE", "run-1")
        assert result["components"]["demand_coverage"] == pytest.approx(0.0)

    def test_cold_temperature_produces_nonzero_c4_risk(self):
        """Temperature below COLD_THRESHOLD_C must produce a non-zero C4 risk."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records(temperature=0.0)  # below 5 °C threshold
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        assert result["components"]["temperature_demand"] > 0.0

    def test_hot_temperature_produces_nonzero_c4_risk(self):
        """Temperature above HOT_THRESHOLD_C must produce a non-zero C4 risk."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records(temperature=35.0)  # above 28 °C threshold
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        assert result["components"]["temperature_demand"] > 0.0

    def test_moderate_temperature_zero_c4_risk(self):
        """Moderate temperature (between thresholds) must produce C4 risk = 0."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records(temperature=18.0)  # between 5 and 28
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        assert result["components"]["temperature_demand"] == pytest.approx(0.0)

    def test_no_records_returns_error(self):
        """If no records are available, error must be set and score must be 0."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=([], 0)):
            result = _compute_risk_for_country("FR", "run-1")
        assert result["error"] is not None
        assert result["score"] == 0.0

    def test_result_has_required_keys(self):
        """Result dict must always contain all required keys."""
        from System2.Analysis.analysis_agent import _compute_risk_for_country
        records = self._make_records()
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _compute_risk_for_country("FR", "run-1")
        for key in ("score", "components", "weights_used",
                    "has_temperature_data", "auxiliary", "error"):
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# TestDetectPatternsForCountry
# ===========================================================================

class TestDetectPatternsForCountry:

    def test_fallback_used_when_insufficient_data(self):
        """fallback_used must be True when actual_window < requested_window."""
        from System2.Analysis.analysis_agent import _detect_patterns_for_country
        records = [
            {"variable": "generation_solar", "value": 100.0,
             "timestamp": datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
             "source_api": "entsoe"},
        ]
        # actual_days=1, requested=30 → fallback
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _detect_patterns_for_country("FR", "run-1", window_days=30)
        assert result["fallback_used"] is True
        assert result["actual_window_days"] == 1

    def test_no_fallback_when_sufficient_data(self):
        """fallback_used must be False when actual_window == requested_window."""
        from System2.Analysis.analysis_agent import _detect_patterns_for_country
        records = [
            {"variable": "generation_solar", "value": float(i),
             "timestamp": datetime(2024, 6, i + 1, 0, tzinfo=timezone.utc),
             "source_api": "entsoe"}
            for i in range(30)
        ]
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 30)):
            result = _detect_patterns_for_country("FR", "run-1", window_days=30)
        assert result["fallback_used"] is False

    def test_empty_records_returns_empty_variables(self):
        """No records → variables dict must be empty."""
        from System2.Analysis.analysis_agent import _detect_patterns_for_country
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=([], 0)):
            result = _detect_patterns_for_country("FR", "run-1", window_days=7)
        assert result["variables"] == {}

    def test_variables_keyed_by_variable_name(self):
        """Result variables dict must be keyed by variable name."""
        from System2.Analysis.analysis_agent import _detect_patterns_for_country
        records = [
            {"variable": "generation_solar", "value": 1000.0,
             "timestamp": datetime(2024, 6, 1, i, tzinfo=timezone.utc),
             "source_api": "entsoe"}
            for i in range(3)
        ]
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _detect_patterns_for_country("FR", "run-1", window_days=1)
        assert "generation_solar" in result["variables"]

    def test_each_variable_has_trend_keys(self):
        """Each variable entry must contain slope, mean, min, max, n."""
        from System2.Analysis.analysis_agent import _detect_patterns_for_country
        records = [
            {"variable": "load_actual_aggregated", "value": float(v),
             "timestamp": datetime(2024, 6, 1, i, tzinfo=timezone.utc),
             "source_api": "entsoe"}
            for i, v in enumerate([30000.0, 31000.0, 32000.0])
        ]
        with patch("System2.Analysis.analysis_agent._fetch_records_for_window",
                   return_value=(records, 1)):
            result = _detect_patterns_for_country("FR", "run-1", window_days=1)
        trend = result["variables"]["load_actual_aggregated"]
        for key in ("slope", "mean", "min", "max", "n"):
            assert key in trend, f"Missing trend key: {key}"


# ===========================================================================
# TestProcessToolMessages
# ===========================================================================

class TestProcessToolMessages:
    """
    _process_tool_messages converts ToolMessages into analysis_results + rag_topics.
    Tests verify parsing and structure without any LLM or DB involvement.
    """

    def test_patterns_message_goes_to_analysis_results(self):
        from System2.Analysis.analysis_agent import _process_tool_messages
        msg = _make_tool_message("detect_patterns",
                                  _make_patterns_result("FR"), "c1")
        results, topics = _process_tool_messages([msg])
        assert "FR" in results
        assert "patterns" in results["FR"]

    def test_risk_message_goes_to_analysis_results(self):
        from System2.Analysis.analysis_agent import _process_tool_messages
        msg = _make_tool_message("compute_risk_indicators",
                                  _make_risk_result("FR"), "c2")
        results, topics = _process_tool_messages([msg])
        assert "FR" in results
        assert "risk" in results["FR"]

    def test_rag_message_goes_to_topics(self):
        from System2.Analysis.analysis_agent import _process_tool_messages
        msg = _make_tool_message("rag_context", _make_rag_result(2), "c3")
        results, topics = _process_tool_messages([msg])
        assert len(topics) == 2

    def test_multiple_countries_separated(self):
        """Results for FR and DE must be stored under separate keys."""
        from System2.Analysis.analysis_agent import _process_tool_messages
        messages = [
            _make_tool_message("detect_patterns",       _make_patterns_result("FR"), "c1"),
            _make_tool_message("compute_risk_indicators", _make_risk_result("FR"),     "c2"),
            _make_tool_message("detect_patterns",       _make_patterns_result("DE"), "c3"),
            _make_tool_message("compute_risk_indicators", _make_risk_result("DE"),     "c4"),
            _make_tool_message("rag_context",            _make_rag_result(1),         "c5"),
        ]
        results, topics = _process_tool_messages(messages)
        assert "FR" in results
        assert "DE" in results
        assert len(topics) == 1

    def test_malformed_tool_message_skipped_gracefully(self):
        """A ToolMessage with non-JSON content must not raise."""
        from System2.Analysis.analysis_agent import _process_tool_messages
        msg = ToolMessage(
            content="Error: connection refused",
            name="detect_patterns",
            tool_call_id="c_err",
        )
        results, topics = _process_tool_messages([msg])
        assert results == {}
        assert topics == []

    def test_empty_messages_returns_empty_results(self):
        from System2.Analysis.analysis_agent import _process_tool_messages
        results, topics = _process_tool_messages([])
        assert results == {}
        assert topics == []


# ===========================================================================
# TestAnalysisNode
# ===========================================================================

class TestAnalysisNode:

    def _make_ai_with_tools(self) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "detect_patterns",
                 "args": {"run_id": "x", "country": "FR", "window_days": 30}},
                {"id": "c2", "name": "compute_risk_indicators",
                 "args": {"run_id": "x", "country": "FR"}},
                {"id": "c3", "name": "rag_context",
                 "args": {"query": "energy France"}},
            ],
        )

    def test_analysis_node_calls_llm(self):
        """analysis_node must invoke the LLM exactly once."""
        state = _make_state()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="done")

        with patch("System2.Analysis.analysis_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System2.Analysis.analysis_agent import analysis_node
            analysis_node(state)

        mock_llm.invoke.assert_called_once()

    def test_analysis_node_stores_provider(self):
        """analysis_node must return llm_provider in the result dict."""
        state = _make_state()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="done")

        with patch("System2.Analysis.analysis_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System2.Analysis.analysis_agent import analysis_node
            result = analysis_node(state)

        assert result["llm_provider"] == "groq"

    def test_analysis_node_appends_ai_message(self):
        """analysis_node must append the LLM response to messages."""
        state = _make_state()
        mock_llm = MagicMock()
        ai_msg = AIMessage(content="done")
        mock_llm.invoke.return_value = ai_msg

        with patch("System2.Analysis.analysis_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System2.Analysis.analysis_agent import analysis_node
            result = analysis_node(state)

        assert ai_msg in result["messages"]

    def test_human_message_contains_run_id(self):
        """The HumanMessage sent to the LLM must contain the run_id."""
        state = _make_state()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="done")

        with patch("System2.Analysis.analysis_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System2.Analysis.analysis_agent import analysis_node
            analysis_node(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        human_content = next(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if (isinstance(m, dict) and m.get("role") == "user")
            or (hasattr(m, "content") and "Human" in type(m).__name__)
        )
        assert state["run_id"] in human_content

    def test_human_message_contains_countries(self):
        """The HumanMessage must mention all countries in the run."""
        state = _make_state(countries=["FR", "DE"])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="done")

        with patch("System2.Analysis.analysis_agent._build_llm_with_tools",
                   return_value=(mock_llm, "groq")):
            from System2.Analysis.analysis_agent import analysis_node
            analysis_node(state)

        call_messages = mock_llm.invoke.call_args[0][0]
        human_content = " ".join(
            m["content"] if isinstance(m, dict) else m.content
            for m in call_messages
            if hasattr(m, "content") or isinstance(m, dict)
        )
        assert "FR" in human_content
        assert "DE" in human_content


# ===========================================================================
# TestProcessEvidenceNode
# ===========================================================================

class TestProcessEvidenceNode:

    def test_builds_analysis_results_from_tool_messages(self):
        """process_evidence_node must populate analysis_results from ToolMessages."""
        state = _make_state()
        state["messages"] = [
            _make_tool_message("detect_patterns",
                                _make_patterns_result("FR"), "c1"),
            _make_tool_message("compute_risk_indicators",
                                _make_risk_result("FR"), "c2"),
            _make_tool_message("rag_context", _make_rag_result(2), "c3"),
        ]

        from System2.Analysis.analysis_agent import process_evidence_node
        result = process_evidence_node(state)

        assert "FR" in result["analysis_results"]
        assert "patterns" in result["analysis_results"]["FR"]
        assert "risk" in result["analysis_results"]["FR"]
        assert len(result["rag_topics"]) == 2

    def test_clears_messages(self):
        """process_evidence_node must clear messages after processing."""
        state = _make_state()
        state["messages"] = [
            _make_tool_message("rag_context", _make_rag_result(1), "c1"),
        ]

        from System2.Analysis.analysis_agent import process_evidence_node
        result = process_evidence_node(state)

        assert result["messages"] == []

    def test_empty_messages_returns_empty_results(self):
        """Empty message list must produce empty analysis_results."""
        state = _make_state()
        state["messages"] = []

        from System2.Analysis.analysis_agent import process_evidence_node
        result = process_evidence_node(state)

        assert result["analysis_results"] == {}
        assert result["rag_topics"] == []


# ===========================================================================
# TestSaveAnalysisNode
# ===========================================================================

class TestSaveAnalysisNode:

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def _state_with_results(self) -> dict:
        return _make_state(
            analysis_results={
                "FR": {
                    "patterns": _make_patterns_result("FR"),
                    "risk":     _make_risk_result("FR"),
                },
                "DE": {
                    "patterns": _make_patterns_result("DE", fallback=True),
                    "risk":     _make_risk_result("DE", has_temp=False),
                },
            },
            rag_topics=[{"id": "t1", "title": "Topic 1"}],
        )

    def test_no_error_on_success(self):
        """analysis_error must be None when the DB update succeeds."""
        state = self._state_with_results()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            result = save_analysis_node(state)

        assert result["analysis_error"] is None

    def test_publishes_to_redis(self):
        """save_analysis_node must publish exactly one message to Redis."""
        state = self._state_with_results()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            save_analysis_node(state)

        mock_redis.publish.assert_called_once()

    def test_redis_event_is_analysis_complete(self):
        """Redis payload event must be 'analysis_complete'."""
        state = self._state_with_results()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            save_analysis_node(state)

        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "validated_data"
        assert json.loads(payload)["event"] == "analysis_complete"

    def test_redis_payload_has_required_fields(self):
        """Redis payload must contain run_id, event, countries, n_countries, timestamp."""
        state = self._state_with_results()
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            save_analysis_node(state)

        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "countries", "n_countries", "timestamp"):
            assert field in parsed, f"Missing field in Redis payload: {field}"

    def test_n_countries_reflects_analysis_results(self):
        """n_countries in Redis payload must equal len(analysis_results)."""
        state = self._state_with_results()  # has FR and DE
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            save_analysis_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["n_countries"] == 2

    def test_error_captured_on_db_failure(self):
        """analysis_error must be set (not raised) when the DB update fails."""
        state = self._state_with_results()
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            result = save_analysis_node(state)

        assert result["analysis_error"] is not None
        assert "DB unavailable" in result["analysis_error"]

    def test_redis_publishes_even_on_db_failure(self):
        """Redis publish must happen even when the DB update fails."""
        state = self._state_with_results()
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System2.Analysis.analysis_agent.engine",    mock_engine),
            patch("System2.Analysis.analysis_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Analysis.analysis_agent import save_analysis_node
            save_analysis_node(state)

        mock_redis.publish.assert_called_once()


# ===========================================================================
# TestRagContextTool
# ===========================================================================

class TestRagContextTool:

    def test_no_rag_url_returns_empty_gracefully(self):
        """If RAG_API_URL is not set, rag_context must return empty topics without raising."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("RAG_API_URL", None)
            # Re-import to pick up the cleared env var
            import importlib
            import System2.Analysis.analysis_agent as mod
            importlib.reload(mod)
            result = mod.rag_context.invoke({"query": "test"})
        assert result["topics"] == []
        assert result["error"] is not None

    def test_http_failure_returns_error_gracefully(self):
        """If the HTTP call fails, rag_context must return empty topics."""
        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key"}),
            patch("httpx.get", side_effect=Exception("timeout")),
        ):
            from System2.Analysis.analysis_agent import rag_context
            result = rag_context.invoke({"query": "energy France"})
        assert result["topics"] == []
        assert result["error"] is not None

    def test_successful_response_returns_topics(self):
        """A successful RAG response must return the topics list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": "t1", "title": "High pressure France", "relevance": 0.9},
            {"id": "t2", "title": "Renewable policy EU",  "relevance": 0.8},
        ]
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"RAG_API_URL": "http://rag.local",
                                      "RAG_API_KEY": "key"}),
            patch("httpx.get", return_value=mock_resp),
        ):
            from System2.Analysis.analysis_agent import rag_context
            result = rag_context.invoke({"query": "energy France"})

        assert len(result["topics"]) == 2
        assert result["n_topics"] == 2
        assert result["error"] is None


# ===========================================================================
# Integration tests — require live credentials + Docker running
# ===========================================================================

@pytest.mark.integration
class TestAnalysisIntegration:
    """
    End-to-end run of the Analysis Agent graph.
    Requires: GROQ_API_KEY, PostgreSQL + Redis running with data in
    energy_climate_records and a pre-existing analysis_runs row.
    """

    def _insert_analysis_run(self, run_id: str) -> None:
        """Insert a minimal analysis_runs row so the UPDATE has a target."""
        from shared.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO analysis_runs
                        (run_id, triggered_by, started_at, status)
                    VALUES
                        (:run_id, 'test', NOW(), 'triggered')
                    ON CONFLICT (run_id) DO NOTHING
                """),
                {"run_id": run_id},
            )
            conn.commit()

    def _delete_analysis_run(self, run_id: str) -> None:
        from shared.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

    def _get_most_recent_run_id(self) -> str | None:
        """Fetch a run_id that has data in energy_climate_records."""
        from shared.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT DISTINCT run_id FROM energy_climate_records LIMIT 1")
            ).fetchone()
        return str(row[0]) if row else None

    def test_full_analysis_run(self):
        """Full run must produce analysis_results with no analysis_error."""
        from System2.Analysis.analysis_agent import invoke_analysis_graph

        run_id = self._get_most_recent_run_id()
        if not run_id:
            pytest.skip("No data in energy_climate_records — run Sistema 1 first")

        self._insert_analysis_run(run_id)
        try:
            state = _make_state(countries=["FR", "DE"])
            state["run_id"] = run_id
            result = invoke_analysis_graph(state)

            assert result["analysis_error"] is None
            assert isinstance(result["analysis_results"], dict)
            assert len(result["analysis_results"]) > 0
        finally:
            self._delete_analysis_run(run_id)

    def test_risk_score_in_valid_range_with_real_data(self):
        """Risk scores must be between 0 and 100 on real data."""
        from System2.Analysis.analysis_agent import invoke_analysis_graph

        run_id = self._get_most_recent_run_id()
        if not run_id:
            pytest.skip("No data in energy_climate_records")

        self._insert_analysis_run(run_id)
        try:
            state = _make_state(countries=["FR", "DE"])
            state["run_id"] = run_id
            result = invoke_analysis_graph(state)

            for country, data in result["analysis_results"].items():
                if "risk" in data:
                    score = data["risk"].get("score", -1)
                    assert 0.0 <= score <= 100.0, (
                        f"Risk score for {country} out of range: {score}"
                    )
        finally:
            self._delete_analysis_run(run_id)

    def test_fallback_used_flag_present(self):
        """Each country's pattern result must contain the fallback_used flag."""
        from System2.Analysis.analysis_agent import invoke_analysis_graph

        run_id = self._get_most_recent_run_id()
        if not run_id:
            pytest.skip("No data in energy_climate_records")

        self._insert_analysis_run(run_id)
        try:
            state = _make_state(countries=["FR"])
            state["run_id"] = run_id
            result = invoke_analysis_graph(state)

            fr_data = result["analysis_results"].get("FR", {})
            if "patterns" in fr_data:
                assert "fallback_used" in fr_data["patterns"]
        finally:
            self._delete_analysis_run(run_id)