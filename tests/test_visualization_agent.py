"""
Tests for System2/visualization_agent.py

Unit tests  : fully mocked — no network, no DB, no LLM calls.
Integration : marked @pytest.mark.integration — require Docker running.

Run unit tests only:
    python -m pytest tests/test_visualization_agent.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_visualization_agent.py -v -m integration
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
    countries: list[str] | None = None,
    analysis_results: dict | None = None,
) -> dict:
    """Return a minimal valid AgentState for visualization tests."""
    return {
        "run_id":           str(uuid.uuid4()),
        "countries":        countries or ["FR", "DE"],
        "date_from":        datetime(2024, 6, 1, 0,  tzinfo=timezone.utc),
        "date_to":          datetime(2024, 6, 1, 3,  tzinfo=timezone.utc),
        "run_type":         "full",
        "messages":         [],
        "records":          [],
        "ingestion_error":  None,
        "llm_provider":     None,
        "tool_results":     [],
        "cycle_count":      0,
        "profile":          {},
        "profile_summary":  "",
        "profiling_error":  None,
        "anomalies":        [],
        "qa_severity":      None,
        "qa_error":         None,
        "qa_summary":       "",
        "rca_evidence":     {},
        "rca_result":       None,
        "rca_sources":      [],
        "rca_error":        None,
        "run_report":       "",
        "reporter_error":   None,
        "analysis_results": analysis_results if analysis_results is not None else _make_analysis_results(),
        "rag_topics":       [],
        "analysis_error":   None,
        "viz_data":         {},
        "viz_error":        None,
    }


def _make_analysis_results(
    countries: list[str] | None = None,
) -> dict:
    """Return a realistic analysis_results dict matching what the Analysis Agent produces."""
    countries = countries or ["FR", "DE"]
    results = {}
    for country in countries:
        has_temp = country == "FR"  # Only FR has Copernicus data in the real DB
        results[country] = {
            "patterns": {
                "actual_window_days":    1,
                "requested_window_days": 30,
                "fallback_used":         True,
                "variables": {
                    "generation_solar": {
                        "slope": 12.5,
                        "mean":  1200.0,
                        "min":   800.0,
                        "max":   1600.0,
                        "n":     3,
                    },
                    "load_actual_aggregated": {
                        "slope": -5.0,
                        "mean":  45000.0,
                        "min":   42000.0,
                        "max":   48000.0,
                        "n":     3,
                    },
                },
            },
            "risk": {
                "score":               42.5,
                "has_temperature_data": has_temp,
                "error":               None,
                "components": {
                    "demand_coverage":        15.0,
                    "renewable_intermittency": 30.0,
                    "hydraulic_buffer":        80.0,
                    "temperature_demand":       0.0 if not has_temp else 20.0,
                },
                "weights_used": {
                    "demand_coverage":        0.375 if not has_temp else 0.30,
                    "renewable_intermittency": 0.3125 if not has_temp else 0.25,
                    "hydraulic_buffer":        0.3125 if not has_temp else 0.25,
                    "temperature_demand":       0.0 if not has_temp else 0.20,
                },
                "auxiliary": {
                    "total_generation_mw": 40000.0,
                    "load_mw":             45000.0,
                    "renewables_mw":       12000.0,
                    "hydro_buffer_mw":     500.0,
                    "temperature_c":       22.5 if has_temp else None,
                },
            },
        }
    return results


def _mock_engine_and_redis(time_series_rows: list | None = None):
    """
    Return (mock_engine, mock_redis) for patching DB and Redis.

    time_series_rows: list of (period_datetime, variable_str, mean_float)
                      returned by the DATE_TRUNC query.
    """
    rows = time_series_rows or [
        (datetime(2024, 6, 1, tzinfo=timezone.utc), "generation_solar",        1200.0),
        (datetime(2024, 6, 1, tzinfo=timezone.utc), "load_actual_aggregated",  45000.0),
        (datetime(2024, 6, 2, tzinfo=timezone.utc), "generation_solar",        1350.0),
        (datetime(2024, 6, 2, tzinfo=timezone.utc), "load_actual_aggregated",  46000.0),
    ]

    mock_conn   = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)

    # First fetchone() call → anchor timestamp; subsequent fetchall() → rows
    anchor_ts = datetime(2024, 6, 2, 23, tzinfo=timezone.utc)
    mock_conn.execute.return_value.fetchone.return_value = (anchor_ts,)
    mock_conn.execute.return_value.fetchall.return_value = rows

    mock_redis = MagicMock()
    return mock_engine, mock_redis


# ===========================================================================
# TestBuildBarStats
# ===========================================================================

class TestBuildBarStats:
    """
    _build_bar_stats is a pure function — exhaustive tests, no mocking.
    """

    def test_returns_dict_keyed_by_country(self):
        from System2.Visualization.visualization_agent import _build_bar_stats
        result = _build_bar_stats(_make_analysis_results(["FR"]))
        assert "FR" in result

    def test_each_variable_has_required_keys(self):
        from System2.Visualization.visualization_agent import _build_bar_stats
        result = _build_bar_stats(_make_analysis_results(["FR"]))
        for key in ("mean", "min", "max", "slope", "n"):
            assert key in result["FR"]["generation_solar"], f"Missing key: {key}"

    def test_values_match_input(self):
        from System2.Visualization.visualization_agent import _build_bar_stats
        result = _build_bar_stats(_make_analysis_results(["FR"]))
        assert result["FR"]["generation_solar"]["mean"] == 1200.0
        assert result["FR"]["generation_solar"]["slope"] == 12.5

    def test_multiple_countries(self):
        from System2.Visualization.visualization_agent import _build_bar_stats
        result = _build_bar_stats(_make_analysis_results(["FR", "DE"]))
        assert "FR" in result
        assert "DE" in result

    def test_empty_analysis_results_returns_empty(self):
        from System2.Visualization.visualization_agent import _build_bar_stats
        assert _build_bar_stats({}) == {}

    def test_missing_patterns_key_handled(self):
        """Country with no 'patterns' key must not raise."""
        from System2.Visualization.visualization_agent import _build_bar_stats
        result = _build_bar_stats({"FR": {"risk": {}}})
        assert result["FR"] == {}


# ===========================================================================
# TestBuildCountryComparison
# ===========================================================================

class TestBuildCountryComparison:
    """
    _build_country_comparison inverts the {country: {variable: stats}} structure.
    """

    def test_inverts_structure(self):
        from System2.Visualization.visualization_agent import _build_country_comparison
        result = _build_country_comparison(_make_analysis_results(["FR", "DE"]))
        # generation_solar should contain both FR and DE
        assert "FR" in result["generation_solar"]
        assert "DE" in result["generation_solar"]

    def test_values_are_means(self):
        from System2.Visualization.visualization_agent import _build_country_comparison
        result = _build_country_comparison(_make_analysis_results(["FR"]))
        assert result["generation_solar"]["FR"] == 1200.0

    def test_empty_input_returns_empty(self):
        from System2.Visualization.visualization_agent import _build_country_comparison
        assert _build_country_comparison({}) == {}

    def test_variable_with_none_mean_excluded(self):
        """Variables with None mean must not appear in comparison."""
        from System2.Visualization.visualization_agent import _build_country_comparison
        analysis = {
            "FR": {
                "patterns": {
                    "variables": {
                        "generation_solar": {"mean": None, "min": 0, "max": 0, "slope": 0, "n": 0}
                    }
                }
            }
        }
        result = _build_country_comparison(analysis)
        assert "generation_solar" not in result or "FR" not in result.get("generation_solar", {})

    def test_all_variables_from_all_countries_present(self):
        from System2.Visualization.visualization_agent import _build_country_comparison
        result = _build_country_comparison(_make_analysis_results(["FR", "DE"]))
        # Both countries share the same variables in our test fixture
        for variable in ("generation_solar", "load_actual_aggregated"):
            assert variable in result
            assert len(result[variable]) == 2


# ===========================================================================
# TestBuildRiskBreakdown
# ===========================================================================

class TestBuildRiskBreakdown:
    """
    _build_risk_breakdown extracts C1–C4 component scores and weights.
    """

    def test_returns_dict_keyed_by_country(self):
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        result = _build_risk_breakdown(_make_analysis_results(["FR"]))
        assert "FR" in result

    def test_total_score_matches_input(self):
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        result = _build_risk_breakdown(_make_analysis_results(["FR"]))
        assert result["FR"]["total_score"] == 42.5

    def test_components_have_score_and_weight(self):
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        result = _build_risk_breakdown(_make_analysis_results(["FR"]))
        for component in result["FR"]["components"].values():
            assert "score"  in component
            assert "weight" in component

    def test_has_temperature_data_flag(self):
        """FR has temperature data, DE does not (per fixture)."""
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        result = _build_risk_breakdown(_make_analysis_results(["FR", "DE"]))
        assert result["FR"]["has_temperature_data"] is True
        assert result["DE"]["has_temperature_data"] is False

    def test_error_country_produces_error_key(self):
        """A country with risk error must produce an error entry, not raise."""
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        analysis = {
            "FR": {"risk": {"error": "No records available for FR"}}
        }
        result = _build_risk_breakdown(analysis)
        assert "error" in result["FR"]

    def test_four_components_present(self):
        from System2.Visualization.visualization_agent import _build_risk_breakdown
        result = _build_risk_breakdown(_make_analysis_results(["FR"]))
        components = result["FR"]["components"]
        expected = {
            "demand_coverage", "renewable_intermittency",
            "hydraulic_buffer", "temperature_demand",
        }
        assert set(components.keys()) == expected


# ===========================================================================
# TestFetchTimeSeries
# ===========================================================================

class TestFetchTimeSeries:
    """
    _fetch_time_series performs one SQL query per country.
    Tests verify the structure of the output and graceful failure handling.
    """

    def test_returns_dict_keyed_by_country(self):
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR"], "run-1", "day")
        assert "FR" in result

    def test_each_variable_is_list_of_t_v_dicts(self):
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR"], "run-1", "day")
        # Check structure for first variable found
        for variable, series in result["FR"].items():
            assert isinstance(series, list)
            assert all("t" in point and "v" in point for point in series)
            break

    def test_values_are_rounded_floats(self):
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR"], "run-1", "day")
        for series in result["FR"].values():
            assert all(isinstance(p["v"], float) for p in series)

    def test_no_anchor_returns_empty_for_country(self):
        """If no records exist for a run_id/country, must return empty dict for that country."""
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (None,)

        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR"], "run-1", "day")
        assert result["FR"] == {}

    def test_db_failure_returns_empty_for_country(self):
        """DB exception must not propagate — must return empty dict for that country."""
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR"], "run-1", "day")
        assert result["FR"] == {}

    def test_multiple_countries_queried(self):
        """Each country in the list must have an entry in the result."""
        from System2.Visualization.visualization_agent import _fetch_time_series
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = _fetch_time_series(["FR", "DE"], "run-1", "day")
        assert "FR" in result
        assert "DE" in result


# ===========================================================================
# TestVizNode
# ===========================================================================

class TestVizNode:
    """
    viz_node must produce all four chart structures.
    The DB query (_fetch_time_series) is mocked; the in-memory transforms
    are tested via their pure-function tests above.
    """

    def test_viz_data_contains_all_keys(self):
        state = _make_state()
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            from System2.Visualization.visualization_agent import viz_node
            result = viz_node(state)
        for key in ("time_series", "bar_stats", "country_comparison",
                    "risk_breakdown", "granularity", "generated_at"):
            assert key in result["viz_data"], f"Missing key: {key}"

    def test_granularity_in_viz_data_matches_config(self):
        from System2.Visualization.visualization_agent import viz_node, VIZ_TIME_GRANULARITY
        state = _make_state()
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = viz_node(state)
        assert result["viz_data"]["granularity"] == VIZ_TIME_GRANULARITY

    def test_generated_at_is_iso_string(self):
        from System2.Visualization.visualization_agent import viz_node
        state = _make_state()
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = viz_node(state)
        # Must be parseable as ISO datetime
        datetime.fromisoformat(result["viz_data"]["generated_at"])

    def test_empty_analysis_results_produces_empty_bar_stats(self):
        """Empty analysis_results must produce empty bar_stats without raising."""
        from System2.Visualization.visualization_agent import viz_node
        state = _make_state(analysis_results={})
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = viz_node(state)
        assert result["viz_data"]["bar_stats"] == {}

    def test_time_series_keyed_by_country(self):
        from System2.Visualization.visualization_agent import viz_node
        state = _make_state(countries=["FR", "DE"])
        mock_engine, _ = _mock_engine_and_redis()
        with patch("System2.Visualization.visualization_agent.engine", mock_engine):
            result = viz_node(state)
        ts = result["viz_data"]["time_series"]
        assert "FR" in ts
        assert "DE" in ts


# ===========================================================================
# TestSaveVizNode
# ===========================================================================

class TestSaveVizNode:
    """
    save_viz_node must UPDATE analysis_runs.viz_json and publish to Redis.
    """

    def _state_with_viz(self) -> dict:
        state = _make_state()
        state["viz_data"] = {
            "time_series":        {"FR": {"generation_solar": [{"t": "2024-06-01", "v": 1200.0}]}},
            "bar_stats":          {"FR": {"generation_solar": {"mean": 1200.0}}},
            "country_comparison": {"generation_solar": {"FR": 1200.0}},
            "risk_breakdown":     {"FR": {"total_score": 42.5}},
            "granularity":        "day",
            "generated_at":       "2024-06-01T00:00:00+00:00",
        }
        return state

    def test_no_error_on_success(self):
        state = self._state_with_viz()
        mock_engine, mock_redis = _mock_engine_and_redis()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            result = save_viz_node(state)
        assert result["viz_error"] is None

    def test_publishes_to_redis(self):
        state = self._state_with_viz()
        mock_engine, mock_redis = _mock_engine_and_redis()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            save_viz_node(state)
        mock_redis.publish.assert_called_once()

    def test_redis_event_is_viz_complete(self):
        state = self._state_with_viz()
        mock_engine, mock_redis = _mock_engine_and_redis()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            save_viz_node(state)
        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["event"] == "viz_complete"

    def test_redis_payload_contains_required_fields(self):
        state = self._state_with_viz()
        mock_engine, mock_redis = _mock_engine_and_redis()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            save_viz_node(state)
        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "countries", "timestamp"):
            assert field in parsed, f"Missing field: {field}"

    def test_error_captured_on_db_failure(self):
        """viz_error must be set (not raised) when DB update fails."""
        state = self._state_with_viz()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        mock_redis = MagicMock()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            result = save_viz_node(state)
        assert result["viz_error"] is not None
        assert "DB unavailable" in result["viz_error"]

    def test_redis_publishes_even_on_db_failure(self):
        """Redis must publish viz_complete even when the DB update fails."""
        state = self._state_with_viz()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")
        mock_redis = MagicMock()
        with (
            patch("System2.Visualization.visualization_agent.engine",    mock_engine),
            patch("System2.Visualization.visualization_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Visualization.visualization_agent import save_viz_node
            save_viz_node(state)
        mock_redis.publish.assert_called_once()


# ===========================================================================
# TestGranularityValidation
# ===========================================================================

class TestGranularityValidation:
    """
    VIZ_TIME_GRANULARITY must only accept whitelisted values.
    Invalid values must fall back to 'day' without raising.
    """

    def test_valid_granularities_accepted(self):
        """'hour', 'day', 'week' are valid and must not trigger fallback."""
        import importlib
        for valid in ("hour", "day", "week"):
            with patch.dict("os.environ", {"VIZ_TIME_GRANULARITY": valid}):
                import System2.Visualization.visualization_agent as mod
                importlib.reload(mod)
                assert mod.VIZ_TIME_GRANULARITY == valid

    def test_invalid_granularity_falls_back_to_day(self):
        """An invalid value like 'month' must fall back to 'day'."""
        import importlib
        with patch.dict("os.environ", {"VIZ_TIME_GRANULARITY": "month"}):
            import System2.Visualization.visualization_agent as mod
            importlib.reload(mod)
            assert mod.VIZ_TIME_GRANULARITY == "day"


# ===========================================================================
# Integration tests — require Docker (PostgreSQL + Redis) running
# ===========================================================================

@pytest.mark.integration
class TestVizIntegration:
    """
    End-to-end run of the Visualization Agent graph.
    Requires: PostgreSQL + Redis running, and a pre-existing analysis_runs
    row with analysis_results in AgentState (seeded directly here).
    """

    def _insert_analysis_run(self, run_id: str) -> None:
        """Insert a minimal analysis_runs row so UPDATE has a target."""
        from shared.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO analysis_runs
                        (run_id, triggered_by, started_at, status)
                    VALUES
                        (:run_id, 'test', NOW(), 'analysis_complete')
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

    def test_full_viz_run(self):
        """Full graph run must produce viz_data and no viz_error."""
        from System2.Visualization.visualization_agent import invoke_viz_graph
        state = _make_state()
        run_id = state["run_id"]
        self._insert_analysis_run(run_id)
        try:
            result = invoke_viz_graph(state)
            assert result["viz_error"] is None
            assert "time_series"        in result["viz_data"]
            assert "bar_stats"          in result["viz_data"]
            assert "country_comparison" in result["viz_data"]
            assert "risk_breakdown"     in result["viz_data"]
        finally:
            self._delete_analysis_run(run_id)

    def test_viz_json_written_to_db(self):
        """viz_json column in analysis_runs must be populated after the run."""
        from shared.db import engine
        from sqlalchemy import text
        from System2.Visualization.visualization_agent import invoke_viz_graph
        state = _make_state()
        run_id = state["run_id"]
        self._insert_analysis_run(run_id)
        try:
            invoke_viz_graph(state)
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT viz_json FROM analysis_runs WHERE run_id = :run_id"),
                    {"run_id": run_id},
                ).fetchone()
            assert row is not None
            assert row[0] is not None
        finally:
            self._delete_analysis_run(run_id)