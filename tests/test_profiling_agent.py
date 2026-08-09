"""
Tests for System1/Profiling/profiling_agent.py

Unit tests  : fully mocked — no network, no DB, no Redis, no LLM calls.
Integration : marked @pytest.mark.integration — require live credentials
              and Docker running.

Run unit tests only:
    python -m pytest tests/test_profiling_agent.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_profiling_agent.py -v -m integration
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_state(
    run_type: str = "full",
    countries: list[str] | None = None,
    profile: dict | None = None,
) -> dict:
    """Return a minimal valid AgentState dict for profiling tests."""
    return {
        "run_id":          str(uuid.uuid4()),
        "countries":       countries or ["FR", "DE"],
        "date_from":       datetime(2024, 6, 1, 0,  tzinfo=timezone.utc),
        "date_to":         datetime(2024, 6, 1, 3,  tzinfo=timezone.utc),
        "run_type":        run_type,
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],
        "cycle_count":     0,
        "profile":         profile or {},
        "profile_summary": "",
        "profiling_error": None,
        # profiling fields must be present in the initial state dict so
        # LangGraph can merge node outputs into them correctly
    }


def _make_records(
    country: str = "FR",
    n: int = 3,
    variable: str = "generation_solar",
    source_api: str = "entsoe",
    base_value: float = 1000.0,
) -> list[dict]:
    """Generate synthetic records compatible with energy_climate_records schema."""
    return [
        {
            "timestamp":  datetime(2024, 6, 1, i, tzinfo=timezone.utc),
            "source_api": source_api,
            "country":    country,
            "variable":   variable,
            "value":      base_value + i * 100.0,
            "unit":       "MW",
            "metadata":   {},
        }
        for i in range(n)
    ]


def _make_full_batch(country: str = "FR") -> list[dict]:
    """
    Return a complete batch for a 'full' run — all 4 expected variable types.
    Used to test schema_diff with no missing variables.
    """
    return (
        _make_records(country, variable="generation_solar",       source_api="entsoe")
        + _make_records(country, variable="load_actual_aggregated", source_api="entsoe")
        + _make_records(country, variable="climate_temperature_2m", source_api="copernicus")
        + _make_records(country, variable="climate_solar_radiation", source_api="copernicus")
    )


# ===========================================================================
# TestComputeSchemaDiff
# ===========================================================================

class TestComputeSchemaDiff:
    """
    Why test schema_diff in isolation?
    It is a pure function with no dependencies — deterministic given records
    and run_type. Exhaustive unit tests here give fast, precise feedback
    without any graph machinery involved.
    """

    def test_no_missing_variables_full_run(self):
        """A complete full-run batch must produce empty missing and unexpected lists."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        records = _make_full_batch("FR")
        result = compute_schema_diff(records, "full", "FR")
        assert result["missing"] == []
        assert result["unexpected"] == []

    def test_missing_copernicus_variables_full_run(self):
        """A full run with only ENTSO-E records must flag both copernicus variables."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        records = (
            _make_records("FR", variable="generation_solar",        source_api="entsoe")
            + _make_records("FR", variable="load_actual_aggregated", source_api="entsoe")
        )
        result = compute_schema_diff(records, "full", "FR")
        assert "copernicus:climate_temperature_2m"  in result["missing"]
        assert "copernicus:climate_solar_radiation" in result["missing"]

    def test_missing_load_variable(self):
        """A batch missing load_actual_aggregated must flag it."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        records = _make_records("FR", variable="generation_solar", source_api="entsoe")
        result = compute_schema_diff(records, "full", "FR")
        assert "entsoe:load_actual_aggregated" in result["missing"]

    def test_missing_generation_prefix(self):
        """A batch with no generation_* variable must flag entsoe:generation_*."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        records = (
            _make_records("FR", variable="load_actual_aggregated", source_api="entsoe")
            + _make_records("FR", variable="climate_temperature_2m", source_api="copernicus")
            + _make_records("FR", variable="climate_solar_radiation", source_api="copernicus")
        )
        result = compute_schema_diff(records, "full", "FR")
        assert "entsoe:generation_*" in result["missing"]

    def test_incremental_run_ignores_copernicus(self):
        """
        An incremental run with only ENTSO-E records must not flag copernicus
        variables as missing — they are not expected for incremental runs.
        """
        from System1.Profiling.profiling_agent import compute_schema_diff
        records = (
            _make_records("FR", variable="generation_solar",        source_api="entsoe")
            + _make_records("FR", variable="load_actual_aggregated", source_api="entsoe")
        )
        result = compute_schema_diff(records, "incremental", "FR")
        assert result["missing"] == []
        copernicus_missing = [m for m in result["missing"] if "copernicus" in m]
        assert copernicus_missing == []

    def test_empty_records_flags_all_expected(self):
        """An empty batch must flag all expected variables as missing."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        result = compute_schema_diff([], "full", "FR")
        assert len(result["missing"]) > 0

    def test_returns_dict_with_required_keys(self):
        """Return value must always contain 'missing' and 'unexpected' keys."""
        from System1.Profiling.profiling_agent import compute_schema_diff
        result = compute_schema_diff(_make_full_batch("FR"), "full", "FR")
        assert "missing"    in result
        assert "unexpected" in result


# ===========================================================================
# TestComputeDistributionStats
# ===========================================================================

class TestComputeDistributionStats:
    """
    Why test stats in isolation?
    The arithmetic is exact and independent of the graph. Testing here lets
    us verify the numpy computations without any mocking overhead.
    """

    def test_returns_dict_keyed_by_variable(self):
        """Result must be a dict keyed by variable name."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=5)
        result = compute_distribution_stats(records, "FR")
        assert "generation_solar" in result

    def test_required_stat_keys_present(self):
        """Each variable dict must contain mean, std, min, max, p25, p50, p75, n."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=5)
        stats = compute_distribution_stats(records, "FR")["generation_solar"]
        for key in ("mean", "std", "min", "max", "p25", "p50", "p75", "n"):
            assert key in stats, f"Missing key: {key}"

    def test_n_equals_record_count(self):
        """n must equal the number of records for that variable."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=7)
        stats = compute_distribution_stats(records, "FR")
        assert stats["generation_solar"]["n"] == 7

    def test_mean_is_correct(self):
        """Mean must match numpy's mean of the input values."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=3, base_value=1000.0)
        # values: 1000, 1100, 1200 → mean = 1100
        stats = compute_distribution_stats(records, "FR")
        assert stats["generation_solar"]["mean"] == pytest.approx(1100.0, abs=1e-6)

    def test_min_max_correct(self):
        """Min and max must match the actual extremes of the values."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=3, base_value=1000.0)
        stats = compute_distribution_stats(records, "FR")["generation_solar"]
        assert stats["min"] == pytest.approx(1000.0)
        assert stats["max"] == pytest.approx(1200.0)

    def test_multiple_variables_returned(self):
        """Stats must be computed independently for each variable."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = (
            _make_records("FR", variable="generation_solar",        n=3)
            + _make_records("FR", variable="load_actual_aggregated", n=3)
        )
        result = compute_distribution_stats(records, "FR")
        assert "generation_solar"        in result
        assert "load_actual_aggregated"  in result

    def test_single_record_does_not_raise(self):
        """A single record must not raise — std will be 0, percentiles equal mean."""
        from System1.Profiling.profiling_agent import compute_distribution_stats
        records = _make_records("FR", variable="generation_solar", n=1)
        result = compute_distribution_stats(records, "FR")
        assert result["generation_solar"]["n"] == 1
        assert result["generation_solar"]["std"] == pytest.approx(0.0)


# ===========================================================================
# TestDetectDrift
# ===========================================================================

class TestDetectDrift:
    """
    Why mock the DB in drift detection tests?
    detect_drift queries PostgreSQL for historical records. We want to test
    the KL divergence logic and the skip conditions without requiring a
    running database. We inject historical values by patching engine.connect.
    """

    def _mock_historical(self, values: list[float]) -> MagicMock:
        """Return a mock engine.connect() context that yields the given values."""
        mock_conn   = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([(v,) for v in values]))
        mock_conn.execute.return_value = mock_result
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        return mock_engine

    def test_skipped_when_insufficient_history(self):
        """
        detect_drift must set skipped=True when fewer than MIN_HISTORICAL_RECORDS
        historical records exist for the variable/country pair.
        """
        from System1.Profiling.profiling_agent import detect_drift
        records = _make_records("FR", variable="generation_solar", n=3)

        # Only 2 historical records — below MIN_HISTORICAL_RECORDS (10)
        mock_engine = self._mock_historical([500.0, 600.0])

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        assert result["skipped"] is True
        assert result["kl"] is None
        assert result["drift_detected"] is False

    def test_no_drift_when_distributions_identical(self):
        """
        Identical current and historical distributions must produce KL ≈ 0
        and drift_detected = False.
        """
        from System1.Profiling.profiling_agent import detect_drift

        values = [1000.0, 1100.0, 1200.0] * 5   # 15 records — above threshold
        records = [
            {"timestamp": datetime(2024, 6, 3, 10, tzinfo=timezone.utc),
             "variable": "generation_solar", "value": v}
            for v in values
        ]
        mock_engine = self._mock_historical(values)   # same distribution

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        assert result["skipped"] is False
        assert result["kl"] == pytest.approx(0.0, abs=1e-6)
        assert result["drift_detected"] is False

    def test_drift_detected_when_distributions_differ_significantly(self):
        """
        A current distribution far from the historical one must produce
        drift_detected = True.
        """
        from System1.Profiling.profiling_agent import detect_drift

        # Historical: values around 1000; current: values around 5000
        historical = [1000.0 + i for i in range(20)]
        current_records = [
            {"timestamp": datetime(2024, 6, 3, 10, tzinfo=timezone.utc),
             "variable": "generation_solar", "value": 5000.0 + i}
            for i in range(10)
        ]
        mock_engine = self._mock_historical(historical)

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(current_records, "run-1", "generation_solar", "FR")

        assert result["skipped"] is False
        assert result["drift_detected"] is True
        assert result["kl"] > 0.1

    def test_returns_required_keys(self):
        """Return dict must always contain all required keys."""
        from System1.Profiling.profiling_agent import detect_drift
        records = _make_records("FR", variable="generation_solar", n=3)
        mock_engine = self._mock_historical([])   # no history → skipped

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        for key in ("kl", "drift_detected", "n_current", "n_historical",
                    "threshold_used", "skipped", "skip_reason"):
            assert key in result, f"Missing key: {key}"

    def test_threshold_used_matches_config(self):
        """threshold_used in the result must equal KL_DRIFT_THRESHOLD."""
        from System1.Profiling.profiling_agent import detect_drift, KL_DRIFT_THRESHOLD
        records = _make_records("FR", variable="generation_solar", n=3)
        mock_engine = self._mock_historical([])

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        assert result["threshold_used"] == KL_DRIFT_THRESHOLD

    def test_db_failure_produces_skipped_result(self):
        """
        If the PostgreSQL query fails, detect_drift must not raise —
        it must return skipped=True with the error context in skip_reason.
        """
        from System1.Profiling.profiling_agent import detect_drift
        records = _make_records("FR", variable="generation_solar", n=5)

        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")

        with patch("System1.Profiling.profiling_agent.engine", mock_engine):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        assert result["skipped"] is True
        assert result["drift_detected"] is False

    def test_empty_current_records_for_variable(self):
        """
        If the variable has no records in the current batch,
        detect_drift must return skipped=True immediately.
        """
        from System1.Profiling.profiling_agent import detect_drift

        # Records exist but for a different variable
        records = _make_records("FR", variable="load_actual_aggregated", n=5)

        with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
            result = detect_drift(records, "run-1", "generation_solar", "FR")

        assert result["skipped"] is True
        assert result["n_current"] == 0


# ===========================================================================
# TestProfilingNode
# ===========================================================================

class TestProfilingNode:
    """
    Why patch _record_store directly?
    profiling_node reads from the in-process store using run_id. We inject
    records directly into the store to decouple the profiling tests from
    the ingestion graph machinery.
    """

    def test_returns_profile_keyed_by_country(self):
        """profile dict must be keyed by country code."""
        state = _make_state(countries=["FR"])

        with patch("System1.Profiling.profiling_agent._record_store",
                   {state["run_id"]: _make_full_batch("FR")}):
            with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
                from System1.Profiling.profiling_agent import profiling_node
                result = profiling_node(state)

        assert "FR" in result["profile"]

    def test_profile_contains_required_sections(self):
        """Each country entry must contain schema, stats, drift, and n_records."""
        state = _make_state(countries=["FR"])

        with patch("System1.Profiling.profiling_agent._record_store",
                   {state["run_id"]: _make_full_batch("FR")}):
            with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
                from System1.Profiling.profiling_agent import profiling_node
                result = profiling_node(state)

        fr_profile = result["profile"]["FR"]
        for key in ("schema", "stats", "drift", "n_records"):
            assert key in fr_profile, f"Missing section: {key}"

    def test_n_records_matches_input(self):
        """n_records in profile must equal the number of records for that country."""
        state = _make_state(countries=["FR"])
        records = _make_full_batch("FR")

        with patch("System1.Profiling.profiling_agent._record_store",
                   {state["run_id"]: records}):
            with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
                from System1.Profiling.profiling_agent import profiling_node
                result = profiling_node(state)

        assert result["profile"]["FR"]["n_records"] == len(records)

    def test_multiple_countries_profiled_independently(self):
        """profiling_node must produce separate profiles for each country."""
        state = _make_state(countries=["FR", "DE"])
        records = _make_full_batch("FR") + _make_full_batch("DE")

        with patch("System1.Profiling.profiling_agent._record_store",
                   {state["run_id"]: records}):
            with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
                from System1.Profiling.profiling_agent import profiling_node
                result = profiling_node(state)

        assert "FR" in result["profile"]
        assert "DE" in result["profile"]

    def test_empty_store_returns_empty_profile(self):
        """If _record_store has no records for run_id, profile must be empty."""
        state = _make_state()

        with patch("System1.Profiling.profiling_agent._record_store", {}):
            from System1.Profiling.profiling_agent import profiling_node
            result = profiling_node(state)

        assert result["profile"] == {}

    def test_drift_key_present_for_each_variable(self):
        """drift dict must contain one entry per variable found in stats."""
        state = _make_state(countries=["FR"])
        records = _make_full_batch("FR")

        with patch("System1.Profiling.profiling_agent._record_store",
                   {state["run_id"]: records}):
            with patch("System1.Profiling.profiling_agent.engine", MagicMock()):
                from System1.Profiling.profiling_agent import profiling_node
                result = profiling_node(state)

        fr = result["profile"]["FR"]
        assert set(fr["drift"].keys()) == set(fr["stats"].keys())


# ===========================================================================
# TestSummaryNode
# ===========================================================================

class TestSummaryNode:
    """
    Why mock chat_complete and not _call_groq directly?
    summary_node calls chat_complete() from shared.llm_client — that is the
    public interface. Mocking at that level is the correct seam: it verifies
    that summary_node uses the shared LLM client without coupling the test
    to the internal provider routing logic.
    """

    def _make_profile(self) -> dict:
        return {
            "FR": {
                "n_records": 12,
                "schema": {"missing": [], "unexpected": []},
                "stats":  {"generation_solar": {"mean": 1200.0, "std": 150.0,
                                                 "min": 900.0,  "max": 1500.0,
                                                 "p25": 1100.0, "p50": 1200.0,
                                                 "p75": 1350.0, "n": 3}},
                "drift":  {"generation_solar": {"kl": 0.05, "drift_detected": False,
                                                 "skipped": False, "skip_reason": None,
                                                 "n_current": 3, "n_historical": 20,
                                                 "threshold_used": 0.1}},
            }
        }

    def test_summary_node_calls_llm(self):
        """summary_node must call chat_complete exactly once."""
        state = _make_state(profile=self._make_profile())

        with patch("System1.Profiling.profiling_agent.chat_complete",
                   return_value=("FR: schema complete, no drift detected.", "groq")) as mock_llm:
            from System1.Profiling.profiling_agent import summary_node
            summary_node(state)

        mock_llm.assert_called_once()

    def test_summary_node_returns_string(self):
        """profile_summary must be a non-empty string."""
        state = _make_state(profile=self._make_profile())

        with patch("System1.Profiling.profiling_agent.chat_complete",
                   return_value=("FR: schema complete, no drift detected.", "groq")):
            from System1.Profiling.profiling_agent import summary_node
            result = summary_node(state)

        assert isinstance(result["profile_summary"], str)
        assert len(result["profile_summary"]) > 0

    def test_summary_node_stores_provider(self):
        """llm_provider must be set from chat_complete return value."""
        state = _make_state(profile=self._make_profile())

        with patch("System1.Profiling.profiling_agent.chat_complete",
                   return_value=("Some summary.", "groq")):
            from System1.Profiling.profiling_agent import summary_node
            result = summary_node(state)

        assert result["llm_provider"] == "groq"

    def test_summary_node_handles_llm_failure_gracefully(self):
        """If chat_complete raises, summary_node must not propagate the exception."""
        state = _make_state(profile=self._make_profile())

        with patch("System1.Profiling.profiling_agent.chat_complete",
                   side_effect=RuntimeError("Both providers failed")):
            from System1.Profiling.profiling_agent import summary_node
            result = summary_node(state)

        assert "profile_summary" in result
        assert result["llm_provider"] is None

    def test_summary_node_empty_profile_skips_llm(self):
        """If profile is empty, summary_node must not call the LLM."""
        state = _make_state(profile={})

        with patch("System1.Profiling.profiling_agent.chat_complete") as mock_llm:
            from System1.Profiling.profiling_agent import summary_node
            result = summary_node(state)

        mock_llm.assert_not_called()
        assert result["profile_summary"] != ""   # fallback message present


# ===========================================================================
# TestSaveProfileNode
# ===========================================================================

class TestSaveProfileNode:
    """
    Why patch engine.connect and get_redis?
    save_profile_node has two external dependencies: PostgreSQL and Redis.
    Patching at the engine level lets us verify the SQL is attempted and
    the correct Redis message is published without requiring live services.
    """

    def _make_profile_with_drift(self) -> dict:
        return {
            "FR": {
                "n_records": 12,
                "schema": {"missing": ["entsoe:load_actual_aggregated"], "unexpected": []},
                "stats":  {"generation_solar": {"mean": 5000.0, "std": 200.0,
                                                 "min": 4600.0, "max": 5400.0,
                                                 "p25": 4800.0, "p50": 5000.0,
                                                 "p75": 5200.0, "n": 12}},
                "drift":  {"generation_solar": {"kl": 0.25, "drift_detected": True,
                                                 "skipped": False, "skip_reason": None,
                                                 "n_current": 12, "n_historical": 50,
                                                 "threshold_used": 0.1}},
            }
        }

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def test_no_error_on_success(self):
        """profiling_error must be None when the DB write succeeds."""
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "FR: drift detected in generation_solar."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            result = save_profile_node(state)

        assert result["profiling_error"] is None

    def test_publishes_to_redis(self):
        """save_profile_node must publish exactly one message to Redis."""
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "FR: drift detected."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            save_profile_node(state)

        mock_redis.publish.assert_called_once()

    def test_redis_payload_contains_required_fields(self):
        """Redis message must contain run_id, event, n_records, n_anomalies, countries."""
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "FR: drift detected."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            save_profile_node(state)

        channel, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        assert channel == "validated_data"
        for field in ("run_id", "event", "n_records", "n_anomalies", "countries", "timestamp"):
            assert field in parsed, f"Missing field in Redis payload: {field}"

    def test_redis_payload_event_is_profiling_complete(self):
        """event field in Redis payload must be 'profiling_complete'."""
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "summary"
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            save_profile_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["event"] == "profiling_complete"

    def test_n_anomalies_counts_drift_and_missing(self):
        """
        n_anomalies in the Redis payload must equal
        len(drift_alerts) + len(missing_variables).
        Profile has 1 drift alert + 1 missing variable = 2 anomalies.
        """
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "summary"
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            save_profile_node(state)

        _, payload = mock_redis.publish.call_args[0]
        assert json.loads(payload)["n_anomalies"] == 2

    def test_error_captured_on_db_failure(self):
        """profiling_error must be set (not raised) when the DB write fails."""
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "summary"
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            result = save_profile_node(state)

        assert result["profiling_error"] is not None
        assert "DB unavailable" in result["profiling_error"]

    def test_redis_still_publishes_on_db_failure(self):
        """
        Redis publish must still happen even when the DB write fails —
        downstream agents should know profiling ran even if persistence failed.
        """
        state = _make_state(profile=self._make_profile_with_drift())
        state["profile_summary"] = "summary"
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System1.Profiling.profiling_agent.engine",    mock_engine),
            patch("System1.Profiling.profiling_agent.get_redis", return_value=mock_redis),
        ):
            from System1.Profiling.profiling_agent import save_profile_node
            save_profile_node(state)

        mock_redis.publish.assert_called_once()


# ===========================================================================
# Integration tests — require live credentials + Docker running
# ===========================================================================

@pytest.mark.integration
class TestProfilingIntegration:
    """
    End-to-end run of the full Profiling Agent graph.
    Requires: GROQ_API_KEY, PostgreSQL + Redis running.
    Records are seeded directly into _record_store to avoid depending on
    the Ingestion Agent running first.
    """

    def test_full_profile_fr(self):
        """Full profiling run for France — profile must contain all sections."""
        from System1.Profiling.profiling_agent import (
            _record_store, invoke_profiling_graph
        )
        state = _make_state(run_type="full", countries=["FR"])
        _record_store[state["run_id"]] = _make_full_batch("FR")

        result = invoke_profiling_graph(state)

        assert "FR" in result["profile"]
        assert result["profiling_error"] is None
        assert isinstance(result["profile_summary"], str)
        assert len(result["profile_summary"]) > 0

    def test_profile_summary_mentions_country(self):
        """LLM summary must mention the country name."""
        from System1.Profiling.profiling_agent import (
            _record_store, invoke_profiling_graph
        )
        state = _make_state(run_type="full", countries=["FR"])
        _record_store[state["run_id"]] = _make_full_batch("FR")

        result = invoke_profiling_graph(state)

        assert "FR" in result["profile_summary"]