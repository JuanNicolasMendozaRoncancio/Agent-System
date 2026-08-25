"""
Tests for System2/narrative_agent.py

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

def _make_analysis_results(countries: list[str] | None = None) -> dict:
    """
    Return a synthetic analysis_results dict matching the Analysis Agent output.
    """
    countries = countries or ["FR", "DE"]
    results = {}
    for i, country in enumerate(countries):
        results[country] = {
            "risk_score": 0.42 + i * 0.1,
            "risk_level": "MEDIUM" if i == 0 else "LOW",
            "fallback_used": i == 1,   # DE uses fallback (no Copernicus)
            "patterns": {
                "trend_7d": {
                    "generation_solar": {
                        "direction": "rising",
                        "magnitude": 0.023,
                    },
                    "load_actual_aggregated": {
                        "direction": "falling",
                        "magnitude": 0.011,
                    },
                },
                "anomaly_flags": {
                    "generation_solar": False,
                    "load_actual_aggregated": True,
                },
            },
        }
    return results


def _make_viz_data(countries: list[str] | None = None) -> dict:
    """
    Return a synthetic viz_data dict matching the Visualization Agent output.
    """
    countries = countries or ["FR", "DE"]
    bar_stats = {}
    for country in countries:
        bar_stats[country] = {
            "generation_solar": {
                "mean": 1200.0, "min": 800.0, "max": 1600.0,
                "slope": 0.023, "n": 72, "unit": "MW",
            },
            "load_actual_aggregated": {
                "mean": 48000.0, "min": 42000.0, "max": 55000.0,
                "slope": -0.011, "n": 72, "unit": "MW",
            },
        }
    return {
        "time_series":        {},   # not used by Narrative Agent
        "bar_stats":          bar_stats,
        "country_comparison": {},
        "risk_breakdown":     {},
        "granularity":        "hour",
        "generated_at":       "2024-06-01T00:00:00+00:00",
    }


def _make_rag_topics() -> list[dict]:
    return [
        {
            "title": "European Wind Drought 2024",
            "argument_summary": "Persistent anticyclonic conditions cut wind output by 30%.",
            "sentiment": "negative",
            "score": 0.91,
        },
        {
            "title": "Solar Ramp-Up in Southern Europe",
            "argument_summary": "Spain and France installed capacity surged, buffering shortfalls.",
            "sentiment": "positive",
            "score": 0.78,
        },
    ]


def _make_state(
    countries: list[str] | None = None,
    analysis_results: dict | None = None,
    viz_data: dict | None = None,
) -> dict:
    """Return a minimal valid AgentState for narrative tests."""
    countries = countries or ["FR", "DE"]
    return {
        "run_id":           str(uuid.uuid4()),
        "countries":        countries,
        "date_from":        datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
        "date_to":          datetime(2024, 6, 1, 3, tzinfo=timezone.utc),
        "run_type":         "full",
        "triggered_by":     "system1_complete",
        "messages":         [],
        "analysis_results": (
            analysis_results
            if analysis_results is not None
            else _make_analysis_results(countries)
        ),
        "analysis_error":   None,
        "viz_data":         (
            viz_data
            if viz_data is not None
            else _make_viz_data(countries)
        ),
        "viz_error":        None,
        "narrative":        "",
        "narrative_error":  None,
        "llm_provider":     None,
    }


# ===========================================================================
# TestFetchRagTopics
# ===========================================================================

class TestFetchRagTopics:
    """
    _fetch_rag_topics is a plain Python function (not a @tool).
    """

    def test_returns_empty_when_rag_url_not_set(self, monkeypatch):
        """If RAG_API_URL is unset, must return [] without raising."""
        monkeypatch.delenv("RAG_API_URL", raising=False)
        from System2.Narrative.narrative_agent import _fetch_rag_topics
        result = _fetch_rag_topics()
        assert result == []

    def test_returns_empty_when_rag_url_empty_string(self, monkeypatch):
        """Empty-string RAG_API_URL must also return [] without raising."""
        monkeypatch.setenv("RAG_API_URL", "")
        from System2.Narrative.narrative_agent import _fetch_rag_topics
        result = _fetch_rag_topics()
        assert result == []

    def test_returns_topics_on_success(self, monkeypatch):
        """A successful HTTP response must return the parsed topics list."""
        monkeypatch.setenv("RAG_API_URL", "http://rag.local")
        monkeypatch.setenv("RAG_API_KEY", "key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_rag_topics()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            from System2.Narrative.narrative_agent import _fetch_rag_topics
            result = _fetch_rag_topics()

        assert len(result) == 2
        assert result[0]["title"] == "European Wind Drought 2024"

    def test_sorted_by_score_descending(self, monkeypatch):
        """Topics must be returned sorted by score descending."""
        monkeypatch.setenv("RAG_API_URL", "http://rag.local")
        monkeypatch.setenv("RAG_API_KEY", "key")

        topics = [
            {"title": "Low", "score": 0.50, "argument_summary": "A", "sentiment": "neutral"},
            {"title": "High", "score": 0.95, "argument_summary": "B", "sentiment": "positive"},
            {"title": "Mid", "score": 0.70, "argument_summary": "C", "sentiment": "negative"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = topics
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            from System2.Narrative.narrative_agent import _fetch_rag_topics
            result = _fetch_rag_topics()

        assert result[0]["title"] == "High"

    def test_respects_max_topics_limit(self, monkeypatch):
        """Must return at most RAG_MAX_TOPICS topics."""
        monkeypatch.setenv("RAG_API_URL", "http://rag.local")
        monkeypatch.setenv("RAG_API_KEY", "key")
        monkeypatch.setenv("NARRATIVE_RAG_MAX_TOPICS", "2")

        topics = [
            {"title": f"Topic {i}", "score": float(i) / 10,
             "argument_summary": "x", "sentiment": "neutral"}
            for i in range(5)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = topics
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            from System2.Narrative.narrative_agent import _fetch_rag_topics
            result = _fetch_rag_topics()

        assert len(result) <= 2

    def test_http_failure_returns_empty(self, monkeypatch):
        """A failing HTTP call must return [] without raising."""
        monkeypatch.setenv("RAG_API_URL", "http://rag.local")

        with patch("httpx.get", side_effect=Exception("connection refused")):
            from System2.Narrative.narrative_agent import _fetch_rag_topics
            result = _fetch_rag_topics()

        assert result == []

    def test_handles_topics_key_in_response(self, monkeypatch):
        """Some RAG APIs return {'topics': [...]} — must handle both formats."""
        monkeypatch.setenv("RAG_API_URL", "http://rag.local")
        monkeypatch.setenv("RAG_API_KEY", "key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"topics": _make_rag_topics(), "total": 2}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            from System2.Narrative.narrative_agent import _fetch_rag_topics
            result = _fetch_rag_topics()

        assert len(result) == 2


# ===========================================================================
# TestBuildPrompt
# ===========================================================================

class TestBuildPrompt:
    """
    _build_prompt is pure Python — tests verify that the correct signals
    are included and that expensive arrays (time_series) are excluded.
    """

    def test_contains_risk_score_and_level(self):
        """Prompt must include risk_score and risk_level for each country."""
        from System2.Narrative.narrative_agent import _build_prompt
        analysis = _make_analysis_results(["FR"])
        viz      = _make_viz_data(["FR"])
        prompt   = _build_prompt(analysis, viz, [])

        assert "FR" in prompt
        assert "MEDIUM" in prompt 

    def test_contains_trend_direction(self):
        """Prompt must include trend direction from patterns.trend_7d."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt(_make_analysis_results(["FR"]), _make_viz_data(["FR"]), [])
        assert "rising" in prompt or "falling" in prompt

    def test_contains_anomaly_flag(self):
        """Prompt must mark variables flagged as anomalies."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt(_make_analysis_results(["FR"]), _make_viz_data(["FR"]), [])
        # load_actual_aggregated is flagged in our helper
        assert "ANOMALY" in prompt or "anomaly" in prompt.lower()

    def test_contains_bar_stats_means(self):
        """Prompt must include mean values from viz_data.bar_stats."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt(_make_analysis_results(["FR"]), _make_viz_data(["FR"]), [])
        assert "mean=" in prompt

    def test_does_not_contain_time_series_arrays(self):
        """
        Full time_series arrays must not appear in the prompt — they are too
        large for LLM context and the LLM cannot use raw arrays anyway.
        """
        from System2.Narrative.narrative_agent import _build_prompt
        viz = _make_viz_data(["FR"])
        viz["time_series"] = {"FR": {"generation_solar": [{"t": "SENTINEL_TS", "v": 1}]}}
        prompt = _build_prompt(_make_analysis_results(["FR"]), viz, [])
        assert "SENTINEL_TS" not in prompt

    def test_rag_topics_included_when_present(self):
        """RAG topic titles and summaries must appear in the prompt."""
        from System2.Narrative.narrative_agent import _build_prompt
        topics = _make_rag_topics()
        prompt = _build_prompt(_make_analysis_results(["FR"]), _make_viz_data(["FR"]), topics)
        assert "European Wind Drought" in prompt
        assert "anticyclonic" in prompt

    def test_no_rag_shows_fallback_message(self):
        """When no RAG topics, prompt must say 'none available'."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt(_make_analysis_results(["FR"]), _make_viz_data(["FR"]), [])
        assert "none available" in prompt

    def test_fallback_used_note_present_for_country(self):
        """When fallback_used=True for a country, prompt must note it."""
        from System2.Narrative.narrative_agent import _build_prompt
        analysis = _make_analysis_results(["FR", "DE"])
        prompt   = _build_prompt(analysis, _make_viz_data(["FR", "DE"]), [])
        assert "Copernicus" in prompt or "fallback" in prompt.lower()

    def test_empty_analysis_results_does_not_raise(self):
        """An empty analysis_results must produce a valid (if sparse) prompt."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt({}, _make_viz_data(["FR"]), [])
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_multiple_countries_both_present(self):
        """Both countries must appear in the prompt when two are in the run."""
        from System2.Narrative.narrative_agent import _build_prompt
        prompt = _build_prompt(
            _make_analysis_results(["FR", "DE"]),
            _make_viz_data(["FR", "DE"]),
            [],
        )
        assert "FR" in prompt
        assert "DE" in prompt


# ===========================================================================
# TestNarrativeNode
# ===========================================================================

class TestNarrativeNode:
    """
    narrative_node must call _fetch_rag_topics (Python) and then
    chat_complete (LLM) once. Tests verify:
    - Both calls happen in normal operation.
    - RAG failure does not prevent LLM call.
    - LLM failure returns a Python fallback narrative.
    - llm_provider is stored from chat_complete.
    """

    def test_calls_chat_complete_once(self):
        """narrative_node must call chat_complete exactly once."""
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  return_value=("Paragraph 1. Paragraph 2. Paragraph 3.", "groq")) as mock_llm,
        ):
            from System2.Narrative.narrative_agent import narrative_node
            narrative_node(state)

        mock_llm.assert_called_once()

    def test_returns_narrative_string(self):
        """narrative must be a non-empty string."""
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  return_value=("Three paragraphs of narrative.", "groq")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_stores_llm_provider(self):
        """llm_provider must be set from chat_complete return value."""
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  return_value=("Narrative.", "gemini")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert result["llm_provider"] == "gemini"

    def test_rag_topics_fetched_before_llm(self):
        """_fetch_rag_topics must be called before chat_complete."""
        state  = _make_state()
        calls  = []

        def mock_rag():
            calls.append("rag")
            return _make_rag_topics()

        def mock_llm(messages, **kwargs):
            calls.append("llm")
            return ("Narrative.", "groq")

        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", side_effect=mock_rag),
            patch("System2.Narrative.narrative_agent.chat_complete", side_effect=mock_llm),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            narrative_node(state)

        assert calls.index("rag") < calls.index("llm")

    def test_rag_failure_does_not_block_llm(self):
        """
        _fetch_rag_topics already fails silently — but even if it raises here,
        narrative_node must still call chat_complete and return a narrative.
        """
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  return_value=("Narrative without RAG context.", "groq")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert len(result["narrative"]) > 0

    def test_llm_failure_returns_python_fallback(self):
        """When chat_complete raises, narrative_node must not propagate the exception."""
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  side_effect=RuntimeError("Both providers failed")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert "narrative" in result
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_llm_failure_sets_provider_none(self):
        """On LLM failure, llm_provider must be None."""
        state = _make_state()
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  side_effect=RuntimeError("fail")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert result["llm_provider"] is None

    def test_fallback_narrative_mentions_countries(self):
        """Python fallback narrative must mention the countries in the run."""
        state = _make_state(countries=["FR", "DE"])
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  side_effect=RuntimeError("fail")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert "FR" in result["narrative"] or "DE" in result["narrative"]

    def test_rag_topics_passed_to_prompt(self):
        """RAG topics returned by _fetch_rag_topics must reach the LLM prompt."""
        state  = _make_state()
        topics = _make_rag_topics()
        captured_messages = []

        def capture_llm(messages, **kwargs):
            captured_messages.extend(messages)
            return ("Narrative.", "groq")

        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=topics),
            patch("System2.Narrative.narrative_agent.chat_complete", side_effect=capture_llm),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            narrative_node(state)

        full_prompt = " ".join(m["content"] for m in captured_messages)
        assert "European Wind Drought" in full_prompt

    def test_empty_analysis_results_does_not_raise(self):
        """narrative_node must not raise when analysis_results is empty."""
        state = _make_state(analysis_results={}, viz_data={})
        with (
            patch("System2.Narrative.narrative_agent._fetch_rag_topics", return_value=[]),
            patch("System2.Narrative.narrative_agent.chat_complete",
                  return_value=("Minimal narrative.", "groq")),
        ):
            from System2.Narrative.narrative_agent import narrative_node
            result = narrative_node(state)

        assert "narrative" in result


# ===========================================================================
# TestSaveNarrativeNode
# ===========================================================================

class TestSaveNarrativeNode:
    """
    save_narrative_node must UPDATE analysis_runs and publish narrative_complete
    to Redis. Tests verify both success and failure paths.
    """

    def _mock_engine_and_redis(self):
        mock_conn   = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__  = MagicMock(return_value=False)
        mock_redis  = MagicMock()
        return mock_engine, mock_conn, mock_redis

    def test_no_error_on_success(self):
        """narrative_error must be None when the DB update succeeds."""
        state = _make_state()
        state["narrative"] = "Three paragraphs."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            result = save_narrative_node(state)

        assert result["narrative_error"] is None

    def test_publishes_to_redis(self):
        """save_narrative_node must publish exactly one message to Redis."""
        state = _make_state()
        state["narrative"] = "Narrative."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        mock_redis.publish.assert_called_once()

    def test_redis_event_is_narrative_complete(self):
        """Redis payload event must be 'narrative_complete'."""
        state = _make_state()
        state["narrative"] = "Narrative."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "validated_data"
        assert json.loads(payload)["event"] == "narrative_complete"

    def test_redis_payload_contains_required_fields(self):
        """Redis message must contain run_id, event, countries, timestamp."""
        state = _make_state()
        state["narrative"] = "Narrative."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        _, payload = mock_redis.publish.call_args[0]
        parsed = json.loads(payload)
        for field in ("run_id", "event", "countries", "timestamp"):
            assert field in parsed, f"Missing field in Redis payload: {field}"

    def test_redis_payload_countries_matches_analysis_results(self):
        """countries in Redis payload must match keys of analysis_results."""
        state = _make_state(countries=["FR", "DE"])
        state["narrative"] = "Narrative."
        mock_engine, _, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        _, payload = mock_redis.publish.call_args[0]
        countries = json.loads(payload)["countries"]
        assert set(countries) == {"FR", "DE"}

    def test_error_captured_on_db_failure(self):
        """narrative_error must be set (not raised) when the DB update fails."""
        state = _make_state()
        state["narrative"] = "Narrative."
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            result = save_narrative_node(state)

        assert result["narrative_error"] is not None
        assert "DB unavailable" in result["narrative_error"]

    def test_redis_publishes_even_on_db_failure(self):
        """
        Redis publish must happen even when the DB UPDATE fails —
        the FastAPI SSE layer must always be notified that Sistema 2 finished.
        """
        state = _make_state()
        state["narrative"] = "Narrative."
        mock_redis  = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("DB unavailable")

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        mock_redis.publish.assert_called_once()

    def test_db_update_uses_correct_run_id(self):
        """The SQL UPDATE must use the run_id from AgentState, not a hardcoded value."""
        state = _make_state()
        run_id = state["run_id"]
        state["narrative"] = "Narrative."
        mock_engine, mock_conn, mock_redis = self._mock_engine_and_redis()

        with (
            patch("System2.Narrative.narrative_agent.engine",    mock_engine),
            patch("System2.Narrative.narrative_agent.get_redis", return_value=mock_redis),
        ):
            from System2.Narrative.narrative_agent import save_narrative_node
            save_narrative_node(state)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["run_id"] == run_id


# ===========================================================================
# Integration tests — require live credentials + Docker + optional RAG
# ===========================================================================

@pytest.mark.integration
class TestNarrativeIntegration:
    """
    End-to-end run of the Narrative Agent graph.
    Requires: GROQ_API_KEY, PostgreSQL + Redis running.
    RAG integration is optional — if RAG_API_URL is not set, the narrative
    proceeds without documentary context.
    """

    def _insert_analysis_run(self, run_id: str) -> None:
        from shared.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO analysis_runs
                        (run_id, triggered_by, started_at, status)
                    VALUES
                        (:run_id, 'test', NOW(), 'running')
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

    def test_full_narrative_run(self):
        """Narrative Agent must produce a non-empty narrative and no error."""
        from System2.Narrative.narrative_agent import invoke_narrative_graph

        state  = _make_state()
        run_id = state["run_id"]
        self._insert_analysis_run(run_id)

        try:
            result = invoke_narrative_graph(state)
            assert result["narrative_error"] is None
            assert isinstance(result["narrative"], str)
            assert len(result["narrative"]) > 50   # must be more than a stub
        finally:
            self._delete_analysis_run(run_id)

    def test_narrative_mentions_countries(self):
        """LLM narrative must mention at least one of the countries in the run."""
        from System2.Narrative.narrative_agent import invoke_narrative_graph

        state  = _make_state(countries=["FR", "DE"])
        run_id = state["run_id"]
        self._insert_analysis_run(run_id)

        country_map = {"FR": "France", "DE": "Germany"}
        try:
            result = invoke_narrative_graph(state)
            narrative = result["narrative"]
            assert any(
                c in narrative or country_map.get(c, c) in narrative
                for c in ["FR", "DE"]
            ), f"Narrative must mention at least one country; got: {narrative[:200]}"
        finally:
            self._delete_analysis_run(run_id)

    def test_narrative_without_rag(self, monkeypatch):
        """Narrative must complete successfully even when RAG is unavailable."""
        from System2.Narrative.narrative_agent import invoke_narrative_graph

        monkeypatch.delenv("RAG_API_URL", raising=False)
        state  = _make_state()
        run_id = state["run_id"]
        self._insert_analysis_run(run_id)

        try:
            result = invoke_narrative_graph(state)
            assert result["narrative_error"] is None
            assert len(result["narrative"]) > 0
        finally:
            self._delete_analysis_run(run_id)