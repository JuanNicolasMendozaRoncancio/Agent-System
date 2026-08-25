"""
Integration test for POST /pipeline/run SSE endpoint.

No mocks — tests against real Redis, PostgreSQL, and LLM APIs.

Requirements
------------
Docker must be running:
    docker-compose up -d postgres redis

All API keys must be set in .env:
    ENTSOE_API_KEY, COPERNICUS_URL, COPERNICUS_API_KEY,
    GROQ_API_KEY, GEMINI_API_KEY
"""
from __future__ import annotations

import json
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from shared.db import engine

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_events(lines: Iterator[str]) -> list[dict]:
    """
    Parse raw SSE lines into a list of event dicts.
    """
    events = []
    for line in lines:
        line = line.strip()
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return events


def _cleanup(run_id: str) -> None:
    """Delete all rows written by this test run."""
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM energy_climate_records WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        conn.execute(
            text("DELETE FROM data_quality_runs WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        conn.execute(
            text("DELETE FROM analysis_runs WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineRunEndpoint:
    """
    End-to-end tests for POST /pipeline/run.
    """

    _PAYLOAD = {
        "countries": ["FR"],
        "date_from": "2024-06-01T00:00:00",
        "date_to":   "2024-06-01T03:00:00",
        "run_type":  "full",
    }

    def test_returns_200_and_event_stream_content_type(self):
        """Endpoint must return 200 with text/event-stream content type."""
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            for _ in resp.iter_lines():
                pass

    def test_all_sistema1_agents_emit_running_and_done(self):
        """Every Sistema 1 agent must emit exactly one 'running' and one 'done' event."""
        expected_agents = {"Ingestion", "Profiling", "QA", "RCA", "Reporter"}

        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        agent_events: dict[str, list[str]] = {}
        for ev in events:
            agent = ev.get("agent")
            status = ev.get("status")
            if agent:
                agent_events.setdefault(agent, []).append(status)

        for agent in expected_agents:
            statuses = agent_events.get(agent, [])
            assert "running" in statuses, f"{agent}: missing 'running' event"
            assert "done" in statuses,    f"{agent}: missing 'done' event"

    def test_sistema2_agents_emit_events(self):
        """Analysis, Visualization, and Narrative must each emit at least 'done'."""
        expected_s2 = {"Analysis", "Visualization", "Narrative"}

        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        done_agents = {
            ev["agent"] for ev in events
            if ev.get("agent") and ev.get("status") == "done"
        }

        for agent in expected_s2:
            assert agent in done_agents, f"Sistema 2 agent '{agent}' never emitted 'done'"

    def test_pipeline_complete_event_is_last(self):
        """'pipeline_complete' must be the final event in the stream."""
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        assert len(events) > 0, "No events received"
        last = events[-1]
        assert last.get("event") == "pipeline_complete", (
            f"Last event is not pipeline_complete: {last}"
        )

    def test_pipeline_complete_contains_run_id_and_elapsed(self):
        """'pipeline_complete' event must contain run_id and total_elapsed_s."""
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        complete = next(
            (ev for ev in events if ev.get("event") == "pipeline_complete"), None
        )
        assert complete is not None
        assert "run_id" in complete
        assert "total_elapsed_s" in complete
        assert complete["total_elapsed_s"] > 0

    def test_ingestion_done_reports_n_records(self):
        """Ingestion 'done' event must include n_records > 0."""
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        ingestion_done = next(
            (ev for ev in events
             if ev.get("agent") == "Ingestion" and ev.get("status") == "done"),
            None,
        )
        assert ingestion_done is not None
        assert ingestion_done.get("n_records", 0) > 0

    def test_no_error_events_in_clean_run(self):
        """A clean run must produce no 'error' status events."""
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        error_events = [ev for ev in events if ev.get("status") == "error"]
        assert error_events == [], f"Unexpected error events: {error_events}"

    def test_db_state_after_pipeline(self):
        """
        After the pipeline completes, both data_quality_runs and analysis_runs
        must have a completed row — same DB assertions as test_e2e_pipeline.py.
        """
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        complete = next(
            (ev for ev in events if ev.get("event") == "pipeline_complete"), None
        )
        assert complete is not None, "pipeline_complete event not received"
        run_id = complete["run_id"]

        try:
            with engine.connect() as conn:
                dq_row = conn.execute(
                    text("""
                        SELECT status, run_report, n_records
                        FROM data_quality_runs
                        WHERE run_id = :run_id
                    """),
                    {"run_id": run_id},
                ).fetchone()

            assert dq_row is not None, "data_quality_runs: no row found"
            assert dq_row[0] == "complete",        f"status={dq_row[0]}"
            assert dq_row[1] and len(dq_row[1]) > 0, "run_report is empty"
            assert dq_row[2] and dq_row[2] > 0,    "n_records is 0"

            # Sistema 2 — analysis_runs
            with engine.connect() as conn:
                ar_row = conn.execute(
                    text("""
                        SELECT status, narrative, viz_json, charts_json
                        FROM analysis_runs
                        WHERE run_id = :run_id
                    """),
                    {"run_id": run_id},
                ).fetchone()

            assert ar_row is not None,              "analysis_runs: no row found"
            assert ar_row[0] == "complete",         f"status={ar_row[0]}"
            assert ar_row[1] and len(ar_row[1]) > 0, "narrative is empty"
            assert ar_row[2] is not None,           "viz_json is null"
            assert ar_row[3] is not None,           "charts_json is null"

        finally:
            _cleanup(run_id)

    def test_event_order_is_sequential(self):
        """
        Agent 'running' must always precede 'done' for the same agent.
        """
        with client.stream("POST", "/pipeline/run", json=self._PAYLOAD) as resp:
            events = _parse_events(resp.iter_lines())

        agent_order: dict[str, dict[str, int]] = {}
        for i, ev in enumerate(events):
            agent = ev.get("agent")
            status = ev.get("status")
            if agent and status in ("running", "done"):
                agent_order.setdefault(agent, {})[status] = i

        for agent, positions in agent_order.items():
            if "running" in positions and "done" in positions:
                assert positions["running"] < positions["done"], (
                    f"{agent}: 'running' (pos {positions['running']}) "
                    f"came after 'done' (pos {positions['done']})"
                )


@pytest.mark.integration
class TestHealthEndpoint:

    def test_health_returns_ok(self):
        """GET /health must return 200 with postgres and redis both True."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["postgres"] is True
        assert body["redis"]    is True
        assert body["status"]   == "ok"