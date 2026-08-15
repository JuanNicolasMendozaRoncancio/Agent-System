"""
End-to-end integration test — System 1 → Redis → System 2.
 
Validates the full Producer-Consumer pipeline as it runs in production:
 
  invoke_ingestion_graph()
      → Ingestion, Profiling, QA, RCA, Reporter agents run sequentially
      → Reporter publishes 'system1_complete' to Redis channel 'validated_data'
 
  _run_validated_data_listener() (daemon thread, started before Sistema 1)
      → Receives 'system1_complete' from Redis
      → Writes analysis_runs trigger row
      → Runs invoke_analysis_graph → invoke_viz_graph → invoke_narrative_graph
      → Narrative Agent publishes 'narrative_complete' to Redis
 
  Test polls 'validated_data' for 'narrative_complete' and then asserts
  the final DB state.
 
Requirements
------------
Docker must be running with PostgreSQL and Redis healthy:
    docker-compose up -d postgres redis
 
All API keys must be set in .env:
    ENTSOE_API_KEY, COPERNICUS_URL, COPERNICUS_API_KEY,
    GROQ_API_KEY, GEMINI_API_KEY
 
Run with:
    pytest tests/test_e2e_pipeline.py -v -m integration -s
 
The -s flag is recommended: Copernicus downloads and LLM calls produce
informative log output that helps diagnose slow runs.
 
Why a 3-hour window for France only
------------------------------------
Copernicus ERA5 downloads scale with the time window requested — a 24h
window can take 3-5 minutes. A 3-hour window keeps the test under the
5-minute timeout while still exercising both ENTSO-E (generation + load)
and Copernicus (temperature + solar radiation), which is the most complete
data path in the system.
 
Germany is excluded because Copernicus coverage for DE was not yet
populated in the DB at the time this test was written. France exercises
the full four-variable path (generation, load, temperature, solar) and
is therefore the most representative choice.
 
Why daemon=True for the listener thread
----------------------------------------
A daemon thread is automatically killed when the pytest process moves on
to the next test or exits. This is simpler and safer than a manual
stop mechanism: we do not need to publish a dummy message to unblock
pubsub.listen(), and we do not risk accidentally triggering the pipeline
a second time during teardown.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
 
_COUNTRIES   = ["FR"]
_DATE_FROM   = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
_DATE_TO     = datetime(2024, 6, 1, 3, tzinfo=timezone.utc)   # 3-hour window
_RUN_TYPE    = "full"

_POLL_ITERATIONS = 600
_POLL_TIMEOUT_S  = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_initial_state(run_id: str) -> dict:
    """
    Build the AgentState that invoke_ingestion_graph expects.
    """
    return {
        "run_id":          run_id,
        "countries":       _COUNTRIES,
        "date_from":       _DATE_FROM,
        "date_to":         _DATE_TO,
        "run_type":        _RUN_TYPE,
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],
        "cycle_count":     0,
    }

def _cleanup(run_id: str) -> None:
    """
    Delete all rows written by this test run from all three tables.
    """
    from shared.db import engine
 
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
 
    logger.info("cleanup: deleted all rows for run_id=%s", run_id)

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestE2EPipeline:
    """
    Full end-to-end test of the Producer-Consumer pipeline.
 
    The listener thread is started BEFORE Sistema 1 runs — exactly as in
    production, where the Sistema 2 subscriber process is already running
    and waiting before any Sistema 1 run is triggered.
    """
 
    def test_full_pipeline_sistema1_redis_sistema2(self):
        """
        Validate the complete chain:
            Sistema 1 (real APIs) → Redis Pub/Sub → Sistema 2 → PostgreSQL
 
        Assertions
        ----------
        Sistema 1 side (data_quality_runs):
          - Row exists with status = 'complete'
          - run_report is not null and not empty
          - n_records > 0
 
        Sistema 2 side (analysis_runs):
          - Row exists with status = 'complete'
          - narrative is not null and not empty
          - viz_json is not null
          - charts_json is not null
        """
        from shared.db import engine
        from shared.redis_client import get_redis
        from System1.Ingestion.ingestion_agent import invoke_ingestion_graph
        from System1.Profiling.profiling_agent import invoke_profiling_graph
        from System1.QA.qa_agent import invoke_qa_graph
        from System1.RCA.rca_agent import invoke_rca_graph
        from System1.Reporter.reporter_agent import invoke_reporter_graph
        from System2.subscriber import _run_validated_data_listener, _stop_event
 
        run_id = str(uuid.uuid4())
        logger.info("=== E2E TEST START — run_id=%s ===", run_id)
 
        # Subscribe to 'validated_data' BEFORE starting anything so we do
        # not miss any messages published during the pipeline run.
        # We listen on the same channel the Reporter and all downstream
        # agents publish to — this is the real production channel.
        redis_client = get_redis()
        pubsub = redis_client.pubsub()
        pubsub.subscribe("validated_data")
 
        # Drain any stale messages from previous test runs that may still
        # be buffered in this pubsub connection.
        for _ in range(10):
            pubsub.get_message(timeout=0.1)
 
        # --- Start the Sistema 2 listener thread ----------------------------
        # daemon=True: pytest kills this thread automatically when the test
        # finishes, without any manual stop mechanism needed.
        _stop_event.clear()
        listener_thread = threading.Thread(
            target=_run_validated_data_listener,
            name="e2e-validated-data-listener",
            daemon=True,
        )
        listener_thread.start()
        logger.info("Sistema 2 listener thread started (daemon=True)")
 
        try:
            # --- Run all 5 Sistema 1 agents in sequence --------------------
            # Ingestion → Profiling → QA → RCA → Reporter.
            # The Reporter is the last agent and the only one that publishes
            # 'system1_complete' to Redis, which triggers Sistema 2.
            logger.info("Starting Sistema 1 — country=FR, window=3h")
            state = _make_initial_state(run_id)
 
            logger.info("[1/5] IngestionAgent")
            state = invoke_ingestion_graph(state)
            assert state.get("ingestion_error") is None, (
                f"IngestionAgent failed: {state.get('ingestion_error')}"
            )
            assert len(state.get("records", [])) > 0, (
                "No records from ingestion — check ENTSO-E / Copernicus credentials"
            )
            logger.info("[1/5] IngestionAgent OK — %d records", len(state["records"]))
 
            logger.info("[2/5] ProfilingAgent")
            from System1.Ingestion.ingestion_agent import _store_records
            _store_records(run_id, state["records"])
            state = invoke_profiling_graph(state)
            assert state.get("profiling_error") is None, (
                f"ProfilingAgent failed: {state.get('profiling_error')}"
            )
            logger.info("[2/5] ProfilingAgent OK")
 
            logger.info("[3/5] QAAgent")
            state = invoke_qa_graph(state)
            assert state.get("qa_error") is None, (
                f"QAAgent failed: {state.get('qa_error')}"
            )
            logger.info("[3/5] QAAgent OK — severity=%s", state.get("qa_severity"))
 
            logger.info("[4/5] RCAAgent")
            state = invoke_rca_graph(state)
            assert state.get("rca_error") is None, (
                f"RCAAgent failed: {state.get('rca_error')}"
            )
            logger.info("[4/5] RCAAgent OK")
 
            logger.info("[5/5] ReporterAgent")
            state = invoke_reporter_graph(state)
            assert state.get("reporter_error") is None, (
                f"ReporterAgent failed: {state.get('reporter_error')}"
            )
            logger.info("[5/5] ReporterAgent OK — 'system1_complete' published to Redis")
 
            # --- Poll Redis for 'system1_complete' then 'narrative_complete' 
            logger.info(
                "Waiting for 'narrative_complete' on Redis "
                "(max %ds)...", _POLL_ITERATIONS * _POLL_TIMEOUT_S
            )
            narrative_received = False
            for _ in range(_POLL_ITERATIONS):
                msg = pubsub.get_message(timeout=_POLL_TIMEOUT_S)
                if msg and msg["type"] == "message":
                    try:
                        payload = json.loads(msg["data"])
                        event   = payload.get("event", "")
                        logger.debug("Redis event received: %s", event)
                        if event == "narrative_complete":
                            logger.info(
                                "narrative_complete received — run_id=%s",
                                payload.get("run_id"),
                            )
                            narrative_received = True
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass  # ignore non-JSON messages
 
            assert narrative_received, (
                "Timed out waiting for 'narrative_complete' from Sistema 2. "
                "Check logs for pipeline errors."
            )
 
            # --- DB assertions: Sistema 1 (data_quality_runs) --------------
            with engine.connect() as conn:
                dq_row = conn.execute(
                    text("""
                        SELECT status, run_report, n_records
                        FROM data_quality_runs
                        WHERE run_id = :run_id
                    """),
                    {"run_id": run_id},
                ).fetchone()
 
            assert dq_row is not None, (
                "data_quality_runs: no row found for run_id — "
                "save_profile_node may have failed"
            )
            assert dq_row[0] == "complete", (
                f"data_quality_runs.status expected 'complete', got '{dq_row[0]}'"
            )
            assert dq_row[1] is not None and len(dq_row[1]) > 0, (
                "data_quality_runs.run_report is null or empty"
            )
            assert dq_row[2] is not None and dq_row[2] > 0, (
                "data_quality_runs.n_records is 0 — profiling may have failed"
            )
            logger.info(
                "data_quality_runs OK — status=%s n_records=%d",
                dq_row[0], dq_row[2],
            )
 
            # --- DB assertions: Sistema 2 (analysis_runs) ------------------
            with engine.connect() as conn:
                ar_row = conn.execute(
                    text("""
                        SELECT status, narrative, viz_json, charts_json
                        FROM analysis_runs
                        WHERE run_id = :run_id
                    """),
                    {"run_id": run_id},
                ).fetchone()
 
            assert ar_row is not None, (
                "analysis_runs: no row found for run_id — "
                "subscriber may not have written the trigger row"
            )
            assert ar_row[0] == "complete", (
                f"analysis_runs.status expected 'complete', got '{ar_row[0]}'"
            )
            assert ar_row[1] is not None and len(ar_row[1]) > 0, (
                "analysis_runs.narrative is null or empty — "
                "Narrative Agent may have failed"
            )
            assert ar_row[2] is not None, (
                "analysis_runs.viz_json is null — Visualization Agent may have failed"
            )
            assert ar_row[3] is not None, (
                "analysis_runs.charts_json is null — Analysis Agent may have failed"
            )
            logger.info(
                "analysis_runs OK — status=%s narrative_len=%d",
                ar_row[0], len(ar_row[1]),
            )
 
            logger.info("=== E2E TEST PASSED — run_id=%s ===", run_id)
 
        finally:
            pubsub.unsubscribe("validated_data")
            _cleanup(run_id)