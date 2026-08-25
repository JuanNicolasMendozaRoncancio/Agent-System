"""
FastAPI — Control API with Server-Sent Events.
 
Exposes POST /pipeline/run which fires the full Producer-Consumer pipeline
and streams SSE progress events back to the caller in real time.
 
Architecture mirrors test_e2e_pipeline.py exactly:
  1. Subscribe to 'validated_data' Redis channel BEFORE sysyem 1 runs.
  2. Start the System 2 subscriber listener as a daemon thread.
  3. Run all 5 sysyem 1 agents sequentially, emitting SSE events per agent.
  4. Reporter publishes 'system1_complete' → subscriber picks it up → runs
     Analysis, Visualization, Narrative agents → publishes 'narrative_complete'.
  5. Poll Redis for 'narrative_complete', emitting SSE events for each
     sysyem 2 step as they complete (detected via intermediate Redis events).
  6. Emit final 'pipeline_complete' SSE event.
"""
from __future__ import annotations

import os
import asyncio
import json
import logging
import threading
import time
import uuid
from asyncio import AbstractEventLoop
from datetime import datetime, timezone
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(os.getenv("ENV_FILE", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate & Energy Agents API",
    description="Control API for the multi-agent Climate & Energy pipeline.",
    version="1.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
 
class PipelineRunRequest(BaseModel):
    countries: list[str] = ["FR"]
    date_from: str = "2024-06-01T00:00:00"
    date_to: str = "2024-06-01T03:00:00"
    run_type: str = "full"

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    """
    Format a dict as an SSE data line.
    """
    return f"data: {json.dumps(data)}\n\n"
 
 
def _event_running(agent: str) -> str:
    return _sse({"agent": agent, "status": "running"})
 
 
def _event_done(agent: str, elapsed_s: float, **extra) -> str:
    return _sse({"agent": agent, "status": "done", "elapsed_s": round(elapsed_s, 2), **extra})
 
 
def _event_error(agent: str, error: str, elapsed_s: float) -> str:
    return _sse({"agent": agent, "status": "error", "error": error, "elapsed_s": round(elapsed_s, 2)})
 
 
def _event_sistema2(agent: str, status: str, **extra) -> str:
    """SSE event for System 2 agents detected via Redis intermediate events."""
    return _sse({"agent": agent, "status": status, **extra})

# ---------------------------------------------------------------------------
# Pipeline generator
# ---------------------------------------------------------------------------

async def _pipeline_generator(req: PipelineRunRequest) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the full pipeline and yields SSE events.
    """
    loop: AbstractEventLoop = asyncio.get_event_loop()
    run_id = str(uuid.uuid4())
    pipeline_start = time.monotonic()
 
    logger.info("=== PIPELINE START — run_id=%s ===", run_id)

    from datetime import datetime as dt
    state: dict = {
        "run_id":          run_id,
        "countries":       req.countries,
        "date_from":       dt.fromisoformat(req.date_from).replace(tzinfo=timezone.utc),
        "date_to":         dt.fromisoformat(req.date_to).replace(tzinfo=timezone.utc),
        "run_type":        req.run_type,
        "messages":        [],
        "records":         [],
        "ingestion_error": None,
        "llm_provider":    None,
        "tool_results":    [],
        "cycle_count":     0,
    }

    # --- Step 0: Subscribe to Redis BEFORE starting anything ----------------
    from shared.redis_client import get_redis
    from System2.subscriber import _run_validated_data_listener, _stop_event
 
    redis_client = get_redis()
    pubsub = redis_client.pubsub()
    pubsub.subscribe("validated_data")
 
    for _ in range(10):
        pubsub.get_message(timeout=0.05)

    # --- Step 1: Start System 2 listener daemon thread --------------------
    _stop_event.clear()
    listener_thread = threading.Thread(
        target=_run_validated_data_listener,
        name=f"api-listener-{run_id[:8]}",
        daemon=True,
    )
    listener_thread.start()
    logger.info("System 2 listener thread started for run_id=%s", run_id)

    # --- System 1 agents ---------------------------------------------------
 
    # 1. IngestionAgent
    yield _event_running("Ingestion")
    t0 = time.monotonic()
    try:
        from System1.Ingestion.ingestion_agent import invoke_ingestion_graph
        state = await loop.run_in_executor(None, invoke_ingestion_graph, state)
        if state.get("ingestion_error"):
            raise RuntimeError(state["ingestion_error"])
        elapsed = time.monotonic() - t0
        yield _event_done("Ingestion", elapsed, n_records=len(state.get("records", [])))
    except Exception as exc:
        elapsed = time.monotonic() - t0
        yield _event_error("Ingestion", str(exc), elapsed)
        yield _sse({"event": "pipeline_failed", "run_id": run_id, "error": str(exc)})
        return
 
    from System1.Ingestion.ingestion_agent import _store_records
    _store_records(run_id, state["records"])
 
    # 2. ProfilingAgent
    yield _event_running("Profiling")
    t0 = time.monotonic()
    try:
        from System1.Profiling.profiling_agent import invoke_profiling_graph
        state = await loop.run_in_executor(None, invoke_profiling_graph, state)
        if state.get("profiling_error"):
            raise RuntimeError(state["profiling_error"])
        n_anomalies = sum(
            len(d.get("drift", {})) for d in state.get("profile", {}).values()
        )
        elapsed = time.monotonic() - t0
        yield _event_done("Profiling", elapsed, n_anomalies=n_anomalies)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        yield _event_error("Profiling", str(exc), elapsed)
        yield _sse({"event": "pipeline_failed", "run_id": run_id, "error": str(exc)})
        return
 
    # 3. QAAgent
    yield _event_running("QA")
    t0 = time.monotonic()
    try:
        from System1.QA.qa_agent import invoke_qa_graph
        state = await loop.run_in_executor(None, invoke_qa_graph, state)
        if state.get("qa_error"):
            raise RuntimeError(state["qa_error"])
        elapsed = time.monotonic() - t0
        yield _event_done("QA", elapsed, severity=state.get("qa_severity"), n_anomalies=len(state.get("anomalies", [])))
    except Exception as exc:
        elapsed = time.monotonic() - t0
        yield _event_error("QA", str(exc), elapsed)
        yield _sse({"event": "pipeline_failed", "run_id": run_id, "error": str(exc)})
        return
 
    # 4. RCAAgent
    yield _event_running("RCA")
    t0 = time.monotonic()
    try:
        from System1.RCA.rca_agent import invoke_rca_graph
        state = await loop.run_in_executor(None, invoke_rca_graph, state)
        if state.get("rca_error"):
            raise RuntimeError(state["rca_error"])
        elapsed = time.monotonic() - t0
        n_hypotheses = len([
            l for l in (state.get("rca_result") or "").split("\n") if l.strip()
        ])
        yield _event_done("RCA", elapsed, n_hypotheses=n_hypotheses, n_rag_sources=len(state.get("rca_sources", [])))
    except Exception as exc:
        elapsed = time.monotonic() - t0
        yield _event_error("RCA", str(exc), elapsed)
        yield _sse({"event": "pipeline_failed", "run_id": run_id, "error": str(exc)})
        return
 
    # 5. ReporterAgent — publishes system1_complete to Redis
    yield _event_running("Reporter")
    t0 = time.monotonic()
    try:
        from System1.Reporter.reporter_agent import invoke_reporter_graph
        state = await loop.run_in_executor(None, invoke_reporter_graph, state)
        if state.get("reporter_error"):
            raise RuntimeError(state["reporter_error"])
        elapsed = time.monotonic() - t0
        yield _event_done("Reporter", elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        yield _event_error("Reporter", str(exc), elapsed)
        yield _sse({"event": "pipeline_failed", "run_id": run_id, "error": str(exc)})
        return
 
    logger.info("System 1 complete — polling Redis for System 2 events")

    # --- Sistema 2: poll Redis for intermediate and final events ------------

 
    _POLL_ITERATIONS = 600      
    _POLL_TIMEOUT_S  = 0.5     
 
    yield _event_sistema2("Analysis", "running")
 
    s2_agent_map = {
        "analysis_complete":  "Analysis",
        "viz_complete":       "Visualization",
        "narrative_complete": "Narrative",
    }
    s2_running_emitted = {"Analysis"}  
    narrative_received = False
 
    for _ in range(_POLL_ITERATIONS):
        msg = await loop.run_in_executor(
            None, lambda: pubsub.get_message(timeout=_POLL_TIMEOUT_S)
        )
 
        if msg and msg["type"] == "message":
            try:
                payload = json.loads(msg["data"])
                event = payload.get("event", "")
                logger.debug("Redis event: %s", event)
 
                if event in s2_agent_map:
                    agent_name = s2_agent_map[event]
 

                    next_agents = {
                        "analysis_complete": "Visualization",
                        "viz_complete":      "Narrative",
                    }
                    if event in next_agents:
                        next_agent = next_agents[event]
                        if next_agent not in s2_running_emitted:
                            yield _event_sistema2(next_agent, "running")
                            s2_running_emitted.add(next_agent)
 
                    yield _event_sistema2(agent_name, "done")
 
                    if event == "narrative_complete":
                        narrative_received = True
                        break
 
            except (json.JSONDecodeError, TypeError):
                pass
 
    pubsub.unsubscribe("validated_data")
 
    if not narrative_received:
        yield _sse({
            "event": "pipeline_failed",
            "run_id": run_id,
            "error": "Timed out waiting for narrative_complete from Sistema 2.",
        })
        return
 
    total_elapsed = time.monotonic() - pipeline_start
    logger.info("=== PIPELINE COMPLETE — run_id=%s total_elapsed=%.1fs ===", run_id, total_elapsed)
 
    yield _sse({
        "event":           "pipeline_complete",
        "run_id":          run_id,
        "total_elapsed_s": round(total_elapsed, 2),
        "llm_provider":    state.get("llm_provider"),
    })

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
 
@app.post("/pipeline/run")
async def pipeline_run(req: PipelineRunRequest) -> StreamingResponse:
    """
    Fire the full Producer-Consumer pipeline and stream SSE progress events.
 
    Returns a text/event-stream response. Each event is a JSON object:
      {"agent": "Ingestion", "status": "running"}
      {"agent": "Ingestion", "status": "done", "elapsed_s": 4.2, "n_records": 48}
      {"event": "pipeline_complete", "run_id": "...", "total_elapsed_s": 42.1}
 
    Why StreamingResponse with media_type text/event-stream:
        FastAPI's StreamingResponse flushes each yielded chunk immediately,
        which is the correct transport for SSE. The client (Streamlit or curl)
        receives each event as soon as the corresponding agent completes,
        without waiting for the full pipeline to finish.
    """
    return StreamingResponse(
        _pipeline_generator(req),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
 
 
@app.get("/health")
async def health() -> dict:
    """Check connectivity to Redis and PostgreSQL."""
    from shared.db import check_connection as pg_ok, engine
    from shared.redis_client import check_connection as redis_ok
    
    pg_status = False
    pg_error = None
    try:
        pg_status = pg_ok()
    except Exception as exc:
        pg_error = str(exc)
        logger.error("PostgreSQL health check failed: %s", exc)
    
    return {
        "postgres": pg_status,
        "postgres_error": pg_error,
        "redis":    redis_ok(),
        "status":   "ok",
    }