"""
System 2 — Redis Subscriber.
 
Entry point of the Consumer side of the Producer-Consumer pattern.
Listens to the 'validated_data' Redis channel published by the System 1
Reporter Agent, filters for 'system1_complete' events, and triggers the
System 2 pipeline (Analysis → Visualization → Narrative).
 
Dead Letter Queue (DLQ)
-----------------------
Any message that fails processing is published to 'failed_messages' with
retry metadata. A separate thread listens to 'failed_messages' and retries
with exponential backoff (default: 60s → 300s → 900s, max 3 attempts).
Messages that exhaust all retries are logged as permanent failures and
dropped.
 
Why two separate threads and not a single loop:
    'validated_data' is the hot path — it must never be blocked by a slow
    retry delay. Running the DLQ listener in a separate daemon thread lets
    both channels be processed concurrently without either blocking the other.
 
Why filter only 'system1_complete':
    The Reporter Agent publishes to 'validated_data', but so do
    save_profile_node, save_qa_node, and save_rca_node — each with their
    own event types. Sistema 2 only needs to start its pipeline once per
    full Sistema 1 run, not once per intermediate agent. Filtering on
    'system1_complete' gives exactly that guarantee.
 
Why write to analysis_runs on receipt:
    Persisting the trigger immediately (before pipeline execution) gives
    a durable audit trail of every Sistema 2 activation. If the process
    crashes after triggering but before the Analysis Agent writes its own
    row, the 'triggered' row makes the gap visible in the dashboard.
 
Public interface
----------------
start_subscriber() -> None
    Starts both daemon threads and blocks the main thread. Call from
    the Sistema 2 worker entrypoint.
 
stop_subscriber() -> None
    Signals both threads to stop gracefully. Call from signal handlers.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from shared.db import engine
from shared.redis_client import get_redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
CHANNEL_VALIDATED_DATA = "validated_data"
CHANNEL_FAILED_MESSAGES = "failed_messages"
 
_raw_delays = os.getenv("DLQ_RETRY_DELAYS", "60,300,900")
DLQ_RETRY_DELAYS: list[int] = [int(d.strip()) for d in _raw_delays.split(",")]
 
DLQ_MAX_RETRIES: int = len(DLQ_RETRY_DELAYS)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
 
_REQUIRED_FIELDS = {"run_id", "n_records", "countries", "timestamp"}

def _validate_message(payload: dict[str, Any]) -> list[str]:
    """
    Validate that a 'system1_complete' payload has all required fields.
 
    Returns a list of missing field names. Empty list means the payload
    is valid.
 
    Why validate here and not trust the publisher:
        The subscriber and publisher are decoupled by design. A future
        change to the Reporter Agent could inadvertently drop a field.
        Validating on receipt makes the contract explicit and produces a
        clear error message instead of a cryptic KeyError downstream.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in payload]
    return missing

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
 
def _write_analysis_trigger(payload: dict[str, Any]) -> None:
    """
    Insert a row into analysis_runs to record that System 2 was triggered.
 
    Why INSERT here and not in the Analysis Agent:
        If the process crashes between receiving the Redis message and the
        Analysis Agent writing its own row, the 'triggered' status makes the
        gap visible. The Analysis Agent (Step 13) will UPDATE this row with
        its own output once it completes.
 
    Why 'triggered' status and not 'running':
        The subscriber thread does not execute the pipeline — it enqueues it.
        'triggered' accurately reflects that the signal was received and
        persisted, but processing has not yet started.
    """
    run_id = payload["run_id"]
    started_at = datetime.now(timezone.utc)

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO analysis_runs
                    (run_id, triggered_by, started_at, status)
                VALUES
                    (:run_id, :triggered_by, :started_at, :status)
                ON CONFLICT (run_id) DO NOTHING
            """),
            {
                "run_id":       run_id,
                "triggered_by": "redis:system1_complete",
                "started_at":   started_at,
                "status":       "triggered",
            },
        )
        conn.commit()

    logger.info(
        "analysis_runs: inserted trigger for run_id=%s (%d records, countries=%s)",
        run_id, payload.get("n_records", 0), payload.get("countries", []),
    )

# ---------------------------------------------------------------------------
# DLQ publishing
# ---------------------------------------------------------------------------
 
def _publish_to_dlq(raw_message: str, error: str, retry_count: int = 0) -> None:
    """
    Publish a failed message to the 'failed_messages' channel with retry metadata.
 
    The DLQ envelope wraps the original raw message string so the retry
    handler can re-parse and re-process it without any information loss.
 
    Parameters
    ----------
    raw_message:
        The original JSON string received from 'validated_data'.
    error:
        Human-readable description of why processing failed.
    retry_count:
        How many times this message has already been retried.
        0 on first failure, incremented by the DLQ handler on each retry.
    """
    envelope = json.dumps({
        "original_message": raw_message,
        "error":            error,
        "retry_count":      retry_count,
        "failed_at":        datetime.now(timezone.utc).isoformat(),
    })
    try:
        get_redis().publish(CHANNEL_FAILED_MESSAGES, envelope)
        logger.warning(
            "Published to DLQ (retry_count=%d): %s", retry_count, error
        )
    except Exception as exc:
        logger.error("DLQ publish failed — message dropped: %s | cause: %s", error, exc)

# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------
 
def _handle_validated_data(raw_message: str) -> None:
    """
    Process a single message from the 'validated_data' channel.
 
    Steps:
    1. Parse JSON.
    2. Filter: only process 'system1_complete' events.
    3. Validate required fields.
    4. Write trigger row to analysis_runs.
    5. Invoke the System 2 pipeline.
 
    Any exception in steps 3-5 routes the message to the DLQ.
 
    Why parse before filtering:
        We need to read the 'event' field to know whether to process the
        message at all. Parsing is cheap; the filter saves all subsequent work.
    """
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        logger.error("Could not parse message from validated_data: %s | raw: %.200s",
                     exc, raw_message)
        return

    event = payload.get("event")
    if event != "system1_complete":
        logger.debug("Ignored event '%s' on validated_data", event)
        return

    logger.info(
        "Received system1_complete: run_id=%s n_records=%s countries=%s",
        payload.get("run_id"), payload.get("n_records"), payload.get("countries"),
    )

    missing = _validate_message(payload)
    if missing:
        error = f"Missing required fields: {missing}"
        logger.error("Message validation failed — %s | payload: %.200s", error, raw_message)
        _publish_to_dlq(raw_message, error)
        return

    try:
        _write_analysis_trigger(payload)
    except Exception as exc:
        error = f"DB write failed: {exc}"
        logger.error("analysis_runs insert failed for run_id=%s: %s",
                     payload.get("run_id"), exc)
        _publish_to_dlq(raw_message, error)
        return

    from System2.Analysis.analysis_agent import invoke_analysis_graph
    from System2.Visualization.visualization_agent import invoke_viz_graph
    from System2.Narrative.narrative_agent import invoke_narrative_graph

    state = {
        "run_id":       payload["run_id"],
        "countries":    payload.get("countries", []),
        "triggered_by": "redis:system1_complete",
        "messages":     [],
    }

    try:
        state = invoke_analysis_graph(state)
        state = invoke_viz_graph(state)
        state = invoke_narrative_graph(state)
        logger.info(
            "Sistema 2 pipeline complete for run_id=%s — provider=%s",
            payload["run_id"], state.get("llm_provider"),
        )
    except Exception as exc:
        error = f"Sistema 2 pipeline failed: {exc}"
        logger.error("Pipeline error for run_id=%s: %s", payload.get("run_id"), exc)
        _publish_to_dlq(raw_message, error)     

# ---------------------------------------------------------------------------
# DLQ handler
# ---------------------------------------------------------------------------
def _handle_failed_message(raw_envelope: str) -> None:
    """
    Process a single message from the 'failed_messages' DLQ channel.
 
    Reads the retry_count from the envelope, waits the appropriate backoff
    delay, then re-attempts _handle_validated_data on the original message.
 
    Why sleep inside the handler thread:
        The DLQ listener is a dedicated daemon thread — sleeping it does not
        block the main 'validated_data' listener. This is simpler and more
        transparent than a separate scheduler for the retry delays.
 
    Why not re-enqueue after max retries:
        Re-enqueueing an exhausted message would create an infinite loop.
        Permanent failures are logged and dropped. A future improvement could
        write them to a PostgreSQL 'dead_messages' table for manual inspection.
    """
    try:
        envelope = json.loads(raw_envelope)
    except json.JSONDecodeError as exc:
        logger.error("Could not parse DLQ envelope: %s | raw: %.200s", exc, raw_envelope)
        return

    original_message = envelope.get("original_message", "")
    retry_count = int(envelope.get("retry_count",0))
    original_error = envelope.get("error","unknow")

    if retry_count >= DLQ_MAX_RETRIES:
        logger.error(
            "Message exhausted all %d retries. Dropping permanently. "
            "Original error: %s | message: %.200s",
            DLQ_MAX_RETRIES, original_error, original_message,
        )
        return

    delay = DLQ_RETRY_DELAYS[retry_count]
    logger.info(
        "DLQ retry %d/%d in %ds (original error: %s)",
        retry_count + 1, DLQ_MAX_RETRIES, delay, original_error,
    )
    time.sleep(delay)

    try:
        _handle_validated_data(original_message)
        logger.info("DLQ retry %d succeeded.", retry_count + 1)
    except Exception as exc:
        error = f"Retry {retry_count + 1} failed: {exc}"
        logger.warning(error)
        _publish_to_dlq(original_message, error, retry_count=retry_count + 1)

# ---------------------------------------------------------------------------
# Listener threads
# ---------------------------------------------------------------------------
_stop_event = threading.Event()

def _run_validated_data_listener() -> None:
    """
    Blocking loop that listens to 'validated_data' and dispatches each message.
 
    Why a threading.Event for stopping:
        pubsub.listen() is a blocking generator. The cleanest way to stop it
        without killing the thread from outside is to check a shared Event
        in the loop and call pubsub.unsubscribe() to break out of listen().
    """
    redis_client = get_redis()
    pubsub = redis_client.pubsub()
    pubsub.subscribe(CHANNEL_VALIDATED_DATA)
    logger.info("Subscribed to channel '%s'", CHANNEL_VALIDATED_DATA)

    for message in pubsub.listen():
        if _stop_event.is_set():
            break
        if message["type"] != "message":
            continue
        try:
            _handle_validated_data(message["data"])
        except Exception as exc:
            logger.error("Unhandled error in validated_data handler: %s", exc)

    pubsub.unsubscribe(CHANNEL_VALIDATED_DATA)
    logger.info("validated_data listener stopped.")


def _run_dlq_listener() -> None:
    """
    Blocking loop that listens to 'failed_messages' and retries each message.
 
    Mirrors _run_validated_data_listener but dispatches to _handle_failed_message.
    Both loops check _stop_event to allow clean shutdown.
    """
    redis_client = get_redis()
    pubsub = redis_client.pubsub()
    pubsub.subscribe(CHANNEL_FAILED_MESSAGES)
    logger.info("Subscribed to DLQ channel '%s'", CHANNEL_FAILED_MESSAGES)

    for message in pubsub.listen():
        if _stop_event.is_set():
            break
        if message["type"] != "message":
            continue
        try:
            _handle_failed_message(message["data"])
        except Exception as exc:
            logger.error("Unhandled error in DLQ handler: %s", exc)
 
    pubsub.unsubscribe(CHANNEL_FAILED_MESSAGES)
    logger.info("DLQ listener stopped.")

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def start_subscriber() -> None:
    """
    Start both listener threads and block the calling thread.
 
    Both threads are daemon threads: they will be killed automatically
    when the main process exits, so no explicit cleanup is needed for
    normal process termination (e.g. Ctrl+C or Docker SIGTERM).
 
    For graceful shutdown during testing or signal handling, call
    stop_subscriber() first.
    """
    _stop_event.clear()

    main_thread = threading.Thread(
        target=_run_validated_data_listener,
        name="validated-data-listener",
        daemon=True)

    dlq_thread= threading.Thread(
        target=_run_dlq_listener,
        name="dlq-listener",
        daemon=True
    )

    main_thread.start()
    dlq_thread.start()
    logger.info("Sistema 2 subscriber started (2 listener threads).")
 
    main_thread.join()
    dlq_thread.join()


def stop_subscriber() -> None:
    """
    Signal both listener threads to stop after their current message.
 
    The threads check _stop_event at the top of each iteration, so they
    will exit cleanly once the current message (if any) is processed.
    This is safe to call from signal handlers or test teardown.
    """
    _stop_event.set()
    logger.info("Stop signal sent to subscriber threads.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    start_subscriber()