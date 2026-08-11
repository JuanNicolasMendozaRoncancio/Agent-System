"""
Tests for System2/subscriber.py

Unit tests  : fully mocked — no Redis, no DB.
Integration : marked @pytest.mark.integration — require Docker running
              with Redis and PostgreSQL live.

Run unit tests only:
    python -m pytest tests/test_subscriber.py -v -m "not integration"

Run integration tests:
    python -m pytest tests/test_subscriber.py -v -m integration
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(
    event: str = "system1_complete",
    run_id: str | None = None,
    n_records: int = 48,
    countries: list[str] | None = None,
    timestamp: str | None = None,
) -> dict:
    """Return a minimal valid system1_complete payload."""
    return {
        "run_id":    run_id or str(uuid.uuid4()),
        "event":     event,
        "n_records": n_records,
        "countries": countries or ["FR", "DE"],
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def _raw(payload: dict) -> str:
    """Serialize a payload dict to the raw JSON string Redis delivers."""
    return json.dumps(payload)


def _make_dlq_envelope(
    original_message: str,
    error: str = "DB write failed: connection refused",
    retry_count: int = 0,
) -> str:
    """Return a raw DLQ envelope string as published by _publish_to_dlq."""
    return json.dumps({
        "original_message": original_message,
        "error":            error,
        "retry_count":      retry_count,
        "failed_at":        datetime.now(timezone.utc).isoformat(),
    })


# ===========================================================================
# TestValidateMessage
# ===========================================================================

class TestValidateMessage:
    """
    _validate_message is a pure function — no mocking needed.
    It returns a list of missing field names (empty = valid).
    """

    def test_valid_payload_returns_empty_list(self):
        from System2.subscriber import _validate_message
        payload = _make_payload()
        assert _validate_message(payload) == []

    def test_missing_run_id_flagged(self):
        from System2.subscriber import _validate_message
        payload = _make_payload()
        del payload["run_id"]
        assert "run_id" in _validate_message(payload)

    def test_missing_n_records_flagged(self):
        from System2.subscriber import _validate_message
        payload = _make_payload()
        del payload["n_records"]
        assert "n_records" in _validate_message(payload)

    def test_missing_countries_flagged(self):
        from System2.subscriber import _validate_message
        payload = _make_payload()
        del payload["countries"]
        assert "countries" in _validate_message(payload)

    def test_missing_timestamp_flagged(self):
        from System2.subscriber import _validate_message
        payload = _make_payload()
        del payload["timestamp"]
        assert "timestamp" in _validate_message(payload)

    def test_multiple_missing_fields_all_returned(self):
        from System2.subscriber import _validate_message
        # Only event present — all required fields missing
        result = _validate_message({"event": "system1_complete"})
        assert set(result) == {"run_id", "n_records", "countries", "timestamp"}


# ===========================================================================
# TestHandleValidatedData
# ===========================================================================

class TestHandleValidatedData:
    """
    _handle_validated_data is the core dispatcher.
    Tests verify filtering, validation, DB write path, and DLQ routing.
    """

    def _mock_engine(self):
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return mock_engine, mock_conn

    # --- Filtering ---

    def test_ignores_non_system1_complete_events(self):
        """Events other than 'system1_complete' must not trigger any DB write."""
        payload = _make_payload(event="profiling_complete")
        mock_engine, mock_conn = self._mock_engine()

        with (
            patch("System2.subscriber.engine", mock_engine),
            patch("System2.subscriber.get_redis"),
        ):
            from System2.subscriber import _handle_validated_data
            _handle_validated_data(_raw(payload))

        mock_conn.execute.assert_not_called()

    def test_ignores_qa_complete_event(self):
        payload = _make_payload(event="qa_complete")
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _handle_validated_data
            _handle_validated_data(_raw(payload))

        mock_conn.execute.assert_not_called()

    def test_processes_system1_complete_event(self):
        """'system1_complete' must trigger the DB write."""
        payload = _make_payload(event="system1_complete")
        mock_engine, mock_conn = self._mock_engine()

        with (
            patch("System2.subscriber.engine", mock_engine),
            patch("System2.subscriber.get_redis"),
        ):
            from System2.subscriber import _handle_validated_data
            _handle_validated_data(_raw(payload))

        mock_conn.execute.assert_called_once()

    # --- Malformed input ---

    def test_malformed_json_is_dropped_silently(self):
        """Non-JSON input must not raise — it is logged and dropped."""
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _handle_validated_data
            # Must not raise
            _handle_validated_data("not valid json {{{{")

        mock_conn.execute.assert_not_called()

    # --- Validation failure → DLQ ---

    def test_missing_field_routes_to_dlq(self):
        """A payload missing a required field must be published to the DLQ."""
        payload = _make_payload()
        del payload["run_id"]   # missing required field
        mock_redis = MagicMock()
        mock_engine, mock_conn = self._mock_engine()

        with (
            patch("System2.subscriber.engine", mock_engine),
            patch("System2.subscriber.get_redis", return_value=mock_redis),
        ):
            from System2.subscriber import _handle_validated_data
            _handle_validated_data(_raw(payload))

        mock_redis.publish.assert_called_once()
        channel, envelope = mock_redis.publish.call_args[0]
        assert channel == "failed_messages"
        parsed = json.loads(envelope)
        assert "run_id" in parsed["error"]
        # DB write must not have been attempted
        mock_conn.execute.assert_not_called()

    # --- DB failure → DLQ ---

    def test_db_failure_routes_to_dlq(self):
        """If _write_analysis_trigger raises, the message must go to the DLQ."""
        payload = _make_payload()
        mock_redis = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")

        with (
            patch("System2.subscriber.engine", mock_engine),
            patch("System2.subscriber.get_redis", return_value=mock_redis),
        ):
            from System2.subscriber import _handle_validated_data
            _handle_validated_data(_raw(payload))

        mock_redis.publish.assert_called_once()
        channel, envelope = mock_redis.publish.call_args[0]
        assert channel == "failed_messages"
        parsed = json.loads(envelope)
        assert parsed["retry_count"] == 0
        assert "connection refused" in parsed["error"]

    # --- DLQ publish itself fails ---

    def test_dlq_publish_failure_does_not_raise(self):
        """If Redis publish to DLQ also fails, must not raise — log and continue."""
        payload = _make_payload()
        del payload["run_id"]   # trigger DLQ
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = Exception("Redis unavailable")

        with (
            patch("System2.subscriber.get_redis", return_value=mock_redis),
            patch("System2.subscriber.engine", MagicMock()),
        ):
            from System2.subscriber import _handle_validated_data
            # Must not raise even when DLQ publish fails
            _handle_validated_data(_raw(payload))


# ===========================================================================
# TestWriteAnalysisTrigger
# ===========================================================================

class TestWriteAnalysisTrigger:
    """
    _write_analysis_trigger inserts a row into analysis_runs.
    Tests verify the SQL parameters and the ON CONFLICT DO NOTHING behaviour.
    """

    def _mock_engine(self):
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return mock_engine, mock_conn

    def test_inserts_with_correct_run_id(self):
        """The run_id from the payload must be passed to the INSERT."""
        payload = _make_payload()
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _write_analysis_trigger
            _write_analysis_trigger(payload)

        mock_conn.execute.assert_called_once()
        _, params = mock_conn.execute.call_args[0]
        assert params["run_id"] == payload["run_id"]

    def test_triggered_by_is_redis_system1_complete(self):
        """triggered_by must always be 'redis:system1_complete'."""
        payload = _make_payload()
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _write_analysis_trigger
            _write_analysis_trigger(payload)

        _, params = mock_conn.execute.call_args[0]
        assert params["triggered_by"] == "redis:system1_complete"

    def test_status_is_triggered(self):
        """status must be 'triggered' — not 'running' or 'complete'."""
        payload = _make_payload()
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _write_analysis_trigger
            _write_analysis_trigger(payload)

        _, params = mock_conn.execute.call_args[0]
        assert params["status"] == "triggered"

    def test_commit_is_called(self):
        """conn.commit() must be called after the INSERT."""
        payload = _make_payload()
        mock_engine, mock_conn = self._mock_engine()

        with patch("System2.subscriber.engine", mock_engine):
            from System2.subscriber import _write_analysis_trigger
            _write_analysis_trigger(payload)

        mock_conn.commit.assert_called_once()


# ===========================================================================
# TestPublishToDlq
# ===========================================================================

class TestPublishToDlq:
    """
    _publish_to_dlq wraps the original message in an envelope and publishes
    it to 'failed_messages'. Tests verify the envelope structure.
    """

    def test_publishes_to_failed_messages_channel(self):
        mock_redis = MagicMock()
        with patch("System2.subscriber.get_redis", return_value=mock_redis):
            from System2.subscriber import _publish_to_dlq
            _publish_to_dlq('{"event":"system1_complete"}', "some error")

        channel, _ = mock_redis.publish.call_args[0]
        assert channel == "failed_messages"

    def test_envelope_contains_original_message(self):
        original = '{"event":"system1_complete","run_id":"abc"}'
        mock_redis = MagicMock()
        with patch("System2.subscriber.get_redis", return_value=mock_redis):
            from System2.subscriber import _publish_to_dlq
            _publish_to_dlq(original, "some error", retry_count=0)

        _, envelope_raw = mock_redis.publish.call_args[0]
        envelope = json.loads(envelope_raw)
        assert envelope["original_message"] == original

    def test_envelope_contains_error(self):
        mock_redis = MagicMock()
        with patch("System2.subscriber.get_redis", return_value=mock_redis):
            from System2.subscriber import _publish_to_dlq
            _publish_to_dlq('{}', "DB write failed: timeout", retry_count=1)

        _, envelope_raw = mock_redis.publish.call_args[0]
        envelope = json.loads(envelope_raw)
        assert envelope["error"] == "DB write failed: timeout"
        assert envelope["retry_count"] == 1

    def test_failed_at_is_present(self):
        mock_redis = MagicMock()
        with patch("System2.subscriber.get_redis", return_value=mock_redis):
            from System2.subscriber import _publish_to_dlq
            _publish_to_dlq('{}', "error")

        _, envelope_raw = mock_redis.publish.call_args[0]
        envelope = json.loads(envelope_raw)
        assert "failed_at" in envelope


# ===========================================================================
# TestHandleFailedMessage
# ===========================================================================

class TestHandleFailedMessage:
    """
    _handle_failed_message is the DLQ retry logic.
    Tests verify retry counting, backoff waiting, re-enqueueing on failure,
    and permanent drop on exhaustion.
    """

    def _dlq_raw(self, retry_count: int = 0, original_payload: dict | None = None) -> str:
        payload = original_payload or _make_payload()
        return _make_dlq_envelope(_raw(payload), retry_count=retry_count)

    def test_below_max_retries_calls_handle_validated_data(self):
        """With retry_count=0, _handle_validated_data must be called after delay."""
        dlq_raw = self._dlq_raw(retry_count=0)

        with (
            patch("System2.subscriber.time.sleep") as mock_sleep,
            patch("System2.subscriber._handle_validated_data") as mock_handle,
        ):
            from System2.subscriber import _handle_failed_message
            _handle_failed_message(dlq_raw)

        mock_sleep.assert_called_once()
        mock_handle.assert_called_once()

    def test_sleep_duration_matches_delay_for_retry_count(self):
        """
        retry_count=0 → DLQ_RETRY_DELAYS[0] seconds sleep.
        retry_count=1 → DLQ_RETRY_DELAYS[1] seconds sleep.
        """
        from System2.subscriber import DLQ_RETRY_DELAYS

        for retry_count in range(len(DLQ_RETRY_DELAYS)):
            dlq_raw = self._dlq_raw(retry_count=retry_count)
            with (
                patch("System2.subscriber.time.sleep") as mock_sleep,
                patch("System2.subscriber._handle_validated_data"),
            ):
                from System2.subscriber import _handle_failed_message
                _handle_failed_message(dlq_raw)

            mock_sleep.assert_called_once_with(DLQ_RETRY_DELAYS[retry_count])

    def test_at_max_retries_message_is_dropped(self):
        """retry_count >= DLQ_MAX_RETRIES must not call _handle_validated_data."""
        from System2.subscriber import DLQ_MAX_RETRIES
        dlq_raw = self._dlq_raw(retry_count=DLQ_MAX_RETRIES)

        with (
            patch("System2.subscriber.time.sleep") as mock_sleep,
            patch("System2.subscriber._handle_validated_data") as mock_handle,
        ):
            from System2.subscriber import _handle_failed_message
            _handle_failed_message(dlq_raw)

        mock_sleep.assert_not_called()
        mock_handle.assert_not_called()

    def test_retry_failure_republishes_with_incremented_count(self):
        """
        If _handle_validated_data raises during retry, the message must be
        re-published to DLQ with retry_count incremented by 1.
        """
        dlq_raw = self._dlq_raw(retry_count=0)
        mock_redis = MagicMock()

        with (
            patch("System2.subscriber.time.sleep"),
            patch("System2.subscriber._handle_validated_data",
                  side_effect=Exception("still failing")),
            patch("System2.subscriber.get_redis", return_value=mock_redis),
        ):
            from System2.subscriber import _handle_failed_message
            _handle_failed_message(dlq_raw)

        mock_redis.publish.assert_called_once()
        _, envelope_raw = mock_redis.publish.call_args[0]
        envelope = json.loads(envelope_raw)
        assert envelope["retry_count"] == 1

    def test_malformed_dlq_envelope_does_not_raise(self):
        """Malformed DLQ envelope must be logged and dropped without raising."""
        with (
            patch("System2.subscriber.time.sleep"),
            patch("System2.subscriber._handle_validated_data"),
        ):
            from System2.subscriber import _handle_failed_message
            # Must not raise
            _handle_failed_message("not valid json")

    def test_retry_success_does_not_republish_to_dlq(self):
        """If _handle_validated_data succeeds on retry, DLQ must not receive another message."""
        dlq_raw = self._dlq_raw(retry_count=0)
        mock_redis = MagicMock()

        with (
            patch("System2.subscriber.time.sleep"),
            patch("System2.subscriber._handle_validated_data"),  # succeeds
            patch("System2.subscriber.get_redis", return_value=mock_redis),
        ):
            from System2.subscriber import _handle_failed_message
            _handle_failed_message(dlq_raw)

        mock_redis.publish.assert_not_called()


# ===========================================================================
# Integration tests — require Docker running (Redis + PostgreSQL)
# ===========================================================================

@pytest.mark.integration
class TestSubscriberIntegration:
    """
    Live tests against real Redis and PostgreSQL.
    Verifies the full round-trip: publish → receive → DB write.

    These tests bypass the listener threads and call the handler directly
    to avoid non-deterministic timing issues with Redis Pub/Sub delivery.
    """

    def test_handle_system1_complete_writes_to_db(self):
        """
        Publishing a valid system1_complete payload and calling the handler
        must produce a row in analysis_runs with status='triggered'.
        """
        from shared.db import engine
        from sqlalchemy import text
        from System2.subscriber import _handle_validated_data

        payload = _make_payload()
        run_id = payload["run_id"]

        _handle_validated_data(_raw(payload))

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status, triggered_by FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).fetchone()

        assert row is not None, "Expected a row in analysis_runs"
        assert row[0] == "triggered"
        assert row[1] == "redis:system1_complete"

        # Cleanup
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

    def test_duplicate_run_id_does_not_raise(self):
        """
        ON CONFLICT DO NOTHING: calling the handler twice with the same
        run_id must not raise — the second call is silently ignored.
        """
        from shared.db import engine
        from sqlalchemy import text
        from System2.subscriber import _handle_validated_data

        payload = _make_payload()
        run_id = payload["run_id"]
        raw = _raw(payload)

        _handle_validated_data(raw)
        _handle_validated_data(raw)  # second call — must not raise

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).fetchone()[0]

        assert count == 1

        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

    def test_non_system1_complete_event_not_written(self):
        """
        A 'profiling_complete' event on the validated_data channel must not
        produce any row in analysis_runs.
        """
        from shared.db import engine
        from sqlalchemy import text
        from System2.subscriber import _handle_validated_data

        payload = _make_payload(event="profiling_complete")
        run_id = payload["run_id"]

        _handle_validated_data(_raw(payload))

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM analysis_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).fetchone()[0]

        assert count == 0

    def test_dlq_round_trip(self):
        """
        Publishing a malformed payload (missing run_id) and calling the
        handler must publish exactly one message to the DLQ channel.
        Verified by subscribing to 'failed_messages' in the test.
        """
        from shared.redis_client import get_redis
        from System2.subscriber import _handle_validated_data

        redis_client = get_redis()
        pubsub = redis_client.pubsub()
        pubsub.subscribe("failed_messages")

        # Drain any existing messages
        for _ in range(5):
            pubsub.get_message(timeout=0.1)

        payload = _make_payload()
        del payload["run_id"]   # intentionally invalid
        _handle_validated_data(_raw(payload))

        # Wait for the DLQ message to arrive
        dlq_message = None
        for _ in range(20):
            msg = pubsub.get_message(timeout=0.5)
            if msg and msg["type"] == "message":
                dlq_message = msg
                break

        pubsub.unsubscribe("failed_messages")

        assert dlq_message is not None, "Expected a message on failed_messages"
        envelope = json.loads(dlq_message["data"])
        assert envelope["retry_count"] == 0
        assert "run_id" in envelope["error"]