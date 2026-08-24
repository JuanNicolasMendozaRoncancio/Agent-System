"""
Tests for shared/llm_client.py.

Two test categories
-------------------
Unit tests (default, no marker):
    Pure mock tests — no network calls, no API keys needed.
    Run automatically in CI.

Integration tests (@pytest.mark.integration):
    Real API calls to Groq primary and Groq fallback models.
    Require GROQ_API_KEY in the environment (.env file).
    Run manually to verify the key is active and both models respond.

How to run
----------
Unit tests only (CI-safe):
    pytest tests/test_llm_client.py -v

Integration tests only:
    pytest tests/test_llm_client.py -v -m integration

All tests:
    pytest tests/test_llm_client.py -v -m "integration or not integration"
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from shared.llm_client import chat_complete


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES = [{"role": "user", "content": "Hello"}]
_SYSTEM = "You are a helpful assistant."


def _make_rate_limit_error() -> RateLimitError:
    """
    Construct a RateLimitError without a real HTTP response.

    RateLimitError inherits from APIStatusError which requires a `response`
    argument. We provide a minimal mock so the constructor does not blow up.
    """
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    return RateLimitError(
        message="rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "rate limit exceeded"}},
    )


# ---------------------------------------------------------------------------
# Unit tests — primary model succeeds
# ---------------------------------------------------------------------------

class TestGroqPrimarySuccess:
    """Primary model (gpt-oss-20b) answers correctly — fallback must not be touched."""

    def test_returns_groq_provider(self):
        with (
            patch("shared.llm_client._call_groq", return_value="hello") as mock_primary,
            patch("shared.llm_client._call_groq_fallback") as mock_fallback,
        ):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert text == "hello"
        assert provider == "groq"
        mock_primary.assert_called_once()
        mock_fallback.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests — rate-limit triggers fallback
# ---------------------------------------------------------------------------

class TestRateLimitFallback:
    """Primary returns HTTP 429 → must switch to gpt-oss-120b fallback."""

    def test_falls_back_on_rate_limit(self):
        with (
            patch("shared.llm_client._call_groq", side_effect=_make_rate_limit_error()),
            patch("shared.llm_client._call_groq_fallback", return_value="fallback text"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert text == "fallback text"
        assert provider == "groq_fallback"

    def test_fallback_called_exactly_once_on_rate_limit(self):
        with (
            patch("shared.llm_client._call_groq", side_effect=_make_rate_limit_error()),
            patch("shared.llm_client._call_groq_fallback", return_value="ok") as mock_fallback,
        ):
            chat_complete(_MESSAGES)

        mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests — generic exception triggers fallback
# ---------------------------------------------------------------------------

class TestGenericErrorFallback:
    """Primary raises a non-429 exception → fallback to gpt-oss-120b."""

    def test_falls_back_on_generic_error(self):
        with (
            patch("shared.llm_client._call_groq", side_effect=Exception("connection timeout")),
            patch("shared.llm_client._call_groq_fallback", return_value="fallback answer"),
        ):
            text, provider = chat_complete(_MESSAGES, timeout=30.0)

        assert provider == "groq_fallback"
        assert text == "fallback answer"


# ---------------------------------------------------------------------------
# Unit tests — both models fail
# ---------------------------------------------------------------------------

class TestBothModelsFail:
    """If primary AND fallback both fail, RuntimeError must be raised."""

    def test_raises_runtime_error(self):
        with (
            patch("shared.llm_client._call_groq", side_effect=_make_rate_limit_error()),
            patch("shared.llm_client._call_groq_fallback", side_effect=Exception("fallback also down")),
        ):
            with pytest.raises(RuntimeError, match="Both Groq models failed"):
                chat_complete(_MESSAGES)


# ---------------------------------------------------------------------------
# Unit tests — LLM_PROVIDER override
# ---------------------------------------------------------------------------

class TestManualOverride:
    """LLM_PROVIDER=groq_fallback env var must skip the primary model entirely."""

    def test_groq_fallback_override_skips_primary(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with (
            patch("shared.llm_client._call_groq") as mock_primary,
            patch("shared.llm_client._call_groq_fallback", return_value="direct fallback"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert provider == "groq_fallback"
        assert text == "direct fallback"
        mock_primary.assert_not_called()

    def test_default_provider_is_groq_primary(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with (
            patch("shared.llm_client._call_groq", return_value="groq answer"),
            patch("shared.llm_client._call_groq_fallback") as mock_fallback,
        ):
            _, provider = chat_complete(_MESSAGES)

        assert provider == "groq"
        mock_fallback.assert_not_called()

    def test_override_returns_text(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with patch("shared.llm_client._call_groq_fallback", return_value="fallback answer"):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert provider == "groq_fallback"
        assert isinstance(text, str)
        assert len(text) > 0

    def test_override_json_mode(self, monkeypatch):
        import json
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with patch("shared.llm_client._call_groq_fallback", return_value='{"status": "ok"}'):
            text, provider = chat_complete(_MESSAGES, json_mode=True)

        assert provider == "groq_fallback"
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Integration tests — real API calls, require GROQ_API_KEY in .env
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGroqPrimaryIntegration:
    """
    Calls the primary Groq model (gpt-oss-20b) for real.

    _call_groq_fallback is mocked here to guarantee the primary is the only
    network call — if Groq itself triggers a rate-limit that would activate
    the fallback, the mock prevents that path from interfering with these
    assertions.
    """

    def test_primary_returns_text_and_provider(self, monkeypatch):
        import time
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        time.sleep(5)

        with patch("shared.llm_client._call_groq_fallback", return_value="mocked fallback"):
            text, provider = chat_complete(
                [{"role": "user", "content": "Reply with exactly one word: OK"}],
                system="You are a concise assistant. Follow instructions literally.",
            )

        assert provider == "groq", f"Expected 'groq', got '{provider}'"
        assert isinstance(text, str)
        assert len(text) > 0

    def test_primary_json_mode(self, monkeypatch):
        import time
        import json
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        time.sleep(5)

        with patch("shared.llm_client._call_groq_fallback", return_value='{"status": "mocked"}'):
            text, provider = chat_complete(
                [{"role": "user", "content": 'Return a JSON object with key "status" set to "ok"'}],
                json_mode=True,
            )

        assert provider == "groq"
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


@pytest.mark.integration
class TestGroqFallbackIntegration:
    """
    Calls the fallback Groq model (gpt-oss-120b) directly via the override,
    verifying it accepts the same request format and returns valid text.

    Why override and not simulate a 429:
        Simulating a 429 in integration mode would require the primary to
        actually fail, which is non-deterministic. Using LLM_PROVIDER=groq_fallback
        gives a deterministic, reproducible way to verify the fallback model
        is reachable and well-behaved without depending on the primary's state.
    """

    def test_fallback_returns_text_and_provider(self, monkeypatch):
        import time
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")
        time.sleep(5)

        text, provider = chat_complete(
            [{"role": "user", "content": "Reply with exactly one word: OK"}],
            system="You are a concise assistant. Follow instructions literally.",
        )

        assert provider == "groq_fallback", f"Expected 'groq_fallback', got '{provider}'"
        assert isinstance(text, str)
        assert len(text) > 0

    def test_fallback_json_mode(self, monkeypatch):
        import time
        import json
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")
        time.sleep(5)

        text, provider = chat_complete(
            [{"role": "user", "content": 'Return a JSON object with key "status" set to "ok"'}],
            json_mode=True,
        )

        assert provider == "groq_fallback"
        parsed = json.loads(text)
        assert isinstance(parsed, dict)