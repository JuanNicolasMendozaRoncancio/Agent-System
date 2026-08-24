"""
Tests for shared/llm_client.py.

Two test categories
-------------------
Unit tests (default, no marker):
    Pure mock tests — no network calls, no API keys needed.
    Run automatically in CI.

Integration tests (@pytest.mark.integration):
    Real API calls to Groq and groq_fallback.  Require GROQ_API_KEY and
    groq_fallback_API_KEY in the environment (.env file).
    Run manually to verify keys are active and response format is correct.

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

# The module under test.
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
    argument.  We provide a minimal mock so the constructor does not blow up.
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
# Tests
# ---------------------------------------------------------------------------

class TestGroqSuccess:
    """Groq answers correctly — groq_fallback must not be touched."""

    def test_returns_groq_provider(self):
        with (
            patch("shared.llm_client._call_groq", return_value="hello") as mock_groq,
            patch("shared.llm_client._call_groq_fallback") as mock_groq_fallback,
        ):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert text == "hello"
        assert provider == "groq"
        mock_groq.assert_called_once()
        mock_groq_fallback.assert_not_called()


class TestRateLimitFallback:
    """Groq returns 429 → must switch to groq_fallback, provider reported as 'groq_fallback'."""

    def test_falls_back_on_rate_limit(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch("shared.llm_client._call_groq_fallback", return_value="fallback text"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert text == "fallback text"
        assert provider == "groq_fallback"

    def test_groq_fallback_called_exactly_once_on_rate_limit(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch(
                "shared.llm_client._call_groq_fallback", return_value="ok"
            ) as mock_groq_fallback,
        ):
            chat_complete(_MESSAGES)

        mock_groq_fallback.assert_called_once()


class TestTimeoutFallback:
    """Groq raises a generic exception → fallback to groq_fallback."""

    def test_falls_back_on_generic_error(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=Exception("connection timeout"),
            ),
            patch("shared.llm_client._call_groq_fallback", return_value="groq_fallback answer"),
        ):
            text, provider = chat_complete(_MESSAGES, timeout=30.0)

        assert provider == "groq_fallback"
        assert text == "groq_fallback answer"


class TestBothProvidersFail:
    """If both Groq and groq_fallback fail, RuntimeError must be raised."""

    def test_raises_runtime_error(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch(
                "shared.llm_client._call_groq_fallback",
                side_effect=Exception("groq_fallback also down"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Both LLM providers failed"):
                chat_complete(_MESSAGES)


class TestManualOverride:
    """LLM_PROVIDER=groq_fallback env var must skip Groq entirely."""

    def test_groq_fallback_override_skips_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with (
            patch("shared.llm_client._call_groq") as mock_groq,
            patch("shared.llm_client._call_groq_fallback", return_value="direct groq_fallback"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert provider == "groq_fallback"
        assert text == "direct groq_fallback"
        mock_groq.assert_not_called()

    def test_default_provider_is_groq(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with (
            patch("shared.llm_client._call_groq", return_value="groq answer"),
            patch("shared.llm_client._call_groq_fallback") as mock_groq_fallback,
        ):
            _, provider = chat_complete(_MESSAGES)

        assert provider == "groq"
        mock_groq_fallback.assert_not_called()

    def test_groq_fallback_override_returns_text(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with patch("shared.llm_client._call_groq_fallback", return_value="groq_fallback answer"):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert provider == "groq_fallback"
        assert isinstance(text, str)
        assert len(text) > 0

    def test_groq_fallback_override_json_mode(self, monkeypatch):
        import json
        monkeypatch.setenv("LLM_PROVIDER", "groq_fallback")

        with patch("shared.llm_client._call_groq_fallback", return_value='{"status": "ok"}'):
            text, provider = chat_complete(_MESSAGES, json_mode=True)

        assert provider == "groq_fallback"
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Integration tests — real API calls, require keys in .env
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGroqIntegration:
    """Call Groq for real and verify the response contract.

    _call_groq_fallback is mocked here to guarantee Groq is the only network call —
    regardless of whether Groq itself triggers a fallback attempt, groq_fallback's
    rate limit cannot interfere with these assertions.
    """

    def test_groq_returns_text_and_provider(self, monkeypatch):
        import time
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        time.sleep(5)

        with patch("shared.llm_client._call_groq_fallback", return_value="mocked fallback"):
            text, provider = chat_complete(
                [{"role": "user", "content": "Reply with exactly one word: OK"}],
                system="You are a concise assistant. Follow instructions literally.",
            )

        assert provider == "groq", f"Expected 'groq', got '{provider}'"
        assert isinstance(text, str), "Response text must be a string"
        assert len(text) > 0, "Response text must not be empty"

    def test_groq_json_mode(self, monkeypatch):
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
        assert isinstance(parsed, dict), "json_mode response must deserialize to a dict"