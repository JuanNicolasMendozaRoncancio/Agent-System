"""
Tests for shared/llm_client.py.

Two test categories
-------------------
Unit tests (default, no marker):
    Pure mock tests — no network calls, no API keys needed.
    Run automatically in CI.

Integration tests (@pytest.mark.integration):
    Real API calls to Groq and Gemini.  Require GROQ_API_KEY and
    GEMINI_API_KEY in the environment (.env file).
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
    """Groq answers correctly — Gemini must not be touched."""

    def test_returns_groq_provider(self):
        with (
            patch("shared.llm_client._call_groq", return_value="hello") as mock_groq,
            patch("shared.llm_client._call_gemini") as mock_gemini,
        ):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert text == "hello"
        assert provider == "groq"
        mock_groq.assert_called_once()
        mock_gemini.assert_not_called()


class TestRateLimitFallback:
    """Groq returns 429 → must switch to Gemini, provider reported as 'gemini'."""

    def test_falls_back_on_rate_limit(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch("shared.llm_client._call_gemini", return_value="fallback text"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert text == "fallback text"
        assert provider == "gemini"

    def test_gemini_called_exactly_once_on_rate_limit(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch(
                "shared.llm_client._call_gemini", return_value="ok"
            ) as mock_gemini,
        ):
            chat_complete(_MESSAGES)

        mock_gemini.assert_called_once()


class TestTimeoutFallback:
    """Groq raises a generic exception → fallback to Gemini."""

    def test_falls_back_on_generic_error(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=Exception("connection timeout"),
            ),
            patch("shared.llm_client._call_gemini", return_value="gemini answer"),
        ):
            text, provider = chat_complete(_MESSAGES, timeout=30.0)

        assert provider == "gemini"
        assert text == "gemini answer"


class TestBothProvidersFail:
    """If both Groq and Gemini fail, RuntimeError must be raised."""

    def test_raises_runtime_error(self):
        with (
            patch(
                "shared.llm_client._call_groq",
                side_effect=_make_rate_limit_error(),
            ),
            patch(
                "shared.llm_client._call_gemini",
                side_effect=Exception("gemini also down"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Both LLM providers failed"):
                chat_complete(_MESSAGES)


class TestManualOverride:
    """LLM_PROVIDER=gemini env var must skip Groq entirely."""

    def test_gemini_override_skips_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")

        with (
            patch("shared.llm_client._call_groq") as mock_groq,
            patch("shared.llm_client._call_gemini", return_value="direct gemini"),
        ):
            text, provider = chat_complete(_MESSAGES)

        assert provider == "gemini"
        assert text == "direct gemini"
        mock_groq.assert_not_called()

    def test_default_provider_is_groq(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with (
            patch("shared.llm_client._call_groq", return_value="groq answer"),
            patch("shared.llm_client._call_gemini") as mock_gemini,
        ):
            _, provider = chat_complete(_MESSAGES)

        assert provider == "groq"
        mock_gemini.assert_not_called()

    def test_gemini_override_returns_text(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")

        with patch("shared.llm_client._call_gemini", return_value="gemini answer"):
            text, provider = chat_complete(_MESSAGES, system=_SYSTEM)

        assert provider == "gemini"
        assert isinstance(text, str)
        assert len(text) > 0

    def test_gemini_override_json_mode(self, monkeypatch):
        import json
        monkeypatch.setenv("LLM_PROVIDER", "gemini")

        with patch("shared.llm_client._call_gemini", return_value='{"status": "ok"}'):
            text, provider = chat_complete(_MESSAGES, json_mode=True)

        assert provider == "gemini"
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Integration tests — real API calls, require keys in .env
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGroqIntegration:
    """Call Groq for real and verify the response contract.

    _call_gemini is mocked here to guarantee Groq is the only network call —
    regardless of whether Groq itself triggers a fallback attempt, Gemini's
    rate limit cannot interfere with these assertions.
    """

    def test_groq_returns_text_and_provider(self, monkeypatch):
        import time
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        time.sleep(5)

        with patch("shared.llm_client._call_gemini", return_value="mocked fallback"):
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

        with patch("shared.llm_client._call_gemini", return_value='{"status": "mocked"}'):
            text, provider = chat_complete(
                [{"role": "user", "content": 'Return a JSON object with key "status" set to "ok"'}],
                json_mode=True,
            )

        assert provider == "groq"
        parsed = json.loads(text)
        assert isinstance(parsed, dict), "json_mode response must deserialize to a dict"