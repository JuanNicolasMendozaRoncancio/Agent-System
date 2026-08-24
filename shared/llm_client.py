"""
Centralized LLM client: Groq gpt-oss-20b (primary) → Groq gpt-oss-120b (fallback).

Both models are served by Groq, so a single GROQ_API_KEY covers both paths.
The fallback is triggered on RateLimitError (HTTP 429) or any other exception
from the primary model.

Public interface
----------------
chat_complete(messages, *, system, json_mode, timeout) -> tuple[str, str]
    Returns (response_text, provider_used).
    provider_used is "groq" or "groq_fallback" — callers store this in PostgreSQL.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_GROQ_MODEL = "openai/gpt-oss-20b"
# Larger Groq-hosted model used as fallback when the primary hits rate limits
# or errors. Same GROQ_API_KEY — no second vendor or extra credential needed.
_GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"
 
_DEFAULT_TIMEOUT = 40.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Groq API not found in env")
    return Groq(api_key=api_key)

def _call_groq(
    messages: list[dict[str, str]],
    *,
    system: str | None,
    json_mode: bool,
    timeout: float,
) -> str:
    """
    Call Groq synchronously and return the response text.
    """
    client = _groq_client()

    full_messages: list[dict[str,str]] = []
    if system:
        full_messages.append({"role":"system",
                              "content": system})
    full_messages.extend(messages)

    kwargs: dict[str, Any] = {
        "model": _GROQ_MODEL,
        "messages": full_messages,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""

def _call_groq_fallback(
    messages: list[dict[str, str]],
    *,
    system: str | None,
    json_mode: bool,
    timeout: float,
) -> str:
    """
    Call the larger Groq fallback model (gpt-oss-120b) using the same
    GROQ_API_KEY as the primary model.

    Why a separate function instead of passing the model name to _call_groq:
        Keeps _call_groq's signature stable (no model parameter leak into
        callers) and makes the fallback path explicit and independently testable.
        The mock target in tests is 'shared.llm_client._call_groq_fallback',
        mirroring the old '_call_gemini' mock target used in test_llm_client.py.
    """
    client = _groq_client()

    full_messages: list[dict[str, str]] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    kwargs: dict[str, Any] = {
        "model": _GROQ_FALLBACK_MODEL,
        "messages": full_messages,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def chat_complete(
    messages: list[dict[str, str]],
    *,
    system: str | None = None,
    json_mode: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """
    Send a chat request and return (response_text, provider_used).
 
    Parameters
    ----------
    messages:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
    system:
        Optional system prompt injected as the first message.
    json_mode:
        When True, both providers are instructed to return valid JSON only.
        The response text is still a plain string — the caller parses it.
    timeout:
        Seconds before a provider attempt is considered failed and the fallback
        is triggered.  Default 30 s matches the plan spec.
 
    Returns
    -------
    (text, provider)
        text     — raw string content from the model.
        provider — "groq" or "groq_fallback"; store this in PostgreSQL so
                   dashboards can show which model served each run.

    Raises
    ------
    RuntimeError
        If both Groq models fail.
    """
    override = os.getenv("LLM_PROVIDER", "").lower()

    if override == "groq_fallback":
        logger.info("LLM_PROVIDER=groq_fallback override; skipping primary model")
        text = _call_groq_fallback(messages, system=system, json_mode=json_mode, timeout=timeout)
        return text, "groq_fallback"

    t0 = time.monotonic()
    try:
        text = _call_groq(messages, system=system,
                          json_mode=json_mode, timeout=timeout)
        elapsed = time.monotonic() - t0
        logger.info("Groq (primary) answered in %.2fs", elapsed)
        return text, "groq"
    except RateLimitError:
        # HTTP 429: primary model rate-limited → fall through to larger model.
        logger.warning("Groq primary returned 429 (rate limit); switching to gpt-oss-120b")

    except Exception as exc:
        elapsed = time.monotonic() - t0
        if elapsed >= timeout:
            logger.warning(
                "Groq primary timed out after %.2fs; switching to gpt-oss-120b", elapsed
            )
        else:
            logger.warning("Groq primary error (%s); switching to gpt-oss-120b", exc)

    try:
        text = _call_groq_fallback(messages, system=system, json_mode=json_mode, timeout=timeout)
        logger.info("Groq fallback (gpt-oss-120b) answered")
        return text, "groq_fallback"

    except Exception as exc:
        raise RuntimeError(
            f"Both Groq models failed. Last error: {exc}"
        ) from exc