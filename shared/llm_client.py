"""
Centralized LLM client: Groq (primary) → Gemini 2.0 Flash (fallback).
 
Public interface
----------------
chat_complete(messages, *, system, json_mode, timeout) -> tuple[str, str]
    Returns (response_text, provider_used).
    provider_used is "groq" or "gemini" — callers store this in PostgreSQL.
 
Fallback logic
--------------
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.1-8b-instant"
_GEMINI_MODEL = "gemini-2.0-flash"

_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
 
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

def _call_gemini(
    messages: list[dict[str, str]],
    *,
    system: str | None,
    json_mode: bool,
    timeout: float,
) -> str:
    """
    Call Gemini 2.0 Flash via its OpenAI-compatible REST endpoint.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Gemini API not found")

    full_messages: list[dict[str, str]] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
 
    payload: dict[str, Any] = {
        "model": _GEMINI_MODEL,
        "messages": full_messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
 
    response = httpx.post(
        _GEMINI_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"] or ""

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
        provider — "groq" or "gemini"; store this in PostgreSQL so dashboards
                   can show which provider served each run.
 
    Raises
    ------
    RuntimeError
        If both providers fail.
    """
    override = os.getenv("LLM_PROVIDER", "").lower()

    if override == "gemini":
        logger.info("LLM_PROVIDER=gemini override; skipping Groq")
        text = _call_gemini(messages, system=system, json_mode=json_mode, timeout=timeout)
        return text, "gemini"

    t0 = time.monotonic()
    try:
        text = _call_groq(messages, system=system,
                          json_mode=json_mode, timeout=timeout)
        elapsed = time.monotonic()-t0
        logger.info("Groq answered in %.2fs", elapsed)
        return text, "groq"
    except RateLimitError:
        # HTTP 429: Groq rate-limit hit → fall through to Gemini immediately.
        logger.warning("Groq returned 429 (rate limit); switching to Gemini Flash")

    except Exception as exc:
        elapsed = time.monotonic() - t0
        if elapsed >= timeout:
            logger.warning(
                "Groq timed out after %.2fs; switching to Gemini Flash", elapsed
            )
        else:
            logger.warning("Groq error (%s); switching to Gemini Flash", exc)

    try:
        text = _call_gemini(messages, system=system, json_mode=json_mode, timeout=timeout)
        logger.info("Gemini Flash answered (fallback)")
        return text, "gemini"
 
    except Exception as exc:
        raise RuntimeError(
            f"Both LLM providers failed. Last Gemini error: {exc}"
        ) from exc