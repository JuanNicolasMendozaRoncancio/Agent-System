"""
SSE client for the Climate & Energy pipeline API.
 
Public interface
----------------
iter_pipeline_events(api_url, payload) -> Iterator[dict]
    Opens a streaming POST request to /pipeline/run and yields one parsed
    dict per SSE event until the stream closes.
 
Why requests.post(stream=True) and not httpx async:
    Streamlit runs each user interaction in a blocking thread. There is no
    running event loop to attach an async client to without spinning up a
    new one via asyncio.run(), which adds complexity for zero benefit here.
    requests with stream=True blocks the Streamlit thread intentionally —
    that is the correct behaviour while the user waits for pipeline progress.
 
Why iter_lines() and not iter_content():
    SSE events are newline-delimited text. iter_lines() splits on b'\n',
    decodes bytes to str, and filters empty lines automatically, giving one
    raw SSE line per iteration without manual boundary parsing.
"""
from __future__ import annotations

from typing import Iterator
import json
import logging

import requests

logger = logging.getLogger(__name__)

_PIPELINE_PATH = "/pipeline/run"
_REQUEST_TIMEOUT = 600

def iter_pipeline_events(api_url: str, payload: dict) -> Iterator[dict]:
    """
    POST to /pipeline/run and yield one parsed event dict per SSE event.
 
    Strips the 'data: ' prefix mandated by the SSE protocol before
    JSON-parsing each line. Lines that cannot be parsed are logged and
    skipped so a single malformed chunk never breaks the whole stream.
 
    Parameters
    ----------
    api_url:
        Base URL of the FastAPI server, e.g. 'http://localhost:8000'.
        Trailing slashes are stripped before appending the path.
    payload:
        JSON-serialisable dict matching PipelineRunRequest:
        {countries, date_from, date_to, run_type}.
 
    Yields
    ------
    dict
        Parsed SSE event. Shape depends on event type:
        - Agent progress : {"agent": str, "status": str, ...}
        - Pipeline done  : {"event": "pipeline_complete", "run_id": str, ...}
        - Pipeline failed: {"event": "pipeline_failed",   "error": str, ...}
 
    Raises
    ------
    requests.exceptions.ConnectionError
        If the API server is unreachable. Callers should catch this and
        display an appropriate error in the UI.
    """
    url = api_url.rstrip("/") + _PIPELINE_PATH
 
    with requests.post(
        url,
        json=payload,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue
            json_str = raw_line[len("data: "):]
            try:
                yield json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse SSE line: %s", raw_line[:200])