"""
Reporter Agent.

Closes the Sistema 1 pipeline by generating a fluent executive report that
integrates all signals produced during the run — profiling summary, QA
anomalies, RCA result — into 2-3 readable paragraphs for the dashboard.

Responsibilities
----------------
1. LLM report      — one call that synthesises profile_summary, anomalies,
                     and rca_result into a cohesive natural-language report.
2. Persistence     — UPDATE on data_quality_runs writing run_report to the
                     new dedicated column (row created by save_profile_node).
3. Redis publish   — event 'system1_complete' signals the Sistema 1 pipeline
                     is fully done; Sistema 2 subscribes to this event.

Graph
-----
START → reporter_node → save_reporter_node → END

Design rationale
----------------
The Reporter is the only agent that holds the complete run context — profile,
QA anomalies, RCA, provider metadata — because it runs last. That makes it
the correct place for the integrative LLM call. All previous agents produced
their own narrower summaries; the Reporter fuses them into one human-readable
narrative without duplicating any earlier computation.

One LLM call, O(n_countries * n_variables) tokens input, O(1) calls regardless
of how many anomalies or countries were processed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(encoding="latin-1")

from langgraph.graph import END, START, StateGraph
from sqlalchemy import text

from shared.llm_client import chat_complete
from shared.db import engine
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis
from System1.Profiling.profiling_agent import AgentState as _ProfilingState
from typing_extensions import TypedDict


class AgentState(_ProfilingState, total=False):
    """
    Extends the Profiling AgentState with fields written by the Reporter Agent.

    run_report:     fluent 2-3 paragraph executive report produced by the LLM.
    reporter_error: set if save_reporter_node fails; never raised.
    """
    run_report:     str
    reporter_error: str | None


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Reporter Agent of a Climate & Energy multi-agent monitoring system.
Your job is to write a clear, fluent executive report of a data pipeline run
for a technical dashboard audience (data engineers, energy analysts).

Write exactly 2-3 paragraphs. Do not use bullet points or headers.
Be specific: mention countries, record counts, variable names, anomaly
severities, and RCA findings when they are available.
If no anomalies were found, say so clearly and positively.
End with one sentence on data quality confidence for this run.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(state: AgentState) -> str:
    """
    Build a compact input prompt for the Reporter LLM call.

    Why build this in Python rather than passing AgentState directly:
    AgentState contains nested dicts (profile, drift results, raw records)
    that would bloat the prompt. We extract only the signals the LLM needs
    to write a fluent report — roughly 400-700 tokens regardless of run size.
    """
    profile = state.get("profile", {})
    anomalies_raw = state.get("anomalies", [])
    rca_result = state.get("rca_result", None)
    profile_summary = state.get("profile_summary", "")
    llm_provider = state.get("llm_provider", "unknown")
    countries = state.get("countries", [])
    date_from = state.get("date_from")
    date_to = state.get("date_to")
    run_type = state.get("run_type", "full")

    # Aggregate record count across countries
    total_records = sum(d.get("n_records", 0) for d in profile.values())

    # Summarise anomalies compactly
    anomaly_lines: list[str] = []
    for a in anomalies_raw:
        severity = a.get("severity", "UNKNOWN")
        variable = a.get("variable", "?")
        country  = a.get("country", "?")
        rule     = a.get("rule", "")
        anomaly_lines.append(f"  - [{severity}] {variable} ({country}): {rule}")
    anomaly_block = "\n".join(anomaly_lines) if anomaly_lines else "  None detected."

    # Drift summary from profile
    drift_lines: list[str] = []
    for country, data in profile.items():
        for var, d in data.get("drift", {}).items():
            if d.get("drift_detected"):
                drift_lines.append(
                    f"  - {var} ({country}): KL={d['kl']:.3f}"
                )
    drift_block = "\n".join(drift_lines) if drift_lines else "  No drift detected."

    rca_block = rca_result if rca_result else "Not triggered (no MEDIUM/CRITICAL anomalies)."

    period = (
        f"{date_from.isoformat()} → {date_to.isoformat()}"
        if date_from and date_to else "unknown period"
    )

    return (
        f"Run type: {run_type}\n"
        f"Period: {period}\n"
        f"Countries: {', '.join(countries)}\n"
        f"Total records ingested: {total_records}\n\n"
        f"Profiling summary:\n{profile_summary}\n\n"
        f"QA anomalies:\n{anomaly_block}\n\n"
        f"Drift alerts:\n{drift_block}\n\n"
        f"RCA result:\n{rca_block}\n\n"
        f"LLM provider used: {llm_provider}\n\n"
        "Write the executive report now."
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def reporter_node(state: AgentState) -> dict:
    """
    Generate the executive run report via a single LLM call.

    Reads from AgentState: profile, profile_summary, anomalies, rca_result,
    llm_provider, countries, date_from, date_to, run_type.

    Returns
    -------
    dict with keys:
        run_report    — fluent 2-3 paragraph string.
        llm_provider  — provider that served this call (may differ from
                        earlier calls if Groq rate-limited mid-run).
    """
    prompt = _build_prompt(state)

    try:
        report_text, provider = chat_complete(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM_PROMPT,
        )
        logger.info("reporter_node: report generated via %s", provider)
    except Exception as exc:
        logger.error("reporter_node: LLM call failed: %s", exc)
        # Fallback: structured plain-text report built entirely from AgentState
        profile = state.get("profile", {})
        total_records = sum(d.get("n_records", 0) for d in profile.values())
        report_text = (
            f"Run completed for {', '.join(state.get('countries', []))}. "
            f"{total_records} records ingested. "
            f"Profile summary: {state.get('profile_summary', 'unavailable')}. "
            "LLM report generation failed — see logs for details."
        )
        provider = None

    return {"run_report": report_text, "llm_provider": provider}


def save_reporter_node(state: AgentState) -> dict:
    """
    Persist the run report and publish the system1_complete event to Redis.

    DB write: UPDATE data_quality_runs SET run_report = :run_report,
              status = 'complete' WHERE run_id = :run_id.
    The row was created by save_profile_node (INSERT); this is the final
    UPDATE that closes the Sistema 1 lifecycle for this run.

    Redis: publishes event 'system1_complete' regardless of DB outcome so
    Sistema 2 is always notified even when persistence fails.

    Why UPDATE and not INSERT:
        save_profile_node already created the data_quality_runs row with
        INSERT. Every subsequent agent (QA, RCA, Reporter) only enriches
        that same row. A second INSERT would violate the UNIQUE constraint
        on run_id.
    """
    run_id     = state["run_id"]
    run_report = state.get("run_report", "")
    profile    = state.get("profile", {})
    llm_provider = state.get("llm_provider")

    total_records = sum(d.get("n_records", 0) for d in profile.values())
    countries = list(profile.keys())
    completed_at = datetime.now(timezone.utc)

    error_msg: str | None = None

    # --- 1. UPDATE data_quality_runs ----------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE data_quality_runs
                       SET run_report    = :run_report,
                           completed_at  = :completed_at,
                           llm_provider  = :llm_provider,
                           status        = 'complete'
                     WHERE run_id = :run_id
                """),
                {
                    "run_id":       run_id,
                    "run_report":   run_report,
                    "completed_at": completed_at,
                    "llm_provider": llm_provider,
                },
            )
            conn.commit()
        logger.info("save_reporter_node: updated data_quality_runs for run_id=%s", run_id)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_reporter_node: DB update failed: %s", exc)

    # --- 2. Publish system1_complete to Redis --------------------------------
    redis_message = json.dumps({
        "run_id":      run_id,
        "event":       "system1_complete",
        "n_records":   total_records,
        "countries":   countries,
        "run_report":  run_report,
        "timestamp":   completed_at.isoformat(),
    })

    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_reporter_node: published system1_complete to Redis")
    except Exception as exc:
        logger.warning("save_reporter_node: Redis publish failed: %s", exc)

    return {"reporter_error": error_msg}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_reporter_graph():
    """
    Compile and return the Reporter Agent as a LangGraph graph.

    Graph: START → reporter_node → save_reporter_node → END

    Linear — no conditional edges. The LLM call always runs; if it fails,
    reporter_node returns a Python-built fallback so save_reporter_node
    always has something to persist.
    """
    graph = StateGraph(AgentState)

    graph.add_node("reporter_node",      reporter_node)
    graph.add_node("save_reporter_node", save_reporter_node)

    graph.add_edge(START,              "reporter_node")
    graph.add_edge("reporter_node",    "save_reporter_node")
    graph.add_edge("save_reporter_node", END)

    return graph.compile(checkpointer=None)


def invoke_reporter_graph(state: AgentState) -> AgentState:
    """Invoke the Reporter Agent graph."""
    graph = build_reporter_graph()
    return graph.invoke(state, config={"recursion_limit": 10})