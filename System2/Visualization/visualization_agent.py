"""
Visualization Agent.

Reads analysis_results from AgentState (patterns + risk indicators already
computed by the Analysis Agent) and produces Streamlit-ready chart structures
stored in analysis_runs.viz_json.

Responsibilities
----------------
1. time_series   — SQL query per country aggregated at configurable granularity
                   (default: daily). One data point per (day, variable) pair,
                   ready for st.line_chart.
2. bar_stats     — Per-country, per-variable descriptive stats (mean, min, max,
                   slope, n) read directly from AgentState. No additional DB I/O.
3. country_comparison — Per-variable cross-country mean comparison, ready for
                        st.bar_chart.
4. risk_breakdown — Per-country C1–C4 component scores with weights and total,
                    ready for a stacked bar or radar chart.

Graph
-----
START → viz_node → save_viz_node → END

"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text

load_dotenv()

from shared.db import engine
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis
from System2.Analysis.analysis_agent import AgentState as _AnalysisState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ALLOWED_GRANULARITIES = {"hour", "day", "week"}

VIZ_TIME_GRANULARITY: str = os.getenv(
    "VIZ_TIME_GRANULARITY", "day"
).lower()

if VIZ_TIME_GRANULARITY not in _ALLOWED_GRANULARITIES:
    logger.warning(
        "VIZ_TIME_GRANULARITY='%s' is not valid. Falling back to 'day'. "
        "Allowed values: %s",
        VIZ_TIME_GRANULARITY,
        _ALLOWED_GRANULARITIES,
    )
    VIZ_TIME_GRANULARITY = "day"

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(_AnalysisState, total=False):
    """
    Extends the Analysis AgentState with fields written by the Visualization Agent.

    viz_data:
        Streamlit-ready chart structures produced by viz_node.
        Written to analysis_runs.viz_json by save_viz_node.

    viz_error:
        Captured exception string from save_viz_node; never raised.
    """
    viz_data:  dict
    viz_error: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_time_series(
    countries: list[str],
    run_id: str,
    granularity: str,
) -> dict[str, dict[str, list[dict]]]:
    """
    Query energy_climate_records and return time-aggregated series per country.

    Uses DATE_TRUNC to aggregate raw hourly records into the requested
    granularity (hour / day / week). The anchor is the MAX(timestamp) of the
    current run — same strategy as _fetch_records_for_window in the Analysis
    Agent — so the window is consistent regardless of when the agent runs.

    Parameters
    ----------
    countries:
        ISO-2 country codes to query.
    run_id:
        Used to anchor the time window to this run's latest timestamp.
    granularity:
        One of 'hour', 'day', 'week' — validated before this call.

    Returns
    -------
    dict keyed by country → variable → list of {"t": ISO str, "v": float}
    """
    result: dict[str, dict[str, list[dict]]] = {}

    for country in countries:
        result[country] = {}
        try:
            with engine.connect() as conn:
                anchor_row = conn.execute(
                    text("""
                        SELECT MAX(timestamp)
                        FROM energy_climate_records
                        WHERE run_id = :run_id AND country = :country
                    """),
                    {"run_id": run_id, "country": country},
                ).fetchone()

                if not anchor_row or anchor_row[0] is None:
                    logger.warning(
                        "_fetch_time_series: no records for %s / run_id=%s",
                        country, run_id,
                    )
                    continue

                anchor_ts = anchor_row[0]
                if anchor_ts.tzinfo is None:
                    anchor_ts = anchor_ts.replace(tzinfo=timezone.utc)

                rows = conn.execute(
                    text(f"""
                        SELECT
                            DATE_TRUNC('{granularity}', timestamp) AS period,
                            variable,
                            AVG(value)                             AS mean_value
                        FROM energy_climate_records
                        WHERE country   = :country
                          AND timestamp <= :anchor
                        GROUP BY period, variable
                        ORDER BY variable, period
                    """),
                    {"country": country, "anchor": anchor_ts},
                ).fetchall()

            by_variable: dict[str, list[dict]] = {}
            for row in rows:
                period, variable, mean_value = row
                t_str = (
                    period.isoformat()
                    if hasattr(period, "isoformat")
                    else str(period)
                )
                by_variable.setdefault(variable, []).append(
                    {"t": t_str, "v": round(float(mean_value), 4)}
                )

            result[country] = by_variable
            logger.info(
                "_fetch_time_series: %s — %d variables, granularity=%s",
                country, len(by_variable), granularity,
            )

        except Exception as exc:
            logger.warning(
                "_fetch_time_series: query failed for %s: %s", country, exc
            )

    return result


def _build_bar_stats(
    analysis_results: dict,
) -> dict[str, dict[str, dict]]:
    """
    Extract per-country, per-variable descriptive stats from analysis_results.

    Reads directly from AgentState — no additional DB query. The Analysis
    Agent already computed slope, mean, min, max, n for every variable.

    Returns
    -------
    {country: {variable: {mean, min, max, slope, n}}}
    """
    bar_stats: dict[str, dict[str, dict]] = {}

    for country, data in analysis_results.items():
        patterns = data.get("patterns", {})
        variables = patterns.get("variables", {})
        bar_stats[country] = {
            variable: {
                "mean":  stats.get("mean"),
                "min":   stats.get("min"),
                "max":   stats.get("max"),
                "slope": stats.get("slope"),
                "n":     stats.get("n"),
            }
            for variable, stats in variables.items()
        }

    return bar_stats


def _build_country_comparison(
    analysis_results: dict,
) -> dict[str, dict[str, float]]:
    """
    Build a cross-country mean comparison per variable.

    Inverts the {country: {variable: stats}} structure into
    {variable: {country: mean}} so Streamlit can render a bar chart
    comparing all countries for each variable side by side.

    Returns
    -------
    {variable: {country: mean_value}}
    """
    comparison: dict[str, dict[str, float]] = {}

    for country, data in analysis_results.items():
        patterns  = data.get("patterns", {})
        variables = patterns.get("variables", {})
        for variable, stats in variables.items():
            mean = stats.get("mean")
            if mean is not None:
                comparison.setdefault(variable, {})[country] = mean

    return comparison


def _build_risk_breakdown(
    analysis_results: dict,
) -> dict[str, dict]:
    """
    Extract the risk score breakdown per country for dashboard visualisation.

    Produces a structure with:
    - total_score    — composite risk index (0–100).
    - has_temperature_data — whether C4 was active for this country.
    - components     — {component_name: {score, weight}} for C1–C4.

    Returns
    -------
    {country: {total_score, has_temperature_data, components: {name: {score, weight}}}}
    """
    breakdown: dict[str, dict] = {}

    for country, data in analysis_results.items():
        risk = data.get("risk", {})
        if not risk or risk.get("error"):
            breakdown[country] = {"error": risk.get("error", "unavailable")}
            continue

        components    = risk.get("components", {})
        weights_used  = risk.get("weights_used", {})

        breakdown[country] = {
            "total_score":          round(risk.get("score", 0.0), 2),
            "has_temperature_data": risk.get("has_temperature_data", False),
            "components": {
                name: {
                    "score":  round(score, 2),
                    "weight": weights_used.get(name, 0.0),
                }
                for name, score in components.items()
            },
        }

    return breakdown


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def viz_node(state: AgentState) -> dict:
    """
    Build all Streamlit-ready chart structures from AgentState and DB.

    Two data sources:
    1. DB query  — time_series via _fetch_time_series (one query per country).
    2. In-memory — bar_stats, country_comparison, risk_breakdown from
                   analysis_results already in AgentState.

    Returns
    -------
    dict with key 'viz_data': the complete chart structure for viz_json.
    """
    run_id           = state["run_id"]
    countries        = state.get("countries", [])
    analysis_results = state.get("analysis_results", {})

    time_series = _fetch_time_series(countries, run_id, VIZ_TIME_GRANULARITY)

    bar_stats = _build_bar_stats(analysis_results)

    country_comparison = _build_country_comparison(analysis_results)

    risk_breakdown = _build_risk_breakdown(analysis_results)

    viz_data: dict[str, Any] = {
        "time_series":        time_series,
        "bar_stats":          bar_stats,
        "country_comparison": country_comparison,
        "risk_breakdown":     risk_breakdown,
        "granularity":        VIZ_TIME_GRANULARITY,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "viz_node: built viz_data for %d countries (granularity=%s)",
        len(countries), VIZ_TIME_GRANULARITY,
    )

    return {"viz_data": viz_data}


def save_viz_node(state: AgentState) -> dict:
    """
    Persist viz_data to analysis_runs.viz_json and publish to Redis.

    Uses UPDATE (not INSERT) — the row was created by the subscriber
    (status='triggered') and enriched by save_analysis_node. This agent
    adds viz_json to that same row.

    Redis publishes event 'viz_complete' regardless of DB outcome so the
    Narrative Agent is always notified.
    """
    run_id       = state["run_id"]
    viz_data     = state.get("viz_data", {})
    completed_at = datetime.now(timezone.utc)
    error_msg: str | None = None

    # --- UPDATE analysis_runs.viz_json --------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE analysis_runs
                       SET viz_json     = :viz_json,
                           completed_at = :completed_at,
                           status       = 'viz_complete'
                     WHERE run_id = :run_id
                """),
                {
                    "run_id":       run_id,
                    "viz_json":     json.dumps(viz_data),
                    "completed_at": completed_at,
                },
            )
            conn.commit()
        logger.info(
            "save_viz_node: updated analysis_runs.viz_json for run_id=%s", run_id
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_viz_node: DB update failed: %s", exc)

    # --- Publish viz_complete to Redis --------------------------------------
    redis_message = json.dumps({
        "run_id":    run_id,
        "event":     "viz_complete",
        "countries": state.get("countries", []),
        "timestamp": completed_at.isoformat(),
    })

    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_viz_node: published viz_complete to Redis")
    except Exception as exc:
        logger.warning("save_viz_node: Redis publish failed: %s", exc)

    return {"viz_error": error_msg}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_viz_graph():
    """
    Compile and return the Visualization Agent as a LangGraph graph.

    Graph: START → viz_node → save_viz_node → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("viz_node",      viz_node)
    graph.add_node("save_viz_node", save_viz_node)

    graph.add_edge(START,         "viz_node")
    graph.add_edge("viz_node",    "save_viz_node")
    graph.add_edge("save_viz_node", END)

    return graph.compile(checkpointer=None)


def invoke_viz_graph(state: AgentState) -> AgentState:
    """
    Invoke the Visualization Agent graph.
    """
    graph = build_viz_graph()
    return graph.invoke(state, config={"recursion_limit": 10})