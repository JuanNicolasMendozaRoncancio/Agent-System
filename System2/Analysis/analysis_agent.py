"""
Analysis Agent — Consumer.
 
First processing node of System 2. Activated by the Redis subscriber after
receiving a 'system1_complete' event. Reads energy_climate_records for the
triggered run, detects temporal patterns, computes an energy supply risk index,
and fetches active documentary topics from the RAG system.
 
Responsibilities
----------------
1. detect_patterns()       — Temporal trend analysis over 24h / 7d / 30d windows.
                             Falls back to available data if the requested window
                             exceeds what is stored in PostgreSQL.
2. compute_risk_indicators() — Composite energy supply risk score (0–100) built
                               from four components derived from ENTSO-E data.
                               If Copernicus temperature data is unavailable for
                               a country, the temperature component weight is
                               redistributed proportionally across the other three.
3. rag_context()           — Calls GET /rag/topics/active on the RAG API to
                             retrieve active documentary topics for narrative
                             enrichment. Returns an empty list gracefully if the
                             endpoint is unreachable or unconfigured.
 
Graph
-----
START → analysis_node → tool_node → process_evidence_node → save_analysis_node → END
 
Risk score components
---------------------
C1 — Demand coverage        30 %  : total_generation / load
C2 — Renewable intermittency 25 % : (wind + solar) / total_generation
C3 — Hydraulic buffer        25 % : (hydro_reservoir + pumped_storage) / load
C4 — Temperature vs demand   20 % : extreme temperature contribution to demand
 
If Copernicus data is unavailable for a country, C4 weight = 0 and the 20 %
is redistributed proportionally: C1 → 37.5 %, C2 → 31.25 %, C3 → 31.25 %.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import text

load_dotenv()
from shared.db import engine
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis

from System1.Reporter.reporter_agent import AgentState as _ReporterState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_RAG_API_URL = os.getenv("RAG_API_URL", "").rstrip("/")
_RAG_API_KEY = os.getenv("RAG_API_KEY", "")
_RAG_TIMEOUT = 15.0  

_WEIGHTS_FULL = {
    "demand_coverage":       0.30,
    "renewable_intermittency": 0.25,
    "hydraulic_buffer":      0.25,
    "temperature_demand":    0.20,
}

_WEIGHTS_NO_TEMP = {
    "demand_coverage":       0.375,
    "renewable_intermittency": 0.3125,
    "hydraulic_buffer":      0.3125,
    "temperature_demand":    0.0,
}

_WIND_VARS = {
    "generation_wind_onshore",
    "generation_wind_offshore",
}
_SOLAR_VARS = {
    "generation_solar",
}
_HYDRO_BUFFER_VARS = {
    "generation_hydro_water_reservoir",
    "generation_hydro_pumped_storage",
}
_HYDRO_RUN_VARS = {
    "generation_hydro_run-of-river_and_poundage",
}
_GENERATION_PREFIX = "generation_"

_COLD_THRESHOLD_C = 5.0
_HOT_THRESHOLD_C  = 28.0

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
class AgentState(_ReporterState, total=False):
    """
    Extends the Reporter AgentState with fields written by the Analysis Agent.
 
    analysis_results:
        Structured dict containing pattern trends and risk indicators per country.
        Written to analysis_runs.charts_json by save_analysis_node.
        Read by the Visualization Agent and Narrative Agent.
 
    rag_topics:
        Active documentary topics from /rag/topics/active.
        Passed to the Narrative Agent for contextual enrichment.
 
    analysis_error:
        Captured exception string from save_analysis_node; never raised.
    """
    analysis_results: dict
    rag_topics:       list[dict]
    analysis_error:   str | None

# ---------------------------------------------------------------------------
# Internal helpers — deterministic compute functions
# ---------------------------------------------------------------------------
def _fetch_records_for_window(
    country: str,
    run_id: str,
    window_days: int,
) -> tuple[list[dict], int]:
    """
    Fetch energy_climate_records for a country over the requested window.
 
    Strategy:
    1. Try to fetch `window_days` of data going back from the latest timestamp
       of the current run.
    2. If the actual span of data is shorter than requested, fall back to
       whatever is available and return the actual span in days.
 
    Returns
    -------
    (records, actual_window_days)
        records            — list of row dicts with keys: variable, value, timestamp.
        actual_window_days — days actually covered (may be < window_days).
    """
    try:
        with engine.connect() as conn:
            # Anchor: latest timestamp for this run in the given country.
            anchor_row = conn.execute(
                text("""
                    SELECT MAX(timestamp)
                    FROM energy_climate_records
                    WHERE run_id = :run_id AND country = :country
                """),
                {"run_id": run_id, "country": country},
            ).fetchone()
 
            if not anchor_row or anchor_row[0] is None:
                return [], 0
 
            anchor_ts: datetime = anchor_row[0]
            if anchor_ts.tzinfo is None:
                anchor_ts = anchor_ts.replace(tzinfo=timezone.utc)
 
            cutoff = anchor_ts - timedelta(days=window_days)
 
            rows = conn.execute(
                text("""
                    SELECT variable, value, timestamp, source_api
                    FROM energy_climate_records
                    WHERE country   = :country
                      AND timestamp >= :cutoff
                      AND timestamp <= :anchor
                    ORDER BY variable, timestamp
                """),
                {"country": country, "cutoff": cutoff, "anchor": anchor_ts},
            ).fetchall()
 
        if not rows:
            return [], 0
 
        records = [
            {
                "variable":   r[0],
                "value":      float(r[1]),
                "timestamp":  r[2],
                "source_api": r[3],
            }
            for r in rows
        ]
 
        timestamps = [r["timestamp"] for r in records]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        if min_ts.tzinfo is None:
            min_ts = min_ts.replace(tzinfo=timezone.utc)
        if max_ts.tzinfo is None:
            max_ts = max_ts.replace(tzinfo=timezone.utc)
 
        actual_days = max(1, int((max_ts - min_ts).total_seconds() / 86400) + 1)
        return records, min(actual_days, window_days)
 
    except Exception as exc:
        logger.warning("_fetch_records_for_window failed [%s/%s]: %s", country, window_days, exc)
        return [], 0


def _compute_trend(values: list[float]) -> dict[str, float]:
    """
    Compute a simple linear trend over a list of values.
 
    Returns slope (change per step), mean, min, max, and n.
    Slope is computed as (last_mean - first_mean) / n_points, where first and
    last refer to the first and last third of the series respectively.
    """
    n = len(values)
    if n == 0:
        return {"slope": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    mean_val = sum(values)/n
    min_val = min(values)
    max_val = max(values)

    if n < 3:
        return {"slope": 0.0, "mean": round(mean_val, 4),
                "min": round(min_val, 4), "max": round(max_val, 4), "n": n}

    third = max(1, n // 3)
    first_mean = sum(values[:third])/third
    last_mean = sum(values[-third:])/third
    slope = (last_mean - first_mean)/n

    return {
        "slope": round(slope, 6),
        "mean":  round(mean_val, 4),
        "min":   round(min_val, 4),
        "max":   round(max_val, 4),
        "n":     n,
    }


def _detect_patterns_for_country(
    country: str,
    run_id: str,
    window_days: int,
) -> dict[str, Any]:
    """
    Detect temporal trends per variable for a single country.
 
    Fetches data for the requested window (falling back to available data if
    the window exceeds what is stored) and computes trend statistics per variable.
 
    Returns
    -------
    dict with keys:
        actual_window_days — days actually covered.
        requested_window_days — what was requested.
        fallback_used — True if actual < requested.
        variables — dict keyed by variable name, each a trend dict.
    """
    records, actual_days = _fetch_records_for_window(country, run_id, window_days)

    result: dict[str, Any] = {
        "actual_window_days":    actual_days,
        "requested_window_days": window_days,
        "fallback_used":         actual_days < window_days,
        "variables":             {},
    }

    if not records:
        return result

    by_variable: dict[str, list[float]] = {}
    for rec in records:
        by_variable.setdefault(rec["variable"], []).append(rec["value"])

    for variable, values in by_variable.items():
        result["variables"][variable] = _compute_trend(values)

    return result

def _compute_risk_for_country(
    country: str,
    run_id: str,
) -> dict[str, Any]:
    """
    Compute the energy supply risk score (0–100) for a single country.
 
    Components
    ----------
    C1 — Demand coverage (30 % or 37.5 % without temperature data)
         ratio = total_generation / load
         risk  = max(0, 1 - ratio) * 100   (0 if generation >= load)
 
    C2 — Renewable intermittency (25 % or 31.25 % without temperature data)
         ratio = (wind + solar) / total_generation
         risk  = ratio * 100   (higher renewable share → higher intermittency risk)
 
    C3 — Hydraulic buffer (25 % or 31.25 % without temperature data)
         ratio = hydro_dispatchable / load
         risk  = max(0, 1 - ratio * 10) * 100   (scaled: 10 % buffer → 0 risk)
 
    C4 — Temperature vs demand (20 % if Copernicus data available, else 0 %)
         Uses mean temperature of the window.
         risk  = 100 if temp < COLD_THRESHOLD or temp > HOT_THRESHOLD, else
                 linear interpolation between 0 and 100 toward the nearest threshold.
 
    Weight redistribution when C4 is unavailable:
        Weights for C1, C2, C3 are scaled proportionally so they sum to 1.0.
 
    Returns
    -------
    dict with keys: score (float 0-100), components (breakdown), weights_used,
    has_temperature_data (bool), error (str | None).
    """
    result: dict[str, Any] = {
        "score":                0.0,
        "components":          {},
        "weights_used":        {},
        "has_temperature_data": False,
        "error":               None,
    }

    try:
        records, _ = _fetch_records_for_window(country, run_id, window_days=1)

        if not records:
            result["error"] = f"No records available for {country}"
            return result

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in records:
            var = rec["variable"]
            sums[var] = sums.get(var, 0) + rec["value"]
            counts[var] = counts.get(var, 0) + 1

        means: dict[str, float] = {
            var: sums[var]/ counts[var]
            for var in sums
        }

        # --- C1: Demand coverage -------------------------------------------
        total_gen = sum(
            v for k, v in means.items()
            if k.startswith(_GENERATION_PREFIX)
        )
        load = means.get("load_actual_aggregated", 0.0)

        if load > 0:
            coverage_ratio = total_gen/load
            c1_risk = max(0.0, 1.0 - coverage_ratio) * 100.0
        else:
            c1_risk = 50.0

        # --- C2: Renewable intermittency ------------------------------------
        renewables = sum(
            means.get(v,0) for v in _WIND_VARS | _SOLAR_VARS
        )

        if total_gen > 0:
            intermittency_ratio = renewables / total_gen
            c2_risk = intermittency_ratio * 100.0
        else:
            c2_risk = 0.0

        # --- C3: Hydraulic buffer ------------------------------------------
        hydro_buffer = sum(
            means.get(v,0) for v in _HYDRO_BUFFER_VARS
        )

        if load > 0:
            buffer_ratio = hydro_buffer / load
            c3_risk = max(0.0, 1.0 - buffer_ratio * 10.0) * 100.0
        else:
            c3_risk = 50.0 

        # --- C4: Temperature vs demand (Copernicus) ------------------------
        temp_mean = means.get("climate_temperature_2m")
        has_temp  = temp_mean is not None
 
        if has_temp:
            if temp_mean < _COLD_THRESHOLD_C:
                c4_risk = min(100.0, (_COLD_THRESHOLD_C - temp_mean) / 25.0 * 100.0)
            elif temp_mean > _HOT_THRESHOLD_C:
                c4_risk = min(100.0, (temp_mean - _HOT_THRESHOLD_C) / 17.0 * 100.0)
            else:
                c4_risk = 0.0
        else:
            c4_risk = 0.0

        # --- Weight selection and score ------------------------------------
        weights = _WEIGHTS_FULL if has_temp else _WEIGHTS_NO_TEMP

        score = (
            weights["demand_coverage"]       * c1_risk
            + weights["renewable_intermittency"] * c2_risk
            + weights["hydraulic_buffer"]      * c3_risk
            + weights["temperature_demand"]    * c4_risk
        )

        result.update({
            "score": round(score, 2),
            "components": {
                "demand_coverage":        round(c1_risk, 2),
                "renewable_intermittency": round(c2_risk, 2),
                "hydraulic_buffer":       round(c3_risk, 2),
                "temperature_demand":     round(c4_risk, 2),
            },
            "weights_used":         weights,
            "has_temperature_data": has_temp,
            "auxiliary": {
                "total_generation_mw": round(total_gen, 2),
                "load_mw":             round(load, 2),
                "renewables_mw":       round(renewables, 2),
                "hydro_buffer_mw":     round(hydro_buffer, 2),
                "temperature_c":       round(temp_mean, 4) if has_temp else None,
            },
        })
 
    except Exception as exc:
        logger.warning("_compute_risk_for_country failed [%s]: %s", country, exc)
        result["error"] = str(exc)[:200]
 
    return result

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def detect_patterns(run_id: str, country: str, window_days: int) -> dict:
    """
    Detect temporal trends in energy and climate variables for a country.
 
    Queries energy_climate_records for the requested window going back from
    the latest timestamp of the current run. If less data is available than
    requested, falls back to the available span and sets fallback_used=True.
 
    Args:
        run_id:      Current run identifier — used to anchor the time window.
        country:     ISO-2 country code (e.g. 'FR', 'DE').
        window_days: Requested analysis window in days (e.g. 1, 7, 30).
 
    Returns:
        Dict with actual_window_days, fallback_used, and per-variable trend
        statistics (slope, mean, min, max, n).
    """
    logger.info("detect_patterns: country=%s window=%dd run_id=%s",
                country, window_days, run_id)
    result = _detect_patterns_for_country(country, run_id, window_days)
    result["country"]    = country.upper()
    result["run_id"]     = run_id
    result["window_days"] = window_days
    return result

@tool
def compute_risk_indicators(run_id: str, country: str) -> dict:
    """
    Compute the energy supply risk index (0–100) for a country.
 
    Four components: demand coverage, renewable intermittency, hydraulic buffer,
    and temperature-driven demand pressure. If Copernicus temperature data is
    unavailable for the country, the temperature component weight is redistributed
    proportionally across the other three components.
 
    Args:
        run_id:  Current run identifier — used to fetch the relevant records.
        country: ISO-2 country code (e.g. 'FR', 'DE').
 
    Returns:
        Dict with score (0–100), component breakdown, weights used, and auxiliary
        generation/demand figures.
    """
    logger.info("compute_risk_indicators: country=%s run_id=%s", country, run_id)
    result = _compute_risk_for_country(country, run_id)
    result["country"] = country.upper()
    result["run_id"]  = run_id
    return result

@tool
def rag_context(query: str) -> dict:
    """
    Fetch active documentary topics from the RAG system.
 
    Calls GET /rag/topics/active on the RAG API. Returns the list of active
    topics for narrative enrichment by the Narrative Agent. Returns an empty
    list gracefully if the endpoint is unreachable or RAG_API_URL is not set.
 
    Args:
        query: Context string describing the current run (e.g. 'energy analysis
               France Germany June 2024'). Sent as a query parameter so the RAG
               API can filter topics by relevance if it supports it.
 
    Returns:
        Dict with keys: topics (list of topic dicts), n_topics, error.
    """
    result: dict[str, Any] = {"topics": [], "n_topics": 0, "error": None}
 
    if not _RAG_API_URL:
        result["error"] = "RAG_API_URL not configured — skipping rag_context"
        logger.info("rag_context: RAG_API_URL not set, returning empty topics")
        return result
 
    try:
        response = httpx.get(
            f"{_RAG_API_URL}/rag/topics/active",
            params={"query": query},
            headers={"X-RAG-Key": _RAG_API_KEY},
            timeout=_RAG_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
 
        topics = data if isinstance(data, list) else data.get("topics", [])
        result["topics"]   = topics
        result["n_topics"] = len(topics)
        logger.info("rag_context: received %d active topics", len(topics))
 
    except Exception as exc:
        logger.warning("rag_context: request failed: %s", exc)
        result["error"] = str(exc)[:200]
 
    return result
 
 
_TOOLS = [detect_patterns, compute_risk_indicators, rag_context]

# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------
def _build_llm_with_tools():
    """
    Return (llm_with_tools, provider_name).
    """
    from langchain_groq import ChatGroq
 
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not found in env.")
 
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=groq_key,
        temperature=0,
    )
    return llm.bind_tools(_TOOLS, parallel_tool_calls=True), "groq"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
 
_SYSTEM_PROMPT = """\
You are the Analysis Agent of a Climate & Energy multi-agent monitoring system.
You receive a run_id and a list of countries. Your job is to collect analytical
evidence by calling the available tools.
 
Rules:
- Call detect_patterns for EACH country with window_days=30. If the tool returns
  fallback_used=True, that is expected — use whatever data is available.
- Call compute_risk_indicators for EACH country.
- Call rag_context ONCE with a brief query describing the run context
  (e.g. 'energy climate analysis France Germany').
- Emit ALL tool calls in a single response — do not wait for intermediate results.
- Do not interpret results — that is the job of the Narrative Agent.
"""

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def analysis_node(state: AgentState) -> dict:
    """
    Orchestration node: LLM emits all tool calls in a single parallel response.
 
    Sends the run context (run_id, countries, date window) to the LLM and
    lets it decide which tool calls to emit. Always: detect_patterns and
    compute_risk_indicators for each country + one rag_context call.
    """
    llm_with_tools, provider = _build_llm_with_tools()
 
    countries    = state.get("countries", [])
    run_id       = state["run_id"]
    date_from    = state.get("date_from")
    date_to      = state.get("date_to")
 
    period = (
        f"{date_from.isoformat()} → {date_to.isoformat()}"
        if date_from and date_to else "unknown period"
    )
 
    human_content = (
        f"Run ID: {run_id}\n"
        f"Countries: {', '.join(countries)}\n"
        f"Period: {period}\n\n"
        "Please collect analytical evidence using all available tools. "
        "Emit all tool calls in a single response."
    )
 
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        HumanMessage(content=human_content),
    ]
 
    response: AIMessage = llm_with_tools.invoke(messages)
    logger.info(
        "analysis_node: LLM emitted %d tool call(s)",
        len(response.tool_calls or []),
    )
 
    return {"messages": [response], "llm_provider": provider}
 
def _process_tool_messages(messages: list[BaseMessage]) -> tuple[dict, list[dict]]:
    """
    Parse ToolMessages produced by ToolNode into structured analysis results.
 
    Returns
    -------
    (analysis_results, rag_topics)
        analysis_results — dict keyed by country with 'patterns' and 'risk' sub-keys.
        rag_topics       — list of topic dicts from rag_context.
    """
    analysis_results: dict[str, Any] = {}
    rag_topics: list[dict] = []
 
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
 
        try:
            content = (
                json.loads(msg.content)
                if isinstance(msg.content, str)
                else msg.content
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("process_evidence: could not parse ToolMessage for %s", msg.name)
            continue
 
        country = content.get("country", "").upper()
 
        if msg.name == "detect_patterns":
            analysis_results.setdefault(country, {})
            analysis_results[country]["patterns"] = content
 
        elif msg.name == "compute_risk_indicators":
            analysis_results.setdefault(country, {})
            analysis_results[country]["risk"] = content
 
        elif msg.name == "rag_context":
            rag_topics = content.get("topics", [])
 
    return analysis_results, rag_topics
 
def process_evidence_node(state: AgentState) -> dict:
    """
    Intermediate node between tool_node and save_analysis_node.
 
    Reads ToolMessages appended by ToolNode, builds analysis_results and
    rag_topics, then clears the message history.
    """
    messages = state.get("messages", [])
    analysis_results, rag_topics = _process_tool_messages(messages)
 
    logger.info(
        "process_evidence_node: %d countries analysed, %d RAG topics",
        len(analysis_results),
        len(rag_topics),
    )
 
    return {
        "analysis_results": analysis_results,
        "rag_topics":       rag_topics,
        "messages":         [],  
    }
 
def save_analysis_node(state: AgentState) -> dict:
    """
    Persist analysis results to PostgreSQL and publish to Redis.
 
    Writes to analysis_runs (row created by the subscriber with status='triggered'):
        UPDATE analysis_runs
        SET charts_json = :charts_json,
            rag_topics_used = :rag_topics_used,
            llm_provider = :llm_provider,
            status = 'analysis_complete'
        WHERE run_id = :run_id
 
    Redis: publishes event 'analysis_complete' regardless of DB outcome so
    downstream agents (Visualization, Narrative) are always notified.
    """
    run_id           = state["run_id"]
    analysis_results = state.get("analysis_results", {})
    rag_topics       = state.get("rag_topics", [])
    llm_provider     = state.get("llm_provider")
    completed_at     = datetime.now(timezone.utc)
 
    error_msg: str | None = None
 
    charts_json = {
        "patterns":              {
            country: data.get("patterns", {})
            for country, data in analysis_results.items()
        },
        "risk_indicators":       {
            country: data.get("risk", {})
            for country, data in analysis_results.items()
        },
        "window_days_analyzed":  30,
        "computed_at":           completed_at.isoformat(),
    }
 
    # --- UPDATE analysis_runs -----------------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE analysis_runs
                    SET charts_json      = :charts_json,
                        rag_topics_used  = :rag_topics_used,
                        llm_provider     = :llm_provider,
                        llm_fallback_used = :llm_fallback_used,
                        completed_at     = :completed_at,
                        status           = 'analysis_complete'
                    WHERE run_id = :run_id
                """),
                {
                    "run_id":           run_id,
                    "charts_json":      json.dumps(charts_json),
                    "rag_topics_used":  json.dumps(rag_topics),
                    "llm_provider":     llm_provider,
                    "llm_fallback_used": llm_provider == "gemini",
                    "completed_at":     completed_at,
                },
            )
            conn.commit()
        logger.info("save_analysis_node: updated analysis_runs for run_id=%s", run_id)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_analysis_node: DB update failed: %s", exc)
 
    # --- Publish analysis_complete to Redis ---------------------------------
    redis_message = json.dumps({
        "run_id":      run_id,
        "event":       "analysis_complete",
        "countries":   list(analysis_results.keys()),
        "n_countries": len(analysis_results),
        "timestamp":   completed_at.isoformat(),
    })
 
    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_analysis_node: published analysis_complete to Redis")
    except Exception as exc:
        logger.warning("save_analysis_node: Redis publish failed: %s", exc)
 
    return {"analysis_error": error_msg}

def risk_node(state: AgentState) -> dict:
    """
    Deterministic risk computation node — no LLM involved.
    """
    run_id           = state["run_id"]
    countries        = state.get("countries", [])
    analysis_results = state.get("analysis_results", {})

    for country in countries:
        risk = _compute_risk_for_country(country, run_id)
        risk["country"] = country.upper()
        risk["run_id"]  = run_id
        analysis_results.setdefault(country, {})
        analysis_results[country]["risk"] = risk
        logger.info("risk_node: computed risk for %s — score=%.2f", country, risk.get("score", 0))

    return {"analysis_results": analysis_results}

def rag_node(state: AgentState) -> dict:
    """
    Deterministic RAG topics node — no LLM involved.
    """
    if not _RAG_API_URL:
        logger.info("rag_node: RAG_API_URL not set — skipping")
        return {"rag_topics": []}

    try:
        response = httpx.get(
            f"{_RAG_API_URL}/rag/topics/active",
            headers={"X-RAG-Key": _RAG_API_KEY},
            timeout=_RAG_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        topics = data if isinstance(data, list) else data.get("topics", [])
        logger.info("rag_node: fetched %d RAG topics", len(topics))
        return {"rag_topics": topics}
    except Exception as exc:
        logger.warning("rag_node: request failed: %s", exc)
        return {"rag_topics": []}
# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
 
def build_analysis_graph():
    """
    Compile and return the Analysis Agent as a LangGraph graph.
 
    Graph:
        START → analysis_node → tool_node → process_evidence_node
              → save_analysis_node → END
    """
    tool_node = ToolNode(_TOOLS)
 
    graph = StateGraph(AgentState)

    graph.add_node("analysis_node",         analysis_node)
    graph.add_node("tool_node",             tool_node)
    graph.add_node("process_evidence_node", process_evidence_node)
    graph.add_node("risk_node",             risk_node)
    graph.add_node("rag_node",              rag_node)
    graph.add_node("save_analysis_node",    save_analysis_node)
 
    graph.add_edge(START,                   "analysis_node")
    graph.add_edge("analysis_node",         "tool_node")
    graph.add_edge("tool_node",             "process_evidence_node")
    graph.add_edge("process_evidence_node", "risk_node")
    graph.add_edge("risk_node",             "rag_node")
    graph.add_edge("rag_node",              "save_analysis_node")
    graph.add_edge("save_analysis_node",    END)
    
    return graph.compile(checkpointer=None)
 
 
def invoke_analysis_graph(state: AgentState) -> AgentState:
    """
    Invoke the Analysis Agent graph.
    """
    graph = build_analysis_graph()
    return graph.invoke(state)