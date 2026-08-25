"""
Profiling Agent.

Computes a structured profile of the batch produced by the Ingestion Agent
and generates a natural-language summary for the dashboard.

Responsibilities
----------------
1. Schema diff       — which expected variables are missing or unexpected.
2. Distribution stats — mean, std, min, max, percentiles per variable/country.
3. Drift detection   — KL divergence between current batch and same-window
                       historical baseline from PostgreSQL.
4. LLM summary       — one call to generate 1-2 human-readable lines per
                       country, written to AgentState for the dashboard.
5. Persistence       — writes profile + summary to data_quality_runs and
                       publishes to Redis 'validated_data'.

Graph
-----
START → profiling_node → summary_node → save_profile_node → END
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.stats import entropy as kl_divergence
from sqlalchemy import text

load_dotenv()

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from shared.db import engine
from shared.llm_client import chat_complete
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis

from System1.Ingestion.ingestion_agent import AgentState as _IngestionState, _record_store
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(_IngestionState, total=False):
    """
    Extends the Ingestion AgentState with fields written by the Profiling Agent.
    """
    profile:         dict          # structured profile — output of profiling_node
    profile_summary: str           # LLM summary — output of summary_node
    profiling_error: str | None    # set if save_profile_node fails

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KL_DRIFT_THRESHOLD = float(os.getenv("KL_DRIFT_THRESHOLD", "0.1"))

MIN_HISTORICAL_RECORDS = int(os.getenv("MIN_HISTORICAL_RECORDS", "10"))

N_BINS = int(os.getenv("KL_N_BINS", "20"))

_EXPECTED_VARIABLES: dict[str, dict[str, list[str]]] = {
    "full": {
        "entsoe":     ["load_actual_aggregated"],   # generation_* checked by prefix
        "copernicus": ["climate_temperature_2m", "climate_solar_radiation"],
    },
    "incremental": {
        "entsoe": ["load_actual_aggregated"],
    },
}

_GENERATION_PREFIX = "generation_"


# ---------------------------------------------------------------------------
# Compute functions — called directly from profiling_node, not as @tool
# ---------------------------------------------------------------------------

def compute_schema_diff(
    records: list[dict],
    run_type: str,
    country: str,
) -> dict[str, list[str]]:
    """
    Compare variables present in the batch against expected variables.

    Parameters
    ----------
    records:
        All records for a single country from the current batch.
    run_type:
        'full' or 'incremental' — determines which variables are expected.
    country:
        ISO-2 country code. Used only for logging context.

    Returns
    -------
    dict with keys:
        'missing'    — expected variables not present in the batch.
        'unexpected' — variables present but not in the expected set.
    """
    present_by_source: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        present_by_source[rec["source_api"]].add(rec["variable"])

    expected = _EXPECTED_VARIABLES.get(run_type, _EXPECTED_VARIABLES["full"])
    missing: list[str] = []
    unexpected: list[str] = []

    for source, expected_vars in expected.items():
        present = present_by_source.get(source, set())

        for var in expected_vars:
            if var not in present:
                missing.append(f"{source}:{var}")

        if source == "entsoe":
            has_generation = any(v.startswith(_GENERATION_PREFIX) for v in present)
            if not has_generation:
                missing.append("entsoe:generation_*")

    all_expected_sources = set(expected.keys())
    for source, vars_present in present_by_source.items():
        if source not in all_expected_sources:
            for var in vars_present:
                unexpected.append(f"{source}:{var}")

    logger.debug(
        "schema_diff [%s, %s]: missing=%s unexpected=%s",
        country, run_type, missing, unexpected,
    )
    return {"missing": missing, "unexpected": unexpected}


def compute_distribution_stats(
    records: list[dict],
    country: str,
) -> dict[str, dict[str, float]]:
    """
    Compute descriptive statistics per variable for a single country.

    Parameters
    ----------
    records:
        Records for a single country.
    country:
        ISO-2 country code. Used only for logging context.

    Returns
    -------
    dict keyed by variable name, each value a stats dict:
        {mean, std, min, max, p25, p50, p75, n}
    """
    values_by_variable: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        values_by_variable[rec["variable"]].append(float(rec["value"]))

    stats: dict[str, dict[str, float]] = {}
    for variable, values in values_by_variable.items():
        arr = np.array(values)
        stats[variable] = {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
            "n":    int(len(arr)),
        }

    logger.debug("distribution_stats [%s]: %d variables", country, len(stats))
    return stats


def detect_drift(
    records: list[dict],
    run_id: str,
    variable: str,
    country: str,
) -> dict[str, Any]:
    """
    Detect distributional drift for a single variable/country pair.

    Compares the current batch distribution against a historical baseline
    drawn from the same day-of-week and hour window in PostgreSQL.

    Parameters
    ----------
    records:
        Current batch records for this variable/country.
    run_id:
        Used only for logging context.
    variable:
        Variable name (e.g. 'generation_solar').
    country:
        ISO-2 country code.

    Returns
    -------
    dict:
        kl              — KL divergence value (float). None if insufficient history.
        drift_detected  — True if kl > KL_DRIFT_THRESHOLD.
        n_current       — number of records in current batch.
        n_historical    — number of historical records used as baseline.
        threshold_used  — the threshold value applied (for auditability).
        skipped         — True if drift detection was skipped (insufficient history).
    """
    current_values = [float(r["value"]) for r in records if r["variable"] == variable]

    if not current_values:
        return {
            "kl": None, "drift_detected": False,
            "n_current": 0, "n_historical": 0,
            "threshold_used": KL_DRIFT_THRESHOLD, "skipped": True,
            "skip_reason": "no current records for variable",
        }
    current_timestamps = [
        r["timestamp"] for r in records
        if r["variable"] == variable and r.get("timestamp")
    ]

    historical_values: list[float] = []

    if current_timestamps:

        try:
            sample_ts = current_timestamps[0]
            if isinstance(sample_ts, str):
                sample_ts = datetime.fromisoformat(sample_ts)

            dow = sample_ts.weekday()   # 0=Monday … 6=Sunday
            hour = sample_ts.hour

            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT value
                        FROM energy_climate_records
                        WHERE variable = :variable
                          AND country  = :country
                          AND EXTRACT(DOW  FROM timestamp) = :dow
                          AND EXTRACT(HOUR FROM timestamp) = :hour
                          AND run_id != :run_id
                        LIMIT 2000
                    """),
                    {
                        "variable": variable,
                        "country":  country,
                        "dow":      dow,
                        "hour":     hour,
                        "run_id":   run_id,
                    },
                )
                historical_values = [float(row[0]) for row in result]

        except Exception as exc:
            logger.warning("detect_drift: DB query failed for %s/%s: %s", variable, country, exc)

    if len(historical_values) < MIN_HISTORICAL_RECORDS:
        return {
            "kl": None, "drift_detected": False,
            "n_current": len(current_values), "n_historical": len(historical_values),
            "threshold_used": KL_DRIFT_THRESHOLD, "skipped": True,
            "skip_reason": f"insufficient history ({len(historical_values)} < {MIN_HISTORICAL_RECORDS})",
        }


    combined = np.array(current_values + historical_values)
    bin_edges = np.linspace(combined.min(), combined.max(), N_BINS + 1)

    p_current, _    = np.histogram(current_values,    bins=bin_edges, density=False)
    p_historical, _ = np.histogram(historical_values, bins=bin_edges, density=False)

    eps = 1e-10
    p_current    = (p_current    + eps) / (p_current.sum()    + eps * N_BINS)
    p_historical = (p_historical + eps) / (p_historical.sum() + eps * N_BINS)

    kl = float(kl_divergence(p_current, p_historical))

    logger.debug(
        "detect_drift [%s/%s]: KL=%.4f threshold=%.4f drift=%s",
        variable, country, kl, KL_DRIFT_THRESHOLD, kl > KL_DRIFT_THRESHOLD,
    )

    return {
        "kl":             kl,
        "drift_detected": kl > KL_DRIFT_THRESHOLD,
        "n_current":      len(current_values),
        "n_historical":   len(historical_values),
        "threshold_used": KL_DRIFT_THRESHOLD,
        "skipped":        False,
        "skip_reason":    None,
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def profiling_node(state: AgentState) -> dict:
    """
    Compute schema diff, distribution stats, and drift for all countries.

    Reads records from _record_store[run_id]

    Returns
    -------
    dict with key 'profile': a nested dict structured as:
        {country: {schema, stats, drift}}
    """
    run_id   = state["run_id"]
    run_type = state["run_type"]
    records  = _record_store.get(run_id, [])

    if not records:
        logger.warning("profiling_node: no records found in store for run_id=%s", run_id)
        return {"profile": {}}

    by_country: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_country[rec["country"]].append(rec)

    profile: dict[str, Any] = {}

    for country, country_records in by_country.items():
        logger.info("Profiling country=%s (%d records)", country, len(country_records))

        schema = compute_schema_diff(country_records, run_type, country)

        stats = compute_distribution_stats(country_records, country)

        drift: dict[str, dict] = {}
        for variable in stats:
            drift[variable] = detect_drift(
                country_records, run_id, variable, country
            )

        profile[country] = {
            "schema": schema,
            "stats":  stats,
            "drift":  drift,
            "n_records": len(country_records),
        }

    logger.info("profiling_node: profile built for countries=%s", list(profile.keys()))
    return {"profile": profile}


def summary_node(state: AgentState) -> dict:
    """
    Generate a natural-language summary of the profile using a single LLM call.

    The LLM receives only the aggregated profile dict — no raw records.
    Token cost is proportional to the number of countries × variables, not
    to the number of records.

    Returns
    -------
    dict with key 'profile_summary': a short string for the dashboard.
    """
    profile = state.get("profile", {})

    if not profile:
        return {"profile_summary": "No data available to profile.", "llm_provider": None}

    compact: dict[str, Any] = {}
    for country, data in profile.items():
        drift_alerts = [
            {"variable": v, "kl": d["kl"]}
            for v, d in data["drift"].items()
            if d.get("drift_detected")
        ]
        compact[country] = {
            "n_records":    data["n_records"],
            "missing_vars": data["schema"]["missing"],
            "drift_alerts": drift_alerts,
        }

    system_prompt = (
        "You are a data quality summarizer for an energy and climate monitoring system. "
        "Given a profiling report, write 1-2 concise sentences per country describing "
        "the data quality status. Mention missing variables and drift alerts if present. "
        "Be factual and brief — this text appears in a monitoring dashboard."
    )

    user_message = (
        f"Profiling report:\n{json.dumps(compact, indent=2)}\n\n"
        "Write a 1-2 sentence summary per country."
    )

    try:
        summary_text, provider = chat_complete(
            [{"role": "user", "content": user_message}],
            system=system_prompt,
        )
        logger.info("summary_node: LLM summary generated via %s", provider)
    except Exception as exc:
        logger.error("summary_node: LLM call failed: %s", exc)
        summary_text = "Summary generation failed."
        provider = None

    return {"profile_summary": summary_text, "llm_provider": provider}


def save_profile_node(state: AgentState) -> dict:
    """
    Persist the profile and summary to PostgreSQL and publish to Redis.

    Writes to data_quality_runs:
        run_id, started_at, completed_at, source_api, n_records,
        n_anomalies, anomalies (JSONB), rca_result (summary text),
        llm_provider, status.
    """
    run_id      = state["run_id"]
    profile     = state.get("profile", {})
    summary     = state.get("profile_summary", "")
    llm_provider = state.get("llm_provider")
    started_at  = datetime.now(timezone.utc)

    # Aggregate counts across countries for the DB record
    total_records  = sum(d["n_records"] for d in profile.values())
    drift_alerts   = [
        {"country": country, "variable": v, "kl": d["kl"]}
        for country, data in profile.items()
        for v, d in data["drift"].items()
        if d.get("drift_detected")
    ]
    missing_vars = [
        item
        for data in profile.values()
        for item in data["schema"]["missing"]
    ]
    n_anomalies = len(drift_alerts) + len(missing_vars)

    anomalies_payload = {
        "drift_alerts": drift_alerts,
        "missing_variables": missing_vars,
    }

    error_msg: str | None = None
    completed_at = datetime.now(timezone.utc)

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO data_quality_runs
                        (run_id, started_at, completed_at, source_api,
                         n_records, n_anomalies, anomalies, rca_result,
                         llm_provider, llm_fallback_used, status)
                    VALUES
                        (:run_id, :started_at, :completed_at, :source_api,
                         :n_records, :n_anomalies, :anomalies, :rca_result,
                         :llm_provider, :llm_fallback_used, :status)
                """),
                {
                    "run_id":           run_id,
                    "started_at":       started_at,
                    "completed_at":     completed_at,
                    "source_api":       "system1_profiling",
                    "n_records":        total_records,
                    "n_anomalies":      n_anomalies,
                    "anomalies":        json.dumps(anomalies_payload),
                    "rca_result":       summary,
                    "llm_provider":     llm_provider,
                    "llm_fallback_used": llm_provider == "gemini",
                    "status":           "completed",
                },
            )
            conn.commit()
        logger.info("save_profile_node: wrote data_quality_runs for run_id=%s", run_id)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_profile_node: DB write failed: %s", exc)

    # Publish to Redis so downstream agents know profiling is done
    redis_message = json.dumps({
        "run_id":       run_id,
        "event":        "profiling_complete",
        "n_records":    total_records,
        "n_anomalies":  n_anomalies,
        "countries":    list(profile.keys()),
        "timestamp":    completed_at.isoformat(),
    })

    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_profile_node: published to Redis")
    except Exception as exc:
        logger.warning("save_profile_node: Redis publish failed: %s", exc)

    return {"profiling_error": error_msg}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_profiling_graph():
    """
    Compile and return the Profiling Agent as a LangGraph graph.

    Graph: START → profiling_node → summary_node → save_profile_node → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("profiling_node",    profiling_node)
    graph.add_node("summary_node",      summary_node)
    graph.add_node("save_profile_node", save_profile_node)

    graph.add_edge(START,             "profiling_node")
    graph.add_edge("profiling_node",  "summary_node")
    graph.add_edge("summary_node",    "save_profile_node")
    graph.add_edge("save_profile_node", END)

    return graph.compile(checkpointer=None)


def invoke_profiling_graph(state: AgentState) -> AgentState:
    """
    Invoke the profiling graph.
    """
    graph = build_profiling_graph()
    return graph.invoke(state)