"""
QA Agent.
 
Reads the profile produced by the Profiling Agent and the raw records from
AgentState, applies configurable business rules, and produces a structured
list of anomalies with severity LOW / MEDIUM / CRITICAL.
 
Responsibilities
----------------
1. Business rule validation  — non-negative variables, configurable via YAML.
2. Completeness check        — ratio of records received vs expected per
                               country / source_api / hour window.
3. Anomaly flagging          — consolidates drift alerts (read from profile,
                               NOT recomputed) and rule violations into a
                               uniform anomaly list with severity.
4. LLM summary               — one call to produce a human-readable dashboard
                               card from the anomaly list.
5. Persistence               — UPDATE data_quality_runs (row written by
                               save_profile_node) with final anomaly counts
                               and severity; publish to Redis 'validated_data'.
 
Graph
-----
START → qa_node → summary_node → save_qa_node → END
 
Design rationale
----------------
No LLM in the compute loop. Business rules are exact arithmetic operations —
Python computes them deterministically and cheaply. Drift is already computed
by the Profiling Agent and stored in the profile; recomputing it here would
violate the principle of not calling a service when the result is already
available in state. The LLM is called exactly once in summary_node to convert
the structured anomaly list into natural language for the dashboard.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text
from typing_extensions import TypedDict

load_dotenv()

from shared.db import engine
from shared.llm_client import chat_complete
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis
from System1.Profiling.profiling_agent import AgentState as _ProfilingState

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).parent / "business_rules.yaml"

_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "CRITICAL": 3}

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
class AgentState(_ProfilingState, total=False):
    """
    Extends ProfilingState with fields written by the QA Agent.
 
    anomalies   : uniform list of anomaly dicts produced by qa_node.
    qa_severity : highest severity found across all anomalies, or None if clean.
    qa_error    : set if save_qa_node fails; does not propagate as an exception.
    qa_summary  : natural-language summary produced by summary_node.
    """
    anomalies: list[dict]
    qa_severity: str | None
    qa_error: str | None
    qa_summary: str


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------
def _load_rules(path: Path = _RULES_PATH) -> dict:
    """
    Load and return the business rules YAML.
 
    Why load at call time and not at module level:
        Module-level loading runs at import time, which means tests that want
        to patch the rules file path would need to patch before the import —
        fragile and order-dependent. Loading inside the function makes the
        seam clean: patch '_load_rules' directly in tests.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------
def validate_business_rules(
    records: list[dict],
    rules: dict,
) -> list[dict]:
    """
    Check each record against the non-negative variable rules.
 
    Why only non-negative rules here and not completeness or drift:
        Completeness requires knowing the expected record count, which depends
        on the time window — that belongs in check_completeness(). Drift is
        already computed in the profile — reading it here avoids recomputation.
        Keeping each function single-responsibility makes testing exact.
 
    Parameters
    ----------
    records : Raw records from AgentState (energy_climate_records schema).
    rules   : Parsed business_rules.yaml dict.
 
    Returns
    -------
    list[dict] of violations, each with keys:
        rule, country, variable, value, severity, detail.
    """
    non_negative = set(rules.get("non_negative_variables", []))
    violations: list[dict] = []

    for rec in records:
        variable = rec.get("variable","")
        value = rec.get("value")
        if variable in non_negative and value is not None and value < 0:
            violations.append({
                "rule":     "non_negative",
                "country":  rec.get("country", "unknown"),
                "variable": variable,
                "value":    value,
                "severity": "CRITICAL",
                "detail":   f"{variable} has negative value {value:.4f} — physically impossible.",    
            }) 

    return violations

def check_completeness(
    records: list[dict],
    countries: list[str],
    date_from: datetime,
    date_to:   datetime,
    rules: dict,
) -> list[dict]:
    """
    Compute completeness per (country, source_api) pair and flag violations.
 
    Expected record count is derived from the time window length in hours.
    Each (country, source_api) pair should produce one record per hour for
    each variable. We count distinct variables per pair and compare against
    the expected hours * n_variables.
 
    Why hours and not a fixed count:
        The time window varies per run (3h incremental vs 24h full). Hardcoding
        an expected count would break for any window size other than the default.
        Deriving from the actual window makes the check window-agnostic.
 
    Parameters
    ----------
    records   : Raw records from AgentState.
    countries : Expected country list from AgentState.
    date_from : Start of the ingestion window.
    date_to   : End of the ingestion window.
    rules     : Parsed business_rules.yaml dict.
 
    Returns
    -------
    list[dict] of completeness violations with severity.
    """
    comp_rules = rules.get("completeness", {})
    critical_threshold = comp_rules.get("min_ratio_critical", 0.50)
    medium_threshold   = comp_rules.get("min_ratio_medium",   0.80)
    low_threshold      = comp_rules.get("min_ratio_low",      0.95)

    window_hours = max(
        1,
        int((date_to - date_from).total_seconds() / 3600),
    )

    counts: dict[tuple[str, str, str], int] = {}
    for rec in records:
        key = (rec.get("country", ""), rec.get("source_api",""), rec.get("variable",""))
        counts[key] = counts.get(key, 0 )+1

    received_by_pair: dict[tuple[str, str], int] = {}
    variables_by_pair: dict[tuple[str, str], set] = {}
    for (country, source_api, variable), n in counts.items():
        pair = (country, source_api)
        received_by_pair[pair] = received_by_pair.get(pair, 0) + n
        variables_by_pair.setdefault(pair, set()).add(variable)

    violations: list[dict] = []
    for country in countries:
        for pair, received in received_by_pair.items():
            if pair[0] != country:
                continue
            source_api = pair[1]
            n_variables = len(variables_by_pair[pair])
            expected    = window_hours * n_variables
            ratio       = received / expected if expected > 0 else 1.0

            if ratio >= low_threshold:
                continue

            if ratio < critical_threshold:
                severity = "CRITICAL"
            elif ratio < medium_threshold:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            violations.append({
                "rule":       "completeness",
                "country":    country,
                "source_api": source_api,
                "severity":   severity,
                "detail": (
                    f"Completeness {ratio:.1%} for {source_api}/{country} "
                    f"({received}/{expected} expected records in {window_hours}h window)."
                ),
                "ratio":    ratio,
                "received": received,
                "expected": expected,
            })
 
    return violations

def flag_anomalies(
    rule_violations: list[dict],
    profile: dict,
    rules: dict,
) -> tuple[list[dict], str | None]:
    """
    Consolidate rule violations, drift alerts, and missing variables into a
    uniform anomaly list and compute the maximum severity.
 
    Why read drift from profile instead of recomputing:
        The Profiling Agent already computed KL divergence for every
        variable/country pair and stored the result in AgentState['profile'].
        Recomputing here would query PostgreSQL again for the same historical
        data — wasted I/O. The QA Agent's job is to interpret the result,
        not reproduce it.
 
    Parameters
    ----------
    rule_violations : Output of validate_business_rules() + check_completeness().
    profile         : state['profile'] from the Profiling Agent.
    rules           : Parsed business_rules.yaml dict.
 
    Returns
    -------
    (anomalies, max_severity)
        anomalies    : uniform list of anomaly dicts.
        max_severity : highest severity found, or None if list is empty.
    """
    drift_severity   = rules.get("drift",             {}).get("severity", "MEDIUM")
    missing_severity = rules.get("missing_variables",  {}).get("severity", "MEDIUM")

    anomalies: list[dict] = list(rule_violations)

    for country, data in profile.items():
        for variable, drift in data.get("drift", {}).items():
            if drift.get("drift_detected"):
                anomalies.append({
                    "rule":     "drift",
                    "country":  country,
                    "variable": variable,
                    "severity": drift_severity,
                    "detail": (
                        f"Distributional drift detected in {variable}/{country}: "
                        f"KL={drift['kl']:.4f} (threshold={drift['threshold_used']:.4f})."
                    ),
                    "kl":       drift["kl"],
                    "n_current":    drift["n_current"],
                    "n_historical": drift["n_historical"],
                })

        for missing in data.get("schema", {}).get("missing", []):
            anomalies.append({
                "rule":     "missing_variable",
                "country":  country,
                "variable": missing,
                "severity": missing_severity,
                "detail":   f"Expected variable not present in batch: {missing} for {country}.",
            })
 
    if not anomalies:
        return anomalies, None
 
    max_severity = max(anomalies, key=lambda a: _SEVERITY_RANK.get(a["severity"], 0))["severity"]
    return anomalies, max_severity


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
 
def qa_node(state: AgentState) -> dict:
    """
    Apply business rules and consolidate anomalies.
 
    Reads from AgentState: records, profile, countries, date_from, date_to.
    Writes to AgentState: anomalies, qa_severity.
 
    No LLM call here — all operations are deterministic Python.
    """
    rules   = _load_rules()
    records = state.get("records", [])
    profile = state.get("profile", {})
 
    # 1. Non-negative rule violations
    rule_violations = validate_business_rules(records, rules)
 
    # 2. Completeness violations
    completeness_violations = check_completeness(
        records,
        state["countries"],
        state["date_from"],
        state["date_to"],
        rules,
    )
 
    # 3. Consolidate with drift + missing from profile
    anomalies, max_severity = flag_anomalies(
        rule_violations + completeness_violations,
        profile,
        rules,
    )
 
    logger.info(
        "qa_node: %d anomalies found, max_severity=%s",
        len(anomalies), max_severity,
    )
 
    return {
        "anomalies":   anomalies,
        "qa_severity": max_severity,
    }
 
 
def summary_node(state: AgentState) -> dict:
    """
    Generate a natural-language summary of the QA results for the dashboard.
 
    Receives only the anomaly list and qa_severity — NOT the raw records.
    Token cost is proportional to the number of anomalies, not the number
    of records. For typical runs (0-10 anomalies) this is ~200-400 tokens.
 
    Returns
    -------
    dict with key 'qa_summary': a short string for the dashboard.
    """
    anomalies   = state.get("anomalies", [])
    qa_severity = state.get("qa_severity")
 
    if not anomalies:
        return {
            "qa_summary":  "All data quality checks passed. No anomalies detected.",
            "llm_provider": None,
        }
 
    # Compact representation — only what the LLM needs to produce the summary.
    compact = {
        "max_severity": qa_severity,
        "n_anomalies":  len(anomalies),
        "anomalies": [
            {"severity": a["severity"], "detail": a["detail"]}
            for a in anomalies
        ],
    }
 
    system_prompt = (
        "You are a data quality analyst for a European energy and climate monitoring system. "
        "Given a QA report, write 2-3 concise sentences summarizing the data quality status. "
        "Mention the most severe issues first. Use plain language suitable for a monitoring "
        "dashboard — avoid technical jargon like 'KL divergence'. Be factual and brief."
    )
 
    user_message = (
        f"QA report:\n{json.dumps(compact, indent=2)}\n\n"
        "Write a 2-3 sentence summary for the dashboard."
    )
 
    try:
        summary_text, provider = chat_complete(
            [{"role": "user", "content": user_message}],
            system=system_prompt,
        )
        logger.info("summary_node: summary generated via %s", provider)
    except Exception as exc:
        logger.error("summary_node: LLM call failed: %s", exc)
        summary_text = f"QA complete: {len(anomalies)} anomalies detected (max severity: {qa_severity})."
        provider     = None
 
    return {"qa_summary": summary_text, "llm_provider": provider}
 
 
def save_qa_node(state: AgentState) -> dict:
    """
    Persist QA results and publish to Redis.
 
    UPDATEs the existing data_quality_runs row (written by save_profile_node)
    with the final anomaly counts, severity, and summary text.
 
    Why UPDATE and not INSERT:
        save_profile_node already created the row for this run_id with
        status='completed'. The QA Agent enriches that same row — anomaly
        counts and RCA text belong together with the profile metadata in a
        single row per run for clean querying.
    """
    run_id      = state["run_id"]
    anomalies   = state.get("anomalies", [])
    qa_severity = state.get("qa_severity")
    qa_summary  = state.get("qa_summary", "")
    llm_provider = state.get("llm_provider")
 
    n_anomalies = len(anomalies)
    error_msg: str | None = None
 
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE data_quality_runs
                    SET
                        n_anomalies       = :n_anomalies,
                        anomalies         = :anomalies,
                        rca_result        = :rca_result,
                        severity          = :severity,
                        llm_provider      = :llm_provider,
                        llm_fallback_used = :llm_fallback_used,
                        status            = 'qa_complete'
                    WHERE run_id = :run_id
                """),
                {
                    "run_id":           run_id,
                    "n_anomalies":      n_anomalies,
                    "anomalies":        json.dumps(anomalies),
                    "rca_result":       qa_summary,
                    "severity":         qa_severity,
                    "llm_provider":     llm_provider,
                    "llm_fallback_used": llm_provider == "gemini",
                },
            )
            conn.commit()
        logger.info("save_qa_node: updated data_quality_runs for run_id=%s", run_id)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_qa_node: DB update failed: %s", exc)
 
    # Publish to Redis regardless of DB outcome
    redis_message = json.dumps({
        "run_id":       run_id,
        "event":        "qa_complete",
        "n_anomalies":  n_anomalies,
        "qa_severity":  qa_severity,
        "countries":    state["countries"],
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    })
 
    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_qa_node: published to Redis")
    except Exception as exc:
        logger.warning("save_qa_node: Redis publish failed: %s", exc)
 
    return {"qa_error": error_msg}

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
 
def build_qa_graph():
    """
    Compile and return the QA Agent as a LangGraph graph.
 
    Graph: START → qa_node → summary_node → save_qa_node → END
 
    No conditional edges — QA always runs all three steps. Even when there
    are no anomalies, summary_node produces a 'clean' message for the dashboard
    and save_qa_node updates the run status.
    """
    graph = StateGraph(AgentState)
 
    graph.add_node("qa_node",      qa_node)
    graph.add_node("summary_node", summary_node)
    graph.add_node("save_qa_node", save_qa_node)
 
    graph.add_edge(START,          "qa_node")
    graph.add_edge("qa_node",      "summary_node")
    graph.add_edge("summary_node", "save_qa_node")
    graph.add_edge("save_qa_node", END)
 
    return graph.compile(checkpointer=None)
 
 
def invoke_qa_graph(state: AgentState) -> AgentState:
    """Invoke the QA graph with a recursion limit appropriate for a linear graph."""
    graph = build_qa_graph()
    return graph.invoke(state)