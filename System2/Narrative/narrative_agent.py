"""
Narrative Agent.
 
Generates a natural-language summary of the European energy market status
by synthesising three information sources already available in AgentState:
  1. analysis_results  — risk scores, patterns, anomaly flags (Analysis Agent).
  2. viz_data          — bar_stats per variable/country (Visualization Agent).
  3. RAG active topics — GET /rag/topics/active (fetched directly in Python,
                         not as an LLM tool — the call is always needed).
 
Why one LLM call and no tools:
    The Narrative Agent does not need to decide *what* to retrieve — the data
    is already in AgentState and the RAG call is unconditional. A tool-based
    ReAct loop would add latency and token overhead with no benefit. A single
    chat_complete() call with a compact prompt keeps input at ~600-900 tokens
    for 2 countries × 5-6 variables, well within Groq's free-tier budget.
 
Why the RAG call is in Python, not a @tool:
    @tool wrappers exist so the LLM can *decide* whether to call something.
    Here the decision is always yes — active topics always enrich the narrative.
    Calling httpx directly in narrative_node is cheaper, simpler, and
    deterministic. Same rationale applied to profiling_node in Sistema 1.
 
Graph:
    START → narrative_node → save_narrative_node → END
 
DB write:
    UPDATE analysis_runs SET narrative = :narrative WHERE run_id = :run_id
    The row is created by the Analysis Agent (save_analysis_node). Every
    downstream Sistema 2 agent only UPDATEs that same row — never INSERTs.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text

load_dotenv()

from shared.db import engine
from shared.llm_client import chat_complete
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis
from System2.Visualization.visualization_agent import AgentState as _VizState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAG_MAX_TOPICS = int(os.getenv("NARRATIVE_RAG_MAX_TOPICS", "3"))
RAG_MIN_TOPIC_SCORE = float(os.getenv("NARRATIVE_RAG_MIN_SCORE", "0.0"))

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
 
class AgentState(_VizState, total=False):
    """
    Extends the Visualization AgentState with fields written by the
    Narrative Agent.
 
    narrative:       Natural-language summary produced by the LLM.
    narrative_error: Captured exception string; never raised.
    """
    narrative:       str
    narrative_error: str | None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
 
_SYSTEM_PROMPT = """\
You are the Narrative Agent of a Climate & Energy monitoring dashboard for
European electricity markets (France and Germany).
 
Your job is to write a concise, fluent narrative summary of the current energy
market status for a technical-but-non-specialist audience (ESG analysts, energy
managers, sustainability teams).
 
Output format — write exactly 3 paragraphs, no headers, no bullet points:
  1. Current status: Overall risk level and the key drivers behind it
     (generation mix, load, climate conditions). Be specific — cite countries,
     variables, and values where they add clarity.
  2. Trends and patterns: What the recent data shows about direction
     (rising/falling demand, improving/worsening renewable output, temperature
     effects). Cite trend directions and anomaly flags if present.
  3. Documentary context: Integrate the RAG active topics naturally into the
     narrative. These are themes from recent energy and climate documents that
     are currently relevant. Mention the source title if it strengthens the
     point. If no RAG topics are available, write a brief general outlook sentence.
 
Tone: factual, concise, professional. Avoid jargon like 'KL divergence' or
'risk index components'. Use plain language: 'renewable supply is under
pressure', 'demand is running above seasonal norms', etc.
Maximum length: 200 words.
"""

# ---------------------------------------------------------------------------
# RAG helper
# ---------------------------------------------------------------------------
 
def _fetch_rag_topics() -> list[dict[str, Any]]:
    """
    Fetch active topics from the RAG API.
 
    Returns up to RAG_MAX_TOPICS topics filtered by RAG_MIN_TOPIC_SCORE.
    Returns an empty list (never raises) when:
      - RAG_API_URL is not configured.
      - The HTTP call fails for any reason.
 
    Why httpx.get with a short timeout and no retry:
        The Narrative Agent is on the critical path of a user-facing dashboard
        request. A slow or unavailable RAG endpoint should degrade gracefully
        (narrative without documentary context) rather than block the pipeline.
        30 s is the same timeout used by the RCA Agent's rag_search tool.
    """
    rag_url = os.getenv("RAG_API_URL", "").rstrip("/")
    rag_key = os.getenv("RAG_API_KEY", "")

    if not rag_url:
        logger.info("_fetch_rag_topics: RAG_API_URL not set — skipping")
        return []

    RAG_MAX_TOPICS = int(os.getenv("NARRATIVE_RAG_MAX_TOPICS", "3"))

    try:
        response = httpx.get(
            f"{rag_url}/rag/topics/active",
            headers={"X-RAG-Key": rag_key},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        topics = data if isinstance(data, list) else data.get("topics", [])

        filtered = [
            t for t in topics
            if t.get("score", 1.0) >= RAG_MIN_TOPIC_SCORE
        ]
        filtered.sort(key= lambda t: t.get("score", 0), reverse=True)

        return filtered[:RAG_MAX_TOPICS]
    
    except Exception as exc:
        logger.warning("_fetch_rag_topics: HTTP call failed: %s", exc)
        return []

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    analysis_results: dict[str, Any],
    viz_data: dict[str, Any],
    rag_topics: list[dict[str, Any]],
) -> str:
    """
    Build a compact prompt from AgentState signals and RAG topics.             
 
    What is deliberately excluded:
      - Full time_series arrays from viz_data (hundreds of data points — the
        LLM cannot use raw arrays; slopes and means from bar_stats suffice).
      - Detailed risk component breakdown (already aggregated into risk_score).
      - Raw records (never enter LLM context in this project).
    """
    lines: list[str] = ["=== ENERGY MARKET SNAPSHOT ===\n"]


    lines.append("Risk assessment:")
    for country, result in analysis_results.items():
        risk_score = result.get("risk_score")
        risk_level = result.get("risk_level", "UNKNOWN")
        fallback = result.get("fallback_used", False)
        score_str = f"{risk_score:.3f}" if risk_score is not None else "N/A"
        fallback_note = " [Copernicus data unavailable — Copernicus fallback weights applied]" if fallback else ""
        lines.append(f"  {country}: {risk_level} (score={score_str}){fallback_note}")

    lines.append("")

    # --- 2. Trend and anomaly signals from patterns -------------------------
    lines.append("Trend signals (7 days window)")
    for country, result in analysis_results.items():
        patterns = result.get("patterns", {})
        trends = patterns.get("trend_7d", {})
        anomalies = patterns.get("anomaly_flags", {})

        if not trends and not anomalies:
            lines.append(f"  {country}: no pattern data available")
            continue

        for variable, trend_info in trends.items():
            direction = trend_info.get("direction", "flat")
            magnitude = trend_info.get("magnitude")
            mag_str   = f", magnitude={magnitude:.3f}" if magnitude is not None else ""
            flag      = " ANOMALY" if anomalies.get(variable) else ""
            lines.append(f"  {country} / {variable}: {direction}{mag_str}{flag}")
 
    lines.append("")

    # --- 3. Bar stats — mean and slope per variable -------------------------
    bar_stats = viz_data.get("bar_stats", {})
    if bar_stats:
        lines.append("Variable averages (current window):")
        for country, variables in bar_stats.items():
            for variable, stats in variables.items():
                mean  = stats.get("mean")
                slope = stats.get("slope")
                unit  = stats.get("unit", "")
                mean_str  = f"{mean:.2f} {unit}".strip() if mean  is not None else "N/A"
                slope_str = f"{slope:+.4f}" if slope is not None else "N/A"
                lines.append(
                    f"  {country} / {variable}: mean={mean_str}, slope={slope_str}"
                )
        lines.append("")

    # --- 4. RAG active topics -----------------------------------------------
    if rag_topics:
        lines.append("Active documentary topics (from RAG knowledge base):")
        for topic in rag_topics:
            title   = topic.get("title", topic.get("topic", "Unknown topic"))
            summary = topic.get("argument_summary", topic.get("summary", ""))
            sentiment = topic.get("sentiment", "")
            sentiment_note = f" [{sentiment}]" if sentiment else ""
            lines.append(f"  - {title}{sentiment_note}: {summary}")
    else:
        lines.append("Active documentary topics: none available.")
 
    lines.append("\n=== END SNAPSHOT ===")
    lines.append(
        "\nWrite a 3-paragraph narrative summary following the format in your instructions."
    )
 
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def narrative_node(state: AgentState) -> dict:
    """
    Fetch RAG topics, build a compact prompt, and call chat_complete() once.
 
    Reads from AgentState:
        analysis_results — risk scores, patterns, anomaly flags.
        viz_data         — bar_stats per variable/country.
 
    The RAG call always happens before the LLM call. It never causes the node
    to raise — a failed RAG call means the narrative proceeds without
    documentary context.
 
    Returns:
        narrative     — 3-paragraph string from the LLM (or Python fallback).
        llm_provider  — 'groq' or 'gemini'.
    """
    analysis_results = state.get("analysis_results", {})
    viz_data         = state.get("viz_data", {})
 
    rag_topics = _fetch_rag_topics()
    logger.info("narrative_node: fetched %d RAG topics", len(rag_topics))
 
    prompt = _build_prompt(analysis_results, viz_data, rag_topics)
    logger.debug("narrative_node: prompt length=%d chars", len(prompt))
 
    try:
        narrative_text, provider = chat_complete(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM_PROMPT,
        )
        logger.info("narrative_node: narrative generated via %s", provider)
    except Exception as exc:
        logger.error("narrative_node: LLM call failed: %s", exc)
        countries = list(analysis_results.keys())
        risk_lines = [
            f"{c}: {analysis_results[c].get('risk_level', 'UNKNOWN')} "
            f"(score={analysis_results[c].get('risk_score', 'N/A')})"
            for c in countries
        ]
        narrative_text = (
            f"Pipeline run completed for {', '.join(countries)}. "
            f"Risk assessment — {'; '.join(risk_lines)}. "
            f"RAG topics retrieved: {len(rag_topics)}. "
            "Narrative generation failed — see logs for details."
        )
        provider = None
 
    return {"narrative": narrative_text, "llm_provider": provider}

def save_narrative_node(state: AgentState) -> dict:
    """
    Persist the narrative to PostgreSQL and publish narrative_complete to Redis.
 
    DB: UPDATE analysis_runs SET narrative = :narrative,
                                 llm_provider = :llm_provider,
                                 status = 'complete'
        WHERE run_id = :run_id
 
    The row was created by save_analysis_node (Analysis Agent INSERT).
    Every downstream Sistema 2 node only UPDATEs — never INSERTs a second row.
 
    Redis: publishes narrative_complete regardless of DB outcome so the
    FastAPI SSE layer is always notified when Sistema 2 finishes.
    """
    run_id       = state["run_id"]
    narrative    = state.get("narrative", "")
    llm_provider = state.get("llm_provider")
    completed_at = datetime.now(timezone.utc)
    error_msg: str | None = None
 
    # --- 1. UPDATE analysis_runs --------------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE analysis_runs
                       SET narrative    = :narrative,
                           llm_provider = :llm_provider,
                           completed_at = :completed_at,
                           status       = 'complete'
                     WHERE run_id = :run_id
                """),
                {
                    "run_id":       run_id,
                    "narrative":    narrative,
                    "llm_provider": llm_provider,
                    "completed_at": completed_at,
                },
            )
            conn.commit()
        logger.info(
            "save_narrative_node: updated analysis_runs for run_id=%s", run_id
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_narrative_node: DB update failed: %s", exc)
 
    # --- 2. Publish narrative_complete to Redis ------------------------------
    redis_message = json.dumps({
        "run_id":    run_id,
        "event":     "narrative_complete",
        "countries": list(state.get("analysis_results", {}).keys()),
        "timestamp": completed_at.isoformat(),
    })
 
    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_narrative_node: published narrative_complete to Redis")
    except Exception as exc:
        logger.warning("save_narrative_node: Redis publish failed: %s", exc)
 
    return {"narrative_error": error_msg}

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
 
def build_narrative_graph():
    """
    Compile and return the Narrative Agent as a LangGraph graph.
 
    Graph: START → narrative_node → save_narrative_node → END
 
    No conditional edges — the graph always runs both nodes. narrative_node
    handles the LLM failure case by returning a Python-built fallback, so
    save_narrative_node always has something to persist.
    """
    graph = StateGraph(AgentState)
 
    graph.add_node("narrative_node",      narrative_node)
    graph.add_node("save_narrative_node", save_narrative_node)
 
    graph.add_edge(START,                "narrative_node")
    graph.add_edge("narrative_node",     "save_narrative_node")
    graph.add_edge("save_narrative_node", END)
 
    return graph.compile(checkpointer=None)
 
 
def invoke_narrative_graph(state: AgentState) -> AgentState:
    """
    Invoke the Narrative Agent graph.
    """
    graph = build_narrative_graph()
    return graph.invoke(state)