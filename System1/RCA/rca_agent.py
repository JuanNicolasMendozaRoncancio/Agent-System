"""
RCA Agent (Root Cause Analysis).
 
Activated only when the QA Agent detected anomalies with severity MEDIUM or
CRITICAL. Reasons over those anomalies using three evidence sources:
historical patterns from PostgreSQL, climate correlations from the same DB,
and documentary context retrieved from the RAG system.
 
Graph
-----
START → rca_node1 → rca_node2 → save_rca_node → END
 
Node responsibilities
---------------------
rca_node1:
    LLM with bind_tools — one pass, parallel_tool_calls=True.
    Receives the filtered MEDIUM/CRITICAL anomalies and decides which tools
    to call. ToolNode executes them. Results are processed in Python and
    written to rca_evidence in AgentState. They never enter the message
    history forwarded to rca_node2.
 
rca_node2:
    Plain LLM call — no tools.
    Receives anomalies + rca_evidence (clean dict). Produces ranked
    hypotheses with cited sources. Single LLM call, no loop.
 
save_rca_node:
    No LLM. UPDATE data_quality_runs, publish Redis event: rca_complete.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
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
from shared.llm_client import chat_complete
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis

from System1.QA.qa_agent import AgentState as _QAState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
HISTORICAL_WINDOW_DAYS = int(os.getenv("RCA_HISTORICAL_WINDOW_DAYS", "30"))
 
MAX_HISTORICAL_ROWS = int(os.getenv("RCA_MAX_HISTORICAL_ROWS", "720"))
 
RAG_TOP_K = 3
RAG_KEEP = 2
 
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.60"))
 
_RCA_SEVERITIES = {"MEDIUM", "CRITICAL"}

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------
 
class AgentState(_QAState, total=False):
    """
    Extends the QA AgentState with fields written by the RCA Agent.
 
    rca_evidence:
        Structured dict built in Python after rca_node1 tool execution.
        Keys: 'historical', 'climate', 'rag_results'.
        Never passes through LLM message history — written directly to state.
 
    rca_result:
        Natural-language ranked hypotheses produced by rca_node2.
 
    rca_sources:
        RAG document snippets used, preserved for the dashboard Tab 4.
 
    rca_error:
        Captured exception string from save_rca_node; never raised.
    """
    rca_evidence: dict
    rca_result:   str | None
    rca_sources:  list[dict]
    rca_error:    str | None

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def query_historical_db(variable: str, country: str) -> dict:
    """
    Query PostgreSQL for the last 30 days of records for a variable/country pair.
 
    Returns aggregated statistics only — mean, std, min, max, record count,
    and how many times this variable appeared as an anomaly in recent runs.
    Raw rows are never returned so the LLM context stays small.
 
    Args:
        variable: Variable name as stored in energy_climate_records
                  (e.g. 'generation_solar', 'load_actual_aggregated').
        country:  ISO-2 country code (e.g. 'FR', 'DE').
 
    Returns:
        Dict with keys: variable, country, mean, std, min, max,
        n_records, anomaly_count_last_30d, window_days.
    """
    country = country.upper()
    result: dict[str, Any] = {
        "variable": variable,
        "country":  country,
        "window_days": HISTORICAL_WINDOW_DAYS,
        "n_records": 0,
        "mean": None, "std": None, "min": None, "max": None,
        "anomaly_count_last_30d": 0,
        "error": None,
    }
    try:
        with engine.connect() as conn:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=HISTORICAL_WINDOW_DAYS)
            row = conn.execute(
                text("""
                        SELECT
                            COUNT(*)              AS n,
                            AVG(value)            AS mean,
                            STDDEV_SAMP(value)    AS std,
                            MIN(value)            AS min_val,
                            MAX(value)            AS max_val
                        FROM energy_climate_records
                        WHERE variable  = :variable
                        AND country   = :country
                        AND timestamp >= :cutoff
                    """),
                {"variable": variable, "country": country, "cutoff": cutoff},
            ).fetchone()
    
            if row and row[0]:
                result["n_records"] = int(row[0])
                result["mean"]      = round(float(row[1]), 4) if row[1] is not None else None
                result["std"]       = round(float(row[2]), 4) if row[2] is not None else None
                result["min"]       = round(float(row[3]), 4) if row[3] is not None else None
                result["max"]       = round(float(row[4]), 4) if row[4] is not None else None
    

            anomaly_row = conn.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM data_quality_runs
                        WHERE started_at >= :cutoff
                        AND anomalies::text LIKE :pattern
                    """),
                    {"cutoff": cutoff, "pattern": f"%{variable}%"},
                ).fetchone()
    
            if anomaly_row:
                result["anomaly_count_last_30d"] = int(anomaly_row[0])
    
    except Exception as exc:
            logger.warning("query_historical_db failed for %s/%s: %s", variable, country, exc)
            result["error"] = str(exc)[:200]
    
    return result

@tool
def correlate_climate_data(country: str, date_from: str, date_to: str) -> dict:
    """
    Retrieve climate variables from energy_climate_records for a country and
    time window, returning aggregated statistics per climate variable.
 
    Used to cross-reference energy anomalies (ENTSO-E) with climate conditions
    (Copernicus): e.g. a solar generation drop correlated with low radiation,
    or a demand spike correlated with high temperature.
 
    Args:
        country:   ISO-2 country code.
        date_from: Start of window in ISO 8601 format.
        date_to:   End of window in ISO 8601 format.
 
    Returns:
        Dict keyed by climate variable name, each value a stats dict:
        {mean, std, min, max, n_records}.
    """
    country = country.upper()
    result: dict[str, Any] = {"country": country, "variables": {}, "error": None}
 
    try:
        start = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        end   = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
 
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        variable,
                        COUNT(*)           AS n,
                        AVG(value)         AS mean,
                        STDDEV_SAMP(value) AS std,
                        MIN(value)         AS min_val,
                        MAX(value)         AS max_val
                    FROM energy_climate_records
                    WHERE country   = :country
                      AND source_api = 'copernicus'
                      AND timestamp BETWEEN :start AND :end
                    GROUP BY variable
                """),
                {"country": country, "start": start, "end": end},
            ).fetchall()
 
        for row in rows:
            var = row[0]
            result["variables"][var] = {
                "n_records": int(row[1]),
                "mean": round(float(row[2]), 4) if row[2] is not None else None,
                "std":  round(float(row[3]), 4) if row[3] is not None else None,
                "min":  round(float(row[4]), 4) if row[4] is not None else None,
                "max":  round(float(row[5]), 4) if row[5] is not None else None,
            }
 
    except Exception as exc:
        logger.warning("correlate_climate_data failed for %s: %s", country, exc)
        result["error"] = str(exc)[:200]
 
    return result

@tool
def rag_search(query: str) -> dict:
    """
    Semantic search over the RAG system's CURATED collection.
 
    Calls GET /rag/search on the RAG API. Returns the top 2 results
    by cosine similarity score, filtered to score >= RAG_MIN_SCORE. Each
    result includes only main_argument, sentiment, score, and source.
 
    If RAG_API_URL is not configured or the call fails, returns an empty
    results list without raising.
 
    Args:
        query: Natural-language query derived from the anomaly context
               (e.g. 'wind generation drop France high pressure episode').
 
    Returns:
        Dict with keys: query, results (list of up to 2 dicts), error.
    """
    result: dict[str, Any] = {"query": query, "results": [], "error": None}
 
    rag_url = os.getenv("RAG_API_URL", "").rstrip("/")
    rag_key = os.getenv("RAG_API_KEY", "")
 
    if not rag_url:
        result["error"] = "RAG_API_URL not configured — skipping RAG search"
        logger.info("rag_search: RAG_API_URL not set, returning empty results")
        return result
 
    try:
        response = httpx.get(
            f"{rag_url}/rag/search",
            params={"query": query, "top_k": RAG_TOP_K},
            headers={"X-RAG-Key": rag_key},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
 
        candidates = sorted(
            data.get("results", []),
            key=lambda r: r.get("score", 0.0),
            reverse=True,
        )
        result["results"] = [
            {
                "main_argument": r.get("main_argument", ""),
                "sentiment":     r.get("sentiment", ""),
                "score":         round(r.get("score", 0.0), 4),
                "source":        r.get("source", ""),
            }
            for r in candidates
            if r.get("score", 0.0) >= RAG_MIN_SCORE
        ][:RAG_KEEP]
 
    except Exception as exc:
        logger.warning("rag_search failed for query '%s': %s", query, exc)
        result["error"] = str(exc)[:200]
 
    return result
 
 
_TOOLS = [query_historical_db, correlate_climate_data, rag_search]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _filter_rca_anomalies(anomalies: list[dict]) -> list[dict]:
    """
    Return only MEDIUM and CRITICAL anomalies.
 
    LOW anomalies are informational — they do not justify a full RCA pass.
    Filtering here keeps the LLM context in rca_node1 minimal.
    """
    return [a for a in anomalies if a.get("severity") in _RCA_SEVERITIES]
 
 
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
 
 
def _process_tool_messages(messages: list[BaseMessage]) -> dict:
    """
    Extract ToolMessage results from the message list and build rca_evidence.
 
    Called inside rca_node1 after ToolNode has appended its results.
    Parses each ToolMessage by tool name and organises results into the
    rca_evidence structure. Errors from individual tools are captured but
    do not stop the process — rca_node2 reasons with whatever is available.
    """
    evidence: dict[str, Any] = {
        "historical":  {},   
        "climate":     {},   
        "rag_results": [],
    }
 
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
 
        try:
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except (json.JSONDecodeError, TypeError):
            logger.warning("rca_node1: could not parse ToolMessage content for %s", msg.name)
            continue
 
        if msg.name == "query_historical_db":
            key = f"{content.get('variable', '?')}/{content.get('country', '?')}"
            evidence["historical"][key] = content
 
        elif msg.name == "correlate_climate_data":
            country = content.get("country", "?")
            evidence["climate"][country] = content.get("variables", {})
 
        elif msg.name == "rag_search":

            evidence["rag_results"].extend(content.get("results", []))
 
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in sorted(evidence["rag_results"], key=lambda x: x.get("score", 0.0), reverse=True):
        arg = r.get("main_argument", "")
        if arg not in seen:
            seen.add(arg)
            deduped.append(r)
    evidence["rag_results"] = deduped[:RAG_KEEP * 2]  # keep at most 2×RAG_KEEP total
 
    return evidence

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
 
_NODE1_SYSTEM_PROMPT = """\
You are the evidence-gathering node of a Root Cause Analysis agent for a
Climate & Energy monitoring system.
 
You will receive a list of data quality anomalies (MEDIUM or CRITICAL severity)
detected by the QA Agent. Your job is to decide which evidence to collect by
calling the available tools.
 
Rules:
- Call query_historical_db for each unique (variable, country) pair in the anomalies.
- Call correlate_climate_data for each unique country if energy anomalies are present
  (generation_* or load_* variables). Use the date window from the anomaly context.
- Call rag_search with a concise natural-language query that captures the core
  anomaly pattern (e.g. 'wind generation drop France}').
  One rag_search call per distinct anomaly pattern — do not repeat similar queries.
- Emit all tool calls in a single response (parallel).
- Do not reason about causes yet — that is the job of the next node.
"""
 
_NODE2_SYSTEM_PROMPT = """\
You are the causal reasoning node of a Root Cause Analysis agent for a
Climate & Energy monitoring system.
 
You will receive:
1. A list of MEDIUM/CRITICAL anomalies detected by the QA Agent.
2. Supporting evidence collected from PostgreSQL (historical statistics,
   climate correlations) and from a RAG system (documentary context).
 
Your job is to produce ranked hypotheses explaining the anomalies.
 
Output format (plain text, no JSON):
- 2 to 4 hypotheses, each on a new line, numbered and ranked by probability.
- Each hypothesis: one concise sentence stating the probable cause.
- If RAG evidence is available, cite the source after the hypothesis
  in parentheses: (source: carbon_brief, score: 0.93) (score is the level of support
  to the hypothesis from the souce, score should  be a numer in ]0,1[).
- End with a one-sentence overall assessment of data reliability
  (e.g. 'Data appears consistent with a documented weather event').
 
Be factual and concise — this text appears in a monitoring dashboard.
Do not invent causes that are not supported by the evidence provided.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
 
def rca_node1(state: AgentState) -> dict:
    """
    Evidence-gathering node: LLM decides which tools to call, ToolNode
    executes them, results are processed into rca_evidence.
    """
    anomalies = _filter_rca_anomalies(state.get("anomalies", []))
 
    if not anomalies:
        # No qualifying anomalies — skip tool calls entirely.
        # rca_node2 will produce a "no anomalies to analyse" response.
        logger.info("rca_node1: no MEDIUM/CRITICAL anomalies — skipping tool calls")
        return {"messages": [AIMessage(content="No MEDIUM or CRITICAL anomalies to analyse.")]}
 
    llm_with_tools, provider = _build_llm_with_tools()
 
    anomaly_text = json.dumps(anomalies, indent=2, default=str)
    human_content = (
        f"Run ID: {state['run_id']}\n"
        f"Date window: {state['date_from'].isoformat()} → {state['date_to'].isoformat()}\n\n"
        f"Anomalies requiring root cause analysis:\n{anomaly_text}\n\n"
        "Please collect the relevant evidence using the available tools."
    )
 
    messages = [
        {"role": "system", "content": _NODE1_SYSTEM_PROMPT},
        HumanMessage(content=human_content),
    ]
 
    response: AIMessage = llm_with_tools.invoke(messages)
    logger.info("rca_node1: LLM emitted %d tool call(s)", len(response.tool_calls or []))
 
    return {"messages": [response], "llm_provider": provider}
 
 
def _process_rca_evidence(state: AgentState) -> dict:
    """
    Intermediate node between ToolNode and rca_node2.
 
    Reads ToolMessages appended by ToolNode, builds rca_evidence, and
    clears the message history so rca_node2 starts with a clean slate.
    """
    evidence = _process_tool_messages(state.get("messages", []))
    logger.info(
        "_process_rca_evidence: historical=%d climate=%d rag=%d",
        len(evidence["historical"]),
        len(evidence["climate"]),
        len(evidence["rag_results"]),
    )
    return {
        "rca_evidence": evidence,
        "rca_sources":  evidence["rag_results"],  # preserve for dashboard
        "messages":     [],                        # clear history
    }
 
 
def rca_node2(state: AgentState) -> dict:
    """
    Causal reasoning node: plain LLM call, no tools.
    """
    anomalies = _filter_rca_anomalies(state.get("anomalies", []))
    evidence  = state.get("rca_evidence", {})
 
    if not anomalies:
        return {"rca_result": "No MEDIUM or CRITICAL anomalies detected — RCA not required."}
 
    # Build a compact evidence summary for the prompt.
    evidence_lines: list[str] = []
 
    if evidence.get("historical"):
        evidence_lines.append("Historical statistics (last 30 days):")
        for key, stats in evidence["historical"].items():
            if stats.get("error"):
                evidence_lines.append(f"  {key}: query failed — {stats['error']}")
            else:
                evidence_lines.append(
                    f"  {key}: mean={stats.get('mean')}, std={stats.get('std')}, "
                    f"n={stats.get('n_records')}, "
                    f"recent_anomaly_count={stats.get('anomaly_count_last_30d')}"
                )
 
    if evidence.get("climate"):
        evidence_lines.append("Climate conditions (same window):")
        for country, vars_ in evidence["climate"].items():
            for var, stats in vars_.items():
                evidence_lines.append(
                    f"  {var}/{country}: mean={stats.get('mean')}, "
                    f"std={stats.get('std')}, n={stats.get('n_records')}"
                )
 
    if evidence.get("rag_results"):
        evidence_lines.append("Documentary context (RAG):")
        for r in evidence["rag_results"]:
            evidence_lines.append(
                f"  [{r['source']} | score={r['score']} | {r['sentiment']}] "
                f"{r['main_argument']}"
            )
 
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."
    anomaly_text  = json.dumps(anomalies, indent=2, default=str)
 
    user_content = (
        f"Anomalies:\n{anomaly_text}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Please produce ranked hypotheses explaining these anomalies."
    )
 
    try:
        rca_text, provider = chat_complete(
            [{"role": "user", "content": user_content}],
            system=_NODE2_SYSTEM_PROMPT,
        )
        logger.info("rca_node2: hypotheses generated via %s", provider)
    except Exception as exc:
        logger.error("rca_node2: LLM call failed: %s", exc)
        rca_text = "RCA reasoning failed — evidence collected but hypotheses could not be generated."
        provider = None
 
    return {"rca_result": rca_text, "llm_provider": provider}
 
 
def save_rca_node(state: AgentState) -> dict:
    """
    Persist RCA results to PostgreSQL and publish to Redis.
    """
    run_id      = state["run_id"]
    rca_result  = state.get("rca_result") or ""
    rca_sources = state.get("rca_sources", [])
    completed_at = datetime.now(timezone.utc)
    error_msg: str | None = None
 
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE data_quality_runs
                    SET rca_result   = :rca_result,
                        completed_at = :completed_at,
                        status       = 'rca_complete'
                    WHERE run_id = :run_id
                """),
                {
                    "rca_result":   rca_result,
                    "completed_at": completed_at,
                    "run_id":       run_id,
                },
            )
            conn.commit()
        logger.info("save_rca_node: updated data_quality_runs for run_id=%s", run_id)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("save_rca_node: DB update failed: %s", exc)

    redis_message = json.dumps({
        "run_id":       run_id,
        "event":        "rca_complete",
        "n_hypotheses": len([l for l in rca_result.split("\n") if l.strip()]),
        "n_rag_sources": len(rca_sources),
        "countries":    state.get("countries", []),
        "timestamp":    completed_at.isoformat(),
    })
 
    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("save_rca_node: published rca_complete to Redis")
    except Exception as exc:
        logger.warning("save_rca_node: Redis publish failed: %s", exc)
 
    return {"rca_error": error_msg}


 
# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
 
def build_rca_graph():
    """
    Compile and return the RCA Agent as a LangGraph graph.
 
    Graph structure:
        START
          → rca_node1          (LLM decides tool calls)
          → tool_node          (ToolNode executes them)
          → _process_rca_evidence  (Python processes results → rca_evidence)
          → rca_node2          (LLM reasons, produces hypotheses)
          → save_rca_node      (persist + Redis)
          → END
    """
    tool_node = ToolNode(_TOOLS)
 
    graph = StateGraph(AgentState)
 
    graph.add_node("rca_node1",               rca_node1)
    graph.add_node("tool_node",               tool_node)
    graph.add_node("_process_rca_evidence",   _process_rca_evidence)
    graph.add_node("rca_node2",               rca_node2)
    graph.add_node("save_rca_node",           save_rca_node)
 
    graph.add_edge(START,                    "rca_node1")
    graph.add_edge("rca_node1",              "tool_node")
    graph.add_edge("tool_node",              "_process_rca_evidence")
    graph.add_edge("_process_rca_evidence",  "rca_node2")
    graph.add_edge("rca_node2",              "save_rca_node")
    graph.add_edge("save_rca_node",          END)
 
    return graph.compile(checkpointer=None)
 
 
def invoke_rca_graph(state: AgentState) -> AgentState:
    """
    Invoke the RCA graph.
    """
    graph = build_rca_graph()
    return graph.invoke(state)