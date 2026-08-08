"""
Ingestion Agent — Step 6.

LangGraph node that orchestrates ENTSO-E and Copernicus data ingestion
via LLM tool use. The LLM decides which endpoints to call based on the
run context (countries, date range, run type).

Responsibilities
----------------
1. Tool use via LLM   — the LLM selects which fetch tools to invoke.
2. PostgreSQL insert  — bulk-inserts validated records into energy_climate_records
                        and writes execution state to agent_state.
3. Redis publish      — publishes a summary message to the 'validated_data' channel
                        so System 2 knows new data is available.

Public interface
----------------
build_ingestion_graph() -> CompiledGraph
    Returns a compiled LangGraph graph ready to invoke.

AgentState
    TypedDict that defines the shared state for the entire System 1 graph.
    Ingestion writes 'records'; downstream agents (Profiling, QA, ...) read it.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from dotenv import load_dotenv

load_dotenv(encoding="latin-1")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from shared.db import engine
from sqlalchemy import text
from shared.redis_client import CHANNEL_VALIDATED_DATA, get_redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state for the entire System 1 graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state propagated through every node in the System 1 LangGraph.

    Fields written by IngestionAgent
    ---------------------------------
    records         : accumulated data records from all fetch tools.
    ingestion_error : set if the agent fails; downstream nodes must check this.
    llm_provider    : 'groq' or 'gemini' — recorded for observability.
    tool_results    : structured summary of each tool execution (status, n_records, error).
                      Built by summarize_node after each tool_node pass and used to
                      construct a fresh HumanMessage on the next ingestion_node call,
                      replacing full message history accumulation.
    cycle_count     : number of ReAct cycles completed. Capped at MAX_CYCLES to prevent
                      infinite retry loops.

    Fields read by IngestionAgent (set by the caller before invoking the graph)
    ---------------------------------------------------------------------------
    run_id, countries, date_from, date_to, run_type, messages.
    """
    run_id: str
    countries: list[str]
    date_from: datetime
    date_to: datetime
    run_type: str                         
    messages: Annotated[list[BaseMessage], add_messages]
    records: list[dict]
    ingestion_error: str | None
    llm_provider: str | None
    tool_results: list[dict]              
    cycle_count: int                      


# Maximum number of ReAct cycles before forcing save_node.
# Each cycle = 1 LLM call + N tool executions.
# With 2 cycles: first pass fetches all data, second pass retries any failures.
MAX_CYCLES = 2
# ---------------------------------------------------------------------------
# In-process record store
#
# Why not pass records through ToolMessages?
# ToolMessages are fed back to the LLM on every ReAct iteration. A 3-hour
# ENTSO-E window already produces ~200 records × ~100 bytes of JSON = ~20k
# tokens, exceeding Groq's free-tier TPM limit. The LLM is the orchestrator,
# not a data transporter — it only needs a short summary ("48 records fetched")
# to decide whether to call more tools or stop.
#
# Records are stored here keyed by run_id, collected by save_node after the
# loop ends, then cleared to avoid memory leaks between runs.
# ---------------------------------------------------------------------------
_record_store: dict[str, list[dict]] = {}


def _store_records(run_id: str, records: list[dict]) -> None:
    """Append records to the in-process store for this run."""
    if run_id not in _record_store:
        _record_store[run_id] = []
    _record_store[run_id].extend(records)


def _collect_records(run_id: str) -> list[dict]:
    """Return and clear all stored records for this run."""
    return _record_store.pop(run_id, [])


@tool
def fetch_generation(run_id: str, country_code: str, date_from: str, date_to: str) -> dict:
    """
    Fetch electricity generation by source (wind, solar, nuclear, hydro, gas)
    from the ENTSO-E Transparency Platform.

    Args:
        run_id:       Current run identifier — used to store records in memory.
        country_code: ISO 3166-1 alpha-2 country code (e.g. 'FR', 'DE', 'ES').
        date_from:    Start of the period in ISO 8601 format (e.g. '2024-06-01T00:00:00').
        date_to:      End of the period in ISO 8601 format (e.g. '2024-06-02T00:00:00').

    Returns:
        A short summary dict — NOT the full records — so the LLM context stays small.
        Records are stored in memory and collected by save_node after the loop ends.
    """
    client = _get_entsoe_client()
    start = datetime.fromisoformat(date_from)
    end = datetime.fromisoformat(date_to)
    records = client.fetch_generation(country_code.upper(), start, end)
    _store_records(run_id, records)
    return {"fetched": len(records), "source": "entsoe", "dataset": "generation", "country": country_code.upper()}


@tool
def fetch_load(run_id: str, country_code: str, date_from: str, date_to: str) -> dict:
    """
    Fetch electricity demand (load) from the ENTSO-E Transparency Platform.

    Args:
        run_id:       Current run identifier — used to store records in memory.
        country_code: ISO 3166-1 alpha-2 country code (e.g. 'FR', 'DE', 'ES').
        date_from:    Start of the period in ISO 8601 format.
        date_to:      End of the period in ISO 8601 format.

    Returns:
        A short summary dict. Records stored in memory.
    """
    client = _get_entsoe_client()
    start = datetime.fromisoformat(date_from)
    end = datetime.fromisoformat(date_to)
    records = client.fetch_load(country_code.upper(), start, end)
    _store_records(run_id, records)
    return {"fetched": len(records), "source": "entsoe", "dataset": "load", "country": country_code.upper()}


@tool
def fetch_temperature(run_id: str, country_code: str, date_from: str, date_to: str) -> dict:
    """
    Fetch 2-metre air temperature data from Copernicus ERA5 reanalysis.

    Args:
        run_id:       Current run identifier — used to store records in memory.
        country_code: ISO 3166-1 alpha-2 country code (e.g. 'FR', 'DE', 'ES').
        date_from:    Start of the period in ISO 8601 format.
        date_to:      End of the period in ISO 8601 format.

    Returns:
        A short summary dict. Records stored in memory.
    """
    client = _get_copernicus_client()
    start = datetime.fromisoformat(date_from)
    end = datetime.fromisoformat(date_to)
    records = client.fetch_temperature(country_code.upper(), start, end)
    _store_records(run_id, records)
    return {"fetched": len(records), "source": "copernicus", "dataset": "temperature", "country": country_code.upper()}


@tool
def fetch_solar_radiation(run_id: str, country_code: str, date_from: str, date_to: str) -> dict:
    """
    Fetch surface solar radiation downwards from Copernicus ERA5 reanalysis.

    Args:
        run_id:       Current run identifier — used to store records in memory.
        country_code: ISO 3166-1 alpha-2 country code (e.g. 'FR', 'DE', 'ES').
        date_from:    Start of the period in ISO 8601 format.
        date_to:      End of the period in ISO 8601 format.

    Returns:
        A short summary dict. Records stored in memory.
    """
    client = _get_copernicus_client()
    start = datetime.fromisoformat(date_from)
    end = datetime.fromisoformat(date_to)
    records = client.fetch_solar_radiation(country_code.upper(), start, end)
    _store_records(run_id, records)
    return {"fetched": len(records), "source": "copernicus", "dataset": "solar_radiation", "country": country_code.upper()}


_TOOLS = [fetch_generation, fetch_load, fetch_temperature, fetch_solar_radiation]


def _get_entsoe_client():
    """Lazy import to avoid credential errors at module load time."""
    import System1.Ingestion.entsoe_client as _ec
    return _ec


def _get_copernicus_client():
    """Lazy import to avoid credential errors at module load time."""
    import System1.Ingestion.copernicus_client as _cc
    return _cc


def _build_llm_with_tools():
    """
    Return (llm_with_tools, provider_name).

    Uses llama-3.1-8b-instant: sufficient for structured tool-call decisions
    (no open-ended reasoning required) and stays within Groq free-tier TPM.
    parallel_tool_calls=True lets the LLM emit all tool calls for all countries
    in a single AIMessage, so tool_node executes them all before returning.
    """
    from langchain_groq import ChatGroq

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not found in env.")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0,
    )
    return llm.bind_tools(_TOOLS, parallel_tool_calls=True), "groq"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Ingestion Agent of a Climate & Energy multi-agent system.
Your job is to collect electricity and climate data for the requested countries
and time window by calling the available tools.

Rules:
- For a "full" run: call fetch_generation, fetch_load, fetch_temperature,
  and fetch_solar_radiation for EACH country.
- For an "incremental" run: call only fetch_generation and fetch_load
  (climate data changes slowly and does not need hourly updates).
- Always use ISO 8601 format for dates (e.g. '2024-06-01T00:00:00').
- Always pass the run_id from the context as the first argument to every tool call.
- Emit ALL required tool calls in a single response — do not wait for intermediate results.
- On a retry pass: call ONLY the tools listed as failed. Do not re-call successful tools.
- When all required tools are OK, respond with a plain text confirmation and no tool calls.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _build_first_human_message(state: AgentState) -> str:
    """
    Build the initial HumanMessage content for the first cycle.

    Contains the full run context so the LLM knows what to fetch.
    """
    return (
        f"Run ID: {state['run_id']}\n"
        f"Run type: {state['run_type']}\n"
        f"Countries: {', '.join(state['countries'])}\n"
        f"Date from: {state['date_from'].isoformat()}\n"
        f"Date to:   {state['date_to'].isoformat()}\n\n"
        "Please fetch all required data using the available tools. "
        "Emit all tool calls in a single response."
    )


def _build_retry_human_message(state: AgentState) -> str:
    """
    Build the fresh HumanMessage content for retry cycles.

    Constructed entirely from state['tool_results'] — no message history needed.
    This is what keeps token consumption flat across cycles: the LLM receives
    a compact structured summary instead of the full accumulated message history.
    """
    lines = [
        f"Run ID: {state['run_id']} — cycle {state['cycle_count']} summary:\n"
    ]
    for tr in state["tool_results"]:
        status = tr["status"].upper()
        if status == "OK":
            lines.append(
                f"  - {tr['tool']} {tr['country']}: OK — {tr['n_records']} records"
            )
        else:
            lines.append(
                f"  - {tr['tool']} {tr['country']}: FAILED — {tr['error']}"
            )

    failed = [tr for tr in state["tool_results"] if tr["status"] != "ok"]
    if failed:
        lines.append(
            "\nSome tools failed. Please retry ONLY the failed tools listed above."
        )
    else:
        lines.append(
            "\nAll tools completed successfully. Respond with a plain confirmation — no tool calls."
        )

    return "\n".join(lines)


def ingestion_node(state: AgentState) -> dict:
    """
    Primary LangGraph node: sends the run context to the LLM and lets it
    decide which fetch tools to call.

    On cycle 0 (first pass): sends full run context, expects all tool calls at once.
    On cycle > 0 (retry pass): sends a fresh summary from state['tool_results'],
    expects only failed tools to be retried.

    Why a fresh HumanMessage instead of accumulated history?
    The LLM only needs to know the current state of the run — which tools
    succeeded and which failed — not the full sequence of past messages.
    Passing a fresh HumanMessage built from AgentState keeps token consumption
    O(1) per cycle instead of O(n_cycles).
    """
    llm_with_tools, provider = _build_llm_with_tools()

    if state["cycle_count"] == 0:
        human_content = _build_first_human_message(state)
    else:
        human_content = _build_retry_human_message(state)

    # Always send only: system prompt + a single fresh HumanMessage.
    # No history accumulation, AgentState carries the run state instead.
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        HumanMessage(content=human_content),
    ]

    response: AIMessage = llm_with_tools.invoke(messages)

    # Replace messages entirely on each cycle, do not accumulate.
    return {
        "messages": [response],
        "llm_provider": provider,
    }


def summarize_node(state: AgentState) -> dict:
    """
    Intermediate node between tool_node and ingestion_node.

    Responsibilities:
    1. Read the ToolMessages just added by tool_node.
    2. Build a structured tool_results list in AgentState from those messages.
    3. Increment cycle_count.
    4. Clear messages so the next ingestion_node call starts with a clean slate.

    Why a separate node and not logic inside ingestion_node?
    tool_node appends ToolMessages to state['messages'] automatically — we
    cannot intercept that. summarize_node runs after tool_node and before
    ingestion_node, giving us a clean place to process results and reset
    the message history without mixing concerns into ingestion_node.
    """
    new_tool_results: list[dict] = []

    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            # ToolMessage.name holds the tool function name
            new_tool_results.append({
                "tool":      msg.name,
                "country":   content.get("country", "unknown"),
                "status":    "ok",
                "n_records": content.get("fetched", 0),
                "error":     None,
            })
        except Exception as exc:
            # Tool raised an exception, ToolNode serializes it as a string
            new_tool_results.append({
                "tool":      msg.name,
                "country":   "unknown",
                "status":    "error",
                "n_records": 0,
                "error":     str(msg.content)[:200],
            })

    # Merge with existing tool_results
    existing = {(tr["tool"], tr["country"]): tr for tr in state.get("tool_results", [])}
    for tr in new_tool_results:
        existing[(tr["tool"], tr["country"])] = tr

    return {
        "tool_results": list(existing.values()),
        "cycle_count":  state["cycle_count"] + 1,
        "messages":     [],   # clear history
    }


def _should_continue(state: AgentState) -> str:
    """
    Routing function after ingestion_node.

    Sends to tool_node if the LLM made tool calls AND we have not hit the
    cycle cap. Otherwise proceeds to save_node.

    Why cap at MAX_CYCLES and not rely solely on the LLM deciding to stop?
    The LLM could theoretically retry indefinitely if errors persist. A hard
    cap guarantees the graph always terminates and save_node always runs,
    persisting whatever records were successfully fetched.
    """
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)

    if has_tool_calls and state["cycle_count"] < MAX_CYCLES:
        return "tool_node"
    return "save_node"


def save_node(state: AgentState) -> dict:
    """
    Final node: persist records to PostgreSQL and publish summary to Redis.

    Two writes happen here:
    1. Bulk INSERT into energy_climate_records (one row per record).
    2. Single INSERT into agent_state (one row per agent run).

    Why bulk INSERT with executemany instead of the SQLAlchemy ORM?
    The ORM emits one INSERT per object. psycopg v3's executemany() sends
    all rows in a single protocol message — one network roundtrip regardless
    of how many records we have. For 48+ records per run the difference is
    measurable.
    """
    started_at = datetime.now(timezone.utc)
    run_id = state["run_id"]
    records = _collect_records(run_id)

    # --- 1. Bulk insert into energy_climate_records -------------------------
    inserted = 0
    error_msg: str | None = None

    if records:
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO energy_climate_records
                            (run_id, timestamp, source_api, country, variable, value, unit, metadata)
                        VALUES
                            (:run_id, :timestamp, :source_api, :country, :variable, :value, :unit, :metadata)
                    """),
                    [
                        {
                            "run_id":     run_id,
                            "timestamp":  rec.get("timestamp"),
                            "source_api": rec.get("source_api"),
                            "country":    rec.get("country"),
                            "variable":   rec.get("variable"),
                            "value":      rec.get("value"),
                            "unit":       rec.get("unit"),
                            "metadata":   json.dumps(rec.get("metadata")) if rec.get("metadata") else None,
                        }
                        for rec in records
                    ],
                )
                conn.commit()
            inserted = len(records)
            logger.info("Inserted %d records into energy_climate_records", inserted)
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Failed to insert records: %s", exc)

    # --- 2. Write agent_state -----------------------------------------------
    completed_at = datetime.now(timezone.utc)
    elapsed_s = (completed_at - started_at).total_seconds()

    sources = list({r.get("source_api") for r in records if r.get("source_api")})
    output_data = {
        "n_records":   inserted,
        "sources":     sources,
        "countries":   state["countries"],
        "cycles_used": state["cycle_count"],
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO agent_state
                        (run_id, system, agent_name, started_at, completed_at,
                        input_data, output_data, error, elapsed_s)
                    VALUES
                        (:run_id, :system, :agent_name, :started_at, :completed_at,
                        :input_data, :output_data, :error, :elapsed_s)
                """),
                {
                    "run_id":       run_id,
                    "system":       "system1",
                    "agent_name":   "IngestionAgent",
                    "started_at":   started_at,
                    "completed_at": completed_at,
                    "input_data":   json.dumps({"countries": state["countries"], "run_type": state["run_type"]}),
                    "output_data":  json.dumps(output_data),
                    "error":        error_msg,
                    "elapsed_s":    elapsed_s,
                },
            )
            conn.commit()
    except Exception as exc:
        logger.error("Failed to write agent_state: %s", exc)

    # --- 3. Publish summary to Redis ----------------------------------------
    redis_message = json.dumps({
        "run_id":    run_id,
        "n_records": inserted,
        "sources":   sources,
        "countries": state["countries"],
        "timestamp": completed_at.isoformat(),
    })

    try:
        get_redis().publish(CHANNEL_VALIDATED_DATA, redis_message)
        logger.info("Published summary to Redis channel '%s'", CHANNEL_VALIDATED_DATA)
    except Exception as exc:
        logger.warning("Failed to publish to Redis: %s", exc)

    return {
        "records":         records,
        "ingestion_error": error_msg,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_ingestion_graph():
    """
    Compile and return the Ingestion Agent as a LangGraph graph.

    Graph structure:
        START → ingestion_node → tool_node → summarize_node → ingestion_node (loop)
                              ↓ (no tool calls or cycle cap reached)
                           save_node → END

    summarize_node is the key addition over the naive ReAct pattern:
    it processes ToolMessages into structured tool_results, clears the
    message history, and increments cycle_count — keeping token consumption
    flat across cycles.
    """
    tool_node = ToolNode(_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("ingestion_node", ingestion_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("save_node", save_node)

    graph.add_edge(START, "ingestion_node")

    graph.add_conditional_edges(
        "ingestion_node",
        _should_continue,
        {"tool_node": "tool_node", "save_node": "save_node"},
    )

    # After tools execute, always pass through summarize_node before
    graph.add_edge("tool_node", "summarize_node")
    graph.add_edge("summarize_node", "ingestion_node")
    graph.add_edge("save_node", END)

    return graph.compile(checkpointer=None)


def invoke_ingestion_graph(state: AgentState) -> AgentState:
    """
    Invoke the ingestion graph with a hard cap on LangGraph node visits.

    Why recursion_limit=20?
    LangGraph counts node visits, not LLM calls. Each cycle visits 3 nodes:
    ingestion_node → tool_node → summarize_node. With MAX_CYCLES=2 we need
    at most 6 cycle nodes + 2 ingestion_node calls + 1 save_node = ~9 visits.
    20 gives comfortable headroom while preventing runaway loops if something
    unexpected happens in the routing logic.
    """
    graph = build_ingestion_graph()
    config = {"recursion_limit": 20}
    return graph.invoke(state, config=config)