"""
Tab 1 — Pipeline Execution Panel.
 
Renders a form to configure and trigger the full Producer-Consumer pipeline
via POST /pipeline/run. Progress is streamed as SSE events and displayed
incrementally using st.status() + st.write(), one line per event.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import streamlit as st

from dashboard.utils.formating import(
    agent_done_line,
    agent_error_line,
    agent_running_line,
    llm_provider_badge,
    pipeline_summary_lines,
)
from dashboard.utils.see_client import iter_pipeline_events

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
 
_DEFAULT_COUNTRIES  = ["FR"]
_DEFAULT_DATE_FROM  = date(2024, 6, 1)
_DEFAULT_DATE_TO    = date(2024, 6, 1)
_DEFAULT_TIME_FROM  = "00:00:00"
_DEFAULT_TIME_TO    = "03:00:00"
_DEFAULT_RUN_TYPE   = "full"
_DEFAULT_API_URL = "https://climate-agents-api-1049167521127.europe-central2.run.app"
 
_AVAILABLE_COUNTRIES = ["FR", "DE", "ES", "BE", "NL", "PT", "IT", "PL", "AT", "CH"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _write_line(container, line: str) -> None:
    """
    Write a single display line into a Streamlit container.
    """
    if "<" in line:
        container.markdown(line, unsafe_allow_html=True)
    else:
        container.write(line)
 
 
def _build_payload(
    countries: list[str],
    date_from: date,
    time_from: str,
    date_to: date,
    time_to: str,
    run_type: str,
) -> dict:
    """Combine date + time inputs into the ISO strings the API expects."""
    dt_from = datetime.combine(date_from, datetime.strptime(time_from, "%H:%M:%S").time())
    dt_to   = datetime.combine(date_to,   datetime.strptime(time_to,   "%H:%M:%S").time())
    return {
        "countries": countries,
        "date_from": dt_from.isoformat(),
        "date_to":   dt_to.isoformat(),
        "run_type":  run_type,
    }

def _run_pipeline(api_url: str, payload: dict) -> None:
    """
    Stream the pipeline SSE events and render progress inside st.status().
    """
    with st.status("Pipeline running...", expanded=True) as status:
        pipeline_failed = False
        final_event: dict = {}
 
        try:
            for event in iter_pipeline_events(api_url, payload):
 
                # --- Agent progress events ----------------------------------
                agent  = event.get("agent")
                estatus = event.get("status")
 
                if agent and estatus == "running":
                    _write_line(status, agent_running_line(agent))
 
                elif agent and estatus == "done":
                    _write_line(status, agent_done_line(agent, event))
 
                elif agent and estatus == "error":
                    _write_line(status, agent_error_line(agent, event))
 
                # --- Terminal events ----------------------------------------
                elif event.get("event") == "pipeline_complete":
                    final_event = event
 
                elif event.get("event") == "pipeline_failed":
                    pipeline_failed = True
                    error_msg = event.get("error", "Unknown error")
                    _write_line(status, f"**Pipeline failed:** {error_msg}")
 
        except Exception as exc:
            pipeline_failed = True
            _write_line(status, f"**Connection error:** {exc}")
 
        # --- Update status expander state -----------------------------------
        if pipeline_failed or not final_event:
            status.update(label="Pipeline failed", state="error", expanded=True)
        else:
            status.update(label="Pipeline complete", state="complete", expanded=False)
 
    # --- Summary card rendered outside the expander ------------------------
    if final_event:
        st.success("Pipeline completed successfully.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total time", f"{final_event.get('total_elapsed_s', 0):.1f} s")
        col2.metric("Run ID", final_event.get("run_id", "—")[:8] + "…")
        provider = final_event.get("llm_provider")
        if provider:
            col3.markdown(
                f"**LLM provider** &nbsp; {llm_provider_badge(provider)}",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------
 
def render(api_url: str = _DEFAULT_API_URL) -> None:
    """
    Render Tab 1: Pipeline Execution Panel.
 
    Parameters
    ----------
    api_url:
        Base URL of the FastAPI server. Passed in from app.py so it can be
        configured via a sidebar input without coupling this tab to global
        Streamlit state.
    """
    st.header("Pipeline Execution")
    st.caption(
        "Triggers the full Producer-Consumer pipeline: "
        "Sistema 1 (Ingestion → Profiling → QA → RCA → Reporter) "
        "→ Redis → Sistema 2 (Analysis → Visualization → Narrative)."
    )
 
    # --- Configuration form -------------------------------------------------
    with st.expander("Run parameters", expanded=True):
        col_l, col_r = st.columns(2)
 
        with col_l:
            countries = st.multiselect(
                "Countries",
                options=_AVAILABLE_COUNTRIES,
                default=_DEFAULT_COUNTRIES,
                help="ISO-2 country codes. At least one required.",
            )
            run_type = st.radio(
                "Run type",
                options=["full", "incremental"],
                index=0,
                horizontal=True,
                help=(
                    "full: fetches generation, load, temperature, solar radiation. "
                    "incremental: ENTSO-E only (no Copernicus)."
                ),
            )
 
        with col_r:
            date_from = st.date_input(
                "Date from",
                value=_DEFAULT_DATE_FROM,
                help="Start of the ingestion window (inclusive).",
            )
            time_from = st.text_input(
                "Time from (HH:MM:SS)",
                value=_DEFAULT_TIME_FROM,
            )
            date_to = st.date_input(
                "Date to",
                value=_DEFAULT_DATE_TO,
                help="End of the ingestion window (exclusive).",
            )
            time_to = st.text_input(
                "Time to (HH:MM:SS)",
                value=_DEFAULT_TIME_TO,
            )
 
    # --- Validation ---------------------------------------------------------
    if not countries:
        st.warning("Select at least one country before running.")
        return
 
    # --- Run button ---------------------------------------------------------
    if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
        payload = _build_payload(countries, date_from, time_from, date_to, time_to, run_type)
        _run_pipeline(api_url, payload)