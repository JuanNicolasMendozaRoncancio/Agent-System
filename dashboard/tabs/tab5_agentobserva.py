"""
Tab 5 — Agent Observability.

Shows a historical table of pipeline runs with key metrics, a detail
view for any selected run, and a link to LangSmith for deep tracing.

Data source: data_quality_runs LEFT JOIN analysis_runs via get_recent_runs().

Why LEFT JOIN and not two separate queries:
    A user looking at run history wants to see both Sistema 1 and Sistema 2
    status in a single row. LEFT JOIN gives us that without two round-trips
    to the DB, and handles the case where Sistema 2 has not yet completed
    for a given run_id.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.utils.db import get_recent_runs

_LANGSMITH_PROJECT_URL = (
    "https://smith.langchain.com/o/agents-climate-energy"
)

_S1_STATUS_ICONS = {
    "complete":    "✅",
    "qa_complete": "🔍",
    "rca_complete":"🔎",
    "running":     "⏳",
}

_S2_STATUS_ICONS = {
    "complete":          "✅",
    "viz_complete":      "📊",
    "analysis_complete": "📈",
    "triggered":         "⏳",
    "running":           "⏳",
}

_SEVERITY_ICONS = {
    "CRITICAL": "🔴",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
}


def _fmt_status(status: str | None, icons: dict) -> str:
    if not status:
        return "—"
    icon = icons.get(status, "⚪")
    return f"{icon} {status}"


def _fmt_elapsed(started_at, completed_at) -> str:
    if started_at is None or completed_at is None:
        return "—"
    delta = completed_at - started_at
    seconds = int(delta.total_seconds())
    return f"{seconds}s"


def render() -> None:
    """Render Tab 5 — Agent Observability."""
    st.header("🤖 Agent Observability")

    # --- LangSmith link ----------------------------------------------------
    st.markdown(
        f"🔗 **Full LLM traces:** [Open in LangSmith]({_LANGSMITH_PROJECT_URL}) "
        f"— token counts, latency per node, provider used, fallback activations."
    )

    st.divider()

    # --- Historical runs table ---------------------------------------------
    st.subheader("Recent Pipeline Runs")

    n_runs = st.slider("Number of runs to show", min_value=5, max_value=50, value=10, step=5)
    runs = get_recent_runs(n=n_runs)

    if not runs:
        st.info("No runs found. Execute the pipeline from Tab 1 to see data here.")
        return

    # Build display DataFrame.
    rows = []
    for r in runs:
        severity = r.get("severity")
        rows.append({
            "Run ID":       str(r["run_id"])[:8] + "…",
            "Started":      r["started_at"].strftime("%Y-%m-%d %H:%M") if r.get("started_at") else "—",
            "Elapsed":      _fmt_elapsed(r.get("started_at"), r.get("completed_at")),
            "Records":      r.get("n_records", 0),
            "Anomalies":    r.get("n_anomalies", 0),
            "Severity":     f"{_SEVERITY_ICONS.get(severity, '✅')} {severity}" if severity else "✅ Clean",
            "LLM":          (r.get("llm_provider") or "—").upper(),
            "Fallback":     "⚠️ Yes" if r.get("llm_fallback_used") else "No",
            "S1 Status":    _fmt_status(r.get("s1_status"), _S1_STATUS_ICONS),
            "S2 Status":    _fmt_status(r.get("s2_status"), _S2_STATUS_ICONS),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Per-run detail selector -------------------------------------------
    st.subheader("Run Detail")

    run_options = {
        f"{r['started_at'].strftime('%Y-%m-%d %H:%M')} — {str(r['run_id'])[:8]}…": r["run_id"]
        for r in runs
        if r.get("started_at")
    }

    if not run_options:
        return

    selected_label = st.selectbox("Select a run to inspect", list(run_options.keys()))
    selected_run_id = run_options[selected_label]
    selected = next((r for r in runs if r["run_id"] == selected_run_id), None)

    if selected is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", selected.get("n_records", 0))
    col2.metric("Anomalies", selected.get("n_anomalies", 0))
    col3.metric("Elapsed", _fmt_elapsed(selected.get("started_at"), selected.get("completed_at")))
    col4.metric("LLM Fallback Used", "Yes ⚠️" if selected.get("llm_fallback_used") else "No")

    st.markdown(f"**Full run ID:** `{selected['run_id']}`")
    st.markdown(
        f"**Sistema 1:** {_fmt_status(selected.get('s1_status'), _S1_STATUS_ICONS)}  |  "
        f"**Sistema 2:** {_fmt_status(selected.get('s2_status'), _S2_STATUS_ICONS)}"
    )

    st.caption(
        "To see the full LLM trace for this run — tool calls, token counts, "
        "provider switches — open LangSmith and filter by run ID above."
    )

    # --- System stats ------------------------------------------------------
    st.divider()
    st.subheader("System Statistics")

    total_records = sum(r.get("n_records", 0) or 0 for r in runs)
    total_anomalies = sum(r.get("n_anomalies", 0) or 0 for r in runs)
    fallback_runs = sum(1 for r in runs if r.get("llm_fallback_used"))
    complete_runs = sum(1 for r in runs if r.get("s1_status") == "complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total_records:,}")
    c2.metric("Total Anomalies", total_anomalies)
    c3.metric("Completed Runs", f"{complete_runs} / {len(runs)}")
    c4.metric("Fallback Activations", fallback_runs)