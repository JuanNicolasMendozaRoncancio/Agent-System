"""
Display formatting helpers for the Streamlit dashboard.
"""
from __future__ import annotations

 
_ICON_RUNNING = "⏳"
_ICON_DONE    = "✅"
_ICON_ERROR   = "❌"

 
_SEVERITY_COLOUR = {
    "CRITICAL": "#d62728",
    "MEDIUM":   "#ff7f0e",
    "LOW":      "#2ca02c",
}

_PROVIDER_COLOUR = {
    "groq":   "#6366f1",   
    "gemini": "#0ea5e9", 
}

def agent_running_line(agent: str) -> str:
    return f"{_ICON_RUNNING} **{agent}** — running"

def agent_done_line(agent: str, event: dict) -> str:
    """
    Return a display line for an agent that completed successfully.
    """
    parts: list[str] = []
 
    elapsed = event.get("elapsed_s")
    if elapsed is not None:
        parts.append(f"{elapsed:.1f} s")
 
    n_records = event.get("n_records")
    if n_records is not None:
        parts.append(f"{n_records} records")
 
    n_anomalies = event.get("n_anomalies")
    if n_anomalies is not None:
        parts.append(f"{n_anomalies} anomalies")
 
    severity = event.get("severity")
    if severity:
        colour = _SEVERITY_COLOUR.get(severity, "#888")
        parts.append(f"severity: <span style='color:{colour};font-weight:bold'>{severity}</span>")
 
    n_hypotheses = event.get("n_hypotheses")
    if n_hypotheses is not None:
        parts.append(f"{n_hypotheses} hypotheses")
 
    n_rag = event.get("n_rag_sources")
    if n_rag is not None:
        parts.append(f"{n_rag} RAG sources")
 
    suffix = "  (" + " · ".join(parts) + ")" if parts else ""
    return f"{_ICON_DONE} **{agent}** — done{suffix}"

def agent_error_line(agent: str, event: dict) -> str:
    """
    Return a display line for an agent that failed.
    """
    error = event.get("error", "unknown error")
    elapsed = event.get("elapsed_s")
    elapsed_str = f"{elapsed:.1f} s · " if elapsed is not None else ""
    return f"{_ICON_ERROR} **{agent}** — error  ({elapsed_str}{error})"

def llm_provider_badge(provider: str | None) -> str:
    """
    Return an HTML badge string for the LLM provider.
 
    Rendered with st.markdown(..., unsafe_allow_html=True).
    """
    if not provider:
        return ""
    label = provider.upper()
    colour = _PROVIDER_COLOUR.get(provider.lower(), "#888")
    return (
        f"<span style='"
        f"background-color:{colour};"
        f"color:white;"
        f"padding:2px 10px;"
        f"border-radius:12px;"
        f"font-size:0.8em;"
        f"font-weight:bold;"
        f"'>{label}</span>"
    )

def pipeline_summary_lines(event: dict) -> list[str]:
    """
    Return a list of Markdown lines for the pipeline_complete event.
 
    Parameters
    ----------
    event:
        The pipeline_complete SSE event dict.
 
    Returns
    -------
    list[str]
        Lines intended to be written sequentially with st.write() /
        st.markdown() inside the st.status() expander after it completes.
    """
    total = event.get("total_elapsed_s", 0)
    run_id = event.get("run_id", "—")
    provider = event.get("llm_provider")
 
    lines = [
        f"**Total time:** {total:.1f} s",
        f"**Run ID:** `{run_id}`",
    ]
    if provider:
        lines.append(f"**LLM provider:** {llm_provider_badge(provider)}")
 
    return lines