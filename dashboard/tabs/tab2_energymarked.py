"""
Tab 2 — Energy Market.
 
Displays the Narrative Agent's summary and energy generation charts built
from analysis_runs.viz_json (bar_stats + time_series).
"""
from __future__ import annotations
 
import streamlit as st
import pandas as pd
 
from dashboard.utils.db import get_latest_analysis_run
 
_CLIMATE_PREFIXES = ("climate_",)
 
_EXCLUDED_VARS = {"load_actual_aggregated"}
 
 
def _is_energy_var(variable: str) -> bool:
    """Return True if the variable belongs in the energy tab."""
    if any(variable.startswith(p) for p in _CLIMATE_PREFIXES):
        return False
    if variable in _EXCLUDED_VARS:
        return False
    return True
 
 
def _format_variable_name(var: str) -> str:
    """Convert 'generation_wind_onshore' → 'Wind Onshore' for display."""
    return var.replace("generation_", "").replace("_", " ").title()
 
 
def render() -> None:
    """Render Tab 2 — Energy Market."""
    st.header("🌍 European Energy Market")
 
    run = get_latest_analysis_run()
 
    if run is None:
        st.info("No analysis data available yet. Run the pipeline from Tab 1 first.")
        return
 
    # --- Narrative ---------------------------------------------------------
    if run.get("narrative"):
        st.subheader("Market Narrative")
        st.markdown(run["narrative"])
    else:
        st.info("Narrative not yet generated for this run.")
 
    st.divider()
 
    # --- Generation bar chart ----------------------------------------------
    viz = run.get("viz_json", {})
    bar_stats = viz.get("bar_stats", {})
 
    if not bar_stats:
        st.warning("No chart data available for this run.")
        return
 
    countries = list(bar_stats.keys())
 
    st.subheader("Generation Mix — Mean Output (MW)")
 
    selected_country = countries[0]
    if len(countries) > 1:
        selected_country = st.selectbox("Country", countries)
 
    country_vars = bar_stats.get(selected_country, {})
 
    energy_vars = {
        _format_variable_name(k): v
        for k, v in country_vars.items()
        if _is_energy_var(k) and v.get("mean", 0) > 0
    }
 
    if not energy_vars:
        st.info(f"No generation data with non-zero output for {selected_country}.")
    else:
        df_bar = pd.DataFrame(
            {"Variable": list(energy_vars.keys()),
             "Mean (MW)": [v["mean"] for v in energy_vars.values()]}
        ).set_index("Variable").sort_values("Mean (MW)", ascending=False)
 
        st.bar_chart(df_bar)
 
    # --- Trend indicators --------------------------------------------------
    st.subheader("Trend Indicators")
 
    trend_data = []
    for var_raw, stats in country_vars.items():
        if not _is_energy_var(var_raw):
            continue
        if stats.get("mean", 0) == 0:
            continue
        slope = stats.get("slope", 0.0)
        trend_data.append({
            "Variable":   _format_variable_name(var_raw),
            "Mean (MW)":  round(stats.get("mean", 0), 1),
            "Min (MW)":   round(stats.get("min", 0), 1),
            "Max (MW)":   round(stats.get("max", 0), 1),
            "Slope":      round(slope, 4),
            "Trend":      "↑ Rising" if slope > 0.001 else ("↓ Falling" if slope < -0.001 else "→ Flat"),
        })
 
    if trend_data:
        st.dataframe(
            pd.DataFrame(trend_data).set_index("Variable"),
            use_container_width=True,
        )
 
    # --- Time series------------------------
    time_series = viz.get("time_series", {})
    country_ts = time_series.get(selected_country, {})
 
    energy_ts = {
        k: v for k, v in country_ts.items()
        if _is_energy_var(k) and len(v) >= 2 and any(p["v"] > 0 for p in v)
    }
 
    if energy_ts:
        st.subheader("Generation Over Time")
 
        all_times = sorted({p["t"] for series in energy_ts.values() for p in series})
        ts_dict = {"Time": all_times}
        for var_raw, points in energy_ts.items():
            point_map = {p["t"]: p["v"] for p in points}
            ts_dict[_format_variable_name(var_raw)] = [
                point_map.get(t, None) for t in all_times
            ]
 
        df_ts = pd.DataFrame(ts_dict).set_index("Time")
        st.line_chart(df_ts)
 
    # --- Run metadata ------------------------------------------------------
    with st.expander("ℹ️ Run metadata"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Run ID", str(run["run_id"])[:8] + "…")
        col2.metric("LLM Provider", run.get("llm_provider", "unknown").upper())
        col3.metric("Status", run.get("status", "unknown"))
        if run.get("started_at"):
            st.caption(f"Run started: {run['started_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")