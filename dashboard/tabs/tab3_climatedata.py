"""
Tab 3 — Climate Data.

Displays Copernicus ERA5 climate variables (temperature, solar radiation)
extracted from analysis_runs.viz_json.bar_stats.

Why filter by prefix instead of a hardcoded list:
    New Copernicus variables (e.g. wind speed, humidity) could be added
    to the ingestion pipeline in the future. Filtering by the 'climate_'
    prefix means this tab picks them up automatically without code changes.

Data source: analysis_runs (direct PostgreSQL read via db.py).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.utils.db import get_latest_analysis_run

_CLIMATE_PREFIX = "climate_"

_VAR_LABELS = {
    "climate_temperature_2m":   ("🌡️ 2m Air Temperature", "°C"),
    "climate_solar_radiation":  ("☀️ Solar Radiation Downwards", "W/m²"),
}


def _format_var(var: str) -> tuple[str, str]:
    """Return (display_label, unit) for a climate variable."""
    if var in _VAR_LABELS:
        return _VAR_LABELS[var]
    label = var.replace(_CLIMATE_PREFIX, "").replace("_", " ").title()
    return label, ""


def render() -> None:
    """Render Tab 3 — Climate Data."""
    st.header("🌡️ Climate Data — Copernicus ERA5")

    run = get_latest_analysis_run()

    if run is None:
        st.info("No analysis data available yet. Run the pipeline from Tab 1 first.")
        return

    viz = run.get("viz_json", {})
    bar_stats = viz.get("bar_stats", {})

    if not bar_stats:
        st.warning("No chart data available for this run.")
        return

    countries = list(bar_stats.keys())
    selected_country = countries[0]
    if len(countries) > 1:
        selected_country = st.selectbox("Country", countries)

    country_vars = bar_stats.get(selected_country, {})

    # Filter to climate variables only.
    climate_vars = {
        k: v for k, v in country_vars.items()
        if k.startswith(_CLIMATE_PREFIX)
    }

    if not climate_vars:
        st.info(
            f"No Copernicus climate data available for **{selected_country}** in this run.\n\n"
            "This is expected when the pipeline runs in incremental mode or when the "
            "Copernicus ERA5 download is skipped. Run the pipeline in **full** mode to "
            "include climate data."
        )
        return

    st.caption(
        f"ERA5 reanalysis data for **{selected_country}** — "
        f"spatial mean over the country bounding box, hourly resolution."
    )

    # --- Metric cards per variable -----------------------------------------
    st.subheader("Summary Statistics")

    cols = st.columns(len(climate_vars))
    for col, (var, stats) in zip(cols, climate_vars.items()):
        label, unit = _format_var(var)
        mean_val = stats.get("mean")
        min_val  = stats.get("min")
        max_val  = stats.get("max")

        with col:
            st.markdown(f"**{label}**")
            if mean_val is not None:
                st.metric(
                    label=f"Mean ({unit})" if unit else "Mean",
                    value=f"{mean_val:.2f}",
                    delta=None,
                )
                st.caption(
                    f"Min: {min_val:.2f} {unit}  |  Max: {max_val:.2f} {unit}"
                    if min_val is not None and max_val is not None
                    else "Min/Max not available"
                )
            else:
                st.metric(label="Mean", value="N/A")

    st.divider()

    # --- Detailed stats table ----------------------------------------------
    st.subheader("Detailed Statistics")

    rows = []
    for var, stats in climate_vars.items():
        label, unit = _format_var(var)
        rows.append({
            "Variable":      label,
            "Unit":          unit,
            "Mean":          round(stats.get("mean", 0), 4) if stats.get("mean") is not None else None,
            "Min":           round(stats.get("min", 0), 4)  if stats.get("min")  is not None else None,
            "Max":           round(stats.get("max", 0), 4)  if stats.get("max")  is not None else None,
            "N Records":     stats.get("n", 0),
            "Trend (slope)": round(stats.get("slope", 0), 6) if stats.get("slope") is not None else None,
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Variable"),
        use_container_width=True,
    )

    # --- Time series -------------------------------------------------------
    time_series = viz.get("time_series", {})
    country_ts  = time_series.get(selected_country, {})
    climate_ts  = {k: v for k, v in country_ts.items()
                   if k.startswith(_CLIMATE_PREFIX) and len(v) >= 2}

    if climate_ts:
        st.subheader("Climate Variables Over Time")

        for var, points in climate_ts.items():
            label, unit = _format_var(var)
            df_ts = pd.DataFrame(points).rename(columns={"t": "Time", "v": f"{label} ({unit})"})
            df_ts = df_ts.set_index("Time")
            st.markdown(f"**{label}**")
            st.line_chart(df_ts)

    # --- Interpretation note -----------------------------------------------
    with st.expander("ℹ️ About this data"):
        st.markdown(
            """
            **Source:** Copernicus Climate Data Store — ERA5 reanalysis dataset
            (`reanalysis-era5-single-levels`).

            **Aggregation:** Spatial mean over the country bounding box.
            Each data point represents the average across the grid cells
            covering the country for that hour.

            **Variables:**
            - **2m Air Temperature** — converted from Kelvin to Celsius (K − 273.15).
            - **Solar Radiation Downwards** — converted from J/m² to W/m² (÷ 3600).
            """
        )