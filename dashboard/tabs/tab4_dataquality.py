"""
Tab 4 — Data Quality & Risk.

Displays QA results (anomaly table, severity), the Reporter Agent's
executive run report, the RCA Agent's hypotheses, and the energy supply
risk breakdown from the Visualization Agent.

Data sources:
    - data_quality_runs: anomalies, severity, run_report, rca_result
    - analysis_runs.viz_json.risk_breakdown: C1–C4 component scores

Why read from two tables here:
    Quality metadata (QA, RCA, report) lives in data_quality_runs written
    by Sistema 1. Risk indicators live in analysis_runs.viz_json written
    by Sistema 2's Visualization Agent. Both share the same run_id so they
    describe the same pipeline execution.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.utils.db import get_latest_quality_run, get_latest_analysis_run

_SEVERITY_COLORS = {
    "CRITICAL": "🔴",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
}

_COMPONENT_LABELS = {
    "demand_coverage":         "C1 — Demand Coverage",
    "renewable_intermittency": "C2 — Renewable Intermittency",
    "hydraulic_buffer":        "C3 — Hydraulic Buffer",
    "temperature_demand":      "C4 — Temperature Demand",
}


def _severity_badge(severity: str | None) -> str:
    if not severity:
        return "✅ Clean"
    icon = _SEVERITY_COLORS.get(severity, "⚪")
    return f"{icon} {severity}"


def render() -> None:
    """Render Tab 4 — Data Quality & Risk."""
    st.header("🔍 Data Quality & Risk")

    quality = get_latest_quality_run()
    analysis = get_latest_analysis_run()

    if quality is None:
        st.info("No quality data available yet. Run the pipeline from Tab 1 first.")
        return

    # --- Top-level metrics -------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records Ingested", quality.get("n_records", 0))
    col2.metric("Anomalies Found", quality.get("n_anomalies", 0))
    col3.metric("Max Severity", _severity_badge(quality.get("severity")))
    col4.metric("LLM Provider", (quality.get("llm_provider") or "—").upper())

    st.divider()

    # --- Anomaly table -----------------------------------------------------
    st.subheader("QA Anomalies")

    anomalies = quality.get("anomalies", [])

    if not anomalies:
        st.success("✅ No anomalies detected in this run.")
    else:
        rows = []
        for a in anomalies:
            severity = a.get("severity", "")
            rows.append({
                "Severity": f"{_SEVERITY_COLORS.get(severity, '⚪')} {severity}",
                "Rule":     a.get("rule", ""),
                "Country":  a.get("country", ""),
                "Variable": a.get("variable", ""),
                "Detail":   a.get("detail", ""),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- Risk breakdown ----------------------------------------------------
    st.subheader("Energy Supply Risk Breakdown")

    if analysis is None:
        st.info("No risk data available — Sistema 2 has not run yet.")
    else:
        viz = analysis.get("viz_json", {})
        risk_breakdown = viz.get("risk_breakdown", {})

        if not risk_breakdown:
            st.info("Risk breakdown not available for this run.")
        else:
            for country, risk in risk_breakdown.items():
                if "error" in risk:
                    st.warning(f"**{country}**: {risk['error']}")
                    continue

                total = risk.get("total_score", 0)
                has_temp = risk.get("has_temperature_data", False)

                # Color the total score.
                if total >= 60:
                    score_color = "🔴"
                elif total >= 30:
                    score_color = "🟡"
                else:
                    score_color = "🟢"

                st.markdown(
                    f"**{country}** — Total Risk Score: {score_color} **{total:.1f} / 100**"
                    + (" *(temperature component active)*" if has_temp else
                       " *(no Copernicus data — C4 weight redistributed)*")
                )

                components = risk.get("components", {})
                comp_rows = []
                for comp_key, comp_vals in components.items():
                    comp_rows.append({
                        "Component": _COMPONENT_LABELS.get(comp_key, comp_key),
                        "Score":     round(comp_vals.get("score", 0), 2),
                        "Weight":    f"{comp_vals.get('weight', 0)*100:.1f}%",
                        "Contribution": round(
                            comp_vals.get("score", 0) * comp_vals.get("weight", 0), 2
                        ),
                    })

                if comp_rows:
                    df_risk = pd.DataFrame(comp_rows).set_index("Component")
                    st.dataframe(df_risk, use_container_width=True)

    st.divider()

    # --- Executive run report ----------------------------------------------
    run_report = quality.get("run_report")
    if run_report:
        with st.expander("📄 Executive Run Report", expanded=False):
            st.markdown(run_report)

    # --- RCA result --------------------------------------------------------
    rca_result = quality.get("rca_result")
    if rca_result:
        with st.expander("🔎 Root Cause Analysis", expanded=False):
            st.markdown(rca_result)
    else:
        with st.expander("🔎 Root Cause Analysis", expanded=False):
            st.info("RCA was not triggered for this run (no MEDIUM/CRITICAL anomalies requiring investigation).")