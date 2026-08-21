"""
Climate & Energy Multi-Agent Dashboard.
 
Entry point for the Streamlit app. Configures the page, renders the
sidebar (API URL + health check), and delegates each tab to its own
module under dashboard/tabs/.
 
How to run
----------
From the project root:
    streamlit run dashboard/app.py
 
Tab structure (Master Plan steps 18-19)
---------------------------------------
Tab 1: Pipeline execution panel — on-demand run with SSE progress.
Tab 2: European energy market — generation, load, narrative.
Tab 3: Climate data (Copernicus) — temperature, solar, drought.
Tab 4: Data quality & risk — anomalies table, risk gauge.
Tab 5: Agent observability — run metrics, LangSmith link.
 
Why a single app.py with tab imports and not multi-page Streamlit:
    Multi-page apps in Streamlit create separate URL routes and do not share
    sidebar state without session_state hacks. A single page with st.tabs()
    keeps the API URL input in the sidebar visible across all tabs and avoids
    page reloads when switching between views.
"""
from __future__ import annotations


import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Climate & Energy Agent System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Sidebar — API configuration and health check
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Climate & Energy")
    st.title("Multi-Agent Monitoring Dashboard")
    st.divider()

    api_url = st.text_input(
        "API URL",
        value="https://climate-agents-api-1049167521127.europe-central2.run.app",
        help="Base URL of the FastAPI control server.",
    )

    if st.button("🔍 Health check", use_container_width=True):
        import requests
        try:
            resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
            data = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("PostgreSQL", "✅" if data.get("postgres") else "❌")
            col2.metric("Redis",      "✅" if data.get("redis")    else "❌")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

    st.divider()
    st.caption("System 1: ENTSO-E + Copernicus → LangGraph")
    st.caption("System 2: Analysis → Viz → Narrative")
    st.caption("LLM: Groq (primary) · Gemini Flash (fallback)")  

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
 
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Pipeline",
    "⚡ Energy Market",
    "📊 Climate Data",
    "🔍 Quality & Risk",
    "👀 Observability",
])
 
with tab1:
    from dashboard.tabs.tab1_pipeline import render as render_tab1
    render_tab1(api_url=api_url)
 
with tab2:
    from dashboard.tabs.tab2_energymarked import render as render_tab2
    render_tab2()
 
with tab3:
    from dashboard.tabs.tab3_climatedata import render as render_tab3
    render_tab3()
 
with tab4:
    from dashboard.tabs.tab4_dataquality import render as render_tab4
    render_tab4()
 
with tab5:
    from dashboard.tabs.tab5_agentobserva import render as render_tab5
    render_tab5()