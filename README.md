# Climate & Energy Multi-Agent System

[![CI](https://github.com/JuanNicolasMendozaRoncancio/agents-climate-energy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuanNicolasMendozaRoncancio/agents-climate-energy/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.14](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Producer-Consumer multi-agent system** for monitoring European energy markets and climate data. System 1 (Producer) ingests ENTSO-E + Copernicus data, runs a full data quality and root cause analysis pipeline, and publishes results to Redis. System 2 (Consumer) picks up the Redis event, runs trend analysis, computes a composite energy risk score, and generates a natural-language market narrative — all streamed live to a Streamlit dashboard via Server-Sent Events.

**Near zero-cost stack in production.** Groq free tier (primary LLM) · Groq gpt-oss-120b (automatic fallback) · PostgreSQL on Neon · Redis on Upstash · FastAPI on Google Cloud Run · Streamlit Community Cloud.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Architecture](#architecture)
  - [High-Level Overview](#high-level-overview)
  - [End-to-End Pipeline Walkthrough](#end-to-end-pipeline-walkthrough)
  - [AgentState: How State Flows Between Agents](#agentstate-how-state-flows-between-agents)
  - [Redis: The Inter-System Bus](#redis-the-inter-system-bus)
  - [PostgreSQL: What Gets Written and When](#postgresql-what-gets-written-and-when)
  - [LLM Client: Groq with Automatic Fallback](#llm-client-groq-with-automatic-fallback)
  - [RAG Integration](#rag-integration)
- [System 1: Data Intelligence Pipeline](#system-1-data-intelligence-pipeline)
- [System 2: Analytics Pipeline](#system-2-analytics-pipeline)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Live Demo

| Component | URL |
|-----------|-----|
| Dashboard (Streamlit) | [Agent System App](https://agent-system-kzkhzpezphegbt5mnrmtgp.streamlit.app) |
| API (FastAPI + SSE) | [Cloud Agent System](https://climate-agents-api-1049167521127.europe-central2.run.app) |
| LangSmith traces | [Langchain](https://smith.langchain.com/o/5ee2c88b-656d-4da6-9835-151cd4fc6cc3/projects?timeModel=%7B"duration"%3A"1d"%7D) |

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Streamlit Tab 1)                       │
│           Selects: countries=[FR], date=2024-06-01, 3h window       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  POST /pipeline/run  (SSE stream opens)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI  (Cloud Run)                           │
│   Spawns System 1 agents sequentially via run_in_executor()         │
│   Streams SSE progress events back: {agent, status, elapsed_s}      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     SYSTEM 1 (Producer)   │
              │  LangGraph · 5 Agents     │
              │  Groq gpt-oss-20b (LLM)   │
              └─────────────┬─────────────┘
                            │ Reporter publishes
                            │ event=system1_complete
                            ▼
              ┌─────────────────────────────┐
              │  Redis Pub/Sub              │
              │  channel: validated_data    │◄── Dead Letter Queue
              │  channel: failed_messages   │    (retry 60/300/900s)
              └─────────────┬───────────────┘
                            │ Subscriber receives
                            │ event=system1_complete
                            ▼
              ┌─────────────────────────┐
              │    SYSTEM 2 (Consumer)  │
              │  LangGraph · 3 Agents   │
              │  Groq gpt-oss-20b (LLM) │
              └─────────────┬───────────┘
                            │ Narrative publishes
                            │ event=narrative_complete
                            ▼
              ┌─────────────────────────┐
              │  FastAPI polls Redis    │
              │  SSE: "Pipeline done"   │
              └─────────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Streamlit Dashboard    │
              │  refreshes 5 tabs       │
              └─────────────────────────┘
```

---

### End-to-End Pipeline Walkthrough

This is the exact sequence of events when a user selects `countries=[FR]`, `date=2024-06-01 00:00→03:00`, `run_type=full` and clicks **Run Complete Pipeline**.

#### Step 1 — API receives the request

```
POST /pipeline/run
{
  "countries": ["FR"],
  "date_from": "2024-06-01T00:00:00",
  "date_to":   "2024-06-01T03:00:00",
  "run_type":  "full"
}
```

FastAPI opens an SSE stream and subscribes to the `validated_data` Redis channel **before** launching any agent. This ensures no events are missed. A `run_id = uuid4()` is generated. The System 2 subscriber daemon thread is started.

---

#### Step 2 — Ingestion Agent (System 1)

**Graph:** `START → ingestion_node → [tool_node → summarize_node]×max2 → save_node → copernicus_node → load_node → END`

The LLM (`gpt-oss-20b` via Groq) receives the run context and decides which tools to call in parallel:

| Tool | Source API | Records produced |
|------|-----------|-----------------|
| `fetch_generation(run_id, "FR", date_from, date_to)` | ENTSO-E | ~30 rows (one per production type per hour) |
| `fetch_load(run_id, "FR", date_from, date_to)` | ENTSO-E | 3 rows (hourly load) |
| `fetch_temperature(run_id, "FR", date_from, date_to)` | Copernicus ERA5 | 3 rows (hourly 2m temp, K→°C) |
| `fetch_solar_radiation(run_id, "FR", date_from, date_to)` | Copernicus ERA5 | 3 rows (hourly SSRD, J/m²→W/m²) |

Tools return **only a summary dict** to the LLM context (e.g. `{"fetched": 48, "source": "entsoe"}`). Raw records go to an in-process `_record_store[run_id]`, never to the LLM. This keeps token consumption flat regardless of how many records are fetched.

A `summarize_node` converts ToolMessages into structured `tool_results` in AgentState and clears the message history, so on a retry cycle the LLM receives a clean summary of what failed.

`copernicus_node` and `load_node` run deterministically after `save_node` to guarantee Copernicus data is always fetched (the LLM occasionally skips it) and `load_actual_aggregated` is always present (required by System 2's risk score C1 component).

**PostgreSQL writes:**
```sql
-- Bulk INSERT (~39 rows for FR, 3h window, full run):
INSERT INTO energy_climate_records
  (run_id, timestamp, source_api, country, variable, value, unit, metadata)
VALUES
  ('abc...', '2024-06-01 00:00+00', 'entsoe',    'FR', 'generation_solar',       1234.5, 'MW',  '{"psr_type_raw": "Solar"}'),
  ('abc...', '2024-06-01 00:00+00', 'entsoe',    'FR', 'load_actual_aggregated', 52000.0,'MW',  null),
  ('abc...', '2024-06-01 00:00+00', 'copernicus','FR', 'climate_temperature_2m', 18.3,   '°C',  '{"dataset":"reanalysis-era5-single-levels","aggregation":"spatial_mean"}'),
  ('abc...', '2024-06-01 00:00+00', 'copernicus','FR', 'climate_solar_radiation', 1.2,   'W/m²','{"dataset":"reanalysis-era5-single-levels","aggregation":"spatial_mean"}'),
  -- ... (one row per variable per hour)

-- Agent execution metadata:
INSERT INTO agent_state
  (run_id, system, agent_name, started_at, completed_at, output_data, elapsed_s)
VALUES
  ('abc...', 'system1', 'IngestionAgent', ..., ..., '{"n_records":39,"cycles_used":1}', 4.2);
```

**Redis publish:**
```json
{
  "run_id": "abc...",
  "n_records": 39,
  "sources": ["entsoe", "copernicus"],
  "countries": ["FR"],
  "timestamp": "2024-06-01T03:05:12Z"
}
```
*(This is an intermediate event — System 2 subscriber ignores it; only `system1_complete` triggers it.)*

**SSE to client:** `{"agent": "Ingestion", "status": "done", "elapsed_s": 4.2, "n_records": 39}`

**AgentState after this step:**
```python
{
  "run_id": "abc...",
  "countries": ["FR"],
  "date_from": datetime(2024,6,1,0, tzinfo=UTC),
  "date_to":   datetime(2024,6,1,3, tzinfo=UTC),
  "run_type": "full",
  "records": [...],          # 39 record dicts, carried forward for QA
  "cycle_count": 1,
  "tool_results": [
    {"tool": "fetch_generation", "country": "FR", "status": "ok", "n_records": 33},
    {"tool": "fetch_load",       "country": "FR", "status": "ok", "n_records": 3},
    {"tool": "fetch_temperature","country": "FR", "status": "ok", "n_records": 3},
    # ...
  ],
  "llm_provider": "groq",
  "ingestion_error": None,
}
```

---

#### Step 3 — Profiling Agent (System 1)

**Graph:** `START → profiling_node → summary_node → save_profile_node → END`

All three nodes are deterministic Python. The LLM is called exactly once in `summary_node` to produce a human-readable dashboard card.

`profiling_node` reads records from `_record_store[run_id]` (in-process, not from DB) and computes for each country:

- **Schema diff** — which of the expected variables (`entsoe:generation_*`, `entsoe:load_actual_aggregated`, `copernicus:climate_temperature_2m`, `copernicus:climate_solar_radiation`) are present or missing.
- **Distribution stats** — mean, std, min, max, p25, p50, p75, n per variable using numpy.
- **Drift detection** — KL divergence between current batch and historical baseline (same day-of-week + hour from `energy_climate_records`), using `scipy.stats.entropy`. Skipped if fewer than 10 historical records exist.

`summary_node` calls `chat_complete()` once with a compact profile dict (~400 tokens input) and receives a 1-2 sentence summary per country.

**PostgreSQL writes:**
```sql
-- One INSERT per run (save_profile_node creates the row that all System 1 agents share):
INSERT INTO data_quality_runs
  (run_id, started_at, source_api, n_records, n_anomalies, anomalies, rca_result, llm_provider, status)
VALUES
  ('abc...', NOW(), 'system1_profiling', 39, 0,
   '{"drift_alerts":[],"missing_variables":[]}',
   'FR: schema complete. No drift detected in solar generation.',
   'groq', 'completed');
```

**AgentState additions:**
```python
{
  "profile": {
    "FR": {
      "n_records": 39,
      "schema":  {"missing": [], "unexpected": []},
      "stats":   {"generation_solar": {"mean": 1234.5, "std": 210.3, "n": 3, ...}, ...},
      "drift":   {"generation_solar": {"kl": 0.04, "drift_detected": False, "skipped": False, ...}, ...}
    }
  },
  "profile_summary": "FR: schema complete. No drift detected in any variable.",
  "llm_provider": "groq",
}
```

---

#### Step 4 — QA Agent (System 1)

**Graph:** `START → qa_node → summary_node → save_qa_node → END`

`qa_node` is pure deterministic Python. It reads `business_rules.yaml` at call time and runs three checks:

1. **Non-negative rule** — flags any `generation_*` or `climate_solar_radiation` record with `value < 0` as CRITICAL.
2. **Completeness check** — computes `received / (window_hours × n_variables)` per (country, source_api) pair. Thresholds: <50% → CRITICAL, <80% → MEDIUM, <95% → LOW.
3. **Anomaly consolidation** — reads drift and missing variables **from `profile` in AgentState** (not recomputed), merges with rule violations into a uniform anomaly list.

**PostgreSQL writes:**
```sql
UPDATE data_quality_runs
SET n_anomalies = 0, anomalies = '[]', rca_result = 'All checks passed.',
    severity = null, status = 'qa_complete'
WHERE run_id = 'abc...';
```

**AgentState additions:**
```python
{
  "anomalies":   [],    # or list of {rule, country, variable, severity, detail}
  "qa_severity": None,  # or "LOW" / "MEDIUM" / "CRITICAL"
  "qa_summary":  "All data quality checks passed. No anomalies detected.",
}
```

---

#### Step 5 — RCA Agent (System 1, conditional)

**Activated only when `qa_severity` is `MEDIUM` or `CRITICAL`.**

**Graph:** `START → rca_node1 → tool_node → _process_rca_evidence → rca_node2 → save_rca_node → END`

`rca_node1` sends the filtered anomaly list to the LLM, which emits parallel tool calls:

| Tool | What it does | Returns to LLM |
|------|-------------|----------------|
| `query_historical_db(variable, country)` | Queries last 30 days of `energy_climate_records`, returns mean/std/n + anomaly count | Stats summary dict |
| `correlate_climate_data(country, date_from, date_to)` | Aggregates Copernicus records by variable for the same window | Per-variable climate stats |
| `rag_search(query)` | `GET /rag/search?query=...` on the RAG API, returns top-2 results by cosine similarity (≥0.60) | `{main_argument, sentiment, score, source}` |

`_process_rca_evidence` processes all ToolMessages in Python, builds `rca_evidence` dict, deduplicates RAG results by `main_argument`, and **clears message history** so `rca_node2` starts fresh.

`rca_node2` receives anomalies + compact evidence summary and produces ranked hypotheses via `chat_complete()`.

**PostgreSQL writes:**
```sql
UPDATE data_quality_runs
SET rca_result = '1. High pressure event reduced wind...\n2. ...', status = 'rca_complete'
WHERE run_id = 'abc...';
```

**AgentState additions:**
```python
{
  "rca_evidence": {
    "historical": {"generation_solar/FR": {"mean": 1100.0, "std": 180.0, "n_records": 720, ...}},
    "climate":    {"FR": {"climate_temperature_2m": {"mean": 19.5, ...}}},
    "rag_results": [{"main_argument": "Anticyclonic conditions...", "score": 0.91, "source": "carbon_brief"}]
  },
  "rca_result":  "1. Persistent anticyclonic conditions...\n2. ...",
  "rca_sources": [{"main_argument": "...", "score": 0.91, "source": "carbon_brief"}],
}
```

---

#### Step 6 — Reporter Agent (System 1)

**Graph:** `START → reporter_node → save_reporter_node → END`

`reporter_node` calls `chat_complete()` once with a compact prompt (~500 tokens) that includes: `profile_summary`, `anomalies`, `rca_result`, drift alerts, total records, and the LLM provider used. Returns a fluent 2-3 paragraph executive report.

**PostgreSQL writes:**
```sql
UPDATE data_quality_runs
SET run_report = 'This run processed 39 records for France...', status = 'complete'
WHERE run_id = 'abc...';
```

**Redis publish — the key event that triggers System 2:**
```json
{
  "run_id":   "abc...",
  "event":    "system1_complete",
  "n_records": 39,
  "countries": ["FR"],
  "run_report": "This run processed 39 records...",
  "timestamp": "2024-06-01T03:05:58Z"
}
```

**SSE to client:** `{"agent": "Reporter", "status": "done", "elapsed_s": 2.1}`

---

#### Step 7 — Redis Subscriber (System 2 trigger)

The System 2 subscriber daemon thread is already blocking on `pubsub.listen()`. It receives the `system1_complete` message, validates required fields (`run_id`, `n_records`, `countries`, `timestamp`), and immediately writes a trigger row:

```sql
INSERT INTO analysis_runs
  (run_id, triggered_by, started_at, status)
VALUES
  ('abc...', 'redis:system1_complete', NOW(), 'triggered')
ON CONFLICT (run_id) DO NOTHING;
```

This row acts as a durable audit trail. If the process crashes after this point, the `triggered` status is visible in Tab 5 (Observability) — the gap between `triggered` and `complete` is immediately apparent.

If the DB write fails, the message goes to the **Dead Letter Queue**:
```json
// published to channel: failed_messages
{
  "original_message": "{\"run_id\":\"abc...\",\"event\":\"system1_complete\",...}",
  "error":            "DB write failed: connection refused",
  "retry_count":      0,
  "failed_at":        "2024-06-01T03:06:00Z"
}
```
The DLQ listener thread retries with exponential backoff: 60s → 300s → 900s (configurable via `DLQ_RETRY_DELAYS`). After 3 failures the message is written to the `dead_messages` table and dropped permanently.

---

#### Step 8 — Analysis Agent (System 2)

**Graph:** `START → analysis_node → tool_node → process_evidence_node → risk_node → rag_node → save_analysis_node → END`

The LLM emits parallel tool calls for all countries:

| Tool | What it does |
|------|-------------|
| `detect_patterns(run_id, "FR", window_days=30)` | Queries `energy_climate_records` for 30d going back from the run's latest timestamp. Computes slope (thirds method), mean, min, max, n per variable. Returns `fallback_used=True` if less data is available than requested. |
| `compute_risk_indicators(run_id, "FR")` | Fetches 1-day records, computes 4-component risk score (0–100). |
| `rag_context(query)` | `GET /rag/topics/active` — retrieves active documentary topics for the Narrative Agent. |

`risk_node` and `rag_node` run deterministically after `process_evidence_node` to guarantee risk computation and RAG topics are always present (the LLM occasionally skips one).

**Risk score components:**

| Component | Weight (with temp) | Weight (no temp) | Formula |
|-----------|-------------------|-----------------|---------|
| C1 — Demand coverage | 30% | 37.5% | `max(0, 1 - total_gen/load) × 100` |
| C2 — Renewable intermittency | 25% | 31.25% | `(wind + solar) / total_gen × 100` |
| C3 — Hydraulic buffer | 25% | 31.25% | `max(0, 1 - hydro_buffer/load × 10) × 100` |
| C4 — Temperature demand | 20% | 0% | 100 if temp < 5°C or > 28°C, else 0 |

**PostgreSQL writes:**
```sql
UPDATE analysis_runs
SET charts_json = '{"patterns": {...}, "risk_indicators": {...}}',
    rag_topics_used = '[{"id":"t1","title":"Wind drought EU",...}]',
    llm_provider = 'groq', status = 'analysis_complete'
WHERE run_id = 'abc...';
```

**AgentState additions:**
```python
{
  "analysis_results": {
    "FR": {
      "patterns": {
        "fallback_used": True,   # only 1 day available vs 30 requested
        "actual_window_days": 1,
        "variables": {
          "generation_solar":       {"slope": 12.5, "mean": 1234.5, "min": 800.0, "max": 1600.0, "n": 3},
          "load_actual_aggregated": {"slope": -5.0, "mean": 52000.0, ...},
        }
      },
      "risk": {
        "score": 42.5,
        "has_temperature_data": True,
        "components": {"demand_coverage": 15.0, "renewable_intermittency": 30.0, ...},
        "weights_used": {"demand_coverage": 0.30, ...},
        "auxiliary": {"total_generation_mw": 48000.0, "load_mw": 52000.0, "temperature_c": 18.3}
      }
    }
  },
  "rag_topics": [{"id": "t1", "title": "European Wind Drought 2024", "relevance": 0.91}],
}
```

---

#### Step 9 — Visualization Agent (System 2)

**Graph:** `START → viz_node → save_viz_node → END`

Pure Python + one SQL aggregation per country.

`viz_node` produces four chart structures:

| Structure | Source | SQL used |
|-----------|--------|---------|
| `time_series` | DB query | `DATE_TRUNC('day', timestamp), AVG(value)` grouped by variable |
| `bar_stats` | AgentState | Direct read from `analysis_results.patterns.variables` |
| `country_comparison` | AgentState | Inverts `{country: {variable: mean}}` to `{variable: {country: mean}}` |
| `risk_breakdown` | AgentState | Extracts C1–C4 scores + weights from `analysis_results.risk` |

**PostgreSQL writes:**
```sql
UPDATE analysis_runs
SET viz_json = '{"time_series":{...},"bar_stats":{...},"country_comparison":{...},"risk_breakdown":{...},"granularity":"day"}',
    status = 'viz_complete'
WHERE run_id = 'abc...';
```

---

#### Step 10 — Narrative Agent (System 2)

**Graph:** `START → narrative_node → save_narrative_node → END`

`narrative_node` first calls `GET /rag/topics/active` directly in Python. Then it calls `chat_complete()` once with a compact prompt (~700 tokens) that includes: risk scores, trend directions, anomaly flags, bar_stats means, and the RAG topics.

The LLM produces a 3-paragraph narrative:
1. Current risk level and drivers
2. Trends and patterns
3. Documentary context (RAG topics)

**PostgreSQL writes:**
```sql
UPDATE analysis_runs
SET narrative = 'France is operating with a moderate supply risk score of 42.5...',
    llm_provider = 'groq', status = 'complete'
WHERE run_id = 'abc...';
```

**Redis publish — the event the FastAPI SSE loop is waiting for:**
```json
{
  "run_id":    "abc...",
  "event":     "narrative_complete",
  "countries": ["FR"],
  "timestamp": "2024-06-01T03:06:45Z"
}
```

**SSE to client:** `{"event": "pipeline_complete", "run_id": "abc...", "total_elapsed_s": 87.3, "llm_provider": "groq"}`

The Streamlit dashboard refreshes all 5 tabs.

---

### AgentState: How State Flows Between Agents

Each agent extends the previous agent's `AgentState` TypedDict via Python inheritance. Every node in the LangGraph graphs returns a dict that is **merged** into the shared state — no full rewrites, only additions.

```
AgentState (Ingestion base)
│  run_id, countries, date_from, date_to, run_type
│  messages, records, ingestion_error, llm_provider
│  tool_results, cycle_count
│
├── + profile, profile_summary, profiling_error          [Profiling]
│
├── + anomalies, qa_severity, qa_error, qa_summary       [QA]
│
├── + rca_evidence, rca_result, rca_sources, rca_error   [RCA]
│
├── + run_report, reporter_error                         [Reporter]
│
├── + analysis_results, rag_topics, analysis_error       [Analysis]
│
├── + viz_data, viz_error                                [Visualization]
│
└── + narrative, narrative_error                         [Narrative]
```

The state dict is passed sequentially through System 1 agents and then initialized fresh for System 2 (the subscriber starts a new state dict with `run_id`, `countries`, and `triggered_by`).

---

### Redis: The Inter-System Bus

**Channel `validated_data`** carries all pipeline events:

| Event | Published by | Consumed by |
|-------|-------------|------------|
| *(no event field)* | Ingestion Agent | Ignored by subscriber |
| `profiling_complete` | Profiling Agent | Ignored by subscriber |
| `qa_complete` | QA Agent | Ignored by subscriber |
| `rca_complete` | RCA Agent | Ignored by subscriber |
| `system1_complete` | Reporter Agent | **Subscriber** — triggers System 2 |
| `analysis_complete` | Analysis Agent | FastAPI SSE polling loop |
| `viz_complete` | Visualization Agent | FastAPI SSE polling loop |
| `narrative_complete` | Narrative Agent | FastAPI SSE polling loop (stops polling) |

**Channel `failed_messages`** (Dead Letter Queue):

```json
{
  "original_message": "{ original JSON string from validated_data }",
  "error":            "DB write failed: connection refused",
  "retry_count":      0,
  "failed_at":        "2024-06-01T03:06:00Z"
}
```

The DLQ listener retries with delays `[60, 300, 900]` seconds (configurable via `DLQ_RETRY_DELAYS`). After 3 failures, the message is written to the `dead_messages` PostgreSQL table.

---

### PostgreSQL

| Table | Written by | Pattern | Key fields |
|-------|-----------|---------|-----------|
| `energy_climate_records` | Ingestion Agent | Bulk INSERT | `run_id, timestamp, source_api, country, variable, value, unit, metadata` |
| `energy_climate_records` | Copernicus node, Load node | Bulk INSERT | Same schema |
| `agent_state` | Ingestion Agent | INSERT | `run_id, system, agent_name, elapsed_s, output_data` |
| `data_quality_runs` | Profiling Agent | **INSERT** (creates the row) | `run_id, n_records, anomalies, rca_result, llm_provider, status` |
| `data_quality_runs` | QA Agent | UPDATE | `n_anomalies, anomalies, severity, qa_summary, status=qa_complete` |
| `data_quality_runs` | RCA Agent | UPDATE | `rca_result, status=rca_complete` |
| `data_quality_runs` | Reporter Agent | UPDATE | `run_report, status=complete` |
| `analysis_runs` | Subscriber | **INSERT** (creates the row) | `run_id, triggered_by, started_at, status=triggered` |
| `analysis_runs` | Analysis Agent | UPDATE | `charts_json, rag_topics_used, status=analysis_complete` |
| `analysis_runs` | Visualization Agent | UPDATE | `viz_json, status=viz_complete` |
| `analysis_runs` | Narrative Agent | UPDATE | `narrative, status=complete` |
| `dead_messages` | DLQ handler | INSERT | `original_message, last_error, retry_count, failed_at` |

**One row per run, enriched progressively.** `data_quality_runs` is created by Profiling and updated by QA, RCA, and Reporter. `analysis_runs` is created by the Subscriber and updated by Analysis, Visualization, and Narrative. This keeps the dashboard query simple: `SELECT ... ORDER BY started_at DESC LIMIT 1`.

---

### LLM Client: Groq with Automatic Fallback

All agents share a single `shared/llm_client.py` module:

```python
chat_complete(messages, system=...) → (text, provider_used)
```

Primary model: `openai/gpt-oss-20b` (Groq)  
Fallback model: `openai/gpt-oss-120b` (Groq)

Fallback triggers on HTTP 429 (rate limit) or any exception with `elapsed >= timeout` (30s). The `provider_used` string (`"groq"` or `"groq_fallback"`) is stored in `llm_provider` column of both `data_quality_runs` and `analysis_runs`.

Override with `LLM_PROVIDER=groq_fallback` in `.env` to force the fallback path directly (useful for testing or when the primary model has a prolonged outage).

---

### RAG Integration

This system is a **consumer** of the [RAG Multi-Agent System](https://github.com/JuanNicolasMendozaRoncancio/RAG-Multi-Agent-System). Two agents make outbound calls:

| Agent | Endpoint | When | What it gets |
|-------|---------|------|-------------|
| RCA Agent | `GET /rag/search?query=<anomaly context>` | When anomalies ≥ MEDIUM exist | Top-2 document excerpts by cosine similarity, filtered at score ≥ 0.60 |
| Analysis Agent | `GET /rag/topics/active` | Every System 2 run | Active topics from the last 7 days |
| Narrative Agent | `GET /rag/topics/active` | Every System 2 run | Same — called again in Python (unconditional) |

Both calls use `X-RAG-Key` header and degrade gracefully: an empty result or a failed request returns `[]` without raising, so the pipeline completes normally with a note in the narrative that no documentary context was available.

---

## System 1: Data Intelligence Pipeline

```
Ingestion Agent   → fetch_generation / fetch_load       (ENTSO-E, entsoe-py)
                  → fetch_temperature / fetch_solar_radiation  (Copernicus CDS, cdsapi)
                  → LLM orchestrates tool calls (gpt-oss-20b, parallel_tool_calls=True)
                  → ReAct loop max 2 cycles, summarize_node clears history each cycle

Profiling Agent   → compute_schema_diff()       (pure Python, no LLM)
                  → compute_distribution_stats() (numpy)
                  → detect_drift()              (scipy KL divergence, same dow+hour baseline)
                  → summary_node               (1 LLM call, ~400 token input)

QA Agent          → validate_business_rules()   (YAML-driven, pure Python)
                  → check_completeness()        (ratio vs window_hours × n_variables)
                  → flag_anomalies()            (reads drift from profile, no recompute)
                  → summary_node               (1 LLM call, only when anomalies exist)

RCA Agent         → query_historical_db()       (30-day historical stats from PostgreSQL)
(MEDIUM/CRITICAL) → correlate_climate_data()   (Copernicus records from PostgreSQL)
                  → rag_search()               (GET /rag/search, top-2 by cosine sim)
                  → rca_node2                  (1 LLM call, ranked hypotheses)

Reporter Agent    → reporter_node              (1 LLM call, ~500 token input)
                  → Publishes system1_complete to Redis
```

---

## System 2: Analytics Pipeline

```
Analysis Agent    → detect_patterns()           (SQL: 30d window, thirds-method slope)
                  → compute_risk_indicators()   (4-component score, C4 adaptive weight)
                  → rag_context()              (GET /rag/topics/active)
                  → risk_node + rag_node       (deterministic, always run)

Visualization     → _fetch_time_series()        (DATE_TRUNC SQL aggregation per country)
Agent             → _build_bar_stats()          (from AgentState, no DB)
                  → _build_country_comparison() (from AgentState, no DB)
                  → _build_risk_breakdown()     (from AgentState, no DB)

Narrative Agent   → _fetch_rag_topics()         (GET /rag/topics/active, Python call)
                  → chat_complete()            (1 LLM call, 3-paragraph narrative)
```

---

## Project Structure

```
├── System1/
│   ├── Ingestion/
│   │   ├── ingestion_agent.py    # LangGraph graph + ReAct loop
│   │   ├── entsoe_client.py      # fetch_generation, fetch_load
│   │   └── copernicus_client.py  # fetch_temperature, fetch_solar_radiation
│   ├── Profiling/
│   │   └── profiling_agent.py    # KL drift detection, schema diff
│   ├── QA/
│   │   ├── qa_agent.py           # business rule validation
│   │   └── business_rules.yaml   # configurable thresholds
│   ├── RCA/
│   │   └── rca_agent.py          # evidence gathering + causal reasoning
│   └── Reporter/
│       └── reporter_agent.py     # executive report generation
├── System2/
│   ├── subscriber.py             # Redis listener + DLQ
│   ├── Analysis/
│   │   └── analysis_agent.py     # patterns, risk score, RAG context
│   ├── Visualization/
│   │   └── visualization_agent.py # chart data structures
│   └── Narrative/
│       └── narrative_agent.py    # market narrative
├── shared/
│   ├── llm_client.py             # Groq primary + fallback, chat_complete()
│   ├── db.py                     # SQLAlchemy engine (PostgreSQL)
│   └── redis_client.py           # Redis client + channel constants
├── api/
│   └── main.py                   # FastAPI SSE endpoints
├── dashboard/
│   ├── app.py                    # Streamlit entry point
│   ├── tabs/                     # tab1-tab5 modules
│   └── utils/
│       ├── db.py                 # lazy engine (Streamlit Cloud secrets)
│       └── see_client.py         # SSE iterator for Tab 1
├── tests/
│   ├── ci/                       # Lightweight connectivity tests (GitHub Actions)
│   │   ├── conftest.py           # overrides root 300s rate-limit pause
│   │   └── test_ci_connectivity.py
│   ├── test_llm_client.py
│   ├── test_ingestion_agent.py
│   └── ... (one test file per agent)
├── infra/
│   ├── init.sql                  # PostgreSQL schema
│   └── verify_connections.py     # local connection check
├── shared/llm_client.py
├── docker-compose.yml            # 6 services, all with healthchecks
├── Dockerfile
└── .github/workflows/ci.yml      # lint + unit tests + connectivity tests
```

---

## Local Setup

**Prerequisites:** Docker, Python 3.11+, a `.env` file with your API keys.

```bash
# 1. Clone
git clone https://github.com/JuanNicolasMendozaRoncancio/agents-climate-energy.git
cd agents-climate-energy

# 2. Configure environment
cp .env.example .env
# Fill in: GROQ_API_KEY, ENTSOE_API_KEY, COPERNICUS_URL, COPERNICUS_API_KEY
# Optional: RAG_API_URL, RAG_API_KEY, LANGSMITH_API_KEY

# 3. Start infrastructure (postgres + redis with healthchecks)
docker compose up postgres redis -d

# 4. Verify connections
python infra/verify_connections.py

# 5. Run the API
uvicorn api.main:app --reload --port 8000

# 6. Run the dashboard (separate terminal)
streamlit run dashboard/app.py

# Or start the full 6-service stack with one command:
docker compose up
```

---

## Running Tests

```bash
# Unit tests only (no API keys needed, CI-safe, ~30s)
pytest tests/test_llm_client.py tests/test_ingestion_agent.py -v -m "not integration"

# CI connectivity tests (requires API keys, ~45s)
pytest tests/ci/ -v -m ci

# Full integration tests (requires Docker + all API keys, ~5min)
pytest tests/ -v -m integration

# End-to-end pipeline test (requires all services running, ~3min)
pytest tests/test_e2e_pipeline.py -v -m integration -s
```

---

## Deployment

| Component | Platform | Notes |
|-----------|---------|-------|
| FastAPI API | Google Cloud Run (`europe-central2`) | Dockerfile in repo root |
| Streamlit Dashboard | Streamlit Community Cloud | Auto-deploys on push to `main` |
| PostgreSQL | Neon (serverless) | Connection via `POSTGRES_HOST` env var |
| Redis | Upstash | TLS, `REDIS_TLS=true` |

All secrets are injected as environment variables. No secrets in the repo.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent orchestration | LangGraph | Conditional edges, persistent state, ToolNode |
| LLM (primary) | Groq `gpt-oss-20b` | Sub-100ms, tool use, 14k req/day free |
| LLM (fallback) | Groq `gpt-oss-120b` | Same GROQ_API_KEY, larger capacity |
| Energy data | ENTSO-E Transparency API + `entsoe-py` | 39 countries, hourly generation + load |
| Climate data | Copernicus CDS + `cdsapi` | ERA5 reanalysis, hourly, 0.25° resolution |
| Data quality | `scipy.stats.entropy` (KL divergence) | Drift detection, seasonality-aware baseline |
| Database | PostgreSQL (Neon) + SQLAlchemy 2.0 | JSONB for profiles, ACID, free tier |
| Pub/Sub | Redis | Decoupled Producer-Consumer, DLQ built-in |
| API | FastAPI + SSE `StreamingResponse` | Non-blocking, real-time progress |
| Dashboard | Streamlit | 5 tabs, `st.status()` SSE rendering |
| Observability | LangSmith | Full traces per run, provider-level visibility |
| CI | GitHub Actions + ruff + pytest | Lint, unit tests, connectivity tests |
| Containers | Docker + docker-compose (6 services) | `docker compose up` starts everything |