CREATE TABLE IF NOT EXISTS energy_climate_records(
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source_api VARCHAR(50) NOT NULL,
    country VARCHAR(20) NOT NULL,
    variable VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    metadata JSONB,
    created_ar TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS data_quality_runs (
    id                   SERIAL PRIMARY KEY,
    run_id               UUID NOT NULL UNIQUE,
    started_at           TIMESTAMPTZ NOT NULL,
    completed_at         TIMESTAMPTZ,
    source_api           VARCHAR(50),
    n_records            INTEGER,
    n_anomalies          INTEGER DEFAULT 0,
    anomalies            JSONB,
    rca_result           TEXT,
    run_report           TEXT,
    severity             VARCHAR(20),   
    llm_provider         VARCHAR(20),   
    llm_fallback_used    BOOLEAN DEFAULT FALSE,
    status               VARCHAR(20) DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL UNIQUE,
    triggered_by    VARCHAR(50),
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    charts_json     JSONB,
    narrative       TEXT,
    rag_topics_used JSONB,
    llm_provider    VARCHAR(20),
    llm_fallback_used BOOLEAN DEFAULT FALSE,
    viz_json            JSONB,
    status          VARCHAR(20) DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS agent_state (
    id          SERIAL PRIMARY KEY,
    run_id      UUID NOT NULL,
    system      VARCHAR(20) NOT NULL,  
    agent_name  VARCHAR(100) NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    input_data  JSONB,
    output_data JSONB,
    error       TEXT,
    elapsed_s   DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_energy_run_id ON energy_climate_records(run_id);
CREATE INDEX IF NOT EXISTS idx_quality_run_id ON data_quality_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_analysis_run_id ON analysis_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_state_run_id ON agent_state(run_id);