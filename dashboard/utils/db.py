"""
PostgreSQL read helpers for the Streamlit dashboard.
 
All functions connect directly to PostgreSQL using the same shared engine
from shared.db. Each function returns plain Python dicts or lists — no
SQLAlchemy ORM objects — so Streamlit can consume them without any
additional serialization step.
"""
from __future__ import annotations
 
import json
import logging
import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
 
from dotenv import load_dotenv
load_dotenv()
 
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def _get_engine():
    """
    Build the SQLAlchemy engine lazily.

    Why lazy and not module-level like shared/db.py:
        shared/db.py builds the engine at import time using os.getenv().
        In Streamlit Cloud, secrets are injected as environment variables
        AFTER the module is first imported, so os.getenv() returns None
        and the engine is built with the default fallback values ('agentes').
        Building the engine inside a function guarantees os.getenv() is
        called at connection time, when secrets are already available.
    """
    database_url = (
        f"postgresql+psycopg://"
        f"{os.getenv('POSTGRES_USER', 'agentes')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'agentes')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'agentes_db')}"
        f"?sslmode={os.getenv('POSTGRES_SSLMODE', 'disable')}"
    )
    return create_engine(database_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
 
def get_latest_quality_run() -> dict | None:
    """
    Return the most recent row from data_quality_runs as a plain dict.
    """
    try:
        with _get_engine().connect() as conn:            row = conn.execute(
                text("""
                    SELECT
                        run_id,
                        started_at,
                        completed_at,
                        n_records,
                        n_anomalies,
                        anomalies,
                        rca_result,
                        run_report,
                        severity,
                        llm_provider,
                        llm_fallback_used,
                        status
                    FROM data_quality_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
            ).fetchone()
 
        if row is None:
            return None
 
        return {
            "run_id":           str(row[0]),
            "started_at":       row[1],
            "completed_at":     row[2],
            "n_records":        row[3],
            "n_anomalies":      row[4],
            "anomalies":        row[5] if isinstance(row[5], list) else (json.loads(row[5]) if row[5] else []),
            "rca_result":       row[6],
            "run_report":       row[7],
            "severity":         row[8],
            "llm_provider":     row[9],
            "llm_fallback_used": row[10],
            "status":           row[11],
        }
 
    except Exception as exc:
        logger.error("get_latest_quality_run failed: %s", exc)
        return None

def get_latest_analysis_run() -> dict | None:
    """
    Return the most recent row from analysis_runs as a plain dict.
 
    viz_json and charts_json are parsed from JSONB into Python dicts.
    rag_topics_used is parsed into a list.
 
    """
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(
                text("""
                    SELECT
                        run_id,
                        started_at,
                        completed_at,
                        triggered_by,
                        narrative,
                        charts_json,
                        vis_json,
                        rag_topics_used,
                        llm_provider,
                        llm_fallback_used,
                        status
                    FROM analysis_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
            ).fetchone()
 
        if row is None:
            return None
 
        def _parse(val):
            if val is None:
                return {}
            if isinstance(val, (dict, list)):
                return val
            return json.loads(val)
 
        return {
            "run_id":           str(row[0]),
            "started_at":       row[1],
            "completed_at":     row[2],
            "triggered_by":     row[3],
            "narrative":        row[4],
            "charts_json":      _parse(row[5]),
            "vis_json":         _parse(row[6]),
            "rag_topics_used":  _parse(row[7]) if isinstance(_parse(row[7]), list) else [],
            "llm_provider":     row[8],
            "llm_fallback_used": row[9],
            "status":           row[10],
        }
 
    except Exception as exc:
        logger.error("get_latest_analysis_run failed: %s", exc)
        return None

def get_recent_runs(n: int = 10) -> list[dict]:
    """
    Return the last N runs joining both tables on run_id.
 
    Used by Tab 5 (observability) to show a historical run table.
    Columns from data_quality_runs take precedence for shared fields
    like llm_provider (the Reporter Agent writes the final provider used).
 
    Returns an empty list if no runs exist or the query fails.
    """
    try:
        with _get_engine().connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        dq.run_id,
                        dq.started_at,
                        dq.completed_at,
                        dq.n_records,
                        dq.n_anomalies,
                        dq.severity,
                        dq.llm_provider,
                        dq.llm_fallback_used,
                        dq.status            AS s1_status,
                        ar.status            AS s2_status
                    FROM data_quality_runs dq
                    LEFT JOIN analysis_runs ar USING (run_id)
                    ORDER BY dq.started_at DESC
                    LIMIT :n
                """),
                {"n": n},
            ).fetchall()
 
        return [
            {
                "run_id":           str(r[0]),
                "started_at":       r[1],
                "completed_at":     r[2],
                "n_records":        r[3],
                "n_anomalies":      r[4],
                "severity":         r[5],
                "llm_provider":     r[6],
                "llm_fallback_used": r[7],
                "s1_status":        r[8],
                "s2_status":        r[9],
            }
            for r in rows
        ]
 
    except Exception as exc:
        logger.error("get_recent_runs failed: %s", exc)
        return []