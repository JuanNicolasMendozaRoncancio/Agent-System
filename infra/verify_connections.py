"""
Verify that PostgreSQL and Redis are reachable.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(encoding='latin-1')

from shared.db import check_connection as pg_ok, engine
from shared.redis_client import check_connection as redis_ok
from sqlalchemy import text, inspect

def verify_postgres() -> None:
    print("Checking PostgreSQL...", end=" ", flush=True)
    if not pg_ok():
        print("FAILED")
        sys.exit(1)
    print("OK")

    inspector = inspect(engine)
    expected_tables = {
        "energy_climate_records",
        "data_quality_runs",
        "analysis_runs",
        "agent_state",
    }
    existing = set(inspector.get_table_names())
    missing = expected_tables - existing
    if missing:
        print(f"  WARNING: missing tables: {missing}")
    else:
        print(f"  Tables OK: {sorted(existing)}")

def verify_redis() -> None:
    print("Cheking Redis...", end=" ", flush=True)
    if not redis_ok():
        print("FAILED")
    print("OK")

if __name__=="__main__":
    verify_postgres()
    verify_redis()
    print("\nAll connections healthy")