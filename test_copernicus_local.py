# test_copernicus_local.py — run from project root
import time
from dotenv import load_dotenv
load_dotenv(encoding="latin-1")

from datetime import datetime, timezone
from System1.Ingestion.copernicus_client import fetch_temperature, fetch_solar_radiation

start_dt = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
end_dt   = datetime(2024, 6, 1, 3, tzinfo=timezone.utc)

print(f"Window: {start_dt.isoformat()} → {end_dt.isoformat()}")
print()

# Temperature
t0 = time.monotonic()
records_temp = fetch_temperature("FR", start_dt, end_dt)
elapsed_temp = time.monotonic() - t0
print(f"fetch_temperature: {len(records_temp)} records in {elapsed_temp:.1f}s")

# Solar radiation
t0 = time.monotonic()
records_solar = fetch_solar_radiation("FR", start_dt, end_dt)
elapsed_solar = time.monotonic() - t0
print(f"fetch_solar_radiation: {len(records_solar)} records in {elapsed_solar:.1f}s")

print()
print(f"Total Copernicus time: {elapsed_temp + elapsed_solar:.1f}s")