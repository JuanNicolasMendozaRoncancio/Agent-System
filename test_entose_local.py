import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from System1.Ingestion.entsoe_client import fetch_load

START = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
END   = datetime(2024, 6, 1, 3, tzinfo=timezone.utc)

print("Calling fetch_load for FR, 2024-06-01 00:00 → 03:00 ...")
records = fetch_load("FR", START, END)

print(f"\nRecords returned: {len(records)}")
if records:
    print(f"First record: {records[0]}")
    variables = {r['variable'] for r in records}
    print(f"Variables found: {variables}")
    
    if "load_actual_aggregated" in variables:
        print("\n✅ load_actual_aggregated IS returned by fetch_load")
    else:
        print("\n❌ load_actual_aggregated NOT in results")
else:
    print("❌ No records returned at all")