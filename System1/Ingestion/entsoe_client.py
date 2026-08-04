"""
ENTSO-E Transparency Platform client.
 
Public interface
----------------
fetch_generation(country_code, start, end) -> list[dict]
fetch_load(country_code, start, end)       -> list[dict]
 
Each dict is compatible with the energy_climate_records table schema:
    {
        "timestamp": datetime (UTC, timezone-aware),
        "source_api": "entsoe",
        "country":    str (e.g. "FR"),
        "variable":   str (e.g. "generation_solar"),
        "value":      float (MW),
        "unit":       "MW",
        "metadata":   dict,
    }
 
Why entsoe-py and not raw httpx + xml.etree
-------------------------------------------
ENTSO-E returns XML documents with nested IEC-62325 namespaces and time
series whose resolution varies by country and document type (PT15M or
PT60M).  Parsing that manually is hundreds of lines of fragile XPath.
entsoe-py handles all of it and returns pandas.Series indexed by UTC
timestamp — exactly what we need to produce the flat dicts above.
 
Why synchronous (requests-based)
---------------------------------
LangGraph agent nodes are plain Python functions called synchronously.
FastAPI wraps them with run_in_executor(), so blocking I/O inside the
agent thread is safe and correct.  Using an async HTTP client here would
require asyncio.run() inside the thread, which raises
"This event loop is already running" in FastAPI's context.
 
Country codes
-------------
Use ENTSO-E area codes, not ISO-2.  The most common ones:
    France      : 10YFR-RTE------C
    Germany     : 10Y1001A1001A83F
    Spain       : 10YES-REE------0
    Colombia is not covered by ENTSO-E (European grid operator only).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


_AREA_CODES: dict[str, str] = {
    "FR": "10YFR-RTE------C",
    "DE": "10Y1001A1001A83F",
    "ES": "10YES-REE------0",
    "BE": "10YBE----------2",
    "NL": "10YNL----------L",
    "PT": "10YPT-REN------W",
    "IT": "10YIT-GRTN-----B",
    "PL": "10YPL-AREA-----S",
    "AT": "10YAT-APG------L",
    "CH": "10YCH-SWISSGRIDZ",
}

_PSR_TYPE_NAMES: dict[str, str] = {
    "B01": "biomass",
    "B02": "fossil_brown_coal",
    "B03": "fossil_gas",
    "B04": "fossil_hard_coal",
    "B05": "fossil_oil",
    "B09": "geothermal",
    "B10": "hydro_pumped_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_water_reservoir",
    "B14": "nuclear",
    "B15": "other_renewable",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B20": "other",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_client():
    """
    Build and return an EntsoePandasClient.
 
    Why lazy construction: the client validates the API key on import if
    we build it at module level.  Lazy construction lets the module import
    cleanly even when ENTSOE_API_KEY is not set (e.g. in unit tests that
    mock this function).
    """
    from entsoe import EntsoePandasClient

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSOE_API_KEY is not set. "
            "Register at https://transparency.entsoe.eu and add the token to .env."
        )
    return EntsoePandasClient(api_key=api_key)


def _resolve_area(country_code: str) -> str:
    """
    Convert ISO-2 country code to ENTSO-E area EIC code.
 
    Raises ValueError for unsupported countries so the caller gets a clear
    message instead of a cryptic ENTSO-E XML error.
    """
    code = _AREA_CODES.get(country_code.upper())
    if code is None:
        supported = ", ".join(sorted(_AREA_CODES.keys()))
        raise ValueError(
            f"Country '{country_code}' is not in the supported list: {supported}"
        )
    return code


def _to_pandas_timestamp(dt: datetime) -> pd.Timestamp:
    """
    Convert a datetime to a timezone-aware pandas.Timestamp in UTC.
 
    Why this conversion: entsoe-py requires pd.Timestamp with tz info.
    Accepting plain datetime objects in our public API is friendlier for
    callers (e.g. LangGraph agent nodes that build datetimes from strings).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")

def _series_to_records(
    series: "pd.Series",
    country_code: str,
    variable: str,
    unit: str = "MW",
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a pandas.Series (index=timestamp, values=float) to a list of
    dicts compatible with energy_climate_records.
 
    Why we drop NaN rows: ENTSO-E sometimes returns NaN for hours where
    a generation type was offline or data was not reported.  Inserting NaN
    into a DOUBLE PRECISION column raises a psycopg type error, and a
    missing reading is correctly represented by the absence of a row.
    """
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"resolution": "inferred", **(extra_metadata or {})}
 
    for ts, value in series.items():
        if pd.isna(value):
            continue
        records.append(
            {
                "timestamp": ts.to_pydatetime().astimezone(timezone.utc),
                "source_api": "entsoe",
                "country": country_code.upper(),
                "variable": variable,
                "value": float(value),
                "unit": unit,
                "metadata": metadata,
            }
        )
    return records

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def fetch_generation(
    country_code: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch actual generation per production type for a country and time range.
 
    Parameters
    ----------
    country_code : ISO-2 code, e.g. "FR", "DE", "ES".
    start        : Start of the period (inclusive).  Naive datetimes assumed UTC.
    end          : End of the period (exclusive).    Naive datetimes assumed UTC.
 
    Returns
    -------
    list[dict] where each dict has keys:
        timestamp, source_api, country, variable, value, unit, metadata.
    variable is formatted as "generation_<psr_type_name>", e.g. "generation_solar".
 
    Why we return all production types in a single call
    ---------------------------------------------------
    ENTSO-E's GL_Dts_ActualGenerationPerProductionType document returns all
    generation types for a country in one HTTP request.  Splitting by type
    would multiply the number of API calls with no benefit.
    """
    area = _resolve_area(country_code)
    ts_start = _to_pandas_timestamp(start)
    ts_end = _to_pandas_timestamp(end)

    logger.info(
        "Fetching generation for %s from %s to %s", country_code, ts_start, ts_end
    )

    client = _get_client()

    df: pd.DataFrame = client.query_generation(
        country_code= area,
        start= ts_start,
        end= ts_end,
        psr_type= None 
    )

    if isinstance(df.columns, pd.MultiIndex):
        df = df.loc[:, df.columns.get_level_values(1) == "Actual Aggregated"]
        df.columns = df.columns.droplevel(1)

    records: list[dict[str, Any]] = []
    for col in df.columns:
        variable = "generation_" + col.lower().replace(" ","_")
        records.extend(
            _series_to_records(
                df[col],
                country_code= country_code,
                variable= variable,
                unit="MW",
                extra_metadata={"psr_type_raw": col},
            )
        )

    logger.info(
        "fetch_generation: %d records for %s", len(records), country_code
    )
    return records

def fetch_load(
    country_code: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch actual total load (demand) for a country and time range.
 
    Parameters
    ----------
    country_code : ISO-2 code, e.g. "FR", "DE", "ES".
    start        : Start of the period (inclusive).
    end          : End of the period (exclusive).
 
    Returns
    -------
    list[dict] with variable = "load_actual_aggregated".
 
    Why load separately from generation
    ------------------------------------
    Load and generation come from different ENTSO-E document types
    (TotalLoadActual vs GL_Dts_ActualGenerationPerProductionType).
    Keeping them as separate functions lets callers fetch only what they
    need — the Ingestion Agent can call both and merge the results, or call
    only one if the other is temporarily unavailable (e.g. data not yet
    published for the current hour).
    """
    area = _resolve_area(country_code)
    ts_start = _to_pandas_timestamp(start)
    ts_end = _to_pandas_timestamp(end)

    logger.info(
        "Fetching load for %s from %s to %s", country_code, ts_start, ts_end   
    )

    client = _get_client()

    df: pd.DataFrame = client.query_load(
        country_code=area,
        start=ts_start,
        end=ts_end
    )

    series: pd.series = df["Actual Load"]
    records = _series_to_records(
        series,
        country_code=country_code,
        variable="load_actual_aggregated",
        unit="MW"
    )

    logger.info("fetch_load: %d records for %s", len(records), country_code)
    return records