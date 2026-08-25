"""
Copernicus Climate Data Store (CDS) client.

Fetches ERA5 reanalysis data (temperature, solar radiation) for European
regions and returns records compatible with the energy_climate_records schema.

Public interface
----------------
fetch_temperature(country_code, start, end)      -> list[dict]
fetch_solar_radiation(country_code, start, end)  -> list[dict]
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import cdsapi
import numpy as np
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_COUNTRY_BBOX: dict[str, tuple[float, float, float, float]] = {
    "FR": (51.1, 41.3, -5.2, 9.6),
    "DE": (55.1, 47.3, 5.9, 15.0),
    "ES": (43.8, 35.9, -9.3, 4.3),
    "BE": (51.5, 49.5, 2.5, 6.4),
    "NL": (53.6, 50.7, 3.3, 7.2),
    "PT": (42.2, 36.8, -9.5, -6.2),
    "IT": (47.1, 37.9, 6.6, 18.5),
    "PL": (54.9, 49.0, 14.1, 24.2),
    "AT": (49.0, 46.4, 9.5, 17.2),
    "CH": (47.8, 45.8, 5.9, 10.5),
}

_NETCDF_VAR_NAME: dict[str, str] = {
    "2m_temperature":                    "t2m",
    "surface_solar_radiation_downwards": "ssrd",
}

_SUPPORTED_COUNTRIES = set(_COUNTRY_BBOX.keys())

_DATASET = "reanalysis-era5-single-levels"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_client() -> cdsapi.Client:
    """Build and return a CDS client using env vars (lazy construction)."""
    url = os.getenv("COPERNICUS_URL")
    key = os.getenv("COPERNICUS_API_KEY")
    if not url:
        raise EnvironmentError("COPERNICUS_URL not found in environment")
    if not key:
        raise EnvironmentError("COPERNICUS_API_KEY not found in environment")
    return cdsapi.Client(url=url, key=key, quiet=True)

def _build_request(
    variable: str,
    start: datetime,
    end: datetime,
    country_code: str,
) -> dict[str, Any]:
    """
    Build the CDS API request dict for a single variable.
    """
    import pandas as pd

    hours = pd.date_range(start=start, end=end, freq="h", tz=timezone.utc)

    years  = sorted({str(h.year)  for h in hours})
    months = sorted({str(h.month).zfill(2) for h in hours})
    days   = sorted({str(h.day).zfill(2)   for h in hours})
    times  = sorted({h.strftime("%H:%M")    for h in hours})

    north, south, west, east = _COUNTRY_BBOX[country_code]

    return {
        "product_type": ["reanalysis"],
        "variable":     [variable],
        "year":         years,
        "month":        months,
        "day":          days,
        "time":         times,
        "area":         [north, west, south, east],
        "data_format":  "netcdf",
    }

def _netcdf_to_records(
    path: str,
    variable_name: str,
    output_variable: str,
    country_code: str,
    unit: str,
    transform: Any = None,
) -> list[dict]:
    """
    Parse a NetCDF file into a list of energy_climate_records-compatible dicts.

    Spatial aggregation: ERA5 returns a grid of points over the bounding box.
    We take the spatial mean (nanmean) to produce a single value per timestamp
    representing the country-level average. This is standard practice for
    country-level climate indicators.

    Parameters
    ----------
    path:
        Path to the downloaded NetCDF file.
    variable_name:
        Name of the variable inside the NetCDF file (ERA5 internal name).
    output_variable:
        Name to use in the output dict (our schema convention).
    country_code:
        ISO country code (uppercase).
    unit:
        Physical unit string to store in the record.
    transform:
        Optional callable applied to each value before storing (e.g. K→°C).
    """
    with xr.open_dataset(path, engine="netcdf4") as ds:

        try:
            da = ds[variable_name]
        except KeyError:
            available = list(ds.data_vars)
            raise KeyError(
                f"Variable '{variable_name}' not found in NetCDF. "
                f"Available: {available}"
            )

        spatial_dims = [d for d in da.dims if d in ("latitude", "longitude")]
        da_mean = da.mean(dim=spatial_dims)

        records: list[dict] = []
        for time_val in da_mean.coords["valid_time"].values:
            value = float(da_mean.sel(valid_time=time_val).values)

            if np.isnan(value):
                continue

            if transform is not None:
                value = transform(value)

            ts = (
                datetime.fromtimestamp(
                    (time_val - np.datetime64("1970-01-01T00:00:00")) /
                    np.timedelta64(1, "s")
                ).replace(tzinfo=timezone.utc)
            )

            records.append({
                "timestamp":  ts,
                "source_api": "copernicus",
                "country":    country_code.upper(),
                "variable":   output_variable,
                "value":      value,
                "unit":       unit,
                "metadata": {
                    "dataset":    _DATASET,
                    "resolution": "hourly",
                    "aggregation": "spatial_mean",
                },
            })

    return records

def _fetch(
    variable: str,
    output_variable: str,
    unit: str,
    country_code: str,
    start: datetime,
    end: datetime,
    transform: Any = None,
) -> list[dict]:
    """Core download-and-parse logic shared by all public fetch functions."""
    country_code = country_code.upper()
    if country_code not in _SUPPORTED_COUNTRIES:
        raise ValueError(
            f"Unsupported country '{country_code}'. "
            f"Supported: {sorted(_SUPPORTED_COUNTRIES)}"
        )

    client = _get_client()
    request = _build_request(variable, start, end, country_code)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info(
            "Requesting ERA5 '%s' for %s [%s → %s]",
            variable, country_code,
            start.isoformat(), end.isoformat(),
        )
        client.retrieve(_DATASET, request, tmp_path)
        netcdf_var = _NETCDF_VAR_NAME.get(variable, variable)
        records = _netcdf_to_records(
            tmp_path, netcdf_var, output_variable, country_code, unit, transform
        )
        logger.info("Fetched %d records (%s, %s)", len(records), output_variable, country_code)
        return records

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def fetch_temperature(
    country_code: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Fetch ERA5 2m air temperature for a country and time window.

    ERA5 stores temperature in Kelvin. We convert to Celsius (K - 273.15)
    because Celsius is the standard unit in European energy market reporting
    and is what operators use when correlating heating/cooling demand.

    Parameters
    ----------
    country_code:
        ISO 3166-1 alpha-2 code (case-insensitive). Supported: FR, DE, ES,
        BE, NL, PT, IT, PL, AT, CH.
    start, end:
        UTC-aware datetimes defining the inclusive time window.

    Returns
    -------
    list[dict]
        Each dict is compatible with the energy_climate_records schema.
        variable = 'climate_temperature_2m', unit = '°C'.
    """
    return _fetch(
        variable="2m_temperature",
        output_variable="climate_temperature_2m",
        unit="°C",
        country_code=country_code,
        start=start,
        end=end,
        transform=lambda k: k - 273.15,
    )


def fetch_solar_radiation(
    country_code: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Fetch ERA5 surface solar radiation downwards for a country and time window.

    ERA5 stores this variable as accumulated energy in J/m² since the start
    of the forecast step. We convert to W/m² (average power flux) by dividing
    by 3600 seconds (one hour), which is standard in the energy sector.

    Parameters
    ----------
    country_code:
        ISO 3166-1 alpha-2 code (case-insensitive).
    start, end:
        UTC-aware datetimes defining the inclusive time window.

    Returns
    -------
    list[dict]
        variable = 'climate_solar_radiation', unit = 'W/m²'.
    """
    return _fetch(
        variable="surface_solar_radiation_downwards",
        output_variable="climate_solar_radiation",
        unit="W/m²",
        country_code=country_code,
        start=start,
        end=end,
        transform=lambda j: j / 3600.0,   # J/m² → W/m²
    )