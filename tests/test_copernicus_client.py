# tests/test_copernicus_client.py
"""
Unit and integration tests for System1/Ingestion/copernicus_client.py.

Unit tests: 0 network calls — cdsapi.Client and client.retrieve are fully
mocked. The NetCDF files used for parsing tests are generated in-memory with
xarray so the actual parsing logic runs against real NetCDF structure.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_netcdf(
    variable_name: str,
    values: list[float],
    n_lat: int = 3,
    n_lon: int = 3,
) -> str:
    """
    Write a minimal valid ERA5-shaped NetCDF file to a temp path and return
    the path.
    """
    times = np.array(
        [
            np.datetime64("2024-06-01T00:00:00"),
            np.datetime64("2024-06-01T01:00:00"),
            np.datetime64("2024-06-01T02:00:00"),
        ]
    )[: len(values)]

    lats = np.linspace(41.3, 51.1, n_lat)
    lons = np.linspace(-5.2, 9.6, n_lon)

    # Shape: (time, lat, lon) — fill each time slice uniformly
    data = np.array(
        [np.full((n_lat, n_lon), v) for v in values], dtype=np.float32
    )

    ds = xr.Dataset(
        {variable_name: (["valid_time", "latitude", "longitude"], data)},
        coords={
            "valid_time":  times,
            "latitude":    lats,
            "longitude":   lons,
        },
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.close()
    ds.to_netcdf(tmp.name)
    return tmp.name


def _make_env(monkeypatch) -> None:
    """Inject dummy credentials so _get_client() does not raise."""
    monkeypatch.setenv("COPERNICUS_URL", "https://cds.climate.copernicus.eu/api")
    monkeypatch.setenv("COPERNICUS_API_KEY", "dummy-token")


# ---------------------------------------------------------------------------
# _get_client() — credential validation
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_raises_without_url(self, monkeypatch):
        monkeypatch.setenv("COPERNICUS_API_KEY", "token")
        _real_getenv = os.getenv
        with patch("System1.Ingestion.copernicus_client.os.getenv",
                side_effect=lambda k, *a: None if k == "COPERNICUS_URL" else _real_getenv(k, *a)):
            from System1.Ingestion.copernicus_client import _get_client
            with pytest.raises(EnvironmentError, match="COPERNICUS_URL"):
                _get_client()

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.setenv("COPERNICUS_URL", "https://cds.climate.copernicus.eu/api")
        _real_getenv = os.getenv
        with patch("System1.Ingestion.copernicus_client.os.getenv",
                side_effect=lambda k, *a: None if k == "COPERNICUS_API_KEY" else _real_getenv(k, *a)):
            from System1.Ingestion.copernicus_client import _get_client
            with pytest.raises(EnvironmentError, match="COPERNICUS_API_KEY"):
                _get_client()

    def test_passes_url_and_key_to_constructor(self, monkeypatch):
        _make_env(monkeypatch)
        with patch("cdsapi.Client") as mock_cls:
            from System1.Ingestion import copernicus_client
            copernicus_client._get_client()
            mock_cls.assert_called_once_with(
                url="https://cds.climate.copernicus.eu/api",
                key="dummy-token",
                quiet=True,
            )


# ---------------------------------------------------------------------------
# fetch_temperature()
# ---------------------------------------------------------------------------

class TestFetchTemperature:
    """Tests for fetch_temperature() — mocked CDS, real NetCDF parsing."""

    _KELVIN = [273.15, 293.15, 303.15]
    _CELSIUS = [0.0, 20.0, 30.0]

    def _run(self, monkeypatch, country: str = "FR") -> list[dict]:
        _make_env(monkeypatch)
        nc_path =  _make_netcdf("t2m", self._KELVIN)
        mock_client = MagicMock()

        def fake_retrieve(dataset, request, target):
            import shutil
            shutil.copy(nc_path, target)

        mock_client.retrieve.side_effect = fake_retrieve

        with patch("cdsapi.Client", return_value=mock_client):
            from System1.Ingestion.copernicus_client import fetch_temperature
            records = fetch_temperature(
                country_code=country,
                start=datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
                end=datetime(2024, 6, 1, 2, tzinfo=timezone.utc),
            )

        os.remove(nc_path)
        return records

    def test_returns_list_of_dicts(self, monkeypatch):
        records = self._run(monkeypatch)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_record_count(self, monkeypatch):
        records = self._run(monkeypatch)
        assert len(records) == 3

    def test_schema_keys(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        expected = {"timestamp", "source_api", "country", "variable", "value", "unit", "metadata"}
        assert expected.issubset(record.keys())

    def test_source_api_is_copernicus(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert record["source_api"] == "copernicus"

    def test_variable_name(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert record["variable"] == "climate_temperature_2m"

    def test_unit_is_celsius(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert record["unit"] == "°C"

    def test_kelvin_to_celsius_conversion(self, monkeypatch):
        records = self._run(monkeypatch)
        values = [round(r["value"], 6) for r in records]
        assert values == pytest.approx(self._CELSIUS, abs=1e-4)

    def test_country_uppercase(self, monkeypatch):
        records = self._run(monkeypatch, country="fr")
        assert all(r["country"] == "FR" for r in records)

    def test_timestamps_are_utc_aware(self, monkeypatch):
        records = self._run(monkeypatch)
        for r in records:
            assert r["timestamp"].tzinfo is not None
            assert r["timestamp"].tzinfo == timezone.utc

    def test_nan_values_are_skipped(self, monkeypatch):
        _make_env(monkeypatch)
        nc_path = _make_netcdf("t2m", [273.15, float("nan"), 303.15])
        mock_client = MagicMock()

        def fake_retrieve(dataset, request, target):
            import shutil
            shutil.copy(nc_path, target)

        mock_client.retrieve.side_effect = fake_retrieve

        with patch("cdsapi.Client", return_value=mock_client):
            from System1.Ingestion.copernicus_client import fetch_temperature
            records = fetch_temperature(
                country_code="FR",
                start=datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
                end=datetime(2024, 6, 1, 2, tzinfo=timezone.utc),
            )

        os.remove(nc_path)
        assert len(records) == 2

    def test_invalid_country_raises(self, monkeypatch):
        _make_env(monkeypatch)
        with patch("cdsapi.Client"):
            from System1.Ingestion.copernicus_client import fetch_temperature
            with pytest.raises(ValueError, match="Unsupported country"):
                fetch_temperature(
                    "XX",
                    datetime(2024, 6, 1, tzinfo=timezone.utc),
                    datetime(2024, 6, 1, 2, tzinfo=timezone.utc),
                )

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("COPERNICUS_URL", "https://cds.climate.copernicus.eu/api")
        monkeypatch.delenv("COPERNICUS_API_KEY", raising=False)
        from System1.Ingestion.copernicus_client import fetch_temperature
        with pytest.raises(EnvironmentError, match="COPERNICUS_API_KEY"):
            fetch_temperature(
                "FR",
                datetime(2024, 6, 1, tzinfo=timezone.utc),
                datetime(2024, 6, 1, 2, tzinfo=timezone.utc),
            )

    def test_metadata_keys(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert "dataset" in record["metadata"]
        assert "resolution" in record["metadata"]
        assert "aggregation" in record["metadata"]


# ---------------------------------------------------------------------------
# fetch_solar_radiation()
# ---------------------------------------------------------------------------

class TestFetchSolarRadiation:
    """Tests for fetch_solar_radiation() — J/m² → W/m² conversion."""

    _JOULES = [3600.0, 7200.0, 0.0]
    _WATTS  = [1.0,    2.0,    0.0]

    def _run(self, monkeypatch, country: str = "DE") -> list[dict]:
        _make_env(monkeypatch)
        nc_path = _make_netcdf("ssrd", self._JOULES)
        mock_client = MagicMock()

        def fake_retrieve(dataset, request, target):
            import shutil
            shutil.copy(nc_path, target)

        mock_client.retrieve.side_effect = fake_retrieve

        with patch("cdsapi.Client", return_value=mock_client):
            from System1.Ingestion.copernicus_client import fetch_solar_radiation
            records = fetch_solar_radiation(
                country_code=country,
                start=datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
                end=datetime(2024, 6, 1, 2, tzinfo=timezone.utc),
            )

        os.remove(nc_path)
        return records

    def test_returns_list_of_dicts(self, monkeypatch):
        records = self._run(monkeypatch)
        assert isinstance(records, list)

    def test_variable_name(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert record["variable"] == "climate_solar_radiation"

    def test_unit_is_watts(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        assert record["unit"] == "W/m²"

    def test_joules_to_watts_conversion(self, monkeypatch):
        records = self._run(monkeypatch)
        values = [round(r["value"], 6) for r in records]
        assert values == pytest.approx(self._WATTS, abs=1e-6)

    def test_schema_keys(self, monkeypatch):
        record = self._run(monkeypatch)[0]
        expected = {"timestamp", "source_api", "country", "variable", "value", "unit", "metadata"}
        assert expected.issubset(record.keys())

    def test_source_api_is_copernicus(self, monkeypatch):
        assert self._run(monkeypatch)[0]["source_api"] == "copernicus"

    def test_country_uppercase(self, monkeypatch):
        records = self._run(monkeypatch, country="de")
        assert all(r["country"] == "DE" for r in records)

    def test_values_are_floats(self, monkeypatch):
        records = self._run(monkeypatch)
        assert all(isinstance(r["value"], float) for r in records)


# ---------------------------------------------------------------------------
# Integration tests — require live CDS credentials
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCopernicusIntegration:
    """
    Live calls to CDS. Run only when COPERNICUS_API_KEY is set.
    Window: 3 hours on 2024-06-01 to keep download small.
    """

    _START = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)
    _END   = datetime(2024, 6, 1, 2, tzinfo=timezone.utc)

    def test_fetch_temperature_fr(self):
        from System1.Ingestion.copernicus_client import fetch_temperature
        records = fetch_temperature("FR", self._START, self._END)
        assert len(records) >= 1
        assert records[0]["unit"] == "°C"
        # Sanity check: FR June temperatures are between -5 and 45 °C
        assert all(-5.0 <= r["value"] <= 45.0 for r in records)

    def test_fetch_solar_radiation_de(self):
        from System1.Ingestion.copernicus_client import fetch_solar_radiation
        records = fetch_solar_radiation("DE", self._START, self._END)
        assert len(records) >= 1
        assert records[0]["unit"] == "W/m²"
        # Solar radiation is always non-negative
        assert all(r["value"] >= 0.0 for r in records)