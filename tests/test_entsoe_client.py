"""
Unit tests for sistema1/ingestion/entsoe_client.py.

All tests are mock-based — zero real HTTP calls.
The integration test (marked 'integration') requires ENTSOE_API_KEY in .env.

Why mock _get_client() and not the HTTP layer
----------------------------------------------
_get_client() is the seam between our code and the entsoe-py library.
Mocking at that level lets us control what query_generation / query_load
return without needing a real API key or a running server.  Mocking at
the HTTP layer would require us to construct valid ENTSO-E XML, which is
brittle and adds no value to these tests.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dotenv import load_dotenv

# Make the module importable from the tests/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import System1.Ingestion.entsoe_client as ec  # adjust import path to your project layout


# ---------------------------------------------------------------------------
# Fixtures — reusable mock data
# ---------------------------------------------------------------------------

def _make_generation_df() -> pd.DataFrame:
    """
    Minimal DataFrame mimicking what entsoe-py returns for query_generation.
    Two production types, three hourly timestamps.
    """
    idx = pd.date_range(
        "2024-01-15 00:00", periods=3, freq="h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "Solar": [100.0, 200.0, float("nan")],  # NaN row to test filtering
            "Wind Onshore": [500.0, 600.0, 700.0],
        },
        index=idx,
    )


def _make_load_df() -> pd.DataFrame:
    """
    Minimal DataFrame mimicking what entsoe-py returns for query_load.
    Single column 'Actual Load', three hourly timestamps.
    """
    idx = pd.date_range(
        "2024-01-15 00:00", periods=3, freq="h", tz="UTC"
    )
    return pd.DataFrame(
        {"Actual Load": [45000.0, 46000.0, 47000.0]},
        index=idx,
    )


def _mock_client(generation_df: pd.DataFrame, load_df: pd.DataFrame) -> MagicMock:
    """Build a mock EntsoePandasClient that returns the given DataFrames."""
    client = MagicMock()
    client.query_generation.return_value = generation_df
    client.query_load.return_value = load_df
    return client


# ---------------------------------------------------------------------------
# fetch_generation tests
# ---------------------------------------------------------------------------

class TestFetchGeneration:
    START = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    END = datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc)

    def test_returns_list_of_dicts(self):
        """fetch_generation must return a list of dicts."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)

    def test_schema_keys_present(self):
        """Every record must contain the energy_climate_records-compatible keys."""
        expected_keys = {
            "timestamp", "source_api", "country", "variable", "value", "unit", "metadata"
        }
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        for record in result:
            assert expected_keys == set(record.keys()), (
                f"Record missing keys: {expected_keys - set(record.keys())}"
            )

    def test_source_api_is_entsoe(self):
        """source_api must always be 'entsoe'."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        assert all(r["source_api"] == "entsoe" for r in result)

    def test_country_code_normalized_to_upper(self):
        """Country code in records must be uppercase regardless of input."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("fr", self.START, self.END)  # lowercase input

        assert all(r["country"] == "FR" for r in result)

    def test_nan_rows_are_dropped(self):
        """Records with NaN value must not appear in the output."""
        gen_df = _make_generation_df()  # contains NaN for Solar at index 2
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        # Solar has 2 valid + 1 NaN → 2 solar records; Wind has 3 → total 5
        assert len(result) == 5
        assert all(not pd.isna(r["value"]) for r in result)

    def test_variable_name_format(self):
        """Variables must be prefixed with 'generation_' and lowercased."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        variables = {r["variable"] for r in result}
        assert "generation_solar" in variables
        assert "generation_wind_onshore" in variables

    def test_unit_is_mw(self):
        """Unit must be 'MW' for all generation records."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        assert all(r["unit"] == "MW" for r in result)

    def test_timestamps_are_utc_aware(self):
        """All timestamps must be timezone-aware in UTC."""
        gen_df = _make_generation_df()
        mock_client = _mock_client(gen_df, _make_load_df())

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        for r in result:
            assert r["timestamp"].tzinfo is not None, "timestamp must be tz-aware"
            assert r["timestamp"].tzinfo == timezone.utc

    def test_multiindex_columns_handled(self):
        """
        query_generation can return MultiIndex columns (Actual Aggregated /
        Actual Consumption for pumped storage).  Must keep only Actual Aggregated.
        """
        idx = pd.date_range("2024-01-15 00:00", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame(
            [[100.0, 50.0], [200.0, 60.0]],
            index=idx,
            columns=pd.MultiIndex.from_tuples(
                [("Solar", "Actual Aggregated"), ("Solar", "Actual Consumption")]
            ),
        )
        mock_client = MagicMock()
        mock_client.query_generation.return_value = df

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_generation("FR", self.START, self.END)

        # Only Actual Aggregated values (100, 200) should appear
        values = {r["value"] for r in result}
        assert 100.0 in values
        assert 200.0 in values
        assert 50.0 not in values
        assert 60.0 not in values

    def test_unsupported_country_raises_value_error(self):
        """An unsupported ISO-2 code must raise ValueError immediately."""
        with pytest.raises(ValueError, match="not in the supported list"):
            ec.fetch_generation("XX", self.START, self.END)

    def test_missing_api_key_raises_environment_error(self):
        """If ENTSOE_API_KEY is unset, _get_client must raise EnvironmentError."""
        with patch.dict(os.environ, {}, clear=True):
            # We call _get_client directly — it must fail before any HTTP call.
            # We need entsoe-py importable for this test.
            try:
                from entsoe import EntsoePandasClient  # noqa: F401
                with pytest.raises(EnvironmentError, match="ENTSOE_API_KEY"):
                    ec._get_client()
            except ImportError:
                pytest.skip("entsoe-py not installed")


# ---------------------------------------------------------------------------
# fetch_load tests
# ---------------------------------------------------------------------------

class TestFetchLoad:
    START = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    END = datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc)

    def test_returns_list_of_dicts(self):
        load_df = _make_load_df()
        mock_client = _mock_client(_make_generation_df(), load_df)

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_load("DE", self.START, self.END)

        assert isinstance(result, list)
        assert len(result) == 3  # 3 timestamps, 0 NaN

    def test_variable_is_load_actual_aggregated(self):
        """fetch_load must produce variable='load_actual_aggregated'."""
        load_df = _make_load_df()
        mock_client = _mock_client(_make_generation_df(), load_df)

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_load("DE", self.START, self.END)

        assert all(r["variable"] == "load_actual_aggregated" for r in result)

    def test_schema_keys_present(self):
        expected_keys = {
            "timestamp", "source_api", "country", "variable", "value", "unit", "metadata"
        }
        load_df = _make_load_df()
        mock_client = _mock_client(_make_generation_df(), load_df)

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_load("DE", self.START, self.END)

        for record in result:
            assert expected_keys == set(record.keys())

    def test_values_are_floats(self):
        load_df = _make_load_df()
        mock_client = _mock_client(_make_generation_df(), load_df)

        with patch.object(ec, "_get_client", return_value=mock_client):
            result = ec.fetch_load("DE", self.START, self.END)

        assert all(isinstance(r["value"], float) for r in result)


# ---------------------------------------------------------------------------
# Integration test — requires real ENTSOE_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEntsoEIntegration:
    """
    Makes a real call to ENTSO-E for a narrow 3-hour window.
    Requires ENTSOE_API_KEY in .env and the key to be active.
    """

    def test_fetch_generation_france(self):
        start = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, 13, 0, tzinfo=timezone.utc)

        result = ec.fetch_generation("FR", start, end)

        assert len(result) > 0
        # Must contain at least one solar record (France has solar in June)
        variables = {r["variable"] for r in result}
        assert any("solar" in v for v in variables), (
            f"Expected at least one solar variable, got: {variables}"
        )

    def test_fetch_load_germany(self):
        start = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, 13, 0, tzinfo=timezone.utc)

        result = ec.fetch_load("DE", start, end)

        assert len(result) > 0
        assert all(r["variable"] == "load_actual_aggregated" for r in result)
        assert all(r["value"] > 0 for r in result)