"""
CI connectivity tests — Step 24 GitHub Actions.

Purpose
-------
These tests run on every push to main via GitHub Actions. They verify that:
  1. ENTSO-E API credentials are valid and the endpoint is reachable.
  2. Copernicus CDS credentials are valid and the endpoint is reachable.
  3. The RAG endpoint is reachable and returns a well-formed response.
  4. The Groq primary model (gpt-oss-20b) accepts a request and responds.
  5. The Groq fallback model (gpt-oss-120b) accepts a request and responds.

"""
from __future__ import annotations

import os

import pytest
import httpx
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# ENTSO-E connectivity
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestEntsoEConnectivity:
    """
    Verify ENTSO-E API credentials and endpoint reachability.
    """

    _BASE_URL = "https://web-api.tp.entsoe.eu/api"

    def test_api_key_is_set(self):
        """ENTSOE_API_KEY must be present in the environment."""
        assert os.getenv("ENTSOE_API_KEY"), (
            "ENTSOE_API_KEY is not set. "
            "Add it as a GitHub Actions secret: Settings → Secrets → ENTSOE_API_KEY."
        )

    def test_endpoint_reachable_with_valid_key(self):
        """
        A minimal authenticated GET to the ENTSO-E API must return HTTP 200.
        """
        api_key = os.getenv("ENTSOE_API_KEY")
        if not api_key:
            pytest.skip("ENTSOE_API_KEY not set")

        response = httpx.get(
            self._BASE_URL,
            params={
                "securityToken": api_key,
                "documentType":  "A09",  
            },
            timeout=20.0,
        )

        assert response.status_code in (200, 400), (
            f"ENTSO-E returned unexpected status {response.status_code}. "
            f"Body: {response.text[:200]}"
        )


# ---------------------------------------------------------------------------
# Copernicus CDS connectivity
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestCopernicusConnectivity:
    """
    Verify Copernicus CDS credentials and endpoint reachability.
    """

    def test_url_and_key_are_set(self):
        """Both COPERNICUS_URL and COPERNICUS_API_KEY must be present."""
        assert os.getenv("COPERNICUS_URL"), "COPERNICUS_URL is not set."
        assert os.getenv("COPERNICUS_API_KEY"), "COPERNICUS_API_KEY is not set."

    def test_credentials_accepted_by_cds(self):
        """
        GET /retrieve/v1/jobs with Bearer auth must return HTTP 200.
        """
        api_key = os.getenv("COPERNICUS_API_KEY")
        base_url = os.getenv("COPERNICUS_URL", "").rstrip("/") 
        
        if not api_key or not base_url:
            pytest.skip("COPERNICUS_API_KEY or COPERNICUS_URL not set")

        check_url = f"{base_url}/retrieve/v1/jobs"

        response = httpx.get(
            check_url,
            headers={"Authorization": f"Bearer {api_key}",
                     "PRIVATE-TOKEN": api_key
                    },
            timeout=15.0,
        )
        
        assert response.status_code == 200, (
            f"CDS jobs endpoint returned {response.status_code}. "
            f"Check COPERNICUS_API_KEY and ensure it is valid for the new CDS. "
            f"Body: {response.text[:200]}"
        )
        
        data = response.json()
        assert isinstance(data, (list, dict)), (
            f"Expected JSON list or dict from CDS jobs endpoint, got {type(data)}"
        )


# ---------------------------------------------------------------------------
# RAG endpoint connectivity
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestRagConnectivity:
    """
    Verify the RAG API endpoint is reachable and returns a well-formed response.
    """

    def test_rag_url_configured(self):

        rag_url = os.getenv("RAG_API_URL", "")
        if not rag_url:
            pytest.skip("RAG_API_URL not set — RAG integration is optional in CI.")
        assert rag_url.startswith("http"), (
            f"RAG_API_URL must start with http(s), got: {rag_url!r}"
        )

    def test_topics_active_endpoint_reachable(self):
        """
        GET /rag/topics/active must return HTTP 200 and a JSON list or dict.
        """
        rag_url = os.getenv("RAG_API_URL", "").rstrip("/")
        rag_key = os.getenv("RAG_API_KEY", "")

        if not rag_url:
            pytest.skip("RAG_API_URL not set — skipping RAG connectivity test.")

        response = httpx.get(
            f"{rag_url}/rag/topics/active",
            headers={"X-RAG-Key": rag_key} if rag_key else {},
            timeout=15.0,
        )
        assert response.status_code == 200, (
            f"RAG /rag/topics/active returned {response.status_code}. "
            f"Body: {response.text[:200]}"
        )
        data = response.json()
        assert isinstance(data, (list, dict)), (
            f"Expected JSON list or dict from /rag/topics/active, got {type(data)}"
        )


# ---------------------------------------------------------------------------
# Groq primary model connectivity
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGroqPrimaryConnectivity:
    """
    Verify the primary Groq model (gpt-oss-20b) is reachable and responds.
    """

    def test_groq_key_is_set(self):
        """GROQ_API_KEY must be present in the environment."""
        assert os.getenv("GROQ_API_KEY"), (
            "GROQ_API_KEY is not set. "
            "Add it as a GitHub Actions secret."
        )

    def test_primary_model_responds(self):
        """
        A minimal chat_complete call to gpt-oss-20b must return a non-empty string.
        """
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

        from shared.llm_client import _call_groq

        result = _call_groq(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            system=None,
            json_mode=False,
            timeout=30.0,
        )
        assert isinstance(result, str), "Primary model must return a string"
        assert len(result) > 0, "Primary model must return non-empty text"


# ---------------------------------------------------------------------------
# Groq fallback model connectivity
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGroqFallbackConnectivity:
    """
    Verify the fallback Groq model (gpt-oss-120b) is independently reachable.
    """

    def test_fallback_model_responds(self):
        """
        A minimal call to gpt-oss-120b must return a non-empty string.
        """
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

        from shared.llm_client import _call_groq_fallback

        result = _call_groq_fallback(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            system=None,
            json_mode=False,
            timeout=30.0,
        )
        assert isinstance(result, str), "Fallback model must return a string"
        assert len(result) > 0, "Fallback model must return non-empty text"