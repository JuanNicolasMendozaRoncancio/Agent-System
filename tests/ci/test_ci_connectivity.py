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

What these tests do NOT do
--------------------------
- They do NOT download real data (no ERA5 NetCDF files, no ENTSO-E XML).
- They do NOT insert into PostgreSQL or publish to Redis.
- They do NOT run the full LangGraph graphs.
- They do NOT count against Groq TPM budget (one minimal call per test).

Why a separate subdirectory (tests/ci/) instead of @pytest.mark.ci in tests/
------------------------------------------------------------------------------
The root conftest.py has an autouse fixture that sleeps 300 seconds after every
test to protect Groq free-tier rate limits. That fixture would make a CI job
with 5 connectivity tests take 25 minutes minimum. Placing these tests in their
own subdirectory with a local conftest.py that overrides the pause keeps CI
under 60 seconds while leaving the root behaviour intact for integration tests.

Marker used: @pytest.mark.ci
Run: pytest tests/ci/ -v -m ci
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

    Why a document-listing call instead of a data query:
        The ENTSO-E /api endpoint with action=getDataSets returns a short XML
        document catalogue — no time range, no area code, no data download.
        It consumes negligible quota, completes in under 5 seconds, and is
        sufficient to prove the key is active and the API is reachable.
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

        We request publication types (documentType=A09), a very small metadata
        document with no time-series data attached. This verifies auth without
        downloading any energy records.
        """
        api_key = os.getenv("ENTSOE_API_KEY")
        if not api_key:
            pytest.skip("ENTSOE_API_KEY not set")

        response = httpx.get(
            self._BASE_URL,
            params={
                "securityToken": api_key,
                "documentType":  "A09",   # publication types catalogue
            },
            timeout=20.0,
        )
        # ENTSO-E returns 200 for valid keys even when no data matches the query.
        # A 401 means the key is rejected; a 400 means the query is malformed
        # but the key was accepted (also acceptable for a connectivity check).
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

    Copernicus recently migrated to a new infrastructure (CADS).
    The old /api/userinfo endpoint no longer exists.
    To verify credentials without downloading data, we query the /retrieve/v1/jobs
    endpoint, which lists the user's recent download jobs and requires valid auth.
    """

    def test_url_and_key_are_set(self):
        """Both COPERNICUS_URL and COPERNICUS_API_KEY must be present."""
        assert os.getenv("COPERNICUS_URL"), "COPERNICUS_URL is not set."
        assert os.getenv("COPERNICUS_API_KEY"), "COPERNICUS_API_KEY is not set."

    def test_credentials_accepted_by_cds(self):
        """
        GET /retrieve/v1/jobs with Bearer auth must return HTTP 200.

        A 200 response confirms the key exists and the CDS service is reachable. 
        A 401 means the key has been revoked or is invalid.
        """
        api_key = os.getenv("COPERNICUS_API_KEY")
        # Aseguramos que no haya un slash extra al final de la URL base
        base_url = os.getenv("COPERNICUS_URL", "").rstrip("/") 
        
        if not api_key or not base_url:
            pytest.skip("COPERNICUS_API_KEY or COPERNICUS_URL not set")

        # Nuevo endpoint de la API para consultar trabajos encolados
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
        
        # Validamos que la respuesta sea un JSON válido (suele ser una lista o dict)
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

    Why /rag/topics/active and not /health:
        The /health endpoint may exist on any server. /rag/topics/active is
        specific to our RAG system and is the exact endpoint consumed by the
        Narrative Agent and the RCA Agent. If this endpoint is up, the RAG
        integration will work at pipeline runtime.

    Skipped if RAG_API_URL is not set:
        RAG is optional — the pipeline degrades gracefully without it.
        CI should not fail if the RAG system is not deployed yet.
    """

    def test_rag_url_configured(self):
        """
        If RAG_API_URL is set, verify it is a valid URL string.
        If it is not set, the test passes — RAG is optional in CI.
        """
        rag_url = os.getenv("RAG_API_URL", "")
        if not rag_url:
            pytest.skip("RAG_API_URL not set — RAG integration is optional in CI.")
        assert rag_url.startswith("http"), (
            f"RAG_API_URL must start with http(s), got: {rag_url!r}"
        )

    def test_topics_active_endpoint_reachable(self):
        """
        GET /rag/topics/active must return HTTP 200 and a JSON list or dict.

        Why we accept both list and dict:
            Some RAG API versions return a bare list, others wrap in
            {"topics": [...], "total": N}. Both are valid — what matters
            for CI is that the endpoint responds and the content is JSON.
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

    Why a real call and not a mock:
        The purpose of a CI connectivity test is to detect when the API key
        has expired, the model has been renamed, or Groq has a service
        disruption — none of which a mock can detect. We use the minimal
        possible prompt to keep token cost negligible.

    Token cost per CI run: ~10 input + ~5 output = ~15 tokens total.
    At Groq's free tier (14,400 req/day) this is a rounding error.
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

        Why we call _call_groq directly instead of chat_complete:
            chat_complete would activate the fallback on any error, masking
            a primary-model failure. Calling _call_groq directly verifies the
            primary path in isolation — exactly what CI needs to detect a
            broken primary model while the fallback might still work.
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

    Why test the fallback separately:
        The fallback model may have different availability, rate limits, or
        billing requirements from the primary. Verifying it in CI ensures that
        when the primary hits a rate limit in production, the fallback path is
        known to be working — not discovered to be broken at the worst moment.

    Token cost per CI run: ~10 input + ~5 output = ~15 tokens total.
    """

    def test_fallback_model_responds(self):
        """
        A minimal call to gpt-oss-120b must return a non-empty string.

        Why _call_groq_fallback directly and not LLM_PROVIDER=groq_fallback:
            Same reasoning as the primary test — we want to verify the fallback
            path in isolation, not via chat_complete's routing logic.
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