"""
conftest.py for the tests/ci/ subdirectory.
 
Overrides the root conftest's autouse _rate_limit_pause fixture, which sleeps
300 seconds after every test. CI connectivity tests must not sleep — they are
designed to complete in under 60 seconds total so the GitHub Actions job does
not time out.
 
Why override here instead of changing the root conftest:
    The root 300s pause exists for good reason — integration tests against
    Groq's free tier would hit rate limits without it. CI tests only make
    one lightweight HTTP call per test, so no pause is needed. Scoping the
    override to this subdirectory's conftest keeps the root behaviour intact.
"""
import pytest
 
 
@pytest.fixture(autouse=True)
def _rate_limit_pause():
    """No-op override: CI connectivity tests do not need inter-test pauses."""
    yield