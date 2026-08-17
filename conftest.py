import os
import sys
import time
import pytest
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def _rate_limit_pause():
    yield
    time.sleep(300)

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))