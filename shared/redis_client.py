"""
Redis client and Pub/Sub helpers.
"""
import os
import redis

_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Return the shared Redis client, creating it on first call."""
    global _client
    if _client is None:
        _client = redis.Redis(
            host = _REDIS_HOST,
            port= _REDIS_PORT,
            decode_responses=True,
        )
    return _client

def check_connection() -> bool:
    """Return True if Redis is reachable, False otherwise."""
    try:
        return get_redis().ping()
    except Exception:
        return False

CHANNEL_VALIDATED_DATA = "validated_data"
CHANNEL_FAILED_MESSAGES = "failed_messages"