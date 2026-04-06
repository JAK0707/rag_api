# ============================================================
# FILE: app/services/cache.py
# PURPOSE: Async Redis cache for question+doc_id → answer lookups
# ============================================================

import hashlib

import redis.asyncio as aioredis

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level client; initialised in main.py lifespan
_redis_client: aioredis.Redis | None = None


def set_redis_client(client: aioredis.Redis) -> None:
    """Store the shared async Redis client for use by cache helpers."""
    global _redis_client
    _redis_client = client


def get_redis_client() -> aioredis.Redis | None:
    """Return the shared async Redis client, or None if uninitialised."""
    return _redis_client


def _build_cache_key(doc_id: str, question: str) -> str:
    """Return a deterministic Redis key for a (doc_id, question) pair."""
    question_hash = hashlib.md5(question.encode()).hexdigest()
    return f"rag:{doc_id}:{question_hash}"


async def get_cached_answer(doc_id: str, question: str) -> str | None:
    """Retrieve a cached answer from Redis; returns None on miss or error."""
    client = get_redis_client()
    if client is None:
        logger.warning("Redis client not initialised — skipping cache lookup")
        return None

    key = _build_cache_key(doc_id, question)
    try:
        value = await client.get(key)
        if value:
            logger.info("Cache HIT for key=%s", key)
            return value.decode("utf-8")
        logger.info("Cache MISS for key=%s", key)
        return None
    except aioredis.RedisError as exc:
        logger.warning("Redis GET failed for key=%s: %s", key, exc)
        return None


async def set_cached_answer(
    doc_id: str, question: str, answer: str, ttl: int
) -> None:
    """Store an answer in Redis with the given TTL (seconds); silently skips on error."""
    client = get_redis_client()
    if client is None:
        logger.warning("Redis client not initialised — skipping cache write")
        return

    key = _build_cache_key(doc_id, question)
    try:
        await client.set(key, answer.encode("utf-8"), ex=ttl)
        logger.info("Cached answer for key=%s with TTL=%ds", key, ttl)
    except aioredis.RedisError as exc:
        logger.warning("Redis SET failed for key=%s: %s", key, exc)
