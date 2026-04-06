# ============================================================
# FILE: app/services/embedder.py
# PURPOSE: Batch embedding generation via OpenAI text-embedding-3-small
# ============================================================

import asyncio

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100          # OpenAI recommends batching up to 2048 inputs
_RETRY_ATTEMPTS = 3
_BACKOFF_BASE = 2.0        # seconds; doubles on each retry


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in batches using OpenAI text-embedding-3-small.

    Args:
        texts: Raw strings to embed.

    Returns:
        List of float vectors, one per input string.
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []

    batches = [texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
    logger.info("Embedding %d text(s) in %d batch(es)", len(texts), len(batches))

    for batch_idx, batch in enumerate(batches):
        embeddings = await _embed_batch_with_retry(client, batch, batch_idx)
        all_embeddings.extend(embeddings)
        if batch_idx < len(batches) - 1:
            await asyncio.sleep(0.1)  # gentle rate-limit buffer between batches

    logger.info("Finished embedding — total vectors: %d", len(all_embeddings))
    return all_embeddings


async def _embed_batch_with_retry(
    client: AsyncOpenAI, batch: list[str], batch_idx: int
) -> list[list[float]]:
    """Attempt to embed a single batch, with exponential-backoff retries.

    Args:
        client:    Async OpenAI client.
        batch:     List of strings to embed.
        batch_idx: Index of this batch (for logging).

    Returns:
        List of float vectors for the batch.

    Raises:
        APIError: If all retry attempts are exhausted.
    """
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = await client.embeddings.create(
                model=_EMBEDDING_MODEL,
                input=batch,
            )
            vectors = [item.embedding for item in response.data]
            logger.info(
                "Batch %d/%d embedded (%d vectors, attempt %d)",
                batch_idx + 1, batch_idx + 1, len(vectors), attempt,
            )
            return vectors

        except RateLimitError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Rate limit on batch %d (attempt %d/%d) — retrying in %.1fs: %s",
                batch_idx, attempt, _RETRY_ATTEMPTS, wait, exc,
            )
            await asyncio.sleep(wait)

        except APIConnectionError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Connection error on batch %d (attempt %d/%d) — retrying in %.1fs: %s",
                batch_idx, attempt, _RETRY_ATTEMPTS, wait, exc,
            )
            await asyncio.sleep(wait)

        except APIError as exc:
            logger.error("Unrecoverable OpenAI API error on batch %d: %s", batch_idx, exc)
            raise

    raise APIError(  # type: ignore[call-arg]
        message=f"Embedding batch {batch_idx} failed after {_RETRY_ATTEMPTS} attempts",
        request=None,  # type: ignore[arg-type]
        body=None,
    )


async def embed_query(text: str) -> list[float]:
    """Embed a single query string and return its vector.

    Args:
        text: The query string to embed.

    Returns:
        Float vector for the query.
    """
    vectors = await embed_texts([text])
    return vectors[0]
