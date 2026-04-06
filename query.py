# ============================================================
# FILE: app/routers/query.py
# PURPOSE: POST /api/v1/query — cached RAG Q&A over indexed documents
# ============================================================

import time
import uuid

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.services import cache, embedder, llm, vector_store
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_document(body: QueryRequest) -> QueryResponse:
    """Answer a question about an indexed document using RAG.

    Args:
        body: QueryRequest containing doc_id and question.

    Returns:
        QueryResponse with answer, source chunks, cache flag, and latency.

    Raises:
        HTTPException 404: If no FAISS index exists for the given doc_id.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()

    logger.info("[%s] Query received: doc_id=%s question='%s'",
                request_id, body.doc_id, body.question[:80])

    # --- 1. Redis cache check ---
    cached_answer = await cache.get_cached_answer(body.doc_id, body.question)
    if cached_answer is not None:
        latency_ms = (time.monotonic() - start_time) * 1000
        logger.info("[%s] Returning cached answer in %.1f ms", request_id, latency_ms)
        return QueryResponse(
            answer=cached_answer,
            sources=[],
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

    # --- 2. Load FAISS index ---
    try:
        index, chunk_texts = vector_store.load_index(body.doc_id)
    except FileNotFoundError as exc:
        logger.warning("[%s] doc_id not found: %s", request_id, body.doc_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # --- 3. Embed question ---
    logger.info("[%s] Embedding question", request_id)
    query_vector = await embedder.embed_query(body.question)

    # --- 4. Retrieve top-k chunks ---
    results = vector_store.search(index, chunk_texts, query_vector, settings.TOP_K_RESULTS)
    context_chunks = [text for text, _score in results]
    logger.info("[%s] Retrieved %d context chunk(s)", request_id, len(context_chunks))

    # --- 5. Generate answer ---
    logger.info("[%s] Calling LLM", request_id)
    answer = await llm.generate_answer(context_chunks, body.question)

    # --- 6. Cache answer ---
    await cache.set_cached_answer(
        body.doc_id, body.question, answer, settings.CACHE_TTL_SECONDS
    )

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info("[%s] Query complete in %.1f ms", request_id, latency_ms)

    return QueryResponse(
        answer=answer,
        sources=context_chunks,
        cached=False,
        latency_ms=round(latency_ms, 2),
    )
