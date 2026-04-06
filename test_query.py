# ============================================================
# FILE: tests/test_query.py
# PURPOSE: Async integration tests for POST /api/v1/query
# ============================================================

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_client():
    """Provide an async HTTPX client wired to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


_FAKE_DOC_ID = "a" * 64  # 64-char hex string like a real SHA-256


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_valid_doc_returns_answer(async_client, monkeypatch):
    """A valid query against an indexed doc should return 200 with a non-empty answer."""
    import faiss
    import numpy as np

    async def fake_get_cached(doc_id, question):
        return None

    async def fake_set_cached(doc_id, question, answer, ttl):
        pass

    def fake_load_index(doc_id):
        idx = faiss.IndexFlatL2(1536)
        idx.add(np.array([[0.1] * 1536], dtype=np.float32))
        return idx, ["This is a sample chunk about the topic."]

    async def fake_embed_query(text):
        return [0.1] * 1536

    def fake_search(index, texts, vec, top_k):
        return [(texts[0], 0.01)]

    async def fake_generate(chunks, question):
        return "The answer is 42."

    monkeypatch.setattr("app.routers.query.cache.get_cached_answer", fake_get_cached)
    monkeypatch.setattr("app.routers.query.cache.set_cached_answer", fake_set_cached)
    monkeypatch.setattr("app.routers.query.vector_store.load_index", fake_load_index)
    monkeypatch.setattr("app.routers.query.embedder.embed_query", fake_embed_query)
    monkeypatch.setattr("app.routers.query.vector_store.search", fake_search)
    monkeypatch.setattr("app.routers.query.llm.generate_answer", fake_generate)

    response = await async_client.post(
        "/api/v1/query",
        json={"doc_id": _FAKE_DOC_ID, "question": "What is the answer?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""
    assert data["cached"] is False
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_query_invalid_doc_returns_404(async_client, monkeypatch):
    """Querying a doc_id that has no FAISS index should return 404."""

    async def fake_get_cached(doc_id, question):
        return None

    def fake_load_index(doc_id):
        raise FileNotFoundError(f"FAISS index not found for doc_id='{doc_id}'")

    monkeypatch.setattr("app.routers.query.cache.get_cached_answer", fake_get_cached)
    monkeypatch.setattr("app.routers.query.vector_store.load_index", fake_load_index)

    response = await async_client.post(
        "/api/v1/query",
        json={"doc_id": "nonexistent" * 6, "question": "Does this exist?"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_query_cache_hit_on_second_request(async_client, monkeypatch):
    """The second identical query should return cached=True."""
    answer_store: dict = {}

    async def fake_get_cached(doc_id, question):
        key = f"{doc_id}:{question}"
        return answer_store.get(key)

    async def fake_set_cached(doc_id, question, answer, ttl):
        key = f"{doc_id}:{question}"
        answer_store[key] = answer

    import faiss
    import numpy as np

    def fake_load_index(doc_id):
        idx = faiss.IndexFlatL2(1536)
        idx.add(np.array([[0.1] * 1536], dtype=np.float32))
        return idx, ["Sample chunk text."]

    async def fake_embed_query(text):
        return [0.1] * 1536

    def fake_search(index, texts, vec, top_k):
        return [(texts[0], 0.01)]

    async def fake_generate(chunks, question):
        return "Cached answer content."

    monkeypatch.setattr("app.routers.query.cache.get_cached_answer", fake_get_cached)
    monkeypatch.setattr("app.routers.query.cache.set_cached_answer", fake_set_cached)
    monkeypatch.setattr("app.routers.query.vector_store.load_index", fake_load_index)
    monkeypatch.setattr("app.routers.query.embedder.embed_query", fake_embed_query)
    monkeypatch.setattr("app.routers.query.vector_store.search", fake_search)
    monkeypatch.setattr("app.routers.query.llm.generate_answer", fake_generate)

    payload = {"doc_id": _FAKE_DOC_ID, "question": "Tell me about the doc."}

    first = await async_client.post("/api/v1/query", json=payload)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = await async_client.post("/api/v1/query", json=payload)
    assert second.status_code == 200
    assert second.json()["cached"] is True
