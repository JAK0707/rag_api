# ============================================================
# FILE: tests/test_upload.py
# PURPOSE: Async integration tests for POST /api/v1/upload
# ============================================================

import io
import os

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


def _minimal_pdf_bytes() -> bytes:
    """Return a minimal valid single-page PDF in raw bytes."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_valid_pdf_returns_200(async_client, monkeypatch):
    """Uploading a valid PDF should return 200 with a doc_id and status=indexed."""

    async def fake_embed(texts):
        return [[0.1] * 1536 for _ in texts]

    def fake_split(path):
        from langchain.schema import Document
        return [Document(page_content="hello world", metadata={})]

    def fake_build(doc_id, vectors, texts):
        pass

    monkeypatch.setattr("app.routers.upload.embedder.embed_texts", fake_embed)
    monkeypatch.setattr("app.routers.upload.chunker.load_and_split", fake_split)
    monkeypatch.setattr("app.routers.upload.vector_store.build_and_save", fake_build)

    pdf_bytes = _minimal_pdf_bytes()
    response = await async_client.post(
        "/api/v1/upload",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert len(data["doc_id"]) == 64  # SHA-256 hex digest
    assert data["status"] == "indexed"
    assert data["chunks_count"] >= 1


@pytest.mark.asyncio
async def test_upload_non_pdf_returns_422(async_client):
    """Uploading a non-PDF file should return 422 Unprocessable Entity."""
    response = await async_client.post(
        "/api/v1/upload",
        files={"file": ("report.txt", io.BytesIO(b"not a pdf"), "text/plain")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_oversized_pdf_returns_413(async_client):
    """Uploading a file larger than 10 MB should return 413 Request Entity Too Large."""
    oversized = b"%PDF-1.4\n" + b"x" * (10 * 1024 * 1024 + 1)
    response = await async_client.post(
        "/api/v1/upload",
        files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
    )
    assert response.status_code == 413
