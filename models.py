# ============================================================
# FILE: app/models.py
# PURPOSE: Pydantic v2 request/response schemas for all endpoints
# ============================================================

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload and indexing."""

    doc_id: str = Field(..., description="SHA-256 hash of uploaded file contents")
    chunks_count: int = Field(..., description="Number of text chunks indexed")
    status: str = Field(default="indexed")


class QueryRequest(BaseModel):
    """Request body for the Q&A endpoint."""

    doc_id: str = Field(..., description="Document ID returned from /upload")
    question: str = Field(..., min_length=1, description="Question to ask about the document")


class QueryResponse(BaseModel):
    """Response returned after answering a question."""

    answer: str = Field(..., description="Generated answer from the LLM")
    sources: list[str] = Field(..., description="Relevant chunk excerpts used as context")
    cached: bool = Field(..., description="Whether the answer was served from Redis cache")
    latency_ms: float = Field(..., description="End-to-end request latency in milliseconds")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = Field(default="ok")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the health check")
