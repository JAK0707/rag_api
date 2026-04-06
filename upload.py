# ============================================================
# FILE: app/routers/upload.py
# PURPOSE: POST /api/v1/upload — PDF ingestion, chunking, and FAISS indexing
# ============================================================

import hashlib
import tempfile
import uuid

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.models import UploadResponse
from app.services import chunker, embedder, vector_store
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF upload, chunk it, embed it, and persist a FAISS index.

    Args:
        request: FastAPI request (used to derive a request ID for logging).
        file:    Multipart PDF file upload.

    Returns:
        UploadResponse containing doc_id, chunk count, and status.

    Raises:
        HTTPException 400: If the file is not a PDF.
        HTTPException 413: If the file exceeds the 10 MB limit.
    """
    request_id = str(uuid.uuid4())[:8]
    log = logger  # alias for per-request prefix
    log.info("[%s] Received upload: filename=%s, content_type=%s",
             request_id, file.filename, file.content_type)

    # --- Validate content type ---
    if file.content_type not in ("application/pdf",) and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        log.warning("[%s] Rejected non-PDF upload: %s", request_id, file.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    # --- Read file bytes and size-check ---
    contents = await file.read()
    if len(contents) > _MAX_FILE_SIZE:
        log.warning("[%s] Rejected oversized file: %d bytes", request_id, len(contents))
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the 10 MB limit ({len(contents)} bytes received).",
        )

    # --- Compute deterministic doc_id ---
    doc_id = hashlib.sha256(contents).hexdigest()
    log.info("[%s] doc_id=%s (%d bytes)", request_id, doc_id, len(contents))

    # --- Persist to temp file for LangChain loader ---
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    async with aiofiles.open(tmp_path, "wb") as afh:
        await afh.write(contents)
    log.info("[%s] Saved temp file: %s", request_id, tmp_path)

    # --- Chunk ---
    log.info("[%s] Starting chunking", request_id)
    chunks = chunker.load_and_split(tmp_path)
    chunk_texts = [doc.page_content for doc in chunks]
    log.info("[%s] Produced %d chunks", request_id, len(chunk_texts))

    # --- Embed ---
    log.info("[%s] Starting embedding", request_id)
    vectors = await embedder.embed_texts(chunk_texts)
    log.info("[%s] Produced %d embedding vectors", request_id, len(vectors))

    # --- Build + persist FAISS index ---
    log.info("[%s] Building FAISS index", request_id)
    vector_store.build_and_save(doc_id, vectors, chunk_texts)
    log.info("[%s] FAISS index persisted for doc_id=%s", request_id, doc_id)

    return UploadResponse(doc_id=doc_id, chunks_count=len(chunk_texts), status="indexed")
