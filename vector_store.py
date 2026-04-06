# ============================================================
# FILE: app/services/vector_store.py
# PURPOSE: FAISS index build/save/load/search with chunk text persistence
# ============================================================

import json
import os

import faiss
import numpy as np

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _index_path(doc_id: str) -> str:
    """Return the filesystem path for a FAISS index file."""
    return os.path.join(settings.FAISS_INDEX_DIR, f"{doc_id}.index")


def _chunks_path(doc_id: str) -> str:
    """Return the filesystem path for a chunk-text JSON file."""
    return os.path.join(settings.FAISS_INDEX_DIR, f"{doc_id}.chunks.json")


def build_and_save(doc_id: str, embeddings: list[list[float]], texts: list[str]) -> None:
    """Build a FAISS IndexFlatL2 from embeddings and persist it alongside chunk texts.

    Args:
        doc_id:     Unique document identifier (used as filename stem).
        embeddings: List of float vectors, one per chunk.
        texts:      Raw chunk texts corresponding to each embedding.
    """
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)

    vectors = np.array(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, _index_path(doc_id))
    logger.info("FAISS index saved to %s (%d vectors, dim=%d)", _index_path(doc_id), len(texts), dimension)

    with open(_chunks_path(doc_id), "w", encoding="utf-8") as fh:
        json.dump(texts, fh, ensure_ascii=False, indent=2)
    logger.info("Chunk texts saved to %s", _chunks_path(doc_id))


def load_index(doc_id: str) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Load a persisted FAISS index and its associated chunk texts from disk.

    Args:
        doc_id: Document identifier to look up.

    Returns:
        Tuple of (faiss index, list of chunk text strings).

    Raises:
        FileNotFoundError: If the index or chunk file for doc_id does not exist.
    """
    idx_path = _index_path(doc_id)
    chk_path = _chunks_path(doc_id)

    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"FAISS index not found for doc_id='{doc_id}'. "
            f"Expected path: {idx_path}"
        )
    if not os.path.exists(chk_path):
        raise FileNotFoundError(
            f"Chunk text file not found for doc_id='{doc_id}'. "
            f"Expected path: {chk_path}"
        )

    index = faiss.read_index(idx_path)
    with open(chk_path, "r", encoding="utf-8") as fh:
        texts = json.load(fh)

    logger.info("Loaded FAISS index for doc_id=%s (%d vectors)", doc_id, index.ntotal)
    return index, texts


def search(
    index: faiss.IndexFlatL2,
    texts: list[str],
    query_vector: list[float],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return the top-k most similar chunks to the query vector.

    Args:
        index:        Loaded FAISS index.
        texts:        Chunk texts aligned with index vectors.
        query_vector: Embedding of the query string.
        top_k:        Number of results to retrieve.

    Returns:
        List of (chunk_text, l2_distance) pairs ordered by similarity.
    """
    query = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query, top_k)

    results: list[tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 when fewer results than top_k exist
        results.append((texts[idx], float(dist)))

    logger.info("FAISS search returned %d result(s) (top_k=%d)", len(results), top_k)
    return results
