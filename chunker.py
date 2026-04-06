# ============================================================
# FILE: app/services/chunker.py
# PURPOSE: PDF loading and recursive text splitting via LangChain
# ============================================================

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_split(pdf_path: str) -> list[Document]:
    """Load a PDF from disk and split it into overlapping text chunks.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects, one per chunk.
    """
    logger.info("Loading PDF from path=%s", pdf_path)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info("Loaded %d page(s) from %s", len(pages), pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    logger.info("Split into %d chunk(s) (chunk_size=%d, overlap=%d)",
                len(chunks), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    return chunks
