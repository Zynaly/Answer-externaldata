"""
loader.py
=========
Responsibility: Load PDF files from docs/ folder and split them into
semantically coherent chunks using SemanticChunker.

Why SemanticChunker over RecursiveCharacterTextSplitter?
  - Fixed-size splitting cuts mid-sentence and mid-thought.
  - SemanticChunker embeds every sentence, measures cosine distance
    between consecutive sentences, and only cuts where the topic actually
    changes. Result: chunks that are self-contained units of meaning.
"""

import hashlib
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

import sys
import os

# Add project root (Answer-externaldata) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config.config import (
    BREAKPOINT_TYPE,
    BREAKPOINT_VALUE,
    DOCS_DIR,
    OPENAI_API_KEY,
)

log = logging.getLogger(__name__)


# ── PDF Loader ─────────────────────────────────────────────────────────────────

def load_pdfs(docs_dir: str = DOCS_DIR) -> list[Document]:
    """
    Recursively load every PDF from docs_dir.
    Each page becomes a Document with enriched metadata:
      - source      : full file path
      - filename    : file name only
      - page        : page number (0-indexed)
      - content_hash: MD5 of page text (used to skip unchanged pages on re-ingest)
    """
    folder = Path(docs_dir)
    if not folder.exists():
        log.error("Docs folder not found: %s", folder.resolve())
        sys.exit(1)

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s", folder.resolve())
        return []

    log.info("Found %d PDF(s) in %s", len(pdf_files), folder.resolve())

    all_pages: list[Document] = []

    for pdf_path in pdf_files:
        log.info("  Loading: %s", pdf_path.name)
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()                      # one Document per page

            for page in pages:
                page.metadata["source"]        = str(pdf_path)
                page.metadata["filename"]      = pdf_path.name
                page.metadata["content_hash"]  = _md5(page.page_content)

            all_pages.extend(pages)
            log.info("    %d page(s)", len(pages))

        except Exception as exc:
            log.warning("    Skipping %s — %s", pdf_path.name, exc)

    log.info("Total pages loaded: %d", len(all_pages))
    return all_pages


# ── Semantic Chunker ───────────────────────────────────────────────────────────

def semantic_chunk(pages: list[Document]) -> list[Document]:
    """
    Convert raw pages into semantic chunks.

    Algorithm (SemanticChunker):
      1. Split text into sentences.
      2. Embed each sentence with OpenAI embeddings.
      3. Compute cosine distance between consecutive sentence embeddings.
      4. Insert a chunk boundary wherever distance exceeds BREAKPOINT_VALUE
         at the BREAKPOINT_TYPE percentile / std-dev / IQR level.

    Each chunk inherits the source metadata from its parent page and gets
    two extra fields:
      - chunk_index : sequential index across all chunks
      - chunk_hash  : MD5 of chunk text (for deduplication)
    """
    if not pages:
        log.warning("No pages to chunk.")
        return []

    log.info(
        "Semantic chunking — breakpoint_type=%s, threshold=%.0f",
        BREAKPOINT_TYPE,
        BREAKPOINT_VALUE,
    )

    # OpenAI embeddings are used here purely for semantic similarity during
    # splitting. The same model is reused in embedder.py for storage.
    splitter_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    chunker = SemanticChunker(
        embeddings=splitter_embeddings,
        breakpoint_threshold_type=BREAKPOINT_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_VALUE,
    )

    chunks: list[Document] = chunker.split_documents(pages)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_hash"]  = _md5(chunk.page_content)

    log.info(
        "Produced %d chunks from %d pages (avg %.1f chunks/page)",
        len(chunks),
        len(pages),
        len(chunks) / max(len(pages), 1),
    )
    return chunks


# ── Helper ─────────────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    pages  = load_pdfs()
    chunks = semantic_chunk(pages)
    print(f"\nSample chunk (index 0):\n{chunks[0].page_content[:400] if chunks else 'no chunks'}")
    print(f"\nMetadata: {chunks[0].metadata if chunks else {}}")