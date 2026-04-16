import logging

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import sys
import os

# Add project root (Answer-externaldata) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from config.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from embedder import batched, get_embedding_model

log = logging.getLogger(__name__)

BATCH_SIZE = 64   # chunks per Qdrant upsert call


# ── Collection bootstrap ───────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        log.info("Collection '%s' already exists — appending.", COLLECTION_NAME)
    else:
        log.info(
            "Creating collection '%s' (dim=%d, metric=Cosine).",
            COLLECTION_NAME,
            EMBEDDING_DIM,
        )
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


# ── Save to Qdrant ─────────────────────────────────────────────────────────────

def save_to_vectorstore(
    chunks: list[Document],
    embedding_model: OpenAIEmbeddings,
) -> QdrantVectorStore:
    """
    Embed chunks and upsert them into Qdrant in batches.

    Batching matters because:
      - OpenAI's embedding API has a per-request token limit.
      - Qdrant performs better with moderate-sized upsert payloads.
      - Progress can be logged per batch.

    Returns the connected QdrantVectorStore so callers (runrag.py) can
    immediately use it without reconnecting.
    """
    if not chunks:
        log.error("No chunks to save.")
        raise ValueError("chunks list is empty")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    _ensure_collection(client)

    log.info("Uploading %d chunks to Qdrant in batches of %d ...", len(chunks), BATCH_SIZE)

    vectorstore: QdrantVectorStore | None = None

    for batch_num, batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        log.info("  Batch %d/%d — %d chunks", batch_num, -(-len(chunks)//BATCH_SIZE), len(batch))
        vectorstore = QdrantVectorStore.from_documents(
            documents=batch,
            embedding=embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            collection_name=COLLECTION_NAME,
        )

    log.info("All chunks saved to collection '%s'.", COLLECTION_NAME)
    return vectorstore  # type: ignore[return-value]


# ── Load existing vectorstore (used by retriever.py) ──────────────────────────

def load_vectorstore() -> QdrantVectorStore:
    """
    Connect to an already-populated Qdrant collection.
    Call this in retriever.py and llm.py — do NOT re-ingest just to query.
    """
    log.info("Connecting to existing collection '%s' at %s", COLLECTION_NAME, QDRANT_URL)
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_model(),
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from loaddoc import load_pdfs, semantic_chunk
    from embedder import embed_chunks

    pages  = load_pdfs()
    chunks = semantic_chunk(pages)
    clean, model = embed_chunks(chunks)
    save_to_vectorstore(clean, model)