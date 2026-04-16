import logging
from typing import Any, Generator

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from config.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

log = logging.getLogger(__name__)

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        log.info("Loading embedding model for the first time: %s", EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # swap to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Embedding model loaded and cached.")
    return _embedding_model


# ── Embed chunks ───────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[Document]) -> tuple[list[Document], Any]:
    if not chunks:
        log.warning("No chunks provided to embed.")
        return [], get_embedding_model()

    # Drop empty chunks
    clean: list[Document] = [c for c in chunks if c.page_content.strip()]
    skipped = len(chunks) - len(clean)
    if skipped:
        log.warning("Dropped %d empty chunk(s).", skipped)

    log.info(
        "Preparing to embed %d chunks with model: %s", len(clean), EMBEDDING_MODEL
    )

    model = get_embedding_model()  # returns cached singleton

    # Smoke-test: embed one chunk to confirm model is healthy
    try:
        sample_vector = model.embed_query(clean[0].page_content[:200])
        log.info(
            "Embedding model OK — vector dim=%d (sample chunk 0)",
            len(sample_vector),
        )
    except Exception as exc:
        log.error("Embedding smoke-test failed: %s", exc)
        raise

    return clean, model


# ── Batching helper ────────────────────────────────────────────────────────────

def batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    test_chunks = [
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "test", "chunk_index": 0},
        ),
        Document(
            page_content="Qdrant is a vector database optimised for similarity search.",
            metadata={"source": "test", "chunk_index": 1},
        ),
    ]
    clean, model = embed_chunks(test_chunks)
    print(f"Ready to embed {len(clean)} chunk(s).")

    # Confirm singleton: second call must NOT reload the model
    model2 = get_embedding_model()
    assert model is model2, "Singleton broken — model was reloaded!"
    print("Singleton check passed: model loaded only once.")