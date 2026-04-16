import logging
from typing import Generator

from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
import os

# Add project root (Answer-externaldata) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from config.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

log = logging.getLogger(__name__)


# ── Embeddings model (singleton, reused across calls) ──────────────────────────

# def get_embedding_model() -> OpenAIEmbeddings:
#     return OpenAIEmbeddings(
#         model=EMBEDDING_MODEL,
#         openai_api_key=OPENAI_API_KEY,
#         chunk_size=EMBEDDING_BATCH_SIZE,
#     )
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # or "cuda" if GPU
        encode_kwargs={"normalize_embeddings": True},
    )

# ── Embed chunks ───────────────────────────────────────────────────────────────
from typing import Any
def embed_chunks(chunks: list[Document]) -> tuple[list[Document], Any]:

    if not chunks:
        log.warning("No chunks provided to embed.")
        return [], get_embedding_model()

    # Filter empty chunks
    clean: list[Document] = [
        c for c in chunks if c.page_content.strip()
    ]
    skipped = len(chunks) - len(clean)
    if skipped:
        log.warning("Dropped %d empty chunk(s).", skipped)

    log.info("Preparing to embed %d chunks with model: %s", len(clean), EMBEDDING_MODEL)

    # Smoke-test: embed one chunk to confirm API key + model work
    model = get_embedding_model()
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

def batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive fixed-size batches."""
    for i in range(0, len(items), size):
        yield items[i : i + size]

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Quick test with a synthetic document
    from langchain_core.documents import Document
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