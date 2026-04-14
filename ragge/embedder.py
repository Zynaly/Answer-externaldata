import os
import logging
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the modern OpenAI embedding model.
    Using text-embedding-3-small as it is the current production standard for cost/performance.
    """
    try:
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("OpenAI Embeddings initialized successfully.")
        return embedder
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise