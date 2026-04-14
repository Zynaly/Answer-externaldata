import logging
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from embedder import get_embedding_model

logger = logging.getLogger(__name__)

def save_chunks_to_qdrant(
    chunks: List[Document], 
    collection_name: str = "production_rag",
    location: str = ":memory:" # Use a path like "./qdrant_data" for persistence
) -> QdrantVectorStore:
    """
    Saves embedded semantic chunks to a Qdrant Vector Database.
    """
    logger.info(f"Connecting to Qdrant at {location}...")
    client = QdrantClient(location=location)
    embedding_model = get_embedding_model()

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        logger.info(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    logger.info(f"Indexing {len(chunks)} chunks into Qdrant...")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    
    vector_store.add_documents(chunks)
    logger.info("Successfully saved chunks to Qdrant.")
    
    return vector_store