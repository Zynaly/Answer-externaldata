import os
import logging
from langchain_core.vectorstores import VectorStore
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

logger = logging.getLogger(__name__)

def get_reranking_retriever(vector_store: VectorStore) -> ContextualCompressionRetriever:
    """
    Creates a retrieval pipeline:
    1. Base Retriever: Fetches top 15 chunks via vector similarity.
    2. Reranker: Uses Cohere to score and compress down to the absolute best 5 chunks.
    """
    logger.info("Setting up Contextual Compression Retriever with Cohere Reranking...")
    
    # Base retriever fetches a wider net of documents
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    # Cohere reranker
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set.")
        
    compressor = CohereRerank(
        top_n=5, 
        model="rerank-english-v3.0", 
        cohere_api_key=cohere_api_key
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever