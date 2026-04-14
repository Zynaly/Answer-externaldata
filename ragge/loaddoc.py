import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from embedder import get_embedding_model

logger = logging.getLogger(__name__)

def load_and_chunk_documents(file_path: str) -> List[Document]:
    """
    Loads a document and applies Semantic Chunking using OpenAI embeddings.
    """
    logger.info(f"Loading document from {file_path}")
    
    # 1. Load Document
    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise

    # 2. Semantic Chunking
    logger.info("Starting semantic chunking. This may take a moment as it calculates sentence embeddings...")
    embedding_model = get_embedding_model()
    
    # Percentile threshold: determines how distinct two sentences must be to split them.
    text_splitter = SemanticChunker(
        embedding_model, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} semantic chunks from {len(docs)} pages.")
    
    return chunks