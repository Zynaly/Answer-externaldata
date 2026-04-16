"""
config.py
=========
Single source of truth for all configuration.
All values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")       
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embeddings ─────────────────────────────────────────────────────────────────
# EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# EMBEDDING_DIM        = int(os.getenv("EMBEDDING_DIM", "1536"))   # text-embedding-3-small = 1536
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384   # IMPORTANT ⚠️
# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# ── Documents & chunking ───────────────────────────────────────────────────────
DOCS_DIR         = os.getenv("DOCS_DIR", "./docs")
BREAKPOINT_TYPE  = os.getenv("BREAKPOINT_TYPE", "percentile")   # percentile | standard_deviation | interquartile
BREAKPOINT_VALUE = float(os.getenv("BREAKPOINT_VALUE", "90"))

# ── Retriever ──────────────────────────────────────────────────────────────────
RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")    # mmr | similarity
RETRIEVER_FETCH_K     = int(os.getenv("RETRIEVER_FETCH_K", "20"))    # candidates before reranking
RETRIEVER_FINAL_K     = int(os.getenv("RETRIEVER_FINAL_K", "5"))     # chunks sent to LLM

# ── Retriever settings ─────────────────────────────────────────────────────────
RERANKER_MODEL       = "BAAI/bge-reranker-base"   # or "bge-reranker-large" for better quality
RETRIEVER_FETCH_K    = 20      # chunks fetched from vectorstore before reranking
RETRIEVER_FINAL_K    = 5       # chunks kept after reranking
RETRIEVER_SEARCH_TYPE = "similarity"   # "similarity" or "mmr"