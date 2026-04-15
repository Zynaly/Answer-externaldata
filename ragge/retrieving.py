"""
retriever.py
============
Responsibility: Given a user query, retrieve the most relevant chunks
from Qdrant and re-rank them for precision before passing to the LLM.

Two-stage retrieval:
  Stage 1 — Vector retriever  (recall)
    Fast approximate nearest-neighbour search in Qdrant.
    Returns top-k candidates by cosine similarity.
    Optimised for recall: we over-fetch (fetch_k > k) on purpose.

  Stage 2 — Cross-encoder reranker  (precision)
    A small cross-encoder model (BAAI/bge-reranker-base) scores every
    (query, chunk) pair together — capturing query-document interaction
    that bi-encoder embeddings miss.
    We then keep only the top final_k results.

Why rerank?
  Embedding similarity is fast but shallow. "Apple revenue" and
  "Apple fruit nutrition" have similar embeddings but very different
  relevance to a financial question. The cross-encoder re-reads both
  query and document together and catches this.

Memory integration:
  Short-term memory — ConversationBufferWindowMemory (last N turns).
    Keeps recent context so follow-up questions like "and what about Q3?"
    resolve correctly without re-stating the topic.

  Long-term memory — VectorStoreRetrieverMemory backed by Qdrant.
    Stores every (input, output) exchange as a vector. On each new query
    the K most semantically similar past exchanges are retrieved and
    injected into the prompt, giving the LLM access to relevant history
    even from sessions days ago.
"""

import logging

from langchain.memory import (
    CombinedMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

from config import (
    RERANKER_MODEL,
    RETRIEVER_FETCH_K,
    RETRIEVER_FINAL_K,
    RETRIEVER_SEARCH_TYPE,
)
from vectorstore import load_vectorstore

log = logging.getLogger(__name__)


# ── Stage 1: Vector retriever ──────────────────────────────────────────────────

def build_vector_retriever(vectorstore=None):
    """
    Wrap the Qdrant vectorstore as a LangChain retriever.

    search_type options:
      "similarity"    → pure cosine distance (default, fastest)
      "mmr"           → Maximal Marginal Relevance — reduces redundancy
                        by penalising chunks too similar to already-selected ones.
                        Better diversity; slightly slower.

    fetch_k > k intentionally: we over-fetch for the reranker to work on.
    """
    vs = vectorstore or load_vectorstore()

    retriever = vs.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,     # "similarity" or "mmr"
        search_kwargs={
            "k": RETRIEVER_FETCH_K,            # how many to fetch before reranking
            **({"fetch_k": RETRIEVER_FETCH_K * 2} if RETRIEVER_SEARCH_TYPE == "mmr" else {}),
        },
    )
    log.info(
        "Vector retriever ready (search_type=%s, fetch_k=%d)",
        RETRIEVER_SEARCH_TYPE,
        RETRIEVER_FETCH_K,
    )
    return retriever


# ── Stage 2: Cross-encoder reranker ────────────────────────────────────────────

def build_reranker_retriever(base_retriever) -> ContextualCompressionRetriever:
    """
    Wrap a base retriever with a cross-encoder reranker.

    BAAI/bge-reranker-base:
      - Runs locally (no API cost).
      - ~568M params, strong on English retrieval benchmarks.
      - Swap for bge-reranker-large for higher quality at ~2× latency.

    The ContextualCompressionRetriever calls base_retriever.get_relevant_documents()
    first, then passes all results through the CrossEncoderReranker which scores
    and re-orders them, finally keeping only top_n=RETRIEVER_FINAL_K.
    """
    log.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor    = CrossEncoderReranker(model=cross_encoder, top_n=RETRIEVER_FINAL_K)

    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    log.info(
        "Reranker ready — fetch_k=%d → rerank → final_k=%d",
        RETRIEVER_FETCH_K,
        RETRIEVER_FINAL_K,
    )
    return reranker_retriever


# ── Short-term memory ──────────────────────────────────────────────────────────

def build_short_term_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """
    Sliding window over the last k conversation turns.

    k=5 means the LLM sees the last 5 human+AI exchanges.
    Prevents the context window from growing unboundedly in long sessions.

    memory_key must match the variable name used in the prompt template
    (see prompts.py → {short_term_history}).
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="short_term_history",
        input_key="question",
        return_messages=True,
    )


# ── Long-term memory ───────────────────────────────────────────────────────────

def build_long_term_memory(vectorstore=None) -> VectorStoreRetrieverMemory:
    """
    Semantic long-term memory backed by Qdrant.

    On every turn:
      SAVE  → the (human input, AI output) pair is embedded and stored in Qdrant
              under a dedicated collection key.
      LOAD  → before generating a response, the K most semantically similar
              past exchanges are retrieved and injected into the prompt.

    This means the model can "remember" relevant facts from previous sessions
    without hitting a context-window limit.

    memory_key must match {long_term_history} in prompts.py.
    """
    vs = vectorstore or load_vectorstore()
    memory_retriever = vs.as_retriever(search_kwargs={"k": 3})
    return VectorStoreRetrieverMemory(
        retriever=memory_retriever,
        memory_key="long_term_history",
        input_key="question",
    )


# ── Combined memory ────────────────────────────────────────────────────────────

def build_memory(vectorstore=None) -> CombinedMemory:
    """
    Merge short-term and long-term memory into one object.

    CombinedMemory passes both memory_keys to the prompt:
      {short_term_history}  — last N raw messages
      {long_term_history}   — semantically relevant past exchanges
    """
    short_mem = build_short_term_memory(k=5)
    long_mem  = build_long_term_memory(vectorstore)
    combined  = CombinedMemory(memories=[short_mem, long_mem])
    log.info("Memory ready: short-term (window=5) + long-term (Qdrant-backed)")
    return combined


# ── Full retrieval pipeline ────────────────────────────────────────────────────

def build_retriever(vectorstore=None) -> ContextualCompressionRetriever:
    """
    Compose the full two-stage retrieval pipeline.
    This is the function llm.py imports.
    """
    base      = build_vector_retriever(vectorstore)
    reranked  = build_reranker_retriever(base)
    return reranked


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    retriever = build_retriever()
    query     = "What is the main topic of the documents?"
    results: list[Document] = retriever.get_relevant_documents(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks after reranking:\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')}")
        print(f"     {doc.page_content[:200]}\n")