# import logging

# from langchain_classic.retrievers import ContextualCompressionRetriever
# from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_core.documents import Document

# from config.config import (
#     RERANKER_MODEL,
#     RETRIEVER_FETCH_K,
#     RETRIEVER_FINAL_K,
#     RETRIEVER_SEARCH_TYPE,
# )
# from vectorstore import load_vectorstore
 
# log = logging.getLogger(__name__)


# # ── Stage 1: Vector retriever ──────────────────────────────────────────────────


# # ── Memory ─────────────────────────────────────────────────────────────────────

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory

# _store = {}

# def build_memory(vectorstore=None):
#     """Simple in-memory session store compatible with RunnableWithMessageHistory."""
#     def get_session_history(session_id: str) -> BaseChatMessageHistory:
#         if session_id not in _store:
#             _store[session_id] = ChatMessageHistory()
#         return _store[session_id]
    
#     log.info("Memory ready (in-memory session store)")
#     return get_session_history

# def build_vector_retriever(vectorstore=None):
#     """
#     Wrap the Qdrant vectorstore as a LangChain retriever.

#     search_type options:
#       "similarity"    → pure cosine distance (default, fastest)
#       "mmr"           → Maximal Marginal Relevance — reduces redundancy
#                         by penalising chunks too similar to already-selected ones.
#                         Better diversity; slightly slower.

#     fetch_k > k intentionally: we over-fetch for the reranker to work on.
#     """
#     vs = vectorstore or load_vectorstore()

#     retriever = vs.as_retriever(
#         search_type=RETRIEVER_SEARCH_TYPE,     # "similarity" or "mmr"
#         search_kwargs={
#             "k": RETRIEVER_FETCH_K,            # how many to fetch before reranking
#             **({"fetch_k": RETRIEVER_FETCH_K * 2} if RETRIEVER_SEARCH_TYPE == "mmr" else {}),
#         },
#     )
#     log.info(
#         "Vector retriever ready (search_type=%s, fetch_k=%d)",
#         RETRIEVER_SEARCH_TYPE,
#         RETRIEVER_FETCH_K,
#     )
#     return retriever


# # ── Stage 2: Cross-encoder reranker ────────────────────────────────────────────

# def build_reranker_retriever(base_retriever) -> ContextualCompressionRetriever:
#     """
#     Wrap a base retriever with a cross-encoder reranker.

#     BAAI/bge-reranker-base:
#       - Runs locally (no API cost).
#       - ~568M params, strong on English retrieval benchmarks.
#       - Swap for bge-reranker-large for higher quality at ~2× latency.

#     The ContextualCompressionRetriever calls base_retriever.invoke()
#     first, then passes all results through the CrossEncoderReranker which scores
#     and re-orders them, finally keeping only top_n=RETRIEVER_FINAL_K.
#     """
#     log.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
#     cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
#     compressor    = CrossEncoderReranker(model=cross_encoder, top_n=RETRIEVER_FINAL_K)

#     reranker_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=base_retriever,
#     )
#     log.info(
#         "Reranker ready — fetch_k=%d → rerank → final_k=%d",
#         RETRIEVER_FETCH_K,
#         RETRIEVER_FINAL_K,
#     )
#     return reranker_retriever


# # ── Full retrieval pipeline ────────────────────────────────────────────────────

# def build_retriever(vectorstore=None) -> ContextualCompressionRetriever:
#     """
#     Compose the full two-stage retrieval pipeline.
#     This is the function llm.py imports.
#     """
#     base      = build_vector_retriever(vectorstore)
#     reranked  = build_reranker_retriever(base)
#     return reranked


# # ── Standalone test ────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
#     retriever = build_retriever()
#     query     = "What is the main topic of the documents?"
    
#     # Modern LangChain uses .invoke() instead of .get_relevant_documents()
#     results: list[Document] = retriever.invoke(query)
    
#     print(f"\nQuery: {query}")
#     print(f"Retrieved {len(results)} chunks after reranking:\n")
#     for i, doc in enumerate(results, 1):
#         print(f"[{i}] {doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')}")
#         print(f"     {doc.page_content[:200]}\n")




import logging

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document

from config.config import (
    RERANKER_MODEL,
    RETRIEVER_FETCH_K,
    RETRIEVER_FINAL_K,
    RETRIEVER_SEARCH_TYPE,
)
from vectorstore import load_vectorstore

log = logging.getLogger(__name__)


# ── Singletons ─────────────────────────────────────────────────────────────────
# Each heavy object is created exactly once and reused for every request.
# This eliminates the ~20-25s cold-start on every /chat/stream call.

_vectorstore      = None   # Qdrant connection + embedding model
_vector_retriever = None   # LangChain retriever wrapping the vectorstore
_reranker_model   = None   # HuggingFace cross-encoder weights
_full_retriever   = None   # Final two-stage pipeline (vector → rerank)

# ── In-memory session store ────────────────────────────────────────────────────

_store: dict[str, ChatMessageHistory] = {}


def build_memory():
    """
    Return a get_session_history callable compatible with
    RunnableWithMessageHistory. The _store dict persists for the lifetime
    of the process — no rebuild needed between requests.
    """
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = ChatMessageHistory()
        return _store[session_id]

    log.info("Memory ready (in-memory session store)")
    return get_session_history


# ── Stage 1: Vector retriever ──────────────────────────────────────────────────

def build_vector_retriever(vectorstore=None):
    """
    Wrap the Qdrant vectorstore as a LangChain retriever.
    The result is cached in _vector_retriever so subsequent calls are free.

    search_type options:
      "similarity"  → pure cosine distance (default, fastest)
      "mmr"         → Maximal Marginal Relevance — reduces redundancy,
                      better diversity, slightly slower.
    """
    global _vectorstore, _vector_retriever

    if _vector_retriever is not None and vectorstore is None:
        # Return cached retriever — no model reload, no HuggingFace HTTP calls
        return _vector_retriever

    vs = vectorstore or _vectorstore or load_vectorstore()
    _vectorstore = vs   # cache for reuse after upload refreshes the chain

    retriever = vs.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVER_FETCH_K,
            **(
                {"fetch_k": RETRIEVER_FETCH_K * 2}
                if RETRIEVER_SEARCH_TYPE == "mmr"
                else {}
            ),
        },
    )
    log.info(
        "Vector retriever ready (search_type=%s, fetch_k=%d)",
        RETRIEVER_SEARCH_TYPE,
        RETRIEVER_FETCH_K,
    )

    _vector_retriever = retriever
    return retriever


# ── Stage 2: Cross-encoder reranker ────────────────────────────────────────────

def _get_reranker_model() -> HuggingFaceCrossEncoder:
    """
    Load BAAI/bge-reranker-base once and cache it.
    Swapping to bge-reranker-large gives higher quality at ~2× latency.
    """
    global _reranker_model
    if _reranker_model is None:
        log.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        _reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        log.info("Reranker model loaded and cached.")
    return _reranker_model


def build_reranker_retriever(
    base_retriever,
) -> ContextualCompressionRetriever:
    """
    Wrap a base retriever with a cached cross-encoder reranker.

    The ContextualCompressionRetriever calls base_retriever.invoke() first,
    then scores and re-orders results with the CrossEncoderReranker, keeping
    only top_n=RETRIEVER_FINAL_K.
    """
    cross_encoder = _get_reranker_model()   # cached — no reload
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


# ── Full retrieval pipeline ────────────────────────────────────────────────────

def build_retriever(vectorstore=None) -> ContextualCompressionRetriever:
    """
    Compose the full two-stage retrieval pipeline (vector search → rerank).

    On first call: loads vectorstore + reranker, caches both.
    On subsequent calls with vectorstore=None: returns instantly from cache.
    Pass a fresh vectorstore (e.g. after /upload) to force a rebuild.
    """
    global _full_retriever

    if _full_retriever is not None and vectorstore is None:
        return _full_retriever

    base     = build_vector_retriever(vectorstore)
    reranked = build_reranker_retriever(base)

    _full_retriever = reranked
    return reranked


def invalidate_retriever_cache() -> None:
    """
    Call this after a new document is ingested so the next request
    rebuilds the retriever against the updated vectorstore.
    Only the vectorstore connection is refreshed; the reranker model
    stays cached (it's document-independent).
    """
    global _vectorstore, _vector_retriever, _full_retriever
    _vectorstore      = None
    _vector_retriever = None
    _full_retriever   = None
    log.info("Retriever cache invalidated — will reload vectorstore on next request.")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    retriever = build_retriever()
    query     = "What is the main topic of the documents?"

    results: list[Document] = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks after reranking:\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')}")
        print(f"     {doc.page_content[:200]}\n")

    # Confirm singleton: second call must not reload anything
    retriever2 = build_retriever()
    assert retriever is retriever2, "Singleton broken — retriever was rebuilt!"
    print("Singleton check passed: retriever built only once.")