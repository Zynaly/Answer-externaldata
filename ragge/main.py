
# import logging
# import os
# import sys
# import shutil
# import uuid
# from pathlib import Path
# from typing import AsyncGenerator

# # ── Path setup ─────────────────────────────────────────────────────────────────
# ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)

# # ── FastAPI ────────────────────────────────────────────────────────────────────
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import HTMLResponse, StreamingResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger(__name__)

# DOCS_DIR = Path(ROOT_DIR) / "docs"
# DOCS_DIR.mkdir(exist_ok=True)

# ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}

# app = FastAPI(title="RAG Assistant API")

# # ── Serve index.html ───────────────────────────────────────────────────────────
# @app.get("/", response_class=HTMLResponse)
# async def serve_index():
#     index_path = Path(ROOT_DIR) / "index.html"
#     if not index_path.exists():
#         raise HTTPException(status_code=404, detail="index.html not found")
#     return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# # ── Upload + Ingest ────────────────────────────────────────────────────────────
# @app.post("/upload")
# async def upload_and_ingest(file: UploadFile = File(...)):
#     """
#     Accepts a PDF or Word document, saves it to docs/, then runs the full
#     ingest pipeline (chunk → embed → Qdrant).
#     """
#     suffix = Path(file.filename).suffix.lower()
#     if suffix not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
#         )

#     # Save to docs/
#     safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
#     dest = DOCS_DIR / safe_name
#     with dest.open("wb") as f:
#         shutil.copyfileobj(file.file, f)
#     log.info("Saved uploaded file → %s", dest)

#     # Run ingest
#     try:
#         from loaddoc import load_pdfs, semantic_chunk
#         from embedder import embed_chunks
#         from vectorstore import save_to_vectorstore

#         pages = load_pdfs(str(DOCS_DIR))
#         if not pages:
#             raise ValueError("No pages loaded from docs/")

#         chunks = semantic_chunk(pages)
#         if not chunks:
#             raise ValueError("No chunks produced from documents")

#         clean_chunks, embedding_model = embed_chunks(chunks)
#         save_to_vectorstore(clean_chunks, embedding_model)

#         log.info("Ingest complete — %d chunks in Qdrant", len(clean_chunks))
#         return {
#             "status": "ok",
#             "filename": file.filename,
#             "saved_as": safe_name,
#             "chunks_indexed": len(clean_chunks),
#         }
#     except Exception as exc:
#         log.exception("Ingest failed: %s", exc)
#         # Remove the saved file so a retry is clean
#         dest.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}")


# # ── Chat ───────────────────────────────────────────────────────────────────────
# class ChatRequest(BaseModel):
#     question: str
#     session_id: str = "default-session"


# @app.post("/chat")
# async def chat(req: ChatRequest):
#     """
#     Non-streaming chat endpoint. Returns the full answer as JSON.
#     """
#     if not req.question.strip():
#         raise HTTPException(status_code=400, detail="Question cannot be empty")

#     try:
#         from agent.llm import build_rag_chain

#         chain = build_rag_chain(vectorstore=None)
#         session_config = {"configurable": {"session_id": req.session_id}}

#         answer_tokens = []
#         for token in chain.stream({"question": req.question}, config=session_config):
#             answer_tokens.append(token)

#         return {"answer": "".join(answer_tokens)}
#     except Exception as exc:
#         log.exception("Chat error: %s", exc)
#         raise HTTPException(status_code=500, detail=f"Chat error: {exc}")


# @app.post("/chat/stream")
# async def chat_stream(req: ChatRequest):
#     """
#     Streaming chat endpoint — tokens are sent as Server-Sent Events.
#     """
#     if not req.question.strip():
#         raise HTTPException(status_code=400, detail="Question cannot be empty")

#     async def token_generator() -> AsyncGenerator[str, None]:
#         try:
#             from agent.llm import build_rag_chain

#             chain = build_rag_chain(vectorstore=None)
#             session_config = {"configurable": {"session_id": req.session_id}}

#             for token in chain.stream({"question": req.question}, config=session_config):
#                 # SSE format
#                 yield f"data: {token}\n\n"

#             yield "data: [DONE]\n\n"
#         except Exception as exc:
#             log.exception("Stream error: %s", exc)
#             yield f"data: [ERROR] {exc}\n\n"

#     return StreamingResponse(token_generator(), media_type="text/event-stream")


# # ── Health ─────────────────────────────────────────────────────────────────────
# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# # ── Entry point ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)




import logging
import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional
import sys
import os

# Add project root (Answer-externaldata) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DOCS_DIR = Path(ROOT_DIR) / "docs"
DOCS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}

app = FastAPI(title="RAG Assistant API")


# ── Singleton RAG chain ────────────────────────────────────────────────────────
# Built once at startup. Reused for every /chat and /chat/stream request.
# Re-built after /upload so new documents are immediately queryable.

_rag_chain = None


def get_rag_chain():
    """Return the cached RAG chain, raising clearly if startup failed."""
    if _rag_chain is None:
        raise RuntimeError(
            "RAG chain not initialised. Check startup logs for errors."
        )
    return _rag_chain


def _build_and_cache_chain() -> None:
    """(Re-)build the RAG chain and store it in the module-level singleton."""
    global _rag_chain
    from agent.llm import build_rag_chain
    _rag_chain = build_rag_chain(vectorstore=None)
    log.info("RAG chain cached and ready.")


# ── Application lifecycle ──────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """
    Pre-load the embedding model, reranker, and RAG chain at server startup.
    This front-loads the ~20-25s cold-start so the first real request is fast.
    """
    log.info("=== Startup: pre-loading RAG chain (embedding model + reranker) ===")
    try:
        _build_and_cache_chain()
        log.info("=== Startup complete — ready to serve requests ===")
    except Exception:
        # Log but don't crash the server; /health will report unhealthy.
        log.exception("Startup RAG chain build failed.")


# ── Serve index.html ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(ROOT_DIR) / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ── Upload + Ingest ────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Accept a PDF or Word document, ingest it into Qdrant, then rebuild
    the RAG chain so new content is immediately queryable.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save to docs/
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest = DOCS_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info("Saved uploaded file → %s", dest)

    try:
        from loaddoc import load_pdfs, semantic_chunk
        from embedder import embed_chunks
        from vectorstore import save_to_vectorstore

        pages = load_pdfs(str(DOCS_DIR))
        if not pages:
            raise ValueError("No pages loaded from docs/")

        chunks = semantic_chunk(pages)
        if not chunks:
            raise ValueError("No chunks produced from documents")

        clean_chunks, embedding_model = embed_chunks(chunks)
        save_to_vectorstore(clean_chunks, embedding_model)
        log.info("Ingest complete — %d chunks in Qdrant", len(clean_chunks))

        # Invalidate stale retriever cache so the rebuild picks up new vectors
        from retrieving import invalidate_retriever_cache
        invalidate_retriever_cache()

        # Rebuild the RAG chain against the updated collection
        log.info("Rebuilding RAG chain after upload...")
        _build_and_cache_chain()

        return {
            "status": "ok",
            "filename": file.filename,
            "saved_as": safe_name,
            "chunks_indexed": len(clean_chunks),
        }

    except Exception as exc:
        log.exception("Ingest failed: %s", exc)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}")


# ── Chat (non-streaming) ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default-session"


@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint. Returns the full answer as JSON."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        chain          = get_rag_chain()
        session_config = {"configurable": {"session_id": req.session_id}}

        answer_tokens = []
        for token in chain.stream({"question": req.question}, config=session_config):
            answer_tokens.append(token)

        return {"answer": "".join(answer_tokens)}

    except Exception as exc:
        log.exception("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")


# ── Chat (streaming) ───────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint — tokens sent as Server-Sent Events.
    Uses the pre-built singleton chain; no model reloading per request.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            chain          = get_rag_chain()
            session_config = {"configurable": {"session_id": req.session_id}}

            for token in chain.stream(
                {"question": req.question}, config=session_config
            ):
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            log.exception("Stream error: %s", exc)
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if _rag_chain is not None else "degraded",
        "chain_ready": _rag_chain is not None,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)