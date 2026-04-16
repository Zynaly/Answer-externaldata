import argparse
import logging
import sys
import os
 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Ingest pipeline ────────────────────────────────────────────────────────────

def run_ingest(docs_dir: str) -> None:
    """Full ingestion: PDF → semantic chunks → embeddings → Qdrant."""
    from loaddoc import load_pdfs, semantic_chunk
    from embedder import embed_chunks
    from vectorstore import save_to_vectorstore

    log.info("=== INGEST MODE ===")

    pages = load_pdfs(docs_dir)
    if not pages:
        log.error("No pages loaded. Add PDF files to %s and retry.", docs_dir)
        sys.exit(1)

    chunks = semantic_chunk(pages)
    if not chunks:
        log.error("No chunks produced. Check your documents.")
        sys.exit(1)

    clean_chunks, embedding_model = embed_chunks(chunks)
    vectorstore = save_to_vectorstore(clean_chunks, embedding_model)

    log.info("Ingest complete. %d chunks are now in Qdrant.", len(clean_chunks))
    return vectorstore


# ── Chat loop ──────────────────────────────────────────────────────────────────

def run_chat(vectorstore=None, one_shot_question: str | None = None) -> None:
    """
    Interactive conversation loop.
    Short-term and long-term memories are active for the full session.
    Type 'exit' or 'quit' to end the session.
    """
    from agent.llm import build_rag_chain

    log.info("=== CHAT MODE ===")
    chain = build_rag_chain(vectorstore)

    # ── Session config required by RunnableWithMessageHistory ─────────────────
    session_config = {"configurable": {"session_id": "main-session"}}

    if one_shot_question:
        # Non-interactive mode: answer one question and exit
        print(f"\nQ: {one_shot_question}\nA: ", end="", flush=True)
        for token in chain.stream({"question": one_shot_question}, config=session_config):
            print(token, end="", flush=True)
        print("\n")
        return

    # Interactive session
    print("\n" + "=" * 60)
    print("  RAG Assistant — type 'exit' to quit")
    print("=" * 60)
    print("  Short-term memory: last 5 turns")
    print("  Long-term memory : Qdrant-backed semantic search")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        print("\nAssistant: ", end="", flush=True)
        try:
            for token in chain.stream({"question": question}, config=session_config):
                print(token, end="", flush=True)
        except Exception as exc:
            log.exception("Chain error: %s", exc)
        print("\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG system — ingest PDFs and chat with your documents"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Load PDFs from docs/, chunk, embed, and save to Qdrant",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive Q&A session",
    )
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Path to folder containing PDF files (default: ./docs)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Ask a single question (non-interactive) and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.ingest and not args.chat and not args.question:
        print("Specify --ingest, --chat, or --question. Use --help for options.")
        sys.exit(0)

    vectorstore = None

    if args.ingest:
        vectorstore = run_ingest(args.docs_dir)

    if args.chat or args.question:
        run_chat(vectorstore=vectorstore, one_shot_question=args.question)
