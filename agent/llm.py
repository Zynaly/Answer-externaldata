import logging
import os
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    GROQ_API_KEY,
)
from agent.prompts import get_condense_prompt, get_rag_prompt
from ragge.retrieving import build_memory, build_retriever
from ragge.vectorstore import load_vectorstore

log = logging.getLogger(__name__)


# ── Format retrieved docs ──────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found in the documents."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[{i}] Source: {source}, Page: {page}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Chat model ─────────────────────────────────────────────────────────────────

def get_chat_model(temperature: float = LLM_TEMPERATURE) -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL, # Make sure this is now set to something like "llama3-8b-8192"
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
        groq_api_key=GROQ_API_KEY, # Or remove this line entirely if you set the API key in your terminal!
        streaming=True,
    )

# ── Build the full RAG chain ───────────────────────────────────────────────────

def build_rag_chain(vectorstore=None):
    """
    Modern LCEL RAG chain using RunnableWithMessageHistory.

    Call with:
        config = {"configurable": {"session_id": "any-string"}}
        chain.invoke({"question": "your question"}, config=config)
        chain.stream({"question": "your question"}, config=config)
    """
    vs                  = vectorstore or load_vectorstore()
    retriever           = build_retriever(vs)
    get_session_history = build_memory()        # returns a callable (session_id -> history)
    llm                 = get_chat_model()
    rag_prompt          = get_rag_prompt()
    condense_prompt     = get_condense_prompt()

    condense_chain = (
        condense_prompt
        | get_chat_model(temperature=0.0)
        | StrOutputParser()
    )

    # ── Core stateless chain ───────────────────────────────────────────────────
    # RunnableWithMessageHistory injects "history" automatically from the session store.
    # We just need to use it here.

    def build_inputs(inputs: dict) -> dict:
        question = inputs.get("question", "")
        history  = inputs.get("history", [])   # injected by RunnableWithMessageHistory

        # Condense follow-up questions using history
        if history:
            try:
                condensed = condense_chain.invoke({
                    "question": question,
                    "short_term_history": history,
                })
            except Exception as e:
                log.warning("Condense chain failed (%s), using raw question", e)
                condensed = question
        else:
            condensed = question

        # Retrieve context using the (possibly condensed) question
        docs    = retriever.invoke(condensed)
        context = _format_docs(docs)

        return {
            "question":           question,
            "context":            context,
            "short_term_history": history,
            "long_term_history":  "",   # extend here if you add vector memory later
        }

    core_chain = (
        RunnableLambda(build_inputs)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # ── Wrap with automatic message history ───────────────────────────────────
    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history,            # callable: session_id -> ChatMessageHistory
        input_messages_key="question",  # which input key is the user message
        history_messages_key="history", # which key receives the injected history
    )

    log.info("RAG chain ready (model=%s, streaming=True)", LLM_MODEL)
    return chain_with_history


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    chain  = build_rag_chain()
    config = {"configurable": {"session_id": "test-session"}}

    questions = [
        "What is the main topic covered in the documents?",
        "Can you elaborate on that?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print("A: ", end="", flush=True)
        for token in chain.stream({"question": q}, config=config):
            print(token, end="", flush=True)
        print()