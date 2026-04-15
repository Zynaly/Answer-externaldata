"""
llm.py
======
Responsibility: Wire everything together into a LangChain LCEL chain.

Architecture:
                        ┌─────────────────────────┐
  user question ──────► │   Condense chain         │  rephrase follow-ups
                        │   (ChatOpenAI + prompt)   │  into standalone queries
                        └────────────┬────────────┘
                                     │ condensed query
                        ┌────────────▼────────────┐
                        │   Retriever              │  Stage 1: vector search
                        │   + Reranker             │  Stage 2: cross-encoder
                        └────────────┬────────────┘
                                     │ top-k reranked chunks
                        ┌────────────▼────────────┐
                        │   RAG chain              │  ChatOpenAI + RAG prompt
                        │   (context + memories)   │  + short-term + long-term
                        └────────────┬────────────┘
                                     │
                              final answer

LCEL (LangChain Expression Language) is used throughout:
  chain = prompt | llm | output_parser
  Chains compose with the pipe operator and are lazy until .invoke() is called.
"""

import logging
from operator import itemgetter

from langchain.memory import CombinedMemory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)
from prompts import get_condense_prompt, get_rag_prompt
from retriever import build_memory, build_retriever
from vectorstore import load_vectorstore

log = logging.getLogger(__name__)


# ── Format retrieved docs into a single context string ────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    Concatenate reranked chunks into a readable context block.
    Each chunk is labelled with its source so the LLM can cite it.
    """
    if not docs:
        return "No relevant context found in the documents."
    parts = []
    for i, doc in enumerate(docs, 1):
        source   = doc.metadata.get("filename", "unknown")
        page     = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source}, Page: {page}\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ── Chat model ─────────────────────────────────────────────────────────────────

def get_chat_model(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    """
    ChatOpenAI wraps the GPT chat completion API.
    We use ChatOpenAI (not OpenAI) because:
      - It speaks the messages API (system / user / assistant roles).
      - It integrates directly with ChatPromptTemplate message lists.
      - streaming=True enables token-by-token output for better UX.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
    )


# ── Build the full RAG chain ───────────────────────────────────────────────────

def build_rag_chain(vectorstore=None):
    """
    Assemble the complete RAG pipeline as an LCEL chain.

    Input expected by .invoke():
        {
          "question": "user's question string",
        }

    Memory is read/written automatically via CombinedMemory:
        short_term_history → last 5 message pairs (ConversationBufferWindowMemory)
        long_term_history  → semantically similar past exchanges (Qdrant-backed)

    Returns a callable chain that accepts {"question": str} and returns str.
    """
    vs       = vectorstore or load_vectorstore()
    retriever = build_retriever(vs)
    memory   = build_memory(vs)
    llm      = get_chat_model()
    rag_prompt     = get_rag_prompt()
    condense_prompt = get_condense_prompt()

    # ── Step 1: Condense follow-up questions ───────────────────────────────────
    # If the user says "tell me more" or "what about costs?", we need to
    # rephrase that into a self-contained query before hitting the retriever.
    condense_chain = (
        condense_prompt
        | get_chat_model(temperature=0.0)   # deterministic rephrasing
        | StrOutputParser()
    )

    def condense_question(inputs: dict) -> str:
        """Rephrase the question only if there is conversation history."""
        history = memory.load_memory_variables(inputs).get("short_term_history", [])
        if history:
            return condense_chain.invoke({
                "question": inputs["question"],
                "short_term_history": history,
            })
        return inputs["question"]

    # ── Step 2: Retrieve + rerank ──────────────────────────────────────────────
    def retrieve_context(inputs: dict) -> str:
        condensed = condense_question(inputs)
        docs      = retriever.get_relevant_documents(condensed)
        return _format_docs(docs)

    # ── Step 3: Load both memories ─────────────────────────────────────────────
    def load_memories(inputs: dict) -> dict:
        mem_vars = memory.load_memory_variables(inputs)
        return {
            "question":           inputs["question"],
            "context":            retrieve_context(inputs),
            "short_term_history": mem_vars.get("short_term_history", []),
            "long_term_history":  mem_vars.get("long_term_history", ""),
        }

    # ── Step 4: RAG chain (prompt | llm | parser) ─────────────────────────────
    rag_chain = (
        RunnableLambda(load_memories)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # ── Step 5: Wrap chain to save memory after each response ──────────────────
    class MemoryAwareChain:
        """
        Thin wrapper that:
          1. Calls rag_chain.invoke(inputs)
          2. Saves (question, answer) to both short and long-term memory
          3. Returns the answer string
        """
        def __init__(self, chain, mem: CombinedMemory):
            self._chain  = chain
            self._memory = mem

        def invoke(self, inputs: dict) -> str:
            answer = self._chain.invoke(inputs)
            self._memory.save_context(
                {"question": inputs["question"]},
                {"output": answer},
            )
            return answer

        def stream(self, inputs: dict):
            """Stream tokens for real-time output."""
            full_answer = ""
            for token in self._chain.stream(inputs):
                full_answer += token
                yield token
            self._memory.save_context(
                {"question": inputs["question"]},
                {"output": full_answer},
            )

    chain = MemoryAwareChain(rag_chain, memory)
    log.info("RAG chain ready (model=%s, streaming=True)", LLM_MODEL)
    return chain


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    chain = build_rag_chain()

    questions = [
        "What is the main topic covered in the documents?",
        "Can you elaborate on that?",      # tests short-term memory / condensation
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print("A: ", end="", flush=True)
        for token in chain.stream({"question": q}):
            print(token, end="", flush=True)
        print()