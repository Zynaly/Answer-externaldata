# import logging
# import os
# import sys

# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_groq import ChatGroq

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(ROOT_DIR)

# from config.config import (
#     LLM_MAX_TOKENS,
#     LLM_MODEL,
#     LLM_TEMPERATURE,
#     GROQ_API_KEY,
# )
# from agent.prompts import (
#     format_tool_results_section,
#     get_condense_prompt,
#     get_rag_prompt,
# )
# from agent.tools import AgentPlanner
# from ragge.retrieving import build_memory, build_retriever
# from ragge.vectorstore import load_vectorstore

# log = logging.getLogger(__name__)

# _planner = AgentPlanner()


# def _format_docs(docs: list[Document]) -> str:
#     if not docs:
#         return "No relevant context found in the documents."
#     parts = []
#     for i, doc in enumerate(docs, 1):
#         source = doc.metadata.get("filename", "unknown")
#         page   = doc.metadata.get("page", "?")
#         parts.append(
#             f"[{i}] Source: {source}, Page: {page}\n{doc.page_content.strip()}"
#         )
#     return "\n\n---\n\n".join(parts)


# def get_chat_model(temperature: float = LLM_TEMPERATURE) -> ChatGroq:
#     return ChatGroq(
#         model=LLM_MODEL,
#         temperature=temperature,
#         max_tokens=LLM_MAX_TOKENS,
#         groq_api_key=GROQ_API_KEY,
#         streaming=True,
#     )


# def build_rag_chain(vectorstore=None):
#     vs                  = vectorstore or load_vectorstore()
#     retriever           = build_retriever(vs)
#     get_session_history = build_memory()
#     llm                 = get_chat_model()
#     rag_prompt          = get_rag_prompt()
#     condense_prompt     = get_condense_prompt()

#     condense_chain = (
#         condense_prompt
#         | get_chat_model(temperature=0.0)
#         | StrOutputParser()
#     )

#     def build_inputs(inputs: dict) -> dict:
#         question = inputs.get("question", "")
#         history  = inputs.get("history", [])

#         # 1. Condense follow-up questions
#         if history:
#             try:
#                 condensed = condense_chain.invoke({
#                     "question": question,
#                     "short_term_history": history,
#                 })
#             except Exception as e:
#                 log.warning("Condense chain failed (%s), using raw question", e)
#                 condensed = question
#         else:
#             condensed = question

#         # 2. Retrieve document context
#         docs    = retriever.invoke(condensed)
#         context = _format_docs(docs)

#         if "No relevant context found" in context:
#             context_hint = "INTERNAL DOCUMENTS DO NOT HAVE THIS INFORMATION."
#         else:
#             context_hint = context

#         agent_resp = _planner.run(
#             question=condensed,
#             rag_context=context_hint, # Tell the planner clearly if RAG is empty
#         )

#         # 3. Agentic planner
#         tool_results: list[dict] = []
#         tools_were_called = False
#         try:
#             agent_resp = _planner.run(
#                 question=condensed,
#                 rag_context=context,
#             )
#             tools_were_called = bool(agent_resp.tools_used)

#             # FIX: use the real raw tool results attached by AgentPlanner.run()
#             # instead of the empty placeholder dicts that were here before.
#             if hasattr(agent_resp, "_raw_tool_results"):
#                 tool_results = agent_resp._raw_tool_results
#             else:
#                 # Fallback: at least we know which tools ran, even if results lost
#                 tool_results = [
#                     {"tool": t, "result": {"error": "Result not captured"}}
#                     for t in agent_resp.tools_used
#                 ]

#         except Exception as exc:
#             log.warning("AgentPlanner failed (%s), continuing without tools", exc)

#         # 4. Format tool results for injection into the RAG prompt
#         tool_results_section = format_tool_results_section(tool_results)

#         # FIX: If tools were called to answer this question (web search, weather, etc.),
#         # suppress the RAG document context so irrelevant novel/PDF content doesn't
#         # contaminate the answer (e.g. fictional "Asim Muneer" from a novel leaking in).
#         if tools_were_called and tool_results_section:
#             effective_context = (
#                 "Note: This question was answered using live tool results below. "
#                 "Ignore document context unless it directly supports the answer."
#             )
#         else:
#             effective_context = context

#         return {
#             "question":             question,
#             "context":              effective_context,
#             "tool_results_section": tool_results_section,
#             "short_term_history":   history,
#             "long_term_history":    "",
#         }

#     core_chain = (
#         RunnableLambda(build_inputs)
#         | rag_prompt
#         | llm
#         | StrOutputParser()
#     )

#     chain_with_history = RunnableWithMessageHistory(
#         core_chain,
#         get_session_history,
#         input_messages_key="question",
#         history_messages_key="history",
#     )

#     log.info("RAG + agentic chain ready (model=%s)", LLM_MODEL)
#     return chain_with_history


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#     )
#     chain  = build_rag_chain()
#     config = {"configurable": {"session_id": "test-session"}}

#     questions = [
#         "What is 1024 * 3.14159?",
#         "What is the weather in Lahore?",
#         "What is the main topic of the documents?",
#         "Can you elaborate on that?",
#     ]

#     for q in questions:
#         print(f"\nQ: {q}")
#         print("A: ", end="", flush=True)
#         for token in chain.stream({"question": q}, config=config):
#             print(token, end="", flush=True)
#         print()


import logging
import os
import sys
from typing import Optional

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
from agent.prompts import (
    format_tool_results_section,
    get_condense_prompt,
    get_rag_prompt,
)
from agent.tools import AgentPlanner
from ragge.retrieving import build_memory, build_retriever
from ragge.vectorstore import load_vectorstore

log = logging.getLogger(__name__)

# ── Singleton planner ──────────────────────────────────────────────────────────
# AgentPlanner is stateless between calls — create once, reuse always.
_planner = AgentPlanner()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found in the documents."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source}, Page: {page}\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def get_chat_model(temperature: float = LLM_TEMPERATURE) -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
        groq_api_key=GROQ_API_KEY,
        streaming=True,
    )


# ── Chain builder ──────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore=None):
    vs                  = vectorstore or load_vectorstore()
    retriever           = build_retriever(vs)
    get_session_history = build_memory()
    llm                 = get_chat_model()
    rag_prompt          = get_rag_prompt()
    condense_prompt     = get_condense_prompt()

    condense_chain = (
        condense_prompt
        | get_chat_model(temperature=0.0)
        | StrOutputParser()
    )

    def build_inputs(inputs: dict) -> dict:
        question = inputs.get("question", "")
        history  = inputs.get("history", [])

        # ── 1. Condense follow-up questions ───────────────────────────────────
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

        # ── 2. Retrieve document context ──────────────────────────────────────
        docs    = retriever.invoke(condensed)
        context = _format_docs(docs)

        if "No relevant context found" in context:
            context_hint = "INTERNAL DOCUMENTS DO NOT HAVE THIS INFORMATION."
        else:
            context_hint = context

        # ── 3. Agentic planner (called ONCE) ──────────────────────────────────
        # Previously called twice — the first orphaned call wasted a Groq
        # request and was the main cause of 429 rate-limit errors.
        tool_results: list[dict] = []
        tools_were_called = False

        try:
            agent_resp = _planner.run(
                question=condensed,
                rag_context=context_hint,
            )
            tools_were_called = bool(agent_resp.tools_used)

            if hasattr(agent_resp, "_raw_tool_results"):
                tool_results = agent_resp._raw_tool_results
            else:
                # Fallback: preserve tool names even if results weren't captured
                tool_results = [
                    {"tool": t, "result": {"error": "Result not captured"}}
                    for t in agent_resp.tools_used
                ]

        except Exception as exc:
            log.warning("AgentPlanner failed (%s), continuing without tools", exc)

        # ── 4. Format tool results for the RAG prompt ─────────────────────────
        tool_results_section = format_tool_results_section(tool_results)

        # If tools provided a live answer, suppress document context so
        # irrelevant PDF content doesn't contaminate the response.
        if tools_were_called and tool_results_section:
            effective_context = (
                "Note: This question was answered using live tool results below. "
                "Ignore document context unless it directly supports the answer."
            )
        else:
            effective_context = context

        return {
            "question":             question,
            "context":              effective_context,
            "tool_results_section": tool_results_section,
            "short_term_history":   history,
            "long_term_history":    "",
        }

    core_chain = (
        RunnableLambda(build_inputs)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    log.info("RAG + agentic chain ready (model=%s)", LLM_MODEL)
    return chain_with_history


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    chain  = build_rag_chain()
    config = {"configurable": {"session_id": "test-session"}}

    questions = [
        "What is 1024 * 3.14159?",
        "What is the weather in Lahore?",
        "What is the main topic of the documents?",
        "Can you elaborate on that?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print("A: ", end="", flush=True)
        for token in chain.stream({"question": q}, config=config):
            print(token, end="", flush=True)
        print()