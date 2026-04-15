"""
prompts.py
==========
All prompt templates in one place.

Design decisions:
  - ChatPromptTemplate (not PromptTemplate) because we use a ChatModel
    (ChatOpenAI), which expects a list of messages, not a raw string.
  - SystemMessage sets the persona and hard rules once.
  - HumanMessagePromptTemplate carries the dynamic variables per turn.
  - Variables injected at runtime:
      {short_term_history}  from ConversationBufferWindowMemory
      {long_term_history}   from VectorStoreRetrieverMemory
      {context}             retrieved + reranked document chunks
      {question}            current user question
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# ── System persona ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a knowledgeable and precise AI assistant that answers \
questions based on the provided document context.

RULES:
1. Base your answer ONLY on the provided context and conversation history.
2. If the context does not contain enough information, say so explicitly — \
   do not hallucinate or guess.
3. Always cite the source document and page number when referencing specific facts, \
   e.g. (Source: report.pdf, page 3).
4. Be concise. Prefer bullet points for lists and numbered steps for procedures.
5. If a follow-up question refers to a previous answer (e.g. "tell me more about that"), \
   resolve the reference using the short-term history below before answering.

---
RELEVANT PAST CONVERSATIONS (long-term memory):
{long_term_history}

RECENT CONVERSATION (short-term memory):
"""

HUMAN_TEMPLATE = """Context from documents:
{context}

Question: {question}

Answer:"""


# ── Main RAG chat prompt ───────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="short_term_history"),   # injected by short-term memory
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
])


# ── Standalone query condensation prompt ───────────────────────────────────────
# Used to rephrase a follow-up question into a self-contained retrieval query.
# Example:
#   History: "Tell me about the revenue in 2023."
#   Follow-up: "And what about costs?"
#   Condensed: "What were the costs in 2023?"

CONDENSE_SYSTEM = """Given a chat history and a follow-up question, rephrase the \
follow-up question to be a fully self-contained search query that does not rely on \
the chat history. Output ONLY the rephrased question — no explanation, no prefix."""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONDENSE_SYSTEM),
    MessagesPlaceholder(variable_name="short_term_history"),
    HumanMessagePromptTemplate.from_template("Follow-up question: {question}\n\nRephrased:"),
])


# ── Convenience accessor ───────────────────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    return RAG_PROMPT


def get_condense_prompt() -> ChatPromptTemplate:
    return CONDENSE_PROMPT