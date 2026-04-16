from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
SYSTEM_TEMPLATE = """You are a precise AI assistant. Answer using the provided context, 
history, and live tool results.

RULES:
1. If tool results are available, PRIORITIZE them for real-world facts (news, people, weather).
2. If context is empty and tools failed, explicitly state you don't know.
3. Use (Source: filename, page N) for PDFs and [Source Name](url) for web.
4. If a tool result says "No results," do not hallucinate information.
5. Ignore document context if it contradicts live tool facts regarding real-world entities.

---
RELEVANT PAST CONVERSATIONS:
{long_term_history}

RECENT CONVERSATION:
"""

HUMAN_TEMPLATE = """Context from documents:
{context}

{tool_results_section}

Question: {question}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="short_term_history"),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
])

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Condense / Search Query Rephraser
# ══════════════════════════════════════════════════════════════════════════════

CONDENSE_SYSTEM = "Rephrase the follow-up question into a standalone search query. Output ONLY the query."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONDENSE_SYSTEM),
    MessagesPlaceholder(variable_name="short_term_history"),
    HumanMessagePromptTemplate.from_template("Follow-up: {question}\nStandalone:"),
])

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Planner prompt (FIXED JSON BRACES)
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_TEMPLATE = """You are a RAG Strategy Planner.
Decide if the user's question needs live tools (web, weather, math) or if the document context is enough.

Available tools:
{tool_descriptions}

PRIORITY LOGIC:
- Use 'web_search' for: Public figures (e.g. Asim Muneer), news, or data NOT in your private PDFs.
- Use 'calculator' for: Any math.
- Use 'weather' for: Location forecasts.
- Empty 'tool_calls' ONLY for: Questions specifically about the PDF documents.

Return ONLY JSON:
{{{{
  "reasoning": "Brief explanation of tool choice",
  "tool_calls": [
    {{{{
      "tool": "tool_name",
      "inputs": {{{{ "key": "value" }}}}
    }}}}
  ]
}}}}"""

PLANNER_HUMAN_TEMPLATE = """User question: {question}

RAG context snapshot:
{rag_context_snippet}

Produce JSON plan:"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(PLANNER_SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(PLANNER_HUMAN_TEMPLATE),
])

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Synthesis prompt (FIXED JSON BRACES)
# ══════════════════════════════════════════════════════════════════════════════

SYNTHESIS_SYSTEM = """You are merging Tool Data and RAG Context into a JSON response.

RULES:
- If a tool has an 'error', report it. Do not guess.
- Cite sources exactly as provided in the results.

Return ONLY JSON:
{{{{
  "answer": "Complete markdown answer here",
  "sources": [{{{{ "name": "Source Name", "url": "URL" }}}}],
  "tools_used": ["tool_names"]
}}}}"""

SYNTHESIS_HUMAN_TEMPLATE = """Original question: {question}

Tool results: {tool_results_json}

RAG context: {rag_context}

Produce final JSON:"""

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYNTHESIS_SYSTEM),
    HumanMessagePromptTemplate.from_template(SYNTHESIS_HUMAN_TEMPLATE),
])

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Accessors and Formatting Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_rag_prompt(): return RAG_PROMPT
def get_condense_prompt(): return CONDENSE_PROMPT
def get_planner_prompt(): return PLANNER_PROMPT
def get_synthesis_prompt(): return SYNTHESIS_PROMPT

def format_tool_results_section(tool_results: list[dict]) -> str:
    if not tool_results:
        return ""

    lines = ["### Live Tool Data ###"]
    for entry in tool_results:
        name = entry.get("tool", "tool").upper()
        res = entry.get("result", {})
        err = res.get("error")

        lines.append(f"[{name}]")
        if err:
            lines.append(f"  ERROR: {err}")
        elif name == "WEB_SEARCH":
            for r in res.get("results", []):
                lines.append(f"- {r['title']} ({r['url']})\n  {r['snippet'][:150]}")
        elif name == "WEATHER":
            lines.append(f"  {res.get('location')}: {res.get('temperature')}°C, {res.get('condition')}")
        elif name == "CALCULATOR":
            lines.append(f"  Result: {res.get('formatted')}")
        lines.append("")

    return "\n".join(lines)