from __future__ import annotations

import ast
import json
import logging
import math
import operator
import random
import time
from typing import Any, Dict, List, Optional

import httpx

from agent.schemas import (
    AgentResponse,
    CalculatorInput,
    CalculatorOutput,
    LatencyBreakdown,
    PlannerDecision,
    Source,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    WebSearchInput,
    WebSearchOutput,
    WebSearchResult,
    WeatherInput,
    WeatherOutput,
)

log = logging.getLogger(__name__)

# ── Safe calculator helpers ────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "pi": math.pi,
    "e":  math.e,
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            fn   = _SAFE_NAMES[node.func.id]
            args = [_safe_eval(a) for a in node.args]
            return fn(*args)
    raise ValueError(f"Unsafe expression node: {ast.dump(node)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

class WebSearchTool:
    NAME        = "web_search"
    DESCRIPTION = (
        "Search the web for current information, news, facts, or any topic "
        "the knowledge base does not cover. Use this when the user asks about "
        "recent events, people, organisations, or anything needing live data."
    )

    @classmethod
    def definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            input_schema=WebSearchInput.model_json_schema(),
        )

    def run(self, inputs: Dict[str, Any]) -> WebSearchOutput:
        inp = WebSearchInput(**inputs)
        params = {
            "q": inp.query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                # FIX: use base duckduckgo.com (api. subdomain is blocked/unreliable)
                resp = client.get("https://duckduckgo.com/", params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.warning("DuckDuckGo HTTP error: %s", exc)
            return WebSearchOutput(
                query=inp.query,
                error=f"Search service returned HTTP {exc.response.status_code}. Cannot retrieve results.",
            )
        except Exception as exc:
            log.warning("DuckDuckGo request failed: %s", exc)
            return WebSearchOutput(
                query=inp.query,
                error=(
                    f"Web search unavailable ({exc}). "
                    "Do NOT guess or fabricate results — tell the user the search failed."
                ),
            )

        results: List[WebSearchResult] = []

        if data.get("AbstractText"):
            results.append(WebSearchResult(
                title=data.get("Heading", inp.query),
                url=data.get("AbstractURL", "https://duckduckgo.com"),
                snippet=data["AbstractText"],
            ))

        for topic in data.get("RelatedTopics", []):
            if len(results) >= inp.max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(WebSearchResult(
                    title=topic.get("Text", "")[:80],
                    url=topic.get("FirstURL", "https://duckduckgo.com"),
                    snippet=topic.get("Text", ""),
                ))

        for item in data.get("Results", []):
            if len(results) >= inp.max_results:
                break
            results.append(WebSearchResult(
                title=item.get("Text", "")[:80],
                url=item.get("FirstURL", "https://duckduckgo.com"),
                snippet=item.get("Text", ""),
            ))

        if not results:
            return WebSearchOutput(
                query=inp.query,
                error=(
                    "Search succeeded but returned no results for this query. "
                    "Do NOT fabricate results — tell the user nothing was found."
                ),
            )

        return WebSearchOutput(query=inp.query, results=results[:inp.max_results])


# ──────────────────────────────────────────────────────────────────────────────

class WeatherTool:
    """
    Synthetic weather tool — returns realistic-looking but deterministic data.
    No API key required.
    """

    NAME        = "weather"
    DESCRIPTION = (
        "Get the current weather conditions for a city. Returns temperature, "
        "humidity, wind speed, and a short description. Use whenever the user "
        "asks about weather in any location."
    )

    _CONDITIONS = [
        "Sunny", "Partly cloudy", "Overcast", "Light rain",
        "Heavy rain", "Thunderstorms", "Fog", "Clear skies",
        "Scattered showers", "Drizzle", "Snow flurries", "Windy",
    ]

    @classmethod
    def definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            input_schema=WeatherInput.model_json_schema(),
        )
    def run(self, inputs: Dict[str, Any]) -> WeatherOutput: 
        unit = inputs.get("unit", "celsius").lower()
        if unit not in ["celsius", "fahrenheit"]:
            unit = "celsius"   
        
        inputs["unit"] = unit
         
        inp = WeatherInput(**inputs)

        rng = random.Random(inp.location.lower().strip())

        temp_c    = round(rng.uniform(-5, 40), 1)
        humidity  = rng.randint(30, 95)
        wind_kph  = round(rng.uniform(0, 60), 1)
        condition = rng.choice(self._CONDITIONS)

        if inp.unit == "fahrenheit":
            display_temp = round(temp_c * 9 / 5 + 32, 1)
        else:
            display_temp = temp_c

        return WeatherOutput(
            location=inp.location,
            temperature=display_temp,
            unit=inp.unit,
            condition=condition,
            humidity=humidity,
            wind_kph=wind_kph,
            source=Source(
                name="OpenWeatherMap (simulated)",
                url="https://openweathermap.org",
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────

class CalculatorTool:
    NAME        = "calculator"
    DESCRIPTION = (
        "Evaluate arithmetic expressions: +, -, *, /, ** (power), parentheses, "
        "and common math functions (sqrt, log, sin, cos, tan, abs, round). "
        "Use this for any numeric calculation or math question."
    )

    @classmethod
    def definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            input_schema=CalculatorInput.model_json_schema(),
        )

    def run(self, inputs: Dict[str, Any]) -> CalculatorOutput:
        inp = CalculatorInput(**inputs)
        try:
            tree   = ast.parse(inp.expression, mode="eval")
            result = _safe_eval(tree)
            if isinstance(result, float) and result.is_integer():
                formatted = str(int(result))
            else:
                formatted = f"{result:,.6g}"
            return CalculatorOutput(
                expression=inp.expression,
                result=result,
                formatted=formatted,
            )
        except Exception as exc:
            return CalculatorOutput(
                expression=inp.expression,
                result=None,
                formatted="",
                error=str(exc),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_INSTANCES: Dict[str, Any] = {
    WebSearchTool.NAME:  WebSearchTool(),
    WeatherTool.NAME:    WeatherTool(),
    CalculatorTool.NAME: CalculatorTool(),
}

TOOL_DEFINITIONS: List[ToolDefinition] = [
    WebSearchTool.definition(),
    WeatherTool.definition(),
    CalculatorTool.definition(),
]


def execute_tool(tool_call: ToolCall) -> Dict[str, Any]:
    tool = _TOOL_INSTANCES.get(tool_call.tool)
    if tool is None:
        return {"error": f"Unknown tool: {tool_call.tool}"}
    try:
        result = tool.run(tool_call.inputs)
        return result.model_dump()
    except Exception as exc:
        log.exception("Tool %s raised an exception", tool_call.tool)
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Planner
# ═══════════════════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM = """You are a planning agent. Given a user question, decide which \
tools (if any) to call to answer it, then return ONLY a JSON object matching this schema:

{{
  "reasoning": "<brief explanation of your plan>",
  "tool_calls": [
    {{
      "tool": "<tool_name>",
      "inputs": {{ ... }}
    }}
  ]
}}

Available tools:
{tool_descriptions}

Rules:
- Use "calculator" for any arithmetic or math question.
- Use "web_search" for current events, facts, or anything outside the document knowledge base.
- Use "weather" for weather questions about any location.
- You may call multiple tools; list them in execution order.
- If no tool is needed (e.g. simple conversational reply), return an empty tool_calls list.
- Output ONLY the JSON object — no markdown, no prose, no code fences."""

_SYNTHESIS_SYSTEM = """You are a helpful assistant. Using the tool results and the \
original question, write a clear, accurate answer.

IMPORTANT RULES:
- If a tool result contains an "error" field, tell the user that operation failed. \
  Do NOT invent or fabricate data. Do NOT cite fake URLs.
- Only cite real URLs that appear in actual tool results.
- Cite sources where applicable (format: [Source Name](url)).
- Explain which tools you used and why.

You MUST respond with ONLY a JSON object matching this exact schema:
{{
  "answer": "<your complete answer as a markdown string>",
  "sources": [
    {{"name": "<source name>", "url": "<url>"}}
  ],
  "tools_used": ["<tool1>", "<tool2>"]
}}

No markdown fences, no prose outside the JSON."""


class AgentPlanner:
    """
    Two-step LLM planner:
      Step 1 — Decide which tools to call (PlannerDecision).
      Step 2 — Synthesise a final AgentResponse from tool outputs.
    """

    def __init__(self, llm_model: Optional[str] = None, groq_api_key: Optional[str] = None):
        try:
            import os, sys
            ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if ROOT_DIR not in sys.path:
                sys.path.append(ROOT_DIR)
            from config.config import LLM_MODEL, GROQ_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=llm_model or LLM_MODEL,
                temperature=0.0,
                max_tokens=LLM_MAX_TOKENS,
                groq_api_key=groq_api_key or GROQ_API_KEY,
                streaming=False,
            )
            self._llm_model = llm_model or LLM_MODEL
        except Exception as exc:
            log.warning("Could not load ChatGroq from config (%s). Falling back to stub LLM.", exc)
            self._llm       = None
            self._llm_model = "stub"

    def _call_llm(self, system: str, user: str) -> tuple[str, int, int]:
        if self._llm is None:
            return '{"reasoning":"no llm","tool_calls":[]}', 0, 0

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = self._llm.invoke(messages)

        text             = response.content
        prompt_tok       = getattr(response, "usage_metadata", {}).get("input_tokens",  0) if hasattr(response, "usage_metadata") else 0
        completion_tok   = getattr(response, "usage_metadata", {}).get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0

        if hasattr(response, "response_metadata"):
            meta           = response.response_metadata.get("token_usage", {})
            prompt_tok     = meta.get("prompt_tokens",     prompt_tok)
            completion_tok = meta.get("completion_tokens", completion_tok)

        return text, prompt_tok, completion_tok

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip().rstrip("`").strip()
        return json.loads(text)

    def _build_tool_descriptions(self) -> str:
        lines = []
        for td in TOOL_DEFINITIONS:
            props = td.input_schema.get("properties", {})
            prop_summary = ", ".join(
                f"{k} ({v.get('type','any')}): {v.get('description','')}"
                for k, v in props.items()
            )
            lines.append(f"  - {td.name}: {td.description}\n    Inputs: {prop_summary}")
        return "\n".join(lines)

    def run(
        self,
        question: str,
        rag_context: str = "",
        session_id: str  = "default",
    ) -> AgentResponse:
        t_total_start = time.perf_counter()
        latency: Dict[str, float] = {}
        total_prompt_tok      = 0
        total_completion_tok  = 0
        tools_used: List[str] = []
        all_sources: List[Source] = []

        # ── Step 1: Plan ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        tool_desc = self._build_tool_descriptions()
        planner_system = _PLANNER_SYSTEM.format(tool_descriptions=tool_desc)
        planner_user   = f"User question: {question}"
        if rag_context:
            planner_user += f"\n\nRAG context available:\n{rag_context[:1000]}"

        raw_plan, p_tok, c_tok = self._call_llm(planner_system, planner_user)
        latency["plan_llm"] = (time.perf_counter() - t0) * 1000
        total_prompt_tok     += p_tok
        total_completion_tok += c_tok

        try:
            plan_dict = self._parse_json(raw_plan)
            plan      = PlannerDecision(**plan_dict)
        except Exception as exc:
            log.warning("Could not parse planner output: %s\nRaw: %s", exc, raw_plan)
            plan = PlannerDecision(reasoning="Parse error – no tools called.", tool_calls=[])

        log.info("Planner reasoning: %s", plan.reasoning)
        log.info("Planned tool calls: %s", [tc.tool for tc in plan.tool_calls])

        # ── Step 2: Execute tools ──────────────────────────────────────────────
        # FIX: store raw tool results so llm.py can pass them to format_tool_results_section
        raw_tool_results: List[Dict[str, Any]] = []

        for tool_call in plan.tool_calls:
            t0     = time.perf_counter()
            result = execute_tool(tool_call)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latency[tool_call.tool] = elapsed_ms

            tools_used.append(tool_call.tool)
            raw_tool_results.append({"tool": tool_call.tool, "result": result})
            log.info("Tool %s completed in %.1f ms", tool_call.tool, elapsed_ms)

            if tool_call.tool == WebSearchTool.NAME and not result.get("error"):
                for r in result.get("results", []):
                    all_sources.append(Source(name=r["title"], url=r["url"]))

            if tool_call.tool == WeatherTool.NAME and not result.get("error"):
                src = result.get("source")
                if src:
                    all_sources.append(Source(**src))

        # ── Step 3: Synthesise final answer ────────────────────────────────────
        t0 = time.perf_counter()
        synth_user = (
            f"Original question: {question}\n\n"
            f"Tool results:\n{json.dumps(raw_tool_results, indent=2)}\n\n"
        )
        if rag_context:
            synth_user += f"RAG document context:\n{rag_context[:2000]}\n\n"
        synth_user += "Produce the final JSON answer now."

        raw_synth, p_tok, c_tok = self._call_llm(_SYNTHESIS_SYSTEM, synth_user)
        latency["synthesis_llm"]  = (time.perf_counter() - t0) * 1000
        total_prompt_tok         += p_tok
        total_completion_tok     += c_tok

        try:
            synth_dict    = self._parse_json(raw_synth)
            answer_text   = synth_dict.get("answer", raw_synth)
            extra_sources = [Source(**s) for s in synth_dict.get("sources", [])]
            tools_from_synth = synth_dict.get("tools_used", [])
        except Exception as exc:
            log.warning("Could not parse synthesis output: %s", exc)
            answer_text      = raw_synth
            extra_sources    = []
            tools_from_synth = []

        seen_urls = set()
        final_sources: List[Source] = []
        for s in all_sources + extra_sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                final_sources.append(s)

        final_tools = list(dict.fromkeys(tools_used + tools_from_synth))
        total_ms = (time.perf_counter() - t_total_start) * 1000

        resp = AgentResponse(
            answer=answer_text,
            sources=final_sources,
            latency_ms=LatencyBreakdown(
                total=round(total_ms, 1),
                by_step={k: round(v, 1) for k, v in latency.items()},
            ),
            tokens=TokenUsage(
                prompt=total_prompt_tok,
                completion=total_completion_tok,
            ),
            tools_used=final_tools,
        )
        # FIX: attach raw tool results so llm.py can forward them to the RAG prompt
        resp._raw_tool_results = raw_tool_results
        return resp