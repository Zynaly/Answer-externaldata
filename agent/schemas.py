from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Shared primitives ──────────────────────────────────────────────────────────

class Source(BaseModel):
    """A citable source attached to an agent answer."""
    name: str = Field(..., description="Human-readable name of the source.")
    url:  str = Field(..., description="URL or identifier of the source.")


class LatencyBreakdown(BaseModel):
    """Wall-clock timing for each pipeline step, all values in milliseconds."""
    total:   float = Field(default=0.0, description="End-to-end latency in ms.")
    by_step: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-step timing, e.g. {'retrieve': 45, 'llm': 60, "
            "'web_search': 120, 'weather': 30, 'calculator': 5}."
        ),
    )


class TokenUsage(BaseModel):
    """LLM token accounting."""
    prompt:     int = Field(default=0, description="Tokens in the prompt.")
    completion: int = Field(default=0, description="Tokens in the completion.")


# ── Final structured response ──────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """
    Canonical response envelope returned by the agent for every query.
    The LLM is instructed to produce JSON that matches this schema.
    """
    answer:     str              = Field(...,  description="The agent's final answer.")
    sources:    List[Source]     = Field(default_factory=list, description="Sources cited.")
    latency_ms: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    tokens:     TokenUsage       = Field(default_factory=TokenUsage)
    tools_used: List[str]        = Field(
        default_factory=list,
        description="Names of tools invoked to produce this answer.",
    )


# ── Tool: Web Search ───────────────────────────────────────────────────────────

class WebSearchInput(BaseModel):
    """Input for the DuckDuckGo web-search tool."""
    query: str = Field(
        ...,
        description="The search query string to send to DuckDuckGo.",
        min_length=1,
        max_length=500,
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )


class WebSearchResult(BaseModel):
    """A single result item from DuckDuckGo."""
    title:   str = Field(..., description="Title of the search result.")
    url:     str = Field(..., description="URL of the result page.")
    snippet: str = Field(default="", description="Short text excerpt.")


class WebSearchOutput(BaseModel):
    """Output from the web-search tool."""
    query:   str                  = Field(..., description="The original query.")
    results: List[WebSearchResult] = Field(default_factory=list)
    error:   Optional[str]        = Field(default=None, description="Error message if any.")


# ── Tool: Weather ──────────────────────────────────────────────────────────────

class WeatherInput(BaseModel):
    """Input for the dummy weather tool."""
    location: str = Field(
        ...,
        description="City name or 'City, CountryCode' (e.g. 'London, GB').",
        min_length=1,
        max_length=100,
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit.",
    )


class WeatherOutput(BaseModel):
    """Output from the dummy weather tool."""
    location:    str            = Field(..., description="Resolved location name.")
    temperature: float          = Field(..., description="Current temperature.")
    unit:        str            = Field(..., description="'celsius' or 'fahrenheit'.")
    condition:   str            = Field(..., description="E.g. 'Partly cloudy'.")
    humidity:    int            = Field(..., description="Relative humidity percent.")
    wind_kph:    float          = Field(..., description="Wind speed in km/h.")
    source:      Source         = Field(..., description="Data source attribution.")
    error:       Optional[str]  = Field(default=None)


# ── Tool: Calculator ──────────────────────────────────────────────────────────

class CalculatorInput(BaseModel):
    """Input for the safe arithmetic calculator tool."""
    expression: str = Field(
        ...,
        description=(
            "A safe arithmetic expression using +, -, *, /, **, (, ), and numbers. "
            "Examples: '2 + 2', '(100 * 1.08) / 12', '2 ** 10'."
        ),
        min_length=1,
        max_length=300,
    )


class CalculatorOutput(BaseModel):
    """Output from the calculator tool."""
    expression: str           = Field(..., description="The evaluated expression.")
    result:     Any           = Field(..., description="Numeric result.")
    formatted:  str           = Field(..., description="Human-readable result string.")
    error:      Optional[str] = Field(default=None)


# ── Tool registry entry ────────────────────────────────────────────────────────

class ToolDefinition(BaseModel):
    """
    Metadata the planner uses to decide which tool to call.
    Mirrors the OpenAI / Groq 'function' calling spec so it can be
    serialised directly into the tools list of a chat-completion request.
    """
    name:        str            = Field(..., description="Unique tool identifier.")
    description: str            = Field(..., description="When and why to use this tool.")
    input_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema of the input model (use model.model_json_schema()).",
    )


# ── Planner intermediate output ────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single tool invocation decided by the planner LLM."""
    tool:   str            = Field(..., description="Tool name to invoke.")
    inputs: Dict[str, Any] = Field(..., description="Arguments matching the tool's input schema.")


class PlannerDecision(BaseModel):
    """
    Structured JSON the planner LLM returns before any tool is executed.
    The agent runner iterates over tool_calls in order.
    """
    reasoning:  str            = Field(..., description="Brief explanation of the plan.")
    tool_calls: List[ToolCall] = Field(
        ...,
        description="Ordered list of tool invocations (may be empty if no tools are needed).",
    )