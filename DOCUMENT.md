# Agentic RAG Assistant Project Documentation

> **Assessment submission for the AI Engineer Take-Home.**
> This document covers the architecture rationale (Section A) and implementation notes (Section B) together, since the code was built to answer both.

---

## What This Actually Is

A FastAPI service that lets you upload documents (PDFs, Word files), indexes them in Qdrant, and then answers questions about them but with a twist. Rather than blind retrieval every time, there's an agent layer that looks at each query and decides: can I answer this from the documents, or do I need to go fetch something live (web search, weather, a calculation)?

The answer streams back token by token. Latency and token counts are tracked per step and included in every response.

---

## Section A Architecture Design

### The Full Production Picture

Below is how I'd build this for real at scale not just for one user, but for multiple tenants with proper isolation, cost controls, and observability. The code in Section B is a working subset of this.

---

### 1. Data Ingestion & Indexing

**Chunking strategy** is where most RAG systems quietly fail. Fixed-size character splitting (`RecursiveCharacterTextSplitter` at 500/50) works fine for continuous prose but breaks badly on tables and structured documents headings get split from their content, table rows land in different chunks. For production I'd move to a layout-aware approach: use `unstructured` or Azure Document Intelligence to parse document structure first, then chunk by semantic section rather than character count. The `SemanticChunker` from LangChain (commented out in `loaddoc.py`) is the right direction but was too slow for iteration during development.

**Embeddings**: `text-embedding-3-small` is the right call for cost/quality balance. For a multitenant system I'd batch at the tenant level, cache embeddings for chunks that haven't changed (using the `content_hash` already computed in `loaddoc.py` it's just not wired up yet), and run embedding jobs async rather than blocking the upload request.

**Hybrid search** is worth adding once you're past MVP. Pure vector search misses exact keyword matches product codes, names, specific numbers. The fix is straightforward: run BM25 alongside the vector search, then fuse results with RRF (Reciprocal Rank Fusion) before the reranker. Azure AI Search does this natively; for self-hosted Qdrant you'd need to run a separate BM25 index (Elasticsearch or Typesense) and merge the result sets in the retriever layer.

**Reranking**: The cross-encoder (`BAAI/bge-reranker-base`) is doing real work here it's more accurate than cosine similarity alone because it sees both the query and the chunk together rather than comparing vectors independently. The ~20s load time on first startup is a pain point but it's a one-time cost; after that the model stays in memory as a singleton.

---

### 2. The Agentic Layer

The planner makes two separate LLM calls per query. The first decides *which tools to call and with what inputs*. The second synthesizes tool results and document context into the final answer. This two-call design is intentional keeping planning and synthesis as separate responsibilities makes both easier to test, debug, and swap out.

**Tool registry** currently has three tools:
- `WebSearchTool` DuckDuckGo instant answers (no API key required, but limited coverage)
- `WeatherTool` deterministic dummy seeded on city name (same city always returns same "weather")  
- `CalculatorTool` expression evaluator using Python's `ast` module with an explicit AST whitelist

For production I'd add:
- **Currency/FX**: hits an open exchange rates endpoint, cached for 5 minutes (rates don't move that fast)
- **SQL tool**: for any tenant who's uploaded structured data, let the agent write and execute parameterized queries against a read-only connection
- **Domain policy layer**: a simple allow/deny list checked before any network call goes out currently there's a note in the code about this being missing

The current planner prompt is a plain string that gets serialized into the LLM call. This works but makes prompt iteration clunky. For scale I'd version prompts separately from code and track which prompt version produced which answer useful for debugging regressions.

---

### 3. Cost, Latency & Caching

A few places where money burns unnecessarily in the current design:

**Prompt caching**: Both the condense step and the planner step include a static system prompt in every request. These are prime candidates for Anthropic's or Groq's prompt caching same prefix, repeated across thousands of requests. The savings aren't dramatic for LLaMA-class models but matter at scale.

**Vector cache**: For popular queries ("what does this document say about X?"), the retrieval result is going to be identical. A Redis cache keyed on `(collection_id, query_embedding_hash)` with a short TTL (say, 10 minutes) would cut Qdrant traffic significantly for repeat questions.

**HTTP cache**: The DuckDuckGo integration has no caching at all right now. The same web search query fired twice in quick succession hits the API twice. A simple `httpx` transport with a `hishel` cache layer would fix this.

**Latency budget**: For a production SLA I'd target < 3s to first token for 95th percentile. Current bottlenecks in order: reranker inference (~200ms), planner LLM call (~400ms), synthesis LLM call (~400ms). The planner and tool execution could run with a timeout and fallback if the planner call takes > 1s, skip tools and go straight to RAG-only. That degrades gracefully rather than timing out.

---

### 4. Security & Multitenancy

**Per-tenant data isolation**: Each tenant gets their own Qdrant collection. Collection names are `tenant_{tenant_id}_documents`. The tenant_id comes from the JWT claims, not from the request body the client can't lie about which collection to read from. No cross-collection queries are possible by construction.

**RBAC**: Three roles are enough to start: `admin` (can ingest, can query, can delete), `analyst` (can query only), `viewer` (can query with result filtering can't see raw chunks, only the final answer). These map cleanly to API endpoint guards using FastAPI's dependency injection.

**Secret handling**: The `.env` file in the current repo contains live API keys. This is a development shortcut that should not survive a PR review. For production: secrets go into Azure Key Vault (or AWS Secrets Manager), the app fetches them at startup via managed identity (no credentials needed to fetch credentials), and secret rotation triggers a restart via the health endpoint. The `.env.example` in the repo is the right artifact to keep the `.env` itself should be in `.gitignore` and rotated immediately.

**Audit logging**: Every query should emit a structured log event with `tenant_id`, `user_id`, `query_hash` (not the raw query PII concerns), `tools_called`, `tokens_used`, and `latency_ms`. This is both a compliance requirement and how you catch abuse patterns. The current code has `stdout` logging at various points but nothing structured or centralized.

---

### 5. Observability

**Tracing**: Each request should carry a `trace_id` from the moment it enters the API through every LLM call, tool execution, and retrieval step. OpenTelemetry is the right abstraction here export to whatever backend (Jaeger, Honeycomb, Azure Monitor). LangChain has callbacks for this; it's a matter of wiring them up.

**Metrics to track**:
- Token usage per tenant per day (feeds billing)
- Retrieval relevance score (the cross-encoder score for the top chunk leading indicator of answer quality)
- Tool call success rate per tool type
- P50/P95/P99 latency by endpoint
- Rate limit hits against Groq (currently these just fail; they should be counted)

**Failure dashboards**: The two failure modes I'd alert on immediately are (1) tool call failures that exceed 5% in a 5-minute window, and (2) any LLM call that returns a non-JSON response when JSON was expected. Both currently fail silently or log to stdout.

**Evals**: For a RAG system, automated evals are the only way to know if a prompt change made things better or worse. I'd maintain a small golden dataset of (question, expected_answer, relevant_chunk_ids) tuples and run retrieval recall and answer correctness checks on every PR that touches prompts or chunking config.

---

### 6. Deployment

**Containerization**: The app is a single FastAPI process. The Dockerfile would be straightforward Python base image, copy requirements, copy source, set the startup command. The reranker model download on first run is the main complication; I'd bake it into the image at build time rather than downloading at startup. That eliminates the 20s cold start and makes deployments deterministic.

**CI/CD**:
- PR → run tests, run evals against golden dataset, build image
- Merge to main → push image to registry, deploy to staging
- Staging → smoke tests, then promote to production via blue/green swap
- Blue/green is the right choice here over canary because the Qdrant state is shared a canary that's 10% of traffic but 100% of writes causes problems if the schema changes

**Scaling**: The app is stateless except for the in-memory session store. The session store needs to move to Redis before you can run more than one instance. After that, horizontal scaling is just adding instances behind a load balancer. Qdrant scales independently.

---

## Section B Implementation Notes

### What's Built and Why

The code is a working implementation of the architecture described above minus the multitenancy, Redis, and full observability stack, which would take more than 2-3 hours to wire up responsibly.

Here's what I'd call out specifically:

**The two-call planner** (`agent/tools.py`) There's a comment in `llm.py` about a bug where the planner was being called twice per request. That was burning Groq rate limit quota and adding ~400ms of unnecessary latency. Fixed by caching the plan result within `build_inputs`.

**The calculator sandbox** The `CalculatorTool` walks the AST manually and only permits a specific whitelist of node types. `import os`, `__import__`, and any other exec-style tricks won't get through. This isn't security theater it's the right approach for letting an LLM-driven system evaluate arbitrary expressions.

**The weather dummy** `random.Random(location.lower())` seeded on the city name. Same city always returns the same fake weather. The `source` field says "OpenWeatherMap (simulated)" honest about what it is. Wiring in the real free-tier OpenWeatherMap API would be maybe 20 lines of changes.

**Streaming** `/chat/stream` returns `text/event-stream`. Each token is `data: <token>\n\n`, errors are `data: [ERROR] <message>\n\n`. The reason errors go inline rather than as HTTP error codes is that SSE connections are long-lived sending a 500 after the headers are already sent is awkward for clients.

**The `_raw_tool_results` hack** In `tools.py`, raw tool results get attached to the `AgentResponse` object as a private attribute that's not in the Pydantic model. This works but it's messy. The right fix is to add it as an `Optional[list]` field with `model_config = ConfigDict(arbitrary_types_allowed=True)`.

---

### Known Gaps and What I'd Fix With More Time

**DuckDuckGo coverage** The instant answer API is good for well-known topics but returns nothing for niche or recent queries. A fallback to Brave Search or SerpAPI would make this reliable. The current code returns an explicit error rather than an empty list when this happens, which is the right behavior better than the LLM fabricating something.

**No deduplication on ingest** Upload the same PDF twice and you get duplicate chunks in Qdrant. The `content_hash` is computed but not used for deduplication. Adding a pre-ingest check against existing chunk hashes would fix this.

**Rate limiting** There's no semaphore or queue around the LLM calls in `AgentPlanner.run()`. Under concurrent load you'll hit Groq's per-minute token limits. `asyncio.Semaphore(5)` around the LLM calls would at least serialize the bursts. A proper solution would be a token bucket per API key.

**Session persistence** Chat history lives in a Python dict. Restart the server and all sessions are gone. For production this needs Redis with a TTL (say, 24 hours per session).

**File validation** Uploads are currently validated by file extension only. For production you'd also check magic bytes (the first few bytes of the file) to catch files that have been renamed to sneak past the extension check. File size limits would also be necessary.

---

### Running It

```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env you need OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN

# Start
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First startup takes about 20-25 seconds while the reranker loads. After that `GET /health` should return `{"status": "ok", "chain_ready": true}` and the chat UI is at `GET /`.

To ingest documents without the UI, drop PDFs into `docs/` and run the ingestion scripts in order (`loaddoc.py` → `embedder.py` → `vectorstore.py`), or just use the `/upload` endpoint once the server is running it does the whole pipeline in one request.

---

### Environment Variables

```env
OPENAI_API_KEY=          # text-embedding-3-small
GROQ_API_KEY=            # LLaMA 3.1 inference
QDRANT_URL=              # e.g. http://localhost:6333
QDRANT_API_KEY=          # leave blank for local Qdrant
HF_TOKEN=                # BAAI/bge-reranker-base download
LLM_MODEL=               # e.g. llama-3.1-8b-instant
EMBEDDING_BATCH_SIZE=512

RETRIEVER_FETCH_K=20     # candidates from vector search before reranking
RETRIEVER_FINAL_K=5      # kept after reranking
RETRIEVER_SEARCH_TYPE=similarity
RERANKER_MODEL=BAAI/bge-reranker-base
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1024
```

---

### Response Format

Every `/chat` response conforms to:

```json
{
  "answer": "...",
  "sources": [{"name": "...", "url": "..."}],
  "latency_ms": {
    "total": 1240,
    "by_step": {"retrieve": 180, "plan": 420, "tools": 95, "synthesize": 545}
  },
  "tokens": {"prompt": 1820, "completion": 312},
  "tools_used": ["web_search"]
}
```

Latency is tracked with `time.perf_counter()` at each step. Token counts come from Groq's response metadata (`response_metadata["token_usage"]`), with `usage_metadata` as a fallback for other providers.

---

*The `.env` file in the repo has live keys in it those should be rotated. For any real deployment, use a secrets manager rather than `.env` files checked into version control.*



## Project Layout

```
Answer-externaldata/
├── agent/
│   ├── llm.py          # Builds the full RAG chain; wires everything together
│   ├── tools.py        # AgentPlanner + WebSearch / Weather / Calculator tool classes
│   ├── schemas.py      # Pydantic models for every input/output shape
│   └── prompts.py      # All prompt templates, plus formatting helpers
├── ragge/
│   ├── vectorstore.py  # Qdrant connection logic, collection bootstrap, upsert
│   ├── retrieving.py   # Two-stage retriever (vector search → cross-encoder rerank)
│   ├── loaddoc.py      # PDF loading + chunking
│   └── embedder.py     # Embedding model setup + batched 
│   └── main.py             # FastAPI app, endpoints, startup 
embed calls
├── config/
│   └── config.py       # Reads .env, exposes typed config constants
lifecycle
├── .env.example        # Required environment variables
└── requirements.txt
```

---

## Architecture

### The High-Level Flow

Every user question goes through this sequence:

```
User question
     │
     ▼
Condense step (if there's chat history)
     │          → rewrites "Can you elaborate?" into a proper standalone query
     ▼
Vector retrieval (Qdrant)
     │          → fetches top-K chunks by cosine similarity
     ▼
Cross-encoder reranking (bge-reranker-base)
     │          → re-scores and keeps only the best RETRIEVER_FINAL_K chunks
     ▼
AgentPlanner (Plan step)
     │          → LLM decides: use tools? which ones?
     ▼
Tool execution (web search / weather / calculator)
     │
     ▼
RAG prompt assembly
     │          → document context + tool results + chat history
     ▼
LLM (streaming) → Final answer
```

### Why Two LLM Calls in the Agent?

The `AgentPlanner` makes two separate LLM calls per invocation. The first is the **planning call** it looks at the question and the RAG context snippet, then returns a JSON object describing which tools to call. The second is the **synthesis call** after tools run, it merges the tool results with any document context into a structured JSON answer.

This separation keeps planning and synthesis clean. The planner doesn't need to know how to write a nice answer; the synthesiser doesn't need to know which tools exist. They're two distinct responsibilities.

There's a comment in `llm.py` about a bug that was fixed: the planner used to be called twice, burning an extra Groq request and hitting rate limits. Now it's called exactly once per `build_inputs` invocation.

### The RAG Chain (`agent/llm.py`)

`build_rag_chain()` is the main assembly function. It:

1. Connects to (or reuses) the Qdrant vectorstore
2. Builds the two-stage retriever
3. Sets up in-memory chat history keyed by `session_id`
4. Creates two ChatGroq instances one at `temperature=0` for condensing, one at the configured temperature for final generation
5. Wraps everything in `RunnableWithMessageHistory` so LangChain handles appending messages automatically

The `build_inputs` function is where the logic lives. It's a plain Python function wrapped in `RunnableLambda` this is intentional, it keeps the control flow readable rather than buried in chain composition operators.

One subtle detail: when tools ran successfully and returned results, `build_inputs` replaces the document context with a short note telling the LLM to prioritise the live tool data. This prevents a situation where the LLM hedges between a fresh web search result and a stale paragraph from a PDF.

---

## The Agent (`agent/tools.py`)

### Tool Registry

Three tools are registered:

**WebSearchTool** Hits the DuckDuckGo instant answer API. Uses `httpx` with a 10-second timeout. Note: the code uses `duckduckgo.com` (the base domain) rather than `api.duckduckgo.com`, because the API subdomain is unreliable. Results come from three places in the DuckDuckGo response: `AbstractText`, `RelatedTopics`, and `Results`. If all three are empty, the tool returns an explicit error rather than an empty list the LLM is instructed not to fabricate results when this happens.

**WeatherTool** This is a deterministic dummy. It uses `random.Random(location.lower())` seeded on the city name, so the same city always returns the same "weather." It supports both Celsius and Fahrenheit. The `source` field is set to "OpenWeatherMap (simulated)" which is honest about what it is.

**CalculatorTool** Parses the expression with Python's `ast` module and walks the AST manually through `_safe_eval`. Only whitelisted node types (binary ops, unary ops, constants, and a few named functions like `sqrt`, `log`, `sin`) are allowed. Anything else raises a `ValueError`. This means `import os` or `__import__` tricks won't work. The result is formatted with `{result:,.6g}` to avoid scientific notation for typical values.

### How `AgentPlanner.run()` Works

```python
# 1. Build tool descriptions from ToolDefinition objects
tool_desc = self._build_tool_descriptions()

# 2. Call LLM → get PlannerDecision (which tools, what inputs)
raw_plan = self._call_llm(planner_system, planner_user)
plan = PlannerDecision(**json.loads(raw_plan))

# 3. Execute each tool in order, collect results + timing
for tool_call in plan.tool_calls:
    result = execute_tool(tool_call)
    raw_tool_results.append({"tool": tool_call.tool, "result": result})

# 4. Synthesise: merge tool results + RAG context into final AgentResponse
raw_synth = self._call_llm(synthesis_system, synth_user)
```

The `_parse_json` method strips markdown fences before parsing because LLMs sometimes wrap JSON in ```json``` blocks even when told not to.

Token counts come from `response.response_metadata["token_usage"]` this is Groq-specific metadata. The code also checks `usage_metadata` as a fallback for other providers.

Timing is tracked with `time.perf_counter()` at each step and accumulated into a `LatencyBreakdown` object with `total` and `by_step` fields. This shows up in the final `AgentResponse`.

---

## Data Ingestion Pipeline

### Document Loading (`ragge/loaddoc.py`)

`load_pdfs()` walks the `docs/` directory recursively, loads each PDF page-by-page with `PyPDFLoader`, and adds metadata: `source` (full path), `filename`, and `content_hash` (MD5 of page content). The hash isn't used for deduplication currently but is there for future delta ingestion.

`semantic_chunk()` currently uses `RecursiveCharacterTextSplitter` with `chunk_size=500, chunk_overlap=50`. There's a commented-out `SemanticChunker` implementation using OpenAI embeddings this was the original design but it was replaced, probably due to cost or latency during development. The recursive splitter is a pragmatic choice that works well enough for most documents.

### Embedding (`ragge/embedder.py`)

Uses OpenAI's `text-embedding-3-small`. Chunks are embedded in batches (configurable via `EMBEDDING_BATCH_SIZE` in `.env`, default 512).

### Qdrant Storage (`ragge/vectorstore.py`)

`save_to_vectorstore()` creates the collection if it doesn't exist (cosine distance, dimension from config), then upserts in batches of 64 using `QdrantVectorStore.from_documents()`. Each batch is a separate upsert call this respects Qdrant's payload size limits and lets you log progress.

`load_vectorstore()` is the read path just connects to the existing collection without touching ingestion.

### Retrieval (`ragge/retrieving.py`)

**Stage 1 Vector search:** `build_vector_retriever()` wraps the Qdrant vectorstore with LangChain's retriever interface. Fetches `RETRIEVER_FETCH_K` candidates. Supports `"similarity"` and `"mmr"` search types.

**Stage 2 Reranking:** `build_reranker_retriever()` wraps the vector retriever with `ContextualCompressionRetriever` backed by `CrossEncoderReranker` (BAAI/bge-reranker-base). The cross-encoder takes each (query, chunk) pair and scores them jointly this is more accurate than cosine similarity alone because it can reason about the relationship between the query and the chunk.

All four objects (`_vectorstore`, `_vector_retriever`, `_reranker_model`, `_full_retriever`) are module-level singletons. After an `/upload`, `invalidate_retriever_cache()` clears the vectorstore and retriever references but keeps the reranker model in memory it's document-independent and takes ~20 seconds to load.

---

## Prompts (`agent/prompts.py`)

Four prompt templates:

**RAG_PROMPT** The main generation prompt. System message defines citation rules and tool result priority. Human message takes `context`, `tool_results_section`, and `question`. Uses `MessagesPlaceholder` for `short_term_history`.

**CONDENSE_PROMPT** Single-purpose: rephrase a follow-up question into a standalone query. Very short system message, outputs only the rephrased query.

**PLANNER_PROMPT** Used internally by `AgentPlanner` (the version in `prompts.py` is a LangChain template; `tools.py` also has a plain string version of the same logic). Describes available tools and instructs the LLM to return JSON only.

**SYNTHESIS_PROMPT** Same deal. Takes tool results + RAG context, returns a structured JSON with `answer`, `sources`, and `tools_used`.

`format_tool_results_section()` is a helper that converts the raw `[{"tool": ..., "result": ...}]` list into a human-readable section that goes into the RAG prompt. Each tool type has its own formatting web results show title + URL + snippet, weather shows location and conditions, calculator shows the formatted result.

---

## API (`main.py`)

### Startup

`@app.on_event("startup")` calls `_build_and_cache_chain()` which loads the embedding model, connects to Qdrant, initialises the reranker, and builds the full RAG chain. This takes ~20-25 seconds the first time (mostly the reranker download). Every subsequent request reuses the cached chain no cold start.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` (the chat UI) |
| `POST` | `/upload` | Accepts PDF/DOCX, runs ingestion, rebuilds chain |
| `POST` | `/chat` | Non-streaming chat, returns `{"answer": "..."}` |
| `POST` | `/chat/stream` | Streaming chat via Server-Sent Events |
| `GET` | `/health` | Returns chain readiness status |

### Upload Flow

`/upload` does the full ingestion sequence in one request: save file → load PDFs → chunk → embed → upsert → invalidate retriever cache → rebuild chain. It deletes the saved file if any step fails, so you don't end up with orphaned files from failed ingests.

### Streaming

`/chat/stream` returns a `StreamingResponse` with `media_type="text/event-stream"`. Each token comes as `data: <token>\n\n`. The stream ends with `data: [DONE]\n\n`. Errors are sent as `data: [ERROR] <message>\n\n` rather than HTTP error codes this is because SSE connections are long-lived and closing them with an error code is awkward for clients.

---

## Data Models (`agent/schemas.py`)

Everything is Pydantic. The main types:

- `AgentResponse` The canonical response envelope. Has `answer`, `sources`, `latency_ms`, `tokens`, `tools_used`.
- `LatencyBreakdown` `total` (ms) + `by_step` dict with per-step timing.
- `TokenUsage` `prompt` + `completion` token counts.
- `Source` `name` + `url`.
- `PlannerDecision` `reasoning` + `tool_calls` list. This is the structured output the planner LLM returns.
- `ToolCall` `tool` name + `inputs` dict.
- `ToolDefinition` Name, description, and JSON Schema of inputs. This is what gets serialised into the planner prompt.
- Input/output models for each tool (`WebSearchInput`, `WebSearchOutput`, `WeatherInput`, `WeatherOutput`, `CalculatorInput`, `CalculatorOutput`).

---

## Configuration

All config lives in `.env`. The `config/config.py` module reads it with `python-dotenv` and exposes typed constants.

Key variables:

```env
OPENAI_API_KEY=...          # For text-embedding-3-small
GROQ_API_KEY=...            # For LLaMA 3.1 inference
QDRANT_URL=...              # e.g. http://localhost:6333
QDRANT_API_KEY=...          # Leave empty for local Qdrant
HF_TOKEN=...                # For BAAI/bge-reranker-base download
LLM_MODEL=...               # e.g. llama-3.1-8b-instant
EMBEDDING_BATCH_SIZE=512
```

Model/retriever tuning is also in config:

```env
RETRIEVER_FETCH_K=20        # Candidates from vector search
RETRIEVER_FINAL_K=5         # Kept after reranking
RETRIEVER_SEARCH_TYPE=similarity
RERANKER_MODEL=BAAI/bge-reranker-base
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1024
```
