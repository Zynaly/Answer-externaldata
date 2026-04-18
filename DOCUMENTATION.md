# Agentic RAG Assistant Technical Documentation

## What This Is

This is a production-minded "Answering over External Data" service. You give it documents (PDFs, Word files), it ingests them into a vector database, and then you can ask it questions. What makes it interesting is that it doesn't just do dumb retrieval it has an agent that decides, per query, whether to pull from your documents or go out and fetch live data from tools like web search, a weather API, or its own calculator. Then it streams the answer back token by token.

The whole thing runs as a FastAPI server backed by Qdrant for vector storage, Groq for LLM inference (LLaMA 3.1 Instant model), and a HuggingFace cross-encoder for reranking.

---

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

---

## Running Locally

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN
```

### 4. Ingest documents (optional needed before querying PDFs)

Drop PDFs into `docs/` then:

```bash
python ragge/loaddoc.py     # chunk documents
python ragge/embedder.py    # embed chunks
python ragge/vectorstore.py # upload to Qdrant
```

Or just use the `/upload` endpoint once the server is running.

### 5. Start the server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will take ~20-25 seconds on startup while it loads the reranker model. After that, `GET /health` should return `{"status": "ok", "chain_ready": true}`.

---

## Known Limitations and What Could Be Better

**DuckDuckGo web search** The instant answer API is free and requires no key, but it's not a full search engine. It returns good results for well-known topics (people, cities, common questions) but can return nothing for niche or recent queries. A fallback to SerpAPI or Brave Search would make this more reliable.

**Weather is simulated** The WeatherTool uses seeded random numbers. It's deterministic per city but not real. Wiring in OpenWeatherMap's free tier would be a small change.

**In-memory session store** Chat history is stored in a Python dict in `retrieving.py`. It disappears on restart and doesn't scale past one process. Redis would be the obvious fix.

**Chunking strategy** `RecursiveCharacterTextSplitter` at 500/50 is a reasonable default but doesn't understand document structure. Tables split awkwardly, headings get separated from their content. The commented-out `SemanticChunker` would handle this better.

**No deduplication on ingest** If you upload the same PDF twice, you get duplicate chunks in Qdrant. The `content_hash` metadata is there but isn't used to skip existing chunks.

**Rate limiting** There's no request queue or semaphore. Under concurrent load, you'll hit Groq's rate limits. Adding a simple asyncio semaphore around the LLM calls in `AgentPlanner.run()` would help.

**The `_raw_tool_results` attribute** In `tools.py`, raw tool results are attached to the `AgentResponse` object as `resp._raw_tool_results = raw_tool_results`. This works but it's not declared in the Pydantic model, so it's not validated or serialised. Cleaner to add it as an optional field with `model_config = ConfigDict(arbitrary_types_allowed=True)` or pass it through a separate return type.

---

## Security Notes

**The `.env` file in the repo contains live API keys.** These should be rotated. For production, use a secrets manager (AWS Secrets Manager, Azure Key Vault, etc.) rather than `.env` files.

The calculator is safely sandboxed via AST walking arbitrary Python can't be executed through it. The web search tool validates the DuckDuckGo response before returning, and the LLM is explicitly instructed not to fabricate results when the tool returns an error.

File uploads are restricted to `.pdf`, `.doc`, `.docx` by extension check. For production, you'd want to also validate file contents (magic bytes) rather than trusting the extension, and limit file size.