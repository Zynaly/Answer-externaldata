# LangChain ReAct Agent with RAG + Live Tools

A production-grade AI agent built with LangChain that combines semantic RAG over PDF books with live web search and weather tools, served via a FastAPI REST API.

---

## Architecture

```
User Query (POST /query)
        │
        ▼
  AgentService
        │
        ▼
  ReAct Agent (LangChain + GPT-4o-mini)
     ├── Tool: search_books  → Qdrant vector store (semantic chunks from PDFs)
     ├── Tool: web_search    → DuckDuckGo instant-answer API
     └── Tool: get_weather   → Dummy weather data (swap for real API)
        │
        ▼
  Structured AgentResponse
  { answer, sources, latency_ms, tokens, tools_used }
```

---

## Features

| Feature | Details |
|---|---|
| **Semantic Chunking** | `SemanticChunker` (OpenAI embeddings) splits PDFs by meaning, not fixed size |
| **Vector Store** | Qdrant with cosine similarity + MMR retrieval |
| **LLM** | OpenAI GPT-4o-mini (configurable) |
| **ReAct Loop** | Reason → Act → Observe, up to 8 iterations |
| **Structured Output** | Every response includes `answer`, `sources`, `latency_ms`, `tokens` |
| **Caching** | Disk-based result cache (diskcache) with configurable TTL |
| **Observability** | Structured JSON logs via structlog |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Qdrant)
- OpenAI API key

### 2. Clone & set up environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Add your PDF books

```bash
cp your_book1.pdf app/rag/documents/
cp your_book2.pdf app/rag/documents/
```

### 6. Ingest PDFs

```bash
# Via Python
python -m app.rag.ingest

# Or via API (after starting the server)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_dir": "app/rag/documents", "force_reingest": false}'
```

### 7. Start the API server

```bash
python app/main.py
# or
uvicorn app.main:app --reload --port 8000
```

### 8. Query the agent

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main themes discussed in the books?"}'
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/tools` | List all registered tools |
| `POST` | `/tools/{name}` | Invoke a tool directly |
| `POST` | `/query` | Run the ReAct agent |
| `POST` | `/ingest` | Ingest PDFs into Qdrant |

### POST /query

**Request**
```json
{
  "query": "What is the weather in London?",
  "session_id": "optional-session-id"
}
```

**Response**
```json
{
  "answer": "The current weather in London is Partly Cloudy with a temperature of 14.2°C...",
  "sources": [
    {"name": "WeatherDummy v1.0", "url": ""}
  ],
  "latency_ms": {
    "total": 1823.4,
    "by_step": {"agent": 1823.4}
  },
  "tokens": {
    "prompt": 512,
    "completion": 128
  },
  "tools_used": ["get_weather"]
}
```

---

## Project Structure

```
.
├── app/
│   ├── main.py                  # FastAPI app + routes
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── agent/
│   │   ├── react_agent.py       # ReAct agent setup + run_agent()
│   │   ├── prompts.py           # System prompt + ChatPromptTemplate
│   │   └── output_parser.py     # Parse Final Answer + JSON metadata block
│   ├── tools/
│   │   ├── web_search.py        # DuckDuckGo search tool
│   │   ├── weather.py           # Dummy weather tool
│   │   ├── rag_tool.py          # Qdrant RAG search tool
│   │   └── tool_registry.py     # ALL_TOOLS list + TOOL_MAP
│   ├── rag/
│   │   ├── ingest.py            # PDF load → semantic chunk → upsert
│   │   ├── retriever.py         # MMR retriever factory
│   │   ├── vector_store.py      # Qdrant client + collection management
│   │   └── documents/           # ← place your PDF files here
│   ├── schemas/
│   │   ├── request.py           # QueryRequest, IngestRequest
│   │   └── response.py          # AgentResponse, Source, LatencyBreakdown, TokenUsage
│   ├── services/
│   │   ├── agent_service.py     # process_query() — cache + agent + response build
│   │   └── tool_service.py      # Direct tool invocation for testing
│   └── utils/
│       ├── logger.py            # structlog JSON logger
│       ├── cache.py             # Disk-based result cache
│       └── timers.py            # LatencyTracker context manager
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Configuration

All configuration is via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | *(empty)* | Qdrant API key (for cloud) |
| `QDRANT_COLLECTION` | `book_vectors` | Collection name |
| `TOP_K` | `5` | Chunks to retrieve per query |
| `MAX_ITERATIONS` | `8` | Max ReAct loop iterations |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |

---

## Adding a New Tool

1. Create `app/tools/my_tool.py` with a `@tool` decorated function.
2. Import it in `app/tools/tool_registry.py` and append to `ALL_TOOLS`.
3. Restart the server — the agent picks it up automatically.

---

## Running with Docker

```bash
# Build image
docker build -t react-agent .

# Run (requires Qdrant running separately)
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  react-agent
```


python .\ragge\run.py --ingest  

python .\ragge\run.py --chat