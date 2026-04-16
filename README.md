# Agentic RAG Assistant

A production-grade Agentic Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and Qdrant. This system uses a dedicated Planner agent to intelligently orchestrate between internal document knowledge and real-time external tools.

## Core Features

* Agentic Planning: Uses a two-step 'Plan-then-Synthesize' approach to determine if a query requires internal docs or live tools.
* Integrated Toolset:
    - Web Search: Live data retrieval via DuckDuckGo.
    - Weather: Real-time location forecasts.
    - Calculator: Secure arithmetic evaluation using Python's AST module.
* Advanced RAG Pipeline:
    - Semantic Chunking: Context-aware document splitting for better retrieval accuracy.
    - Reranking: Implements BAAI/bge-reranker-base to optimize the context window.
    - Session Memory: Redis-ready chat history management for multi-turn conversations.

## Project Structure

/agent
  ├── llm.py        # RAG chain logic and synthesis
  ├── tools.py      # AgentPlanner and tool implementations
  └── schemas.py    # Pydantic models for type safety
/ragge
  ├── vectorstore.py # Qdrant connection and upsert logic
  └── retrieving.py  # Retriever and reranker setup
main.py              # FastAPI entry point and streaming logic
config.py            # Environment-based configuration

## Setup and Installation

1. Environment Variables:
   Create a .env file in the root directory:
   GROQ_API_KEY=your_api_key
   QDRANT_URL=http://localhost:6333
   LLM_MODEL=llama-3.1-8b-instant

2. Running with Docker Compose:
   The application requires Qdrant to function.
   docker-compose up --build

3. Manual Execution:
   If running without compose, ensure a Qdrant instance is accessible:
   pip install -r requirements.txt
   python main.py

## API Endpoints

POST /upload
- Accepts PDF/Docx files.
- Triggers semantic chunking and vector embedding.

POST /chat
- Standard request-response for queries.

POST /chat/stream
- Server-Sent Events (SSE) for real-time token streaming.

GET /health
- Monitors RAG chain readiness and database connectivity.

## Implementation Details

The system avoids common RAG pitfalls by:
1. Validating tool results before synthesis to prevent hallucinations.
2. Using a 'Condense Question' step to handle follow-up queries in context.
3. Pre-loading embedding models at startup to minimize API latency on first-hit requests.



### Start Qdrant (Database)
Run the official Qdrant image from Docker Hub:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
 

 