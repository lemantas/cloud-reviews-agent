# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 📋 CLAUDE.md Guidelines

**What to include in this file:**
1. **Architecture mental model** - How data flows through the system
2. **Key patterns and constraints** - What to follow when making changes
3. **Domain context** - Vendors, tools, specialized terminology
4. **Essential commands** - How to run/test/deploy
5. **File organization** - Where to find things quickly
6. **Technology stack** - What libraries and why

**What NOT to include:**
- Implementation details with line numbers (read the actual code instead)
- Example conversations or UI walkthroughs (discoverable via testing)
- Verbose descriptions of every feature (let code and docstrings speak)
- Redundant information already clear from code structure

---

## Project Overview

Agentic RAG application for analyzing cloud hosting provider reviews. The LLM autonomously selects and chains analysis tools (sentiment, aspect extraction, JTBD) based on user queries. Supports multi-step reasoning with dynamic retrieval and conversational memory.

## Architecture

### Data Flow
```
CSV reviews → SQLite (OLTP) → Chunked docs → Chroma (vector store)
                                     ↓
User query → Agent → LLM selects tools → Tool execution → Response synthesis
```

### Chunking Strategy
- **Review-level:** One chunk per review (title + body)
- **Sentence-level:** Individual sentences with parent_id linking back to review
- **Deduplication:** By review_id to avoid duplicate chunks from same review

### Agent Architecture
Uses **LangGraph** (state graph) for autonomous tool selection and transparent execution:

**Graph Structure:**
- **Entry point:** `agent` node (LLM reasoning with tools bound)
- **Conditional routing:** `should_continue()` checks for tool calls
- **Tool execution:** `tools` node executes requested tools
- **Loop back:** Tools → Agent for multi-step reasoning
- **State management:** AgentState TypedDict with messages, tool_outputs, snippets
- **Checkpointing:** SQLite-based conversation persistence via thread_id

**Execution Flow:**
1. User query enters as HumanMessage
2. Agent node: LLM decides which tools to invoke
3. If tool calls exist, route to tools node
4. Tools node: Execute tools, track outputs and snippets
5. Loop back to agent node for response synthesis
6. End when no more tool calls

**Key Pattern:** Agent typically calls `retrieve_reviews` first, then analysis tools as needed.

**Migration Note:** Previously used LangChain's deprecated `create_agent()`. Now uses LangGraph for better control flow, built-in persistence, and transparent debugging.

### Retrieval (MMR)
- TOP_K=12, FETCH_K=30, LAMBDA_MMR=0.5
- Metadata filtering by vendor and chunk_type
- Deduplication by review_id

## Technology Stack

- **Python 3.12** (required for library compatibility)
- **uv** - Modern package manager (replaces pip/poetry)
- **LangGraph** - State graph framework for agentic workflows with built-in checkpointing
- **LangChain** - Core abstractions for LLMs and tools
- **Chroma** - Vector database with persistent storage
- **SQLite** - Structured review data (OLTP) + agent checkpoints
- **OpenAI** - text-embedding-3-small + GPT-4o-mini (temp=0.2)
- **Streamlit** - UI with session state management
- **Docker** - Containerized deployment with volume persistence

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # Add your OPENAI_API_KEY

# Note: uv auto-manages .venv/ - use 'uv run' prefix for commands
```

**Required:** `OPENAI_API_KEY`
**Optional:** `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (for LangSmith tracing)

### Data Ingestion
```bash
# Process CSV → SQLite → Chroma vector index
uv run python app/ingest.py
```

### Running Locally
```bash
uv run streamlit run app/app.py
# Access at http://localhost:8501
```

### Docker Deployment
```bash
# Recommended: Docker Compose
docker-compose up --build

# Manual Docker
docker build -t agentic-rag .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  agentic-rag
```

**Docker notes:** Python 3.12-slim base, health checks on `/_stcore/health`, data persisted via volume mounts

### Web Scraping
```bash
uv run python app/scrape_reviews.py
# Outputs: reviews.csv (name, country, date, review_header, review_body)
```

## Key Implementation Patterns

### Agentic Tools (4 total)
All tools use LLM with structured output (Pydantic validation):

1. **sentiment_analysis** - Analyzes overall sentiment, ratings, themes (returns Sentiment model)
2. **aspect_extraction** - Identifies product features with frequency/sentiment (returns AspectAnalysis model)
3. **jtbd_analysis** - Extracts Jobs-to-Be-Done patterns (returns JTBD model)
4. **retrieve_reviews** - Dynamically fetches snippets from Chroma (returns RetrievalResult model)

**Efficiency rule:** Analysis tools include docstring: "Do not call this tool more than once unless you retrieved new snippets"

### Conversational Memory
- **LangGraph checkpointing** - Automatic conversation persistence via SQLite
- **Thread-based** - Each Streamlit session gets unique thread_id for checkpoint isolation
- **Streamlit session state** - Stores conversation for UI display (synced with checkpoints)
- **Persistent across restarts** - Conversations survive app restarts via checkpoint DB
- Enables follow-up questions and pronoun resolution

### Multi-Tool Support
- Agent can call same tool multiple times with different parameters (e.g., compare vendors)
- Tool outputs stored as list to preserve order
- UI displays numbered instances ("Sentiment Analysis #1", "#2")

### Token Tracking
- Session-scoped limit: 100k tokens
- Real-time display with progress bar
- Blocks execution when limit exceeded
- Implemented via callback in `app/token_tracker.py`

## Domain Specialization

**Supported Vendors:** cherry_servers, ovh, hetzner, digital_ocean, scaleway, vultr

**Aspect Taxonomy:** performance, pricing, support, reliability, setup, features, documentation, interface

**Agent Intelligence:** LLM understands domain context and selects tools based on query type:
- Pricing/features → aspect_extraction
- Satisfaction/ratings → sentiment_analysis
- Motivations/goals → jtbd_analysis

## File Organization

### Core Application
- `app/app.py` - Streamlit UI with session management and thread_id tracking
- `app/graph.py` - LangGraph state graph definition (AgentState, nodes, routing)
- `app/chains.py` - Orchestration layer (`agentic_response`, `simple_rag_response`)
- `app/tools.py` - LangChain tool definitions with structured output
- `app/models.py` - Pydantic schemas (Sentiment, AspectAnalysis, JTBD, Snippet, etc.)
- `app/retrieval.py` - MMR retrieval with deduplication
- `app/clients.py` - Lazy-loaded LLM/embeddings/vector store connections
- `app/prompts.py` - Prompt template loaders
- `app/token_tracker.py` - Token tracking callback
- `app/ingest.py` - Data ingestion pipeline
- `app/scrape_reviews.py` - Trustpilot web scraper

### Data & Config
- `data/prompts/*.txt` - External prompt files (rag_system, agent_system, sentiment_analysis, aspects_analysis, jtbd_analysis)
- `data/reviews/*.csv` - Sample review data
- `data/sqlite.db` - SQLite database for reviews (gitignored)
- `data/agent_checkpoints.db` - LangGraph conversation checkpoints (gitignored)
- `data/chroma_db/` - Chroma vector store (gitignored)
- `.env` - Environment variables (gitignored)
- `Dockerfile`, `docker-compose.yml` - Container configuration

### Configuration
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions

## Development Constraints

### When Making Changes
1. **Maintain agentic pattern** - Let LLM decide tool selection, avoid hardcoded routing
2. **Use structured output** - All analysis tools return Pydantic models, not raw strings
3. **Preserve conversation context** - LangGraph checkpointing handles conversation persistence via thread_id
4. **Respect token limits** - Session capped at 100k tokens
5. **Handle tool outputs as lists** - Support multiple calls to same tool
6. **LangGraph state patterns** - Update state via partial dict returns in nodes, use reducers for accumulation
7. **Python 3.12 required** - Docker and dependencies require this version

### Testing
- Use LangSmith tracing to debug agent reasoning (set `LANGCHAIN_TRACING_V2=true`)
- Test with example questions in UI
- Verify token tracking doesn't block legitimate usage

### Deployment
- Data persistence requires volume mounts for `data/` directory
- Health checks ensure Streamlit is responsive
- Environment variables must include OPENAI_API_KEY minimum
