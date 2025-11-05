# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Agentic RAG (Retrieval-Augmented Generation) application for analyzing customer reviews with intelligent, LLM-driven tool selection. The project specializes in cloud hosting provider reviews and implements a complete agentic pipeline where the LLM autonomously decides which analysis tools to invoke based on user queries. The system supports multi-step reasoning with domain-specific tools for sentiment analysis, aspect extraction, Jobs-to-Be-Done (JTBD) analysis, and dynamic retrieval.

## Architecture

**Data Flow:**
- **Ingestion:** Raw CSV reviews → SQLite database (OLTP) → Chunked documents → Chroma vector store
- **Chunking:** Two-level strategy (review-level: title + body; sentence-level: individual sentences with parent_id)
- **Retrieval:** MMR (Maximal Marginal Relevance) with metadata filtering (vendor, chunk_type) and deduplication by review_id
- **Agentic Flow:** User query → Initial retrieval → Agent creation → LLM-driven tool selection → Tool execution (may include additional retrieval) → Response synthesis

**Core Components:**
- `app/scrape_reviews.py` - Web scraper for Trustpilot reviews (Trustpilot --> CSV)
- `app/ingest.py` - Data ingestion pipeline (CSV --> SQLite --> vector indexing)
- `app/clients.py` - Lazy-loaded database connections (LLM, embeddings, vector store) with callback support
- `app/retrieval.py` - Document retrieval with MMR, deduplication by review_id, and metadata processing
- `app/tools.py` - LangChain tool definitions (4 tools: sentiment_analysis, aspect_extraction, jtbd_analysis, retrieve_reviews) - all using LLM with structured output returning JSON
- `app/models.py` - Pydantic schemas for structured I/O (Sentiment, AspectAnalysis, JTBD, Snippet, RetrievalResult, RetrievalInput, ToolInput)
- `app/prompts.py` - Prompt template loaders and constructors
- `data/prompts/` - External prompt files (rag_system.txt, agent_system.txt, sentiment_analysis.txt, aspects_analysis.txt, jtbd_analysis.txt)
- `app/chains.py` - Agentic RAG implementation using LangChain's create_agent() with LLM-driven tool selection (simple_rag_response, agentic_response)
- `app/token_tracker.py` - Session-scoped token tracking and rate limiting
- `app/app.py` - Streamlit UI with tool output formatting and real-time token usage display

**Technology Stack:**
- **Package Manager:** uv (modern Python package manager)
- **Vector DB:** Chroma with persistent client
- **Chunking:** NLTK sentence tokenization for two-level chunking
- **OLTP DB:** SQLite for structured review data
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o (temperature=0.2) for agentic reasoning and structured output
- **Framework:** LangChain with Agents API (`create_agent()`) for autonomous tool selection
- **UI:** Streamlit with session state management and real-time token tracking

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (automatically manages virtual environment)
uv sync

# Set up environment variables
cp .env.example .env  # Create .env file with your OPENAI_API_KEY

# Note: uv automatically creates and manages a .venv/ directory
# No need to manually activate - use 'uv run' prefix for commands
```

### Data Ingestion
```bash
# Run the ingestion pipeline to process CSV data
uv run python app/ingest.py

# The process:
# 1. Loads data/reviews/*.csv
# 2. Creates SQLite database (data/sqlite.db)
# 3. Generates review-level and sentence-level chunks
# 4. Indexes chunks in Chroma vector store (data/chroma_db/)
```

### Running the Application
```bash
# Start the Streamlit UI
uv run streamlit run app/app.py

# The app provides:
# - Text input for customer review queries
# - Two analysis modes:
#   1. Simple Q&A: Direct RAG without tools (fast responses using rag_chain)
#   2. Agent Mode: LLM-driven tool selection with autonomous multi-step reasoning
# - Sidebar filters (cloud provider, chunk type, MMR parameters)
# - Real-time token usage tracking with progress bar and warnings
# - Tool outputs displayed as formatted sections (sentiment, aspects, JTBD)
# - Retrieved context with source citations and metadata (rating, date, vendor, review_header)
# - CSV export of results with full metadata
```

### Web Scraping
```bash
# Scrape reviews from Trustpilot
uv run python app/scrape_reviews.py

# Outputs: reviews.csv with columns:
# name, country, date, review_header, review_body
```

## Key Implementation Details

### Chunking Strategy
- **Primary chunks:** One per review (title + body), level="review"
- **Mini chunks:** Sentence-level from body text, level="sentence" with parent_id metadata
- **Metadata:** name, country, date, score, vendor, review_id, chunk_type, chunk_level, parent_id, sentence_id, review_header
- **IDs:** Deterministic format: `{review_id}_s{i}` for sentence chunks

### Agentic Tools
The system implements four LangChain tools that the agent can autonomously invoke. All analysis tools use LLM-based structured output with Pydantic validation:

1. **sentiment_analysis** (`@tool` decorator with ToolInput schema)
   - Analyzes overall sentiment, mean rating, positive/negative share and emotional themes
   - Returns: Sentiment model (total_reviews, mean_rating, positive_share, negative_share, positive_themes, negative_themes)
   - Input: List of snippets + user question

2. **aspect_extraction** (`@tool` decorator with ToolInput schema)
   - Identifies and ranks product aspects with frequency, sentiment scores, and example quotes
   - Returns: AspectAnalysis model (total_aspects, aspects list with name, frequency, sentiment_score, examples)
   - Input: List of snippets + user question

3. **jtbd_analysis** (`@tool` decorator with ToolInput schema)
   - Extracts Jobs-to-Be-Done patterns: job, situation, motivation, expected outcomes, frustrations
   - Returns: JTBD model with supporting quotes and total reviews analyzed
   - Input: List of snippets + user question

4. **retrieve_reviews** (`@tool` decorator with RetrievalInput schema) - NEW TOOL
   - Dynamically retrieves relevant review snippets from ChromaDB vector store
   - Enables multi-step reasoning: agent can retrieve → analyze → retrieve again with different parameters
   - Returns: RetrievalResult model (snippets list, count)
   - Input: question, chunk_type (sentence/review), vendor filter, top_k, fetch_k
   - Features: Deduplication by review_id, metadata normalization (text, rating, date, source, vendor, review_header)

### Agentic Architecture (LLM-Driven Tool Selection)
The system uses **LangChain's create_agent()** framework for autonomous tool selection and multi-step reasoning:

**Agent Implementation (`agentic_response` in chains.py:43-116):**
1. Initial retrieval with `retrieve_documents()` to check for relevant reviews
2. If reviews found, creates agent with:
   - Model: GPT-4o from `get_llm()`
   - Tools: `[summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews]`
   - System prompt: Agent-specific instructions from `agent_system.txt`
3. Agent invoked with user message via new Agents API
4. LLM autonomously decides which tools to call based on:
   - User question intent
   - Tool descriptions and docstrings
   - Current context and previous tool outputs
5. Tool outputs collected from ToolMessage entries in message history
6. Final response synthesized from last AI message

**Key Features:**
- No manual keyword routing - LLM decides tool selection
- Multi-step reasoning: Agent can call `retrieve_reviews` multiple times with different parameters
- Handles ambiguous queries intelligently (e.g., "analyze Scaleway" → agent may choose all 3 analysis tools)
- Natural conversation flow with tool chaining

**MMR Retrieval Configuration:**
- TOP_K=12 (results returned)
- FETCH_K=30 (candidates before diversification)
- LAMBDA_MMR=0.5 (0=most diverse, 1=most relevant)
- Deduplication by review_id to avoid duplicate chunks from same review

### Security Considerations
- Prompt injection protection (treat review text as untrusted data)
- Environment variable management for API keys

## File Structure Context

### Data Files
- `data/sqlite.db` - SQLite database with structured review data
- `data/chroma_db/` - Persisted Chroma vector store
- `data/reviews/` - Sample review data for ingestion

### Configuration
- `.env` - Environment variables (API keys, tracing settings)
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions

### Prompts
- `data/prompts/rag_system.txt` - System prompt for simple RAG mode (focus on citations and evidence)
- `data/prompts/agent_system.txt` - Agent system prompt with tool selection guidance and multi-step reasoning instructions
- `data/prompts/sentiment_analysis.txt` - Structured output prompt for sentiment analysis tool (Sentiment model)
- `data/prompts/aspects_analysis.txt` - Structured output prompt for aspect extraction tool (AspectAnalysis model)
- `data/prompts/jtbd_analysis.txt` - Structured output prompt for JTBD analysis tool (JTBD model)
- Note: `tool_selection_system.txt` has been removed (replaced by agent-native tool selection)

### Development Files
- `.env` - Environment variables for local development
- `.venv/` - Virtual environment managed by uv (excluded from version control)

## Domain Specialization

The application is specialized for cloud hosting provider reviews with:
- **Supported Vendors:** cherry_servers, ovh, hetzner, digital_ocean, scaleway, vultr
- **Aspect Taxonomy:** performance, pricing, support, reliability, setup, features, documentation, interface
- **Citation Format:** `[source | YYYY-MM-DD]` with metadata (rating, vendor, review_header)
- **Focus:** IaaS/bare-metal hosting customer insights with structured sentiment, aspect, and JTBD analysis
- **Agent Intelligence:** LLM understands domain context and selects appropriate tools based on query intent (e.g., pricing questions → aspect_extraction, satisfaction questions → sentiment_analysis, motivations → jtbd_analysis)

## Monitoring and Evaluation

- **LangSmith integration** for tracing agent execution and tool calls (set LANGCHAIN_TRACING_V2=1)
- **Token Tracking:** Session-scoped tracking via TokenTracker callback (50k token limit per session)
  - Real-time usage display in sidebar with progress bar
  - Warning when 90% limit reached
  - Prevents execution when limit exceeded
- **RAG framework** for RAG evaluation (Answer Faithfulness, Context Precision/Recall)
- **Error handling** with user-friendly messages in Streamlit UI
- **Rate limiting** and retry policies for API calls (max_retries=3 for LLM and embeddings)