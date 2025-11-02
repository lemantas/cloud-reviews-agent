# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Customer Reviews RAG (Retrieval-Augmented Generation) application for analyzing customer reviews with function calling capabilities. The project specializes in cloud hosting provider reviews and implements a complete RAG pipeline with domain-specific tools for sentiment analysis, aspect extraction, and Jobs-to-Be-Done (JTBD) analysis.

## Architecture

**Data Flow:**
- Raw CSV reviews --> SQLite database (OLTP) --> Chunked documents --> Chroma vector store
- Two-level chunking strategy: review-level chunks (title + body) and sentence-level mini-chunks
- Retrieval uses MMR (Maximal Marginal Relevance) with query-aware filtering based on sentiment

**Core Components:**
- `app/scrape_reviews.py` - Web scraper for Trustpilot reviews (Trustpilot --> CSV)
- `app/ingest.py` - Data ingestion pipeline (CSV --> SQLite --> vector indexing)
- `app/clients.py` - Lazy-loaded database connections (LLM, embeddings, vector store)
- `app/retrieval.py` - Document retrieval with MMR and sentiment-aware filtering
- `app/tools.py` - Domain analysis tools (sentiment, aspects, JTBD) - all using LLM with structured output returning JSON
- `app/models.py` - Pydantic schemas for structured tool outputs (Sentiment, AspectAnalysis, JTBD)
- `app/prompts.py` - Prompt template loaders and constructors
- `data/prompts/` - External prompt files (rag_system.txt, agent_system.txt, sentiment_analysis.txt, aspects_analysis.txt, jtbd_analysis.txt)
- `app/chains.py` - RAG chains with keyword-based tool routing (simple_rag_response, agentic_response)
- `app/token_tracker.py` - Session-scoped token tracking and rate limiting
- `app/app.py` - Streamlit UI with tool output formatting

**Technology Stack:**
- **Package Manager:** uv (modern Python package manager)
- **Vector DB:** Chroma
- **Chunking:** NLTK sentence tokenization
- **OLTP DB:** SQLite for structured review data
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** GPT-4o for chat and function calling
- **Framework:** LangChain for orchestration
- **UI:** Streamlit

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
#   1. Simple Q&A: Direct RAG without tools (fast responses)
#   2. Insights Agent: Keyword-based tool routing with structured analysis
# - Sidebar filters (cloud provider, chunk type)
# - Tool outputs displayed as formatted sections (sentiment, aspects, JTBD)
# - Retrieved context with source citations
# - CSV export of results
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

### Function Calling Tools
The system implements three core domain tools that return structured JSON. All tools use LLM-based analysis with Pydantic validation:
1. **summarize_sentiment** - Uses LLM with Sentiment model to analyze overall sentiment, mean rating, positive/negative share and themes
2. **extract_top_aspects** - Uses LLM with AspectAnalysis model to identify ranked product aspects with frequency, sentiment scores, and examples
3. **infer_jtbd** - Uses LLM with JTBD model to extract Jobs-to-Be-Done patterns

### Query Routing
The system uses keyword-based routing in `agentic_response()`:
- Sentiment keywords ("sentiment", "feel", "positive", "negative") --> `summarize_sentiment` tool
- Aspect keywords ("feature", "aspect", "mention", "performance", "pricing") --> `extract_top_aspects` tool
- JTBD keywords ("job", "accomplish", "goal", "trying to", "need to") --> `infer_jtbd` tool
- No specific keywords detected --> Runs all three tools for comprehensive analysis

Sentiment-aware retrieval filtering:
- Positive queries ("happiest", "love", "best") --> bias to rating ≥ 4
- Negative queries ("missing", "issues", "hate") --> bias to rating ≤ 3
- MMR retrieval with configurable parameters (TOP_K=12, FETCH_K=30, LAMBDA_MMR=0.5)

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
- `data/prompts/agent_system.txt` - System prompt for agentic mode (synthesizing tool outputs with review context)
- `data/prompts/sentiment_analysis.txt` - Prompt template for sentiment analysis tool
- `data/prompts/aspects_analysis.txt` - Prompt template for aspect extraction tool
- `data/prompts/jtbd_analysis.txt` - Prompt template for JTBD analysis tool

### Development Files
- `.env` - Environment variables for local development
- `.venv/` - Virtual environment managed by uv (excluded from version control)

## Domain Specialization

The application is specialized for cloud hosting provider reviews with:
- Aspect taxonomy: performance, pricing, support, reliability, setup, features, documentation, interface
- Citation format: `[source | YYYY-MM-DD]` with URL links
- Focus on IaaS/bare-metal hosting customer insights

## Monitoring and Evaluation

- **LangSmith integration** for tracing (set LANGCHAIN_TRACING_V2=1)
- **RAG framework** for RAG evaluation (Answer Faithfulness, Context Precision/Recall)
- **Error handling** with user-friendly messages in Streamlit UI
- **Rate limiting** and retry policies for API calls