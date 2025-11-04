## What It Does

This is an AI-powered customer review analysis platform that helps businesses understand customer feedback about cloud hosting providers. Users can ask natural language questions like "What are customers saying about pricing?" and get intelligent insights backed by real review data.


## Core Architecture

### 1. Data Pipeline

- Input: CSV files with customer reviews (name, rating, date, review text)
- Storage: SQLite for structured data + Chroma for semantic search
- Chunking: Two levels - full reviews AND individual sentences for precise retrieval

###  2. Retrieval-Augmented Generation (RAG)

- Smart Retrieval: Uses semantic similarity + sentiment-aware filtering
    - "What do customers love?" (Bias toward 4-5 star reviews)
    - "What issues do they face?" (Bias toward 1-3 star reviews)
- Context Assembly: Finds most relevant review snippets with source citations
- Answer Generation: GPT-4o creates responses with proper attribution

### 3. Domain Tools & Function Calling

The system has specialized analysis tools that activate based on user intent. All tools use LLM-based analysis with Pydantic validation for structured outputs:
- Sentiment Analysis: Statistics, ratings breakdown, positive/negative themes
- Aspect Extraction: Identifies top discussed features (pricing, support, performance)
- Jobs-to-Be-Done: Analyzes what customers are trying to accomplish

### 4. Two Analysis Modes

Simple Q&A Mode:
- Direct question --> retrieval --> answer
- Fast, straightforward responses
- No additional tool analysis

Insights Agent Mode:
- LLM-based routing to analysis tools
- Provides deeper, structured insights with JSON output
- Example: "Analyze Scaleway reviews" --> Automatically runs sentiment analysis + aspect extraction + JTBD
- All three tools validated by Pydantic models (Sentiment, AspectAnalysis, JTBD) for structured output

## Agentic refactoring

### 1. LLM-Based Tool Selection

LLM-Based Tool Selection
- LLM decides which tools to use
- More accurate tool selection (no manual keyword lists)
- Handles ambiguous queries better ("analyze Scaleway" → agent decides to run all 3 tools)
- User doesn't need to know which tool does what

Changes needed:
  - Convert your 3 tools (summarize_sentiment, extract_top_aspects, infer_jtbd) to LangChain tool format
  - Use ChatOpenAI.bind_tools() to let GPT-4o decide which tools to invoke
  - Remove route_query_to_tools() function
  - Let the agent autonomously choose 0, 1, 2, or all 3 tools based on the question

### 2. Multi-Step Reasoning

ReAct-style agent that can perform multi-step analysis:
- Handles comparative questions
- Can drill down based on findings ("I see pricing issues, let me get more pricing reviews")
- More natural conversation flow

Changes needed:
- Use LangChain's create_react_agent() or AgentExecutor
- Add a "retrieve_reviews" tool that the agent can call dynamically
- Agent decides: retrieve → analyze → retrieve again → synthesize

Example flow:
  1. User asks: "Compare pricing sentiment between OVH and Scaleway"
  2. Agent thinks: "I need sentiment analysis for each vendor"
  3. Agent retrieves OVH reviews → runs sentiment tool
  4. Agent retrieves Scaleway reviews → runs sentiment tool
  5. Agent synthesizes comparison


### 3. Follow-Up Capabilities

Let the agent ask clarifying questions or suggest follow-ups:
- Makes the agent feel interactive and helpful
- Guides non-technical users
- Natural conversation flow

Example:
- User: "Tell me about customer issues"
- Agent: "I found 3 main issue categories. Would you like me to analyze sentiment for each category
separately?"

Changes needed:
  - Modify agent prompt to include follow-up question generation
  - Add session memory (store conversation history in st.session_state)
  - Display suggested follow-up questions as clickable buttons

## Technical Implementation

### Modular Architecture

  - **clients.py**: Database connections with lazy loading (LLM, embeddings, vector store)
  - **retrieval.py**: Smart document retrieval with MMR and sentiment-aware filtering
  - **tools.py**: Domain-specific analysis functions (sentiment, aspects, JTBD) - all using LLM with structured output
  - **models.py**: Pydantic schemas for structured tool outputs (Sentiment, AspectAnalysis, JTBD)
  - **prompts.py**: Prompt template loaders and constructors
  - **prompts/**: External prompt text files (rag_system.txt, agent_system.txt, sentiment_analysis.txt, aspects_analysis.txt, jtbd_analysis.txt)
  - **chains.py**: High-level orchestration (simple RAG + keyword-based tool routing)
  - **token_tracker.py**: Session-scoped token tracking and rate limiting
  - **app.py**: Streamlit user interface with formatting functions

### Key Technologies

  - LangChain: AI workflow orchestration
  - OpenAI: GPT-4o for reasoning + text-embedding-3-small for search
  - Chroma: Vector database for semantic search
  - SQLite: Traditional database for structured queries
  - Streamlit: Web interface


## User Experience

### Input Options

- Natural language questions
- Filter by cloud provider (OVH, Scaleway, etc.)
- Choose analysis depth (simple vs. comprehensive/agentic)

### Output Format

- Main Response: Bullet points with source citations
- Tool Analysis: Structured insights (sentiment breakdown, aspect rankings, jobs-to-be-done)
- Retrieved Context: Source review snippets for transparency
- Export: Download data as CSV


## Domain Specialization

Focused on Cloud Hosting Reviews:
- Understands hosting-specific aspects (uptime, pricing, support)
- Recognizes cloud provider names and services
- Uses appropriate terminology and context


## Smart Features

### Query Routing

- Keyword-based detection of user intent
- Routes to appropriate analysis tools automatically:
  - Sentiment keywords: `summarize_sentiment`
  - Aspect keywords: `extract_top_aspects`
  - JTBD keywords: `infer_jtbd`
- Applies sentiment filters to retrieval based on query keywords
- If no specific tool detected, runs all three for comprehensive analysis

### Citation & Trust

  - Every claim backed by specific review citations
  - Source transparency with dates and customer names
  - Handles review text as untrusted data (security)


## Use Cases

1. Product Managers: "What features are customers requesting?"
2. Customer Success: "What are the main pain points?"
3. Marketing: "What do customers love most about our service?"
4. Competitive Analysis: "How do customers compare us to competitors?"


## Unique Value Proposition

- Scale: Analyzes thousands of reviews instantly
- Intelligence: Goes beyond keyword search to understand meaning
- Accuracy: Cites specific sources for every insight
- Flexibility: Handles both quick questions and deep analysis
- Domain-Aware: Understands cloud hosting business context

The system transforms unstructured customer feedback into actionable business insights through the combination of modern AI retrieval, domain expertise, and intelligent analysis tools.