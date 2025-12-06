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
- Answer Generation: GPT-4o-mini creates responses with proper attribution (optimized for faster responses)

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

## Advanced Agentic Features

The system implements sophisticated AI agent capabilities that enable intelligent, context-aware analysis:

### 1. Autonomous Tool Selection

The LLM intelligently decides which analysis tools to invoke based on query intent - no keyword matching required:

- **Smart Routing:** "Analyze Scaleway reviews" → Agent autonomously runs sentiment analysis, aspect extraction, and JTBD analysis
- **Intent Understanding:** "What frustrates customers?" → Agent chooses JTBD analysis without being explicitly told
- **Flexible Combinations:** Agent can invoke 0, 1, 2, or all tools depending on what's needed for the query
- **User-Friendly:** Users ask natural questions without needing to know which tools exist

**Implementation:** LangChain's `create_agent()` framework with GPT-4o-mini making real-time tool selection decisions

### 2. Multi-Step Reasoning & Dynamic Retrieval

ReAct-style agent that iterates through analysis cycles for comprehensive insights:

- **Comparative Analysis:** "Compare pricing sentiment between OVH and Scaleway" → Agent retrieves OVH reviews → analyzes → retrieves Scaleway reviews → analyzes → synthesizes comparison
- **Iterative Refinement:** Can retrieve reviews with different parameters based on initial findings
- **Drill-Down Capability:** Discovers pricing issues → retrieves more pricing-specific reviews → provides deeper analysis
- **Natural Flow:** Multi-turn reasoning without requiring user to break down the request

**Example Agent Flow:**
```
User: "Compare pricing sentiment between OVH and Scaleway"
→ retrieve_reviews(vendor="ovh", question="pricing")
→ sentiment_analysis(ovh_reviews)
→ retrieve_reviews(vendor="scaleway", question="pricing")
→ sentiment_analysis(scaleway_reviews)
→ Synthesize comparison with citations
```

### 3. Conversational Memory & Follow-Ups

Full conversation context tracking enables natural multi-turn interactions:

- **Context Preservation:** "Tell me about Scaleway" → [response] → "How does that compare to OVH?" (agent understands "that" refers to Scaleway)
- **Follow-Up Suggestions:** Agent can ask clarifying questions when queries are ambiguous
- **Iterative Exploration:** Users can drill down into findings across multiple turns without repeating context
- **Session Management:** Clear conversation button to start fresh analyses

**Benefits:** Makes the agent feel interactive and helpful, guiding users through complex analyses naturally

## Technical Implementation

### Modular Architecture

  - **clients.py**: Database connections with lazy loading (LLM, embeddings, vector store)
  - **retrieval.py**: Smart document retrieval with MMR and sentiment-aware filtering
  - **tools.py**: Domain-specific analysis functions (sentiment, aspects, JTBD) - all using LLM with structured output
  - **models.py**: Pydantic schemas for structured tool outputs (Sentiment, AspectAnalysis, JTBD)
  - **prompts.py**: Prompt template loaders and constructors
  - **prompts/**: External prompt text files (rag_system.txt, agent_system.txt, sentiment_analysis.txt, aspects_analysis.txt, jtbd_analysis.txt)
  - **chains.py**: High-level orchestration (simple RAG + agentic response with LLM-driven tool selection)
  - **token_tracker.py**: Session-scoped token tracking and rate limiting (100k limit)
  - **app.py**: Streamlit user interface with conversation state management and formatting functions

### Key Technologies

  - LangChain: AI workflow orchestration with Agents API
  - OpenAI: GPT-4o-mini for reasoning (faster responses) + text-embedding-3-small for search
  - Chroma: Vector database for semantic search
  - SQLite: Traditional database for structured queries
  - Streamlit: Web interface with session state management


## User Experience

### Input Options

- Natural language questions
- Filter by cloud provider (OVH, Scaleway, etc.)
- Choose analysis depth (simple vs. comprehensive/agentic)

### Output Format

- **Main Response:** Synthesized insights with inline source citations `[source | YYYY-MM-DD]`
- **Tool Analysis:** Structured insights in expandable sections
  - Sentiment breakdown (ratings, themes, statistics)
  - Aspect rankings (frequency, sentiment scores, example quotes)
  - Jobs-to-be-done patterns (motivations, frustrations, outcomes)
  - Multi-tool support: Same tool called multiple times numbered (#1, #2, etc.)
- **Retrieved Context:** Source review snippets with full metadata (rating, date, vendor, review header)
- **Database Statistics (Sidebar):**
  - Total review count
  - Per-vendor breakdown with percentages
  - Example: `ovh: 2,234 (18.1%)`
- **Token Usage Display:**
  - Real-time tracking with progress bar (100k limit per session)
  - Warning at 90% threshold
  - Message count indicator
- **Interactive Features:**
  - Clear conversation button
  - Example questions for quick demos
  - Collapsible tool output expanders
- **Export:** Download complete results as CSV with metadata


## Domain Specialization

Focused on Cloud Hosting Reviews:
- Understands hosting-specific aspects (uptime, pricing, support)
- Recognizes cloud provider names and services
- Uses appropriate terminology and context


## Smart Features

### Intelligent Tool Selection

The agent uses LLM-driven reasoning to select appropriate analysis tools - no hardcoded keywords required:

- **Natural Language Understanding:** GPT-4o-mini interprets query intent and autonomously chooses which tools to invoke
- **Adaptive Analysis:** "What frustrates customers?" → Agent recognizes JTBD analysis needed; "How's the pricing?" → Aspect extraction + sentiment analysis
- **Multi-Tool Orchestration:** Complex queries trigger multiple tools in sequence (e.g., "Analyze Scaleway" → all 3 analysis tools)
- **Dynamic Retrieval:** Agent can call `retrieve_reviews` multiple times with different parameters for comparative or iterative analysis

**Example Intelligence:**
```
Query: "How do customers feel about Scaleway pricing?"
Agent Reasoning:
  1. Recognizes vendor filter needed (Scaleway)
  2. Identifies pricing aspect focus
  3. Determines sentiment analysis required
  4. Calls: retrieve_reviews(vendor="scaleway", question="pricing")
     → aspect_extraction() → sentiment_analysis()
  5. Synthesizes pricing-specific sentiment insights
```

**Advantages Over Keyword Routing:**
- Handles ambiguous phrasing naturally ("What's the vibe?" → sentiment analysis)
- No maintenance of keyword lists needed
- Understands context from conversation history
- Gracefully handles queries requiring multiple tools

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

---

## Future Roadmap

 1. Code Quality & Maintainability

  Priority: Low-Medium | Effort: Low

  - ✅ Make variable names more descriptive, eliminate unnecessary comments
    - Clarification: Focus on complex logic in chains.py, retrieval.py, tools.py
    - Recommendation: Do this incrementally during other refactors to avoid dedicated effort
    - Impact: Moderate improvement in maintainability

  ---
  2. Architecture Overhaul

  Priority: Medium-High | Effort: High

  - ✅ Refactor with LangGraph
    - Clarification: Replace create_agent() with LangGraph's state machine for better control flow
    - Benefits: Better observability, easier debugging, explicit state management
    - Consideration: Breaking change - requires full rewrite of chains.py
    - Recommendation: Do this BEFORE multi-agent work (LangGraph makes multi-agent easier)
  - ✅ Refactor to web services + separate frontend
    - Clarification:
        - Backend: FastAPI REST API with agent execution endpoints
      - Frontend: React/Vue SPA or keep Streamlit separate
    - Benefits: Better scalability, independent deployment, API for other clients
    - Consideration: Significantly increases complexity and deployment surface
    - Recommendation: Only if you need API access or have multiple frontend clients

  ---
  3. User Experience Enhancements ✨

  Priority: High | Effort: Low-Medium

  - ✅ Stream LLM messages in real-time
    - Clarification: Use stream() instead of invoke() in LangChain
    - Benefits: Better perceived performance, users see progress
    - Implementation: Streamlit has st.write_stream() for this
    - Recommendation: HIGH PRIORITY - quick win with big UX impact
  - ✅ Implement Evals (thumbs up/down rating)
    - Clarification: User feedback on agent responses
    - Answer to your question: YES, LangSmith supports feedback via SDK:
    from langsmith import Client
  client = Client()
  client.create_feedback(run_id, key="user_score", score=1)  # 1=👍, 0=👎
    - Recommendation: Add this early - feedback data informs future improvements

  ---
  4. Tool & Integration Expansion 🔧

  Priority: Medium | Effort: Low-Medium

  - ✅ Use MCP tools (web_search, etc.)
    - Clarification: Which MCP tools? Suggestions:
        - web_search - Lookup vendor incidents/news in real-time
      - filesystem - Read/write reports locally
      - time - For date-based queries
    - Consideration: Each tool adds complexity to agent decision-making
    - Recommendation: Add 1-2 tools max; more tools = worse tool selection accuracy
  - ✅ Create review_response_generator tool
    - Clarification: Generate vendor responses to negative reviews based on JTBD?
    - Use case: "Write a response to this 2-star review about pricing"
    - Recommendation: Good idea - naturally extends JTBD analysis

  ---
  5. Multi-Agent Architecture

  Priority: Medium | Effort: Very High

  Current ideas breakdown:

  - ✅ Supervisor agent coordinates specialists
    - Clarification: Manager decides which specialist(s) to consult
    - Pattern: LangGraph's "supervisor" or "hierarchical" pattern
    - Recommendation: IF you do multi-agent, use this pattern
  - ✅ Parallel tool execution
    - Clarification: Run sentiment + aspect + JTBD simultaneously
    - Benefits: Faster comparative analysis (e.g., compare 3 vendors)
    - Implementation: LangGraph's parallel edges or asyncio
    - Recommendation: HIGH VALUE - significant speed improvement for comparisons
  - ✅ Human-in-the-loop
    - Clarification: Agent asks clarifying questions mid-execution
    - Example: "Do you want performance aspects only, or all aspects?"
    - Implementation: LangGraph interrupt() + Streamlit input
    - Recommendation: MEDIUM PRIORITY - good for ambiguous queries

  ---
  6. Data Collection & Processing

  Priority: Medium | Effort: Medium

  - ✅ Automate Trustpilot scraping
    - Clarification: Scheduled scraping? Trigger-based? Real-time?
    - Options:
        - Cron job / GitHub Actions daily
      - Webhook on new review
      - On-demand via Streamlit button
    - Consideration: Trustpilot may rate-limit or block
    - Recommendation: Start with simple scheduled job
  - ✅ Date restrictions in ChromaDB queries
    - Clarification: Filter by review date range (e.g., "reviews from last 6 months")
    - Implementation: Add date to metadata filter in retrieval.py
    - Recommendation: HIGH VALUE - enables trend analysis
  - ✅ Vendor trends over time with monthly overviews
    - Clarification: Time-series analysis of sentiment/aspects by month
    - Dependencies: Requires date restrictions (above)
    - Implementation: New tool or Streamlit dashboard with charts
    - Recommendation: GREAT FEATURE - adds longitudinal insights

## My Summary List of Future Features
What I would like for the capstone project:
- Make code variable names more descriptive, and eliminate code comments, where possible. Only use code comments where really necessary.
- Refactor architecture with LangGraph
- Stream LLM messages in real-time
- Refactor the app based on web services, and create separate front-end
- Use a few new MCP tools like web_search, and others that make sense here.
- Build multi-agent architecture. Possible ideas:
    - Specialist agents: Sentiment Specialist, Aspect Specialist, JTBD Specialist
    -  Supervisor agent coordinates specialist agents for complex queries
    - Parallel tool execution for faster comparative analysis
    - Agent collaboration for multi-vendor deep dives
    - Multi-agent debate: two agents analyze same query, synthesize best answer
    - Human-in-the-loop: agent pauses to ask clarifying questions mid-analysis
- Automate Trustpilot scraping
- Implement Evals (thumbs up/down rating; can this be saved in LangSmith?)
- Implement date restrictions when querying ChromaDB
- Implement Vendor review trends over time with monthly overviews and mean scores
- Authentication and long-term memory
- Create review_response_generator tool based on JTBD analysis.