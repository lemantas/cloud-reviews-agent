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


## Limitations & Future Work

### Current Limitations

**1. Session-Based Token Constraints**
- 100k token limit per session prevents analysis of extremely large datasets in a single conversation
- No persistent conversation history across app restarts
- Token tracking resets when session ends, even though it makes sense to give each registered used a limited pool of tokens

**2. Single Model Provider**
- Currently locked to OpenAI (GPT-4o-mini and text-embedding-3-small)
- No fallback options if OpenAI API experiences downtime
- Cost optimization limited to single provider's pricing structure
- Cannot leverage strengths of different models (e.g., Claude for analysis, Gemini for summarization)

**3. Static Knowledge Base**
- Review data requires manual ingestion via CSV files
- No real-time integration with Trustpilot or other review platforms
- Cannot automatically update when new reviews are posted
- Scraper (`scrape_reviews.py`) must be run manually
- Scraper is custom-coded, even though there must be more robust open source scrapers already available 

**4. No User Management**
- No authentication or user accounts
- Cannot save user preferences, favorite queries, or analysis history
- All users share the same database and token pool
- No personalization based on user role or industry

**5. Performance Bottlenecks**
- No caching mechanism for frequently asked queries
- Every query triggers fresh vector search and LLM calls
- Popular queries (e.g., "What are common complaints?") re-analyzed each time
- Agent tool calls can be slow for complex multi-step reasoning
- Tool outputs must be streamed to the user in real time, as one now needs to wait more than a minute to receive an answer.

**6. Limited Feedback Loop**
- No mechanism for users to rate response quality
- Cannot learn which tool combinations work best for specific query types
- No A/B testing for prompt variations
- Manual prompt engineering required for improvements

**7. Scalability Constraints**
- Running on local machine or single Streamlit instance
- No load balancing for multiple concurrent users
- Vector database (Chroma) not optimized for distributed deployment
- SQLite database single-threaded and not suitable for high-concurrency scenarios

**9. Retrieval Limitations**
- MMR parameters (TOP_K=12, FETCH_K=30) are hardcoded
- No hybrid search (semantic + keyword BM25). Keyword search might suit aspects_analysis well.
- Cannot handle queries requiring temporal analysis (e.g., "How has sentiment changed over time?"), nor put date restrictions.
- Deduplication by review_id may miss important repeated mentions across reviews

---

### Future Enhancements

#### **Near-Term Improvements (1-2 weeks)**

**1. Enhanced User Experience**
- [ ] Add LLM model selector dropdown (GPT-4o-mini, GPT-4o, Claude Sonnet)
- [ ] Implement query caching with Redis for common questions (30-60s TTL)
- [ ] Add "Save Analysis" button to export full conversation as PDF/Markdown
- [ ] Create onboarding tutorial with interactive walkthrough

**2. Retrieval Optimization**
- [ ] Implement hybrid search (semantic embeddings + BM25 keyword matching)
- [ ] Add temporal filtering (date range picker for review analysis)

**3. Feedback & Learning**
- [ ] Add thumbs up/down rating for each agent response
- [ ] Store feedback in SQLite for prompt optimization analysis
- [ ] A/B test different system prompts based on feedback scores

#### **Mid-Term Enhancements (1-2 months)**

**4. Multi-Model Support**
- [ ] Add Anthropic Claude (Sonnet/Opus) for deeper analysis
- [ ] Integrate Google Gemini as cost-effective alternative
- [ ] Implement automatic failover if primary model is unavailable
- [ ] Model routing: use GPT-4o-mini for simple Q&A, GPT-4o for complex reasoning

**5. Real-Time Data Integration**
- [ ] Build scheduled Trustpilot scraper (runs daily/weekly)
- [ ] Add webhook support for real-time review ingestion
- [ ] Implement incremental vector store updates (avoid full rebuild)
- [ ] Add data freshness indicator in UI ("Last updated: 2 hours ago")

**6. Advanced Analytics**
- [ ] Temporal sentiment analysis (trend charts over time)
- [ ] Comparative dashboards (vendor A vs vendor B side-by-side)
- [ ] Topic modeling for discovering emergent themes
- [ ] Anomaly detection for sudden sentiment shifts

**7. User Management & Personalization**
- [ ] Implement authentication (OAuth, email/password)
- [ ] Save user query history and favorite analyses
- [ ] Role-based access (Product Manager sees different defaults than Support team)
- [ ] Personal knowledge (each user is able to put their goals into the system)

#### **Long-Term Vision (3-6 months)**

**8. Production Deployment**
- [ ] Deploy to AWS/GCP/Azure with auto-scaling
- [ ] Replace SQLite with PostgreSQL for concurrent access
- [ ] Deploy Chroma on dedicated vector DB service (Pinecone, Weaviate, or Qdrant)
- [ ] Implement CDN caching for static assets
- [ ] Add monitoring with Prometheus + Grafana
- [ ] Set up CI/CD pipeline with automated testing

**9. Multi-Agent Workflows**
- [ ] Specialist agents: Sentiment Specialist, Aspect Specialist, JTBD Specialist
- [ ] Supervisor agent coordinates specialist agents for complex queries
- [ ] Parallel tool execution for faster comparative analysis
- [ ] Agent collaboration for multi-vendor deep dives

**10. Domain Expansion**
- [ ] Make aspect taxonomy configurable (upload custom aspect list)
- [ ] Auto-detect domain from reviews (hospitality vs SaaS vs e-commerce)
- [ ] Build domain adapter framework for easy vertical expansion
- [ ] Create marketplace for pre-trained domain configurations

**11. Advanced RAG Techniques**
- [ ] Fine-tune embedding model on domain-specific reviews
- [ ] Implement graph RAG for relationship discovery (e.g., "pricing complaints → churn risk")
- [ ] Add citation verification (check if LLM claims match retrieved context)
- [ ] Implement RAPTOR (recursive abstractive processing) for hierarchical summarization (perhaps with web search tool for home page analysis?; need pieces of content with broader context than reviews)

**12. Intelligence Enhancements**
- [ ] Implement retrieval-augmented fine-tuning (RAFT)
- [ ] Add reasoning traces (show agent's thought process step-by-step)
- [ ] Self-critique mechanism (agent reviews its own responses before showing user)

---

### Research & Experimental Ideas

**1. Agentic Collaboration**
- Multi-agent debate: two agents analyze same query, synthesize best answer
- Human-in-the-loop: agent pauses to ask clarifying questions mid-analysis
- Recursive analysis: agent generates follow-up questions and answers them autonomously

**2. Advanced Evaluation**
- Implement RAGAs framework for systematic quality measurement
- Create golden dataset of query-answer pairs for regression testing
- A/B test different chunking strategies (sentence vs full review)
- Measure retrieval precision/recall with human-labeled relevance scores

**3. Cost Optimization**
- Prompt compression (reduce token usage without losing quality)
- Intelligent caching with semantic similarity (cache similar queries, not just exact matches)
- Batch processing for non-real-time queries
- Model distillation: use smaller local open source model to mimic GPT-4o-mini behavior

**4. Novel Features**
- Compare user perception against vendor positioning to find company positioning gaps
- "Ask Me Anything" mode: agent proactively suggests unexplored analysis angles
- Competitive intelligence: auto-generate competitor comparison reports and send them via email
- Predictive insights: "Based on trends, expect pricing complaints to increase"
- Review response generator: draft responses to negative reviews using JTBD insights