from langchain.tools import tool
from clients import get_llm
from prompts import get_jtbd_prompt, get_aspects_prompt, get_sentiment_prompt
from models import JTBD, AspectAnalysis, Sentiment, Snippet, RetrievalResult, RetrievalInput, ToolInput
from retrieval import retrieve_documents
import json


def _normalize_snippets(snippets):
    """Normalize snippet inputs to a uniform list of dicts with keys 'text' and 'rating'.

    - Accepts a list of strings or dicts; ignores empty/invalid items
    - For strings: maps to {"text": <str>, "rating": None}
    - For dicts: preserves 'text' and 'rating' if available
    - For Pydantic models (e.g., Snippet), uses model_dump()
    """
    normalized = []
    if not isinstance(snippets, list):
        return normalized
    for item in snippets:
        if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
            try:
                item = item.model_dump()
            except Exception:
                item = {}
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append({"text": text, "rating": None})
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                normalized.append({"text": text.strip(), "rating": item.get("rating")})
    return normalized


@tool("sentiment_analysis", args_schema=ToolInput)
def summarize_sentiment(snippets: list[str | dict], question: str) -> dict:
    """Analyze overall sentiment and emotional tone of customer reviews.
    Requires snippets from retrieve_reviews. Do not call this tool more than once unless you retrieved new snippets.

    Use this tool when the user asks about:
    - Overall sentiment, feelings, or opinions (e.g., "How do customers feel?")
    - Satisfaction levels or happiness indicators
    - Positive vs negative feedback distribution
    - Rating statistics and trends
    - Emotional themes in reviews

    Returns: Dictionary with mean_rating, positive_share, negative_share, and key themes.
    """
    try:
        snippets = _normalize_snippets(snippets)
        if not snippets:
            return {"error": "No review data available for sentiment analysis."}

        llm = get_llm()
        sentiment_prompt = get_sentiment_prompt()
        formatted_prompt = sentiment_prompt.format(reviews=json.dumps(snippets), question=question)

        # Use structured output with Sentiment model
        response = llm.with_structured_output(Sentiment).invoke(formatted_prompt)

        # Check if no reviews were analyzed
        if response.total_reviews == 0:
            return {"error": "No review data available for sentiment analysis."}

        # Convert Sentiment model to dictionary
        return {
            "total_reviews": response.total_reviews,
            "mean_rating": response.mean_rating,
            "positive_share": response.positive_share,
            "negative_share": response.negative_share,
            "positive_themes": response.positive_themes,
            "negative_themes": response.negative_themes
        }

    except Exception as e:
        return {"error": f"Error analyzing sentiment: {str(e)}"}

@tool("aspect_extraction", args_schema=ToolInput)
def extract_top_aspects(snippets: list[str | dict], question: str) -> dict:
    """Identify and rank specific product/service features mentioned in customer reviews.
    Requires snippets from retrieve_reviews. Do not call this tool more than once unless you retrieved new snippets.

    Use this tool when the user asks about:
    - Specific features or aspects (e.g., "What features do customers discuss?")
    - Product attributes like performance, pricing, support, reliability
    - What customers talk about most frequently
    - Feature-level sentiment (which features are loved/hated)
    - Topic distribution across reviews

    Returns: Dictionary with ranked aspects including frequency, sentiment scores, and example quotes.
    """
    try:
        snippets = _normalize_snippets(snippets)
        if not snippets:
            return {"error": "No review data available for aspect extraction."}

        llm = get_llm()
        aspects_prompt = get_aspects_prompt()
        formatted_prompt = aspects_prompt.format(reviews=json.dumps(snippets), question=question)

        # Use structured output with AspectAnalysis model
        response = llm.with_structured_output(AspectAnalysis).invoke(formatted_prompt)

        # Check if no aspects were found
        if response.total_aspects == 0 or not response.aspects:
            return {"error": "No specific aspects were identified in the reviews."}

        # Convert AspectAnalysis model to dictionary
        return {
            "total_aspects": response.total_aspects,
            "aspects": [
                {
                    "name": aspect.name,
                    "frequency": aspect.frequency,
                    "sentiment_score": aspect.sentiment_score,
                    "positive_examples": aspect.positive_examples,
                    "neutral_examples": aspect.neutral_examples,
                    "negative_examples": aspect.negative_examples
                }
                for aspect in response.aspects
            ]
        }

    except Exception as e:
        return {"error": f"Error extracting aspects: {str(e)}"}

@tool("jtbd_analysis", args_schema=ToolInput)
def infer_jtbd(snippets: list[str | dict], question: str) -> dict:
    """Analyze customer goals, motivations, and Jobs-to-Be-Done from reviews.
    Requires snippets from retrieve_reviews. Do not call this tool more than once unless you retrieved new snippets.

    Use this tool when the user asks about:
    - Why customers choose the service (e.g., "What are customers trying to accomplish?")
    - Customer goals, needs, or objectives
    - Use cases and scenarios
    - Problems customers are trying to solve
    - Customer motivations and expected outcomes
    - Pain points and frustrations

    Returns: Dictionary with job description, situation, motivation, expected outcomes, and frustrations.
    """
    try:
        snippets = _normalize_snippets(snippets)
        if not snippets:
            return {"error": "No review data available for JTBD analysis."}

        llm = get_llm()

        jtbd_prompt = get_jtbd_prompt()
        formatted_prompt = jtbd_prompt.format(reviews=json.dumps(snippets), question=question)

        # Use structured output with JTBD model
        response = llm.with_structured_output(JTBD).invoke(formatted_prompt)

        return {
            "job": response.job,
            "situation": response.situation,
            "motivation": response.motivation,
            "expected_outcome": response.expected_outcome,
            "frustrations": response.frustrations,
            "quotes": response.quotes,
            "total_reviews": len(snippets)
        }

    except Exception as e:
        return {"error": f"Error performing JTBD analysis: {str(e)}"}


@tool("retrieve_reviews", args_schema=RetrievalInput)
def retrieve_reviews(
    question: str,
    chunk_type: str = "sentence",
    vendor: str | None = None,
    top_k: int | None = None,
    fetch_k: int | None = None,
) -> dict:
    """Retrieve relevant review snippets from the ChromaDB vector store to best answer the question. 
    Always use this tool to best answer the question. Optionally, use it between analyses to get more context.
    """
    try:
        raw_snippets = retrieve_documents(
            question=question,
            chunk_type=chunk_type,
            vendor=vendor,
            top_k=top_k,
            fetch_k=fetch_k,
        )
        # Normalize into Snippet models
        snippets: list[Snippet] = []
        for snip in raw_snippets:
            if isinstance(snip, dict) and snip.get("text"):
                snippets.append(Snippet(
                    text=str(snip.get("text", "")),
                    rating=snip.get("rating"),
                    date=snip.get("date"),
                    source=snip.get("source"),
                    vendor=snip.get("vendor"),
                    review_header=snip.get("review_header"),
                ))
            elif isinstance(snip, str) and snip.strip():
                snippets.append(Snippet(text=snip.strip()))

        payload = RetrievalResult(snippets=snippets, count=len(snippets))
        return payload.model_dump()
    except Exception as e:
        return {"error": f"Error retrieving reviews: {str(e)}"}