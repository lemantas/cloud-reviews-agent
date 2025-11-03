from langchain.tools import tool
from clients import get_llm
from prompts import get_jtbd_prompt, get_aspects_prompt, get_sentiment_prompt
from models import JTBD, AspectAnalysis, Sentiment
import json

@tool
def summarize_sentiment(snippets_data: str, question: str) -> dict:
    """Analyze overall sentiment and emotional tone of customer reviews.

    Use this tool when the user asks about:
    - Overall sentiment, feelings, or opinions (e.g., "How do customers feel?")
    - Satisfaction levels or happiness indicators
    - Positive vs negative feedback distribution
    - Rating statistics and trends
    - Emotional themes in reviews

    Returns: Dictionary with mean_rating, positive_share, negative_share, and key themes.
    """
    try:
        snippets_list = json.loads(snippets_data)
        snippets = [{'text': s.get('text'), 'rating': s.get('rating')} for s in snippets_list if s.get('text')]

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

@tool
def extract_top_aspects(snippets_data: str, question: str) -> dict:
    """Identify and rank specific product/service features mentioned in customer reviews.

    Use this tool when the user asks about:
    - Specific features or aspects (e.g., "What features do customers discuss?")
    - Product attributes like performance, pricing, support, reliability
    - What customers talk about most frequently
    - Feature-level sentiment (which features are loved/hated)
    - Topic distribution across reviews

    Returns: Dictionary with ranked aspects including frequency, sentiment scores, and example quotes.
    """
    try:
        snippets_list = json.loads(snippets_data)
        snippets = [{'text': s.get('text'), 'rating': s.get('rating')} for s in snippets_list if s.get('text') and s.get('rating')]

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

@tool
def infer_jtbd(snippets_data: str, question: str) -> dict:
    """Analyze customer goals, motivations, and Jobs-to-Be-Done from reviews.

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
        snippets_list = json.loads(snippets_data)
        snippets = [{'text': s.get('text'), 'rating': s.get('rating')} for s in snippets_list if s.get('text')]

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