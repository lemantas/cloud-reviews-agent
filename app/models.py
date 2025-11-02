from pydantic import BaseModel, Field
from typing import List, Optional

# Not used anywhere yet, but keeping for now
class Snippet(BaseModel):
    """A review snippet with metadata."""
    text: str
    rating: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    vendor: Optional[str] = None
    review_header: Optional[str] = None

# Not used anywhere yet, but keeping for now
class ToolInput(BaseModel):
    """Input for analysis tools."""
    snippets: List[Snippet]
    question: Optional[str] = None

class Sentiment(BaseModel):
    """Sentiment analysis summary of reviews."""
    total_reviews: int = Field(0, description="Total number of reviews analyzed")
    mean_rating: Optional[float] = Field(None, description="Average rating score")
    positive_share: Optional[float] = Field(None, description="Percentage of positive reviews (rating >= 4)")
    negative_share: Optional[float] = Field(None, description="Percentage of negative reviews (rating <= 2)")
    positive_themes: List[str] = Field(default_factory=list, description="Top positive themes/quotes")
    negative_themes: List[str] = Field(default_factory=list, description="Top negative themes/quotes")

class Aspect(BaseModel):
    """A single product/service aspect mentioned in reviews."""
    name: str = Field(description="Name of the aspect (e.g., 'performance', 'pricing')")
    frequency: int = Field(description="Number of times mentioned")
    sentiment_score: Optional[float] = Field(None, description="Average sentiment score for this aspect")
    positive_examples: List[str] = Field(default_factory=list, description="Positive mentions")
    neutral_examples: List[str] = Field(default_factory=list, description="Neutral mentions")
    negative_examples: List[str] = Field(default_factory=list, description="Negative mentions")

class AspectAnalysis(BaseModel):
    """Complete aspect analysis results from reviews."""
    total_aspects: int = Field(description="Total number of aspects found")
    aspects: List[Aspect] = Field(description="List of analyzed aspects with details")

class JTBD(BaseModel):
    """Jobs-to-Be-Done insight from customer reviews."""
    job: str = Field(description="The functional job customers are trying to accomplish")
    situation: str = Field(description="The context/situation when this job arises")
    motivation: str = Field(description="Why customers want to accomplish this job")
    expected_outcome: str = Field(description="What success looks like for customers")
    frustrations: List[str] = Field(default_factory=list, description="Common pain points")
    quotes: List[str] = Field(default_factory=list, description="Supporting customer quotes")