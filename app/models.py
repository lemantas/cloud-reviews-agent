from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class RetrievalInput(BaseModel):
    """Validated inputs for the retrieval tool."""
    question: str
    chunk_type: Literal["sentence", "review"] = Field(
        default="sentence",
        description='Retrieval granularity: "sentence" for precision; "review" for broader context',
    )
    vendor: Optional[Literal["ovh", "scaleway", "hetzner", "digital_ocean", "vultr", "cherry_servers"]] = Field(
        default=None,
        description="Restrict to a known provider when specified",
    )
    top_k: int = Field(
        default=12,
        ge=10,
        le=100,
        description="Results to return (~10–30 for review-level; ~50–200 for sentence-level)",
    )
    fetch_k: int = Field(
        default=30,
        ge=15,
        le=300,
        description="Candidate pool before diversification (~1.5–3× bigger than top_k)",
    )

class Snippet(BaseModel):
    """A review snippet with metadata."""
    text: str
    rating: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    vendor: Optional[str] = None
    review_header: Optional[str] = None

class RetrievalResult(BaseModel):
    """Normalized retrieval payload returned by retrieve_documents function."""
    snippets: List[Snippet] = Field(..., description="List of review snippets to analyze")
    count: int = Field(..., description="Total number of snippets analyzed")

class ToolInput(BaseModel):
    """Standardized input payload for analysis tools."""
    snippets: List[Snippet] = Field(..., description="List of review snippets to analyze")
    question: str = Field(..., description="Question to analyze the snippets for")

class Sentiment(BaseModel):
    """Complete sentiment analysis result from reviews."""
    total_reviews: int = Field(0, description="Total number of reviews analyzed")
    mean_rating: Optional[float] = Field(None, description="Average rating score")
    positive_share: Optional[float] = Field(None, description="Percentage of positive reviews (rating >= 4)")
    negative_share: Optional[float] = Field(None, description="Percentage of negative reviews (rating <= 2)")
    positive_themes: List[str] = Field(default_factory=list, description="Top positive themes/quotes")
    negative_themes: List[str] = Field(default_factory=list, description="Top negative themes/quotes")

class Aspect(BaseModel):
    """A single product/service aspect analysed from reviews."""
    name: str = Field(description="Name of the aspect (e.g., 'performance', 'pricing')")
    frequency: int = Field(description="Number of times mentioned")
    sentiment_score: Optional[float] = Field(None, description="Average sentiment score for this aspect")
    positive_examples: List[str] = Field(default_factory=list, description="Positive mentions")
    neutral_examples: List[str] = Field(default_factory=list, description="Neutral mentions")
    negative_examples: List[str] = Field(default_factory=list, description="Negative mentions")

class AspectAnalysis(BaseModel):
    """Complete aspect analysis results from a list of aspects from reviews."""
    total_aspects: int = Field(description="Total number of aspects found")
    aspects: List[Aspect] = Field(description="List of analyzed aspects with details")

class JTBD(BaseModel):
    """Complete Jobs-to-Be-Done analysis result from reviews."""
    job: str = Field(description="The functional job customers are trying to accomplish")
    situation: str = Field(description="The context/situation when this job arises")
    motivation: str = Field(description="Why customers want to accomplish this job")
    expected_outcome: str = Field(description="What success looks like for customers")
    frustrations: List[str] = Field(default_factory=list, description="Common pain points")
    quotes: List[str] = Field(default_factory=list, description="Supporting customer quotes")