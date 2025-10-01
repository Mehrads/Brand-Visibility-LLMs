"""
Pydantic models for data validation and structure
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from urllib.parse import urlparse


class Citation(BaseModel):
    """Represents a citation found in the response"""
    text: str = Field(..., description="Context around the citation")
    url: str = Field(..., description="URL of the citation")
    entity: Optional[str] = Field(None, description="Entity being cited")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate that the URL is properly formatted"""
        try:
            result = urlparse(v)
            if not result.scheme or not result.netloc:
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")


class Mention(BaseModel):
    """Represents a brand mention found in the response"""
    text: str = Field(..., description="Context around the mention")
    position: int = Field(..., description="Character position in the response")
    exact_match: str = Field(..., description="Exact text that was matched")
    variation: Optional[str] = Field(None, description="Variation of the brand name found")


class SearchResult(BaseModel):
    """Represents a web search result"""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Snippet or summary of the content")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate that the URL is properly formatted"""
        try:
            result = urlparse(v)
            if not result.scheme or not result.netloc:
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")


class Metadata(BaseModel):
    """Metadata about the analysis process"""
    token_count: int = Field(..., description="Total tokens used in the response")
    searches_performed: int = Field(..., description="Number of web searches performed")
    max_searches: int = Field(..., description="Maximum allowed searches")
    sources_included: int = Field(..., description="Number of sources included")
    max_sources: int = Field(..., description="Maximum allowed sources")
    citations_found: int = Field(..., description="Number of citations found")
    mentions_found: int = Field(..., description="Number of brand mentions found")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for processing")
    model_used: Optional[str] = Field(None, description="LLM model used for generation")


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    human_response_markdown: str = Field(..., description="Human-readable response in markdown")
    citations: List[Citation] = Field(default_factory=list, description="All citations found")
    mentions: List[Mention] = Field(default_factory=list, description="All brand mentions found")
    owned_sources: List[str] = Field(default_factory=list, description="URLs owned by the brand")
    sources: List[str] = Field(default_factory=list, description="External URLs used")
    metadata: Metadata = Field(..., description="Analysis metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return self.dict()


class AnalysisRequest(BaseModel):
    """Request parameters for analysis"""
    brand: str = Field(..., description="Brand name to analyze")
    url: HttpUrl = Field(..., description="Brand website URL")
    question: str = Field(..., description="Question to ask the LLM")
    max_searches: int = Field(default=5, ge=0, le=20, description="Maximum web searches")
    max_sources: int = Field(default=10, ge=0, le=50, description="Maximum sources to include")
    
    @field_validator('brand')
    @classmethod
    def validate_brand(cls, v):
        """Validate brand name is not empty"""
        if not v or not v.strip():
            raise ValueError("Brand name cannot be empty")
        return v.strip()
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate question is not empty"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
