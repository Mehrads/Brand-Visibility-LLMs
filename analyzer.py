"""
Main analyzer for brand visibility in LLM responses
"""
import time
from typing import List, Set, Union
from models import AnalysisRequest, AnalysisResult, Metadata, SearchResult
from llm_client import LLMClient
from llm_client_gemma import OpenSourceLLMClient
from web_search import WebSearchEngine


class BrandAnalyzer:
    """
    Main analyzer for brand visibility in LLM-generated answers
    
    Orchestrates the entire analysis process:
    1. Web search with budget constraints
    2. LLM response generation with token optimization
    3. Citation and mention detection using LLM
    4. Source categorization
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", use_open_source: bool = False,
                 enable_llmlingua: bool = False):
        """
        Initialize analyzer with either OpenAI or open-source client
        
        Args:
            api_key: API key for OpenAI/OpenRouter
            model: Model name (for OpenAI) or open-source model name
            use_open_source: Whether to use open-source model instead of OpenAI
            enable_llmlingua: Enable LLMLingua compression (LLM decides when to use it)
        """
        if use_open_source:
            self.llm_client = OpenSourceLLMClient(api_key=api_key, model=model, enable_llmlingua=enable_llmlingua)
        else:
            self.llm_client = LLMClient(api_key=api_key, model=model, enable_llmlingua=enable_llmlingua)
        self.web_search = None  # Will be initialized with request parameters
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform complete brand visibility analysis
        
        Args:
            request: AnalysisRequest with all parameters
            
        Returns:
            AnalysisResult with complete analysis
        """
        start_time = time.time()
        
        # Initialize web search with budget constraints
        self.web_search = WebSearchEngine(
            max_searches=request.max_searches,
            max_sources=request.max_sources
        )
        
        # Step 1: Perform web search if budget allows
        search_results = self._perform_web_search(request)
        
        # Step 2: Generate LLM response with token optimization
        human_response = self._generate_response(request, search_results)
        
        # Step 3: Analyze response for citations and mentions using LLM
        citations, mentions = self._analyze_citations_and_mentions(
            human_response, request.brand
        )
        
        # Step 4: Collect and categorize sources
        all_sources = self._collect_all_sources(search_results, citations)
        owned_sources, external_sources = self._categorize_sources(
            all_sources, str(request.url)
        )
        
        # Step 5: Create metadata
        processing_time = time.time() - start_time
        metadata = self._create_metadata(
            human_response, search_results, citations, mentions, processing_time
        )
        
        return AnalysisResult(
            human_response_markdown=human_response,
            citations=citations,
            mentions=mentions,
            owned_sources=owned_sources,
            sources=external_sources,
            metadata=metadata
        )
    
    def _perform_web_search(self, request: AnalysisRequest) -> List[SearchResult]:
        """Perform web search with budget constraints"""
        if not self.web_search.can_search_more():
            return []
        
        # Create search query combining brand and question
        search_query = f"{request.brand} {request.question}"
        
        # Perform search
        search_results = self.web_search.search_web(search_query, num_results=5)
        
        return search_results
    
    def _generate_response(self, request: AnalysisRequest, 
                          search_results: List[SearchResult]) -> str:
        """Generate LLM response with token optimization"""
        return self.llm_client.generate_response(
            question=request.question,
            brand_name=request.brand,
            brand_url=str(request.url),
            search_results=search_results,
            max_tokens=1000  # Response length management strategy
        )
    
    def _analyze_citations_and_mentions(self, response_text: str, 
                                      brand_name: str) -> tuple[List, List]:
        """Analyze citations and mentions using LLM"""
        return self.llm_client.analyze_citations_and_mentions(
            response_text, brand_name
        )
    
    def _collect_all_sources(self, search_results: List[SearchResult], 
                           citations: List) -> List[str]:
        """Collect all source URLs from search results and citations"""
        all_sources = set()
        
        # Add URLs from search results
        for result in search_results:
            all_sources.add(result.url)
        
        # Add URLs from citations
        for citation in citations:
            all_sources.add(citation.url)
        
        return list(all_sources)
    
    def _categorize_sources(self, all_sources: List[str], 
                          brand_url: str) -> tuple[List[str], List[str]]:
        """Categorize sources as owned or external using LLM"""
        return self.llm_client.categorize_sources(all_sources, brand_url)
    
    def _create_metadata(self, response_text: str, search_results: List[SearchResult],
                        citations: List, mentions: List, 
                        processing_time: float) -> Metadata:
        """Create metadata about the analysis process"""
        return Metadata(
            token_count=self.llm_client.count_tokens(response_text),
            searches_performed=self.web_search.search_count,
            max_searches=self.web_search.max_searches,
            sources_included=len(search_results) + len(citations),
            max_sources=self.web_search.max_sources,
            citations_found=len(citations),
            mentions_found=len(mentions),
            processing_time_seconds=processing_time,
            model_used=self.llm_client.model
        )
