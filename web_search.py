"""
Web search module with budget constraints
"""
import requests
import os
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from models import SearchResult
from urllib.parse import urlparse
from tavily import TavilyClient


class WebSearchEngine:
    """
    Web search engine with budget constraints
    
    Uses Tavily Search API for real web search results.
    Enforces hard caps on searches and sources as required.
    """
    
    def __init__(self, max_searches: int = 5, max_sources: int = 10):
        self.max_searches = max_searches
        self.max_sources = max_sources
        self.search_count = 0
        self.sources_used = 0
        self.api_key = os.getenv('TAVILY_API_KEY')
        
        # Initialize Tavily client if API key is available
        self.tavily_client = None
        if self.api_key:
            try:
                self.tavily_client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Tavily client: {e}")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def can_search_more(self) -> bool:
        """Check if more searches are allowed within budget"""
        return self.search_count < self.max_searches and self.sources_used < self.max_sources
    
    def search_web(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform web search with budget constraints using Tavily Search API
        """
        if not self.can_search_more():
            return []
        
        # Require an API key and Tavily client explicitly; do not use mock results
        if not self.api_key or not self.tavily_client:
            return []

        # Calculate how many results we can return within budget
        available_sources = self.max_sources - self.sources_used
        results_to_return = min(num_results, available_sources, 10)
        
        if results_to_return <= 0:
            return []
        
        try:
            results = self._search_tavily(query, results_to_return)
            
            # Update budget counters
            self.search_count += 1
            self.sources_used += len(results)
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            # On error, return no results (no mock fallback)
            return []
    
    def _search_tavily(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using Tavily Search API"""
        try:
            # Tavily search with content extraction
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",  # or "advanced" for deeper search
                max_results=min(num_results, 10),  # Tavily max is 10 per request
                include_answer=False,  # We'll extract content ourselves
                include_raw_content=True,  # Get full content
                include_domains=[],
                exclude_domains=[]
            )
            
            results = []
            for item in response.get('results', []):
                result = SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('url', ''),
                    snippet=item.get('content', 'No description')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    
    # Mock search removed per requirements; real API is mandatory
    
    def extract_content(self, url: str) -> tuple[str, str]:
        """
        Extract content from a URL
        
        Used for processing search results and extracting full content.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract content from URLs with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract main content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return title_text, text
            
        except Exception as e:
            return f"Error extracting from {url}", str(e)
    
    def get_budget_status(self) -> Dict[str, int]:
        """Get current budget status"""
        return {
            "searches_performed": self.search_count,
            "max_searches": self.max_searches,
            "sources_used": self.sources_used,
            "max_sources": self.max_sources,
            "searches_remaining": max(0, self.max_searches - self.search_count),
            "sources_remaining": max(0, self.max_sources - self.sources_used)
        }
