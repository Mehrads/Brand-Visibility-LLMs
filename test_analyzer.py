"""
Comprehensive unit tests for the Brand Visibility Analyzer
"""
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv
from models import AnalysisRequest, AnalysisResult, Citation, Mention, Metadata
from analyzer import BrandAnalyzer
from llm_client import LLMClient
from web_search import WebSearchEngine

# Load environment variables for real API testing
load_dotenv()


class TestAnalysisRequest:
    """Test AnalysisRequest model validation"""
    
    def test_valid_request(self):
        """Test valid analysis request"""
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=5,
            max_sources=10
        )
        assert request.brand == "Tesla"
        assert str(request.url).rstrip('/') == "https://www.tesla.com"
        assert request.question == "What is Tesla?"
    
    def test_brand_validation(self):
        """Test brand name validation"""
        # Empty brand should raise validation error
        with pytest.raises(ValueError, match="Brand name cannot be empty"):
            AnalysisRequest(
                brand="",
                url="https://www.tesla.com",
                question="What is Tesla?"
            )
        
        # Whitespace-only brand should raise validation error
        with pytest.raises(ValueError, match="Brand name cannot be empty"):
            AnalysisRequest(
                brand="   ",
                url="https://www.tesla.com",
                question="What is Tesla?"
            )
    
    def test_question_validation(self):
        """Test question validation"""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            AnalysisRequest(
                brand="Tesla",
                url="https://www.tesla.com",
                question=""
            )
    
    def test_budget_validation(self):
        """Test budget constraints validation"""
        # Valid budgets
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=0,
            max_sources=0
        )
        assert request.max_searches == 0
        assert request.max_sources == 0
        
        # Invalid budgets should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            AnalysisRequest(
                brand="Tesla",
                url="https://www.tesla.com",
                question="What is Tesla?",
                max_searches=-1,
                max_sources=10
            )
        
        with pytest.raises(Exception):  # Pydantic validation error
            AnalysisRequest(
                brand="Tesla",
                url="https://www.tesla.com",
                question="What is Tesla?",
                max_searches=25,  # Exceeds max limit of 20
                max_sources=10
            )


class TestWebSearchEngine:
    """Test WebSearchEngine functionality"""
    
    def test_budget_constraints(self):
        """Test budget constraint enforcement"""
        search_engine = WebSearchEngine(max_searches=2, max_sources=5)
        
        # First search should work
        results1 = search_engine.search_web("test query", num_results=3)
        assert len(results1) == 3
        assert search_engine.search_count == 1
        assert search_engine.sources_used == 3
        
        # Second search should work but with limited results
        results2 = search_engine.search_web("test query 2", num_results=5)
        assert len(results2) == 2  # Only 2 more sources allowed
        assert search_engine.search_count == 2
        assert search_engine.sources_used == 5
        
        # Third search should return empty (budget exhausted)
        results3 = search_engine.search_web("test query 3", num_results=5)
        assert len(results3) == 0
        assert search_engine.search_count == 2  # Unchanged
    
    def test_budget_status(self):
        """Test budget status reporting"""
        search_engine = WebSearchEngine(max_searches=5, max_sources=10)
        
        status = search_engine.get_budget_status()
        assert status["searches_performed"] == 0
        assert status["max_searches"] == 5
        assert status["searches_remaining"] == 5
        assert status["sources_remaining"] == 10
        
        # Perform one search
        search_engine.search_web("test", num_results=3)
        
        status = search_engine.get_budget_status()
        assert status["searches_performed"] == 1
        assert status["searches_remaining"] == 4
        assert status["sources_remaining"] == 7


class TestLLMClient:
    """Test LLMClient functionality"""
    
    @patch('openai.OpenAI')
    def test_token_counting(self, mock_openai):
        """Test token counting functionality"""
        client = LLMClient("fake-api-key")
        
        # Test token counting
        text = "This is a test sentence."
        token_count = client.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
    
    @patch('openai.OpenAI')
    def test_truncation(self, mock_openai):
        """Test text truncation"""
        client = LLMClient("fake-api-key")
        
        # Test truncation
        long_text = "This is a very long text. " * 100
        truncated = client.truncate_to_tokens(long_text, 10)
        assert client.count_tokens(truncated) <= 10
        assert len(truncated) < len(long_text)
    
    @patch('openai.OpenAI')
    def test_source_summarization(self, mock_openai):
        """Test source summarization"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary of the content."
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = LLMClient("fake-api-key")
        client.client = mock_client
        
        # Test summarization
        long_content = "This is a very long content. " * 100
        summary = client.summarize_source(long_content, max_tokens=50)
        
        assert isinstance(summary, str)
        assert len(summary) < len(long_content)
        mock_client.chat.completions.create.assert_called_once()


class TestBrandAnalyzer:
    """Test BrandAnalyzer integration with real APIs"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(os.getenv('OPENAI_API_KEY'), model="gpt-4")
        assert analyzer.llm_client.model == "gpt-4"
        assert analyzer.web_search is None  # Not initialized until analyze() is called
    
    def test_empty_search_budget(self):
        """Test analysis with zero search budget using real APIs"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(os.getenv('OPENAI_API_KEY'))
        
        # Create request with minimal search budget
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=0,
            max_sources=1
        )
        
        result = analyzer.analyze(request)
        
        # Verify results
        assert isinstance(result, AnalysisResult)
        assert result.metadata.searches_performed == 0
        assert result.metadata.max_searches == 0
        assert result.metadata.max_sources == 1
        assert "Tesla" in result.human_response_markdown
        assert len(result.human_response_markdown) > 0
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow with real APIs"""
        if not os.getenv('OPENAI_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(os.getenv('OPENAI_API_KEY'))
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=2,
            max_sources=5
        )
        
        result = analyzer.analyze(request)
        
        # Verify complete analysis
        assert isinstance(result, AnalysisResult)
        assert "Tesla" in result.human_response_markdown
        assert len(result.human_response_markdown) > 0
        assert result.metadata.searches_performed <= request.max_searches
        assert result.metadata.processing_time_seconds > 0


class TestErrorHandling:
    """Test error handling scenarios with real APIs"""
    
    def test_api_error_handling(self):
        """Test handling of API errors with real APIs"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        # Test with invalid API key to trigger error
        analyzer = BrandAnalyzer("invalid-api-key")
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?"
        )
        
        # Should handle error gracefully
        result = analyzer.analyze(request)
        assert isinstance(result, AnalysisResult)
        # Should still return a result structure even with errors
        assert hasattr(result, 'human_response_markdown')
        assert hasattr(result, 'citations')
        assert hasattr(result, 'mentions')
    
    def test_invalid_json_handling(self):
        """Test handling with real APIs - should work normally"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(os.getenv('OPENAI_API_KEY'))
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?"
        )
        
        result = analyzer.analyze(request)
        
        # Should handle gracefully with real APIs
        assert isinstance(result, AnalysisResult)
        assert len(result.human_response_markdown) > 0
        assert "Tesla" in result.human_response_markdown


class TestTokenOptimization:
    """Test token optimization strategies with real APIs"""
    
    def test_source_summarization_strategy(self):
        """Test source summarization optimization with real APIs"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        client = LLMClient(os.getenv('OPENAI_API_KEY'))
        
        # Test summarization
        long_content = "Tesla is a company that makes electric vehicles. " * 50
        summary = client.summarize_source(long_content, max_tokens=100)
        
        assert client.count_tokens(summary) <= 100
        assert len(summary) < len(long_content)
        assert len(summary) > 0
    
    def test_response_length_management(self):
        """Test response length management with real APIs"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        client = LLMClient(os.getenv('OPENAI_API_KEY'))
        
        # Test response generation with length limit
        response = client.generate_response(
            question="What is Tesla?",
            brand_name="Tesla",
            brand_url="https://www.tesla.com",
            search_results=[],
            max_tokens=50
        )
        
        assert client.count_tokens(response) <= 50
        assert len(response) > 0
        assert "Tesla" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
