"""
Integration tests for the complete Brand Visibility Analyzer system
"""
import pytest
import json
import os
from dotenv import load_dotenv
from models import AnalysisRequest, AnalysisResult
from analyzer import BrandAnalyzer

# Load environment variables for real API calls
load_dotenv()


class TestIntegrationScenarios:
    """Test various integration scenarios with real API calls"""
    
    def test_tesla_brand_analysis(self):
        """Test complete analysis for Tesla brand with real APIs"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla and what do they do?",
            max_searches=2,
            max_sources=5
        )
        
        result = analyzer.analyze(request)
        
        # Verify comprehensive results
        assert isinstance(result, AnalysisResult)
        assert "Tesla" in result.human_response_markdown
        assert len(result.human_response_markdown) > 50  # Should have substantial content
        
        # Verify citations (should have at least some)
        assert len(result.citations) >= 0
        for citation in result.citations:
            assert citation.url.startswith('http')
            assert len(citation.text) > 0
        
        # Verify mentions (should find Tesla mentions)
        assert len(result.mentions) > 0
        tesla_mentions = [m for m in result.mentions if "Tesla" in m.exact_match]
        assert len(tesla_mentions) > 0
        
        # Verify source categorization
        assert len(result.owned_sources) >= 0
        assert len(result.sources) >= 0
        
        # Verify metadata
        assert result.metadata.searches_performed <= request.max_searches
        assert result.metadata.sources_included >= 0
        assert result.metadata.citations_found >= 0
        assert result.metadata.mentions_found >= 0
        assert result.metadata.model_used == "gpt-4"
        assert result.metadata.processing_time_seconds > 0
    
    def test_competitor_analysis_scenario(self):
        """Test analysis when competitors are mentioned with real APIs"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="Who are Tesla's main competitors?",
            max_searches=2,
            max_sources=5
        )
        
        result = analyzer.analyze(request)
        
        # Verify competitor analysis
        assert isinstance(result, AnalysisResult)
        assert len(result.human_response_markdown) > 50
        
        # Should mention competitors (Ford, GM, etc.)
        response_lower = result.human_response_markdown.lower()
        competitor_mentioned = any(comp in response_lower for comp in ['ford', 'general motors', 'gm', 'rivian', 'bmw', 'mercedes'])
        assert competitor_mentioned, "Should mention at least one competitor"
        
        # Verify brand mentions are detected
        assert len(result.mentions) > 0
        tesla_mentions = [m for m in result.mentions if "Tesla" in m.exact_match]
        assert len(tesla_mentions) > 0
    
    def test_minimal_budget_scenario(self):
        """Test analysis with minimal budget constraints using real APIs"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=0,
            max_sources=0
        )
        
        result = analyzer.analyze(request)
        
        # Verify minimal budget results
        assert isinstance(result, AnalysisResult)
        assert result.metadata.searches_performed == 0
        assert result.metadata.max_searches == 0
        assert result.metadata.max_sources == 0
        assert len(result.human_response_markdown) > 0
        assert "Tesla" in result.human_response_markdown
    
    def test_error_recovery_scenario(self):
        """Test system behavior with real APIs - should handle gracefully"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?"
        )
        
        result = analyzer.analyze(request)
        
        # Verify error recovery - should always return a valid result
        assert isinstance(result, AnalysisResult)
        assert len(result.human_response_markdown) > 0
        assert "Tesla" in result.human_response_markdown
        # Should have some analysis results even if partial
        assert result.metadata.processing_time_seconds > 0


class TestTokenOptimizationIntegration:
    """Test token optimization in real scenarios"""
    
    def test_source_summarization_integration(self):
        """Test source summarization in complete workflow with real APIs"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY') or not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?",
            max_searches=1,
            max_sources=3
        )
        
        result = analyzer.analyze(request)
        
        # Verify results are valid
        assert isinstance(result, AnalysisResult)
        assert len(result.human_response_markdown) > 0
        assert result.metadata.token_count > 0
        assert result.metadata.processing_time_seconds > 0
    
    def test_response_length_management_integration(self):
        """Test response length management in complete workflow with real APIs"""
        # Skip if no API keys available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("API keys not available for real API testing")
        
        analyzer = BrandAnalyzer(
            api_key=os.getenv('OPENAI_API_KEY'), 
            model="gpt-4"
        )
        
        request = AnalysisRequest(
            brand="Tesla",
            url="https://www.tesla.com",
            question="What is Tesla?"
        )
        
        result = analyzer.analyze(request)
        
        # Verify response length management
        assert isinstance(result, AnalysisResult)
        assert result.metadata.token_count > 0
        assert result.metadata.token_count <= 1000  # Should respect max_tokens limit
        assert len(result.human_response_markdown) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
