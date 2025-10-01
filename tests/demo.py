#!/usr/bin/env python3
"""
Demo script for Brand Visibility Analyzer
Shows sample output without requiring API calls
"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import AnalysisResult, Citation, Mention, Metadata


def demo_analysis():
    """Demonstrate the analysis functionality with mock data"""
    
    # Sample analysis result
    result = AnalysisResult(
        human_response_markdown="""**Tesla** is a leading electric vehicle and clean energy company founded by Elon Musk. 

The company has revolutionized the automotive industry with its innovative electric vehicles and sustainable energy solutions.

**Key Information about Tesla:**

- **Company**: Tesla, Inc.
- **Founded**: 2003
- **CEO**: Elon Musk
- **Headquarters**: Austin, Texas
- **Website**: https://www.tesla.com

**Products and Services:**
- Electric vehicles (Model S, Model 3, Model X, Model Y)
- Energy storage systems (Powerwall, Megapack)
- Solar panels and solar roof tiles
- Supercharger network

**Recent Developments:**
Tesla continues to expand its global presence and improve its technology. The company has been focusing on full self-driving capabilities and expanding its charging infrastructure worldwide.

For the most current information, visit [Tesla's official website](https://www.tesla.com) or check their latest quarterly reports.""",
        
        citations=[
            Citation(
                text="visit [Tesla's official website](https://www.tesla.com)",
                url="https://www.tesla.com",
                entity="Tesla"
            )
        ],
        
        mentions=[
            Mention(
                text="**Tesla** is a leading electric vehicle and clean energy company",
                position=0,
                exact_match="Tesla"
            ),
            Mention(
                text="**Company**: Tesla, Inc.",
                position=156,
                exact_match="Tesla"
            ),
            Mention(
                text="Tesla continues to expand its global presence",
                position=580,
                exact_match="Tesla"
            )
        ],
        
        owned_sources=["https://www.tesla.com"],
        
        sources=[
            "https://example1.com/article",
            "https://example2.com/article",
            "https://example3.com/article"
        ],
        
        metadata=Metadata(
            token_count=284,
            searches_performed=1,
            max_searches=5,
            sources_included=4,
            max_sources=10,
            citations_found=1,
            mentions_found=3,
            processing_time_seconds=2.34,
            model_used="gpt-4"
        )
    )
    
    return result


def main():
    """Run the demo"""
    print("Brand Visibility Analyzer - Demo Output")
    print("=" * 50)
    
    # Get demo result
    result = demo_analysis()
    
    # Convert to dict and output as JSON
    output = result.to_dict()
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
