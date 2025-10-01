#!/usr/bin/env python3
"""
Brand Visibility in LLM Answers - CLI Application

This tool analyzes how a brand appears in LLM-generated answers by:
1. Generating ChatGPT-style responses to questions
2. Tracking brand mentions, citations, and source ownership using LLM analysis
3. Optimizing token usage while maintaining response quality
"""
import os
import sys
import json
import argparse
from typing import Optional

from dotenv import load_dotenv
from src.core.models import AnalysisRequest
from src.core.analyzer import BrandAnalyzer
from src.clients.llm_client import LLMClient
from src.clients.llm_client_gemma import OpenSourceLLMClient


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze brand visibility in LLM-generated answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --brand "Tesla" --url "https://www.tesla.com" --question "What is Tesla?"
  python main.py --brand "OpenAI" --url "https://openai.com" --question "What does OpenAI do?" --max-searches 3
  python main.py --brand "Tesla" --url "https://www.tesla.com" --question "What is Tesla?" --model "google/gemma-3-4b-it:free"
        """
    )
    
    parser.add_argument(
        "--brand", 
        required=True, 
        help="Brand name to analyze"
    )
    
    parser.add_argument(
        "--url", 
        required=True, 
        help="Brand website URL"
    )
    
    parser.add_argument(
        "--question", 
        required=True, 
        help="Question to ask the LLM"
    )
    
    parser.add_argument(
        "--max-searches", 
        type=int, 
        default=5, 
        help="Maximum web searches to perform (default: 5)"
    )
    
    parser.add_argument(
        "--max-sources", 
        type=int, 
        default=10, 
        help="Maximum sources to include (default: 10)"
    )
    
    parser.add_argument(
        "--api-key", 
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo-preview", "google/gemma-3-4b-it:free"],
        help="LLM model to use (default: gpt-4o)"
    )
    
    
    
    parser.add_argument(
        "--output", 
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--enable-llmlingua",
        action="store_true",
        help="Enable LLMLingua compression (LLM decides when to use it based on complexity)"
    )
    
    return parser.parse_args()


def get_api_key(args: argparse.Namespace) -> str:
    """Get API key from arguments or environment"""
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.", file=sys.stderr)
        print("Set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
        sys.exit(1)
    return api_key




def main():
    """Main CLI entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    if args.verbose:
        print(f"Starting analysis for brand: {args.brand}", file=sys.stderr)
        print(f"Question: {args.question}", file=sys.stderr)
        print(f"Model: {args.model}", file=sys.stderr)
    
    try:
        # Get API key
        api_key = get_api_key(args)
        
        # Create analysis request
        request = AnalysisRequest(
            brand=args.brand,
            url=args.url,
            question=args.question,
            max_searches=args.max_searches,
            max_sources=args.max_sources
        )
        
        if args.verbose:
            print("Creating analyzer...", file=sys.stderr)
        
        # Create analyzer
        # Auto-detect based on model name
        if args.model.startswith('google/gemma') or args.model.startswith('meta-llama'):
            # Use open-source client for Gemma and Llama models
            analyzer = BrandAnalyzer(
                model=args.model, 
                use_open_source=True,
                enable_llmlingua=args.enable_llmlingua
            )
        else:
            # Use OpenAI client for GPT models
            analyzer = BrandAnalyzer(
                api_key=api_key, 
                model=args.model,
                enable_llmlingua=args.enable_llmlingua
            )
        
        if args.verbose:
            print("Performing analysis...", file=sys.stderr)
        
        # Perform analysis
        result = analyzer.analyze(request)
        
        if args.verbose:
            print(f"Analysis completed in {result.metadata.processing_time_seconds:.2f} seconds", file=sys.stderr)
            print(f"Tokens used: {result.metadata.token_count}", file=sys.stderr)
            print(f"Searches performed: {result.metadata.searches_performed}/{result.metadata.max_searches}", file=sys.stderr)
            print(f"Sources included: {result.metadata.sources_included}/{result.metadata.max_sources}", file=sys.stderr)
        
        # Output result
        output_data = result.to_dict()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            if args.verbose:
                print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(output_data, indent=2))
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_result = {
            "error": str(e),
            "human_response_markdown": "",
            "citations": [],
            "mentions": [],
            "owned_sources": [],
            "sources": [],
            "metadata": {
                "token_count": 0,
                "searches_performed": 0,
                "max_searches": args.max_searches,
                "sources_included": 0,
                "max_sources": args.max_sources,
                "citations_found": 0,
                "mentions_found": 0,
                "processing_time_seconds": 0,
                "model_used": args.model
            }
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(error_result, f, indent=2)
        else:
            print(json.dumps(error_result, indent=2), file=sys.stderr)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
