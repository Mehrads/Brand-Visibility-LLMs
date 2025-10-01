"""
LLM client for open-source models via OpenRouter
Supports both text generation and analysis tasks using open-source models through OpenRouter
"""
import json
import time
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import tiktoken
from src.core.models import SearchResult, Citation, Mention, AnalysisResult, Metadata
from urllib.parse import urlparse

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class OpenSourceLLMClient:
    """
    Open-source LLM client wrapper for brand visibility analysis
    
    Uses open-source models via OpenRouter for both text generation and analysis.
    Provides the same interface as OpenAI client but with free open-source models.
    """
    
    def __init__(self, api_key: str = None, model: str = "google/gemma-3-4b-it:free", enable_llmlingua: bool = False):
        # Use OpenRouter for open-source model calls
        # Default to GEMMA3_4B environment variable if no API key provided
        if api_key is None:
            api_key = os.environ.get("GEMMA3_4B")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_tokens_used = 0
        self.enable_llmlingua = enable_llmlingua
        self.llmlingua_compressor = None
        
        # Lazy load LLMLingua only if enabled
        if enable_llmlingua:
            self._initialize_llmlingua()
        
        # Open-source models available via OpenRouter:
        # - "google/gemma-3-4b-it:free" (Gemma 3 4B Instruct - Free tier)
        # - "google/gemma-2b-it" (Gemma 2B Instruct)
        # - "google/gemma-7b-it" (Gemma 7B Instruct)
        # - "meta-llama/llama-2-7b-chat" (Llama 2 7B Chat)
        # - "meta-llama/llama-2-13b-chat" (Llama 2 13B Chat)
        # - "microsoft/dialo-gpt-medium" (DialoGPT Medium)
        # - "microsoft/dialo-gpt-large" (DialoGPT Large)
    
    def _initialize_llmlingua(self):
        """Lazy initialization of LLMLingua compressor"""
        logger.info("Initializing LLMLingua compressor for Gemma...")
        try:
            from llmlingua import PromptCompressor
            self.llmlingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                device_map="cpu"
            )
            logger.info("✓ LLMLingua compressor initialized successfully (Gemma)")
        except Exception as e:
            logger.warning(f"Could not initialize LLMLingua: {e}")
            logger.info("Falling back to semantic chunking only")
            self.enable_llmlingua = False
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text using semantic chunking to preserve important information
        
        This method uses intelligent semantic preservation rather than fragile regex:
        1. Identifies semantic boundaries (paragraphs, natural breaks)
        2. Preserves complete thoughts and context
        3. Prioritizes key information over verbosity
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Strategy 1: Try paragraph-level chunking first (most semantic)
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            result_tokens = []
            result_text = []
            
            for para in paragraphs:
                para_tokens = self.encoding.encode(para)
                if len(result_tokens) + len(para_tokens) <= max_tokens:
                    result_tokens.extend(para_tokens)
                    result_text.append(para)
                else:
                    if result_text:
                        return '\n\n'.join(result_text)
                    break
        
        # Strategy 2: Sentence-level chunking (fallback)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        if len(sentences) > 1:
            result_tokens = []
            result_text = []
            
            for sentence in sentences:
                sent_tokens = self.encoding.encode(sentence)
                if len(result_tokens) + len(sent_tokens) <= max_tokens:
                    result_tokens.extend(sent_tokens)
                    result_text.append(sentence)
                else:
                    if result_text:
                        return ' '.join(result_text)
                    break
        
        # Strategy 3: Hard truncation (last resort)
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 1.0, response_format: Optional[Dict] = None) -> str:
        """Helper to call the open-source LLM and track token usage"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format:
                params["response_format"] = response_format

            response = self.client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            self.total_tokens_used += response.usage.total_tokens
            return content
        except Exception as e:
            print(f"Open-source LLM call error: {e}")
            return ""
    
    def should_use_llmlingua(self, content: str, max_tokens: int) -> bool:
        """
        Use LLM to decide if content is complex enough to warrant LLMLingua compression
        
        Args:
            content: The content to analyze
            max_tokens: Target token count
            
        Returns:
            True if LLMLingua should be used, False for semantic chunking
        """
        logger.info("=== Compression Decision Stage (Gemma) ===")
        
        if not self.enable_llmlingua:
            logger.info("LLMLingua not enabled, using semantic chunking")
            return False
            
        current_tokens = self.count_tokens(content)
        compression_ratio = max_tokens / current_tokens if current_tokens > 0 else 1.0
        
        logger.info(f"Content analysis: {current_tokens} tokens → {max_tokens} tokens ({compression_ratio:.1%} compression)")
        
        # Quick decision: if content is short or doesn't need much compression, skip LLM call
        if current_tokens < 1000:
            logger.info(f"Quick decision: Content is short ({current_tokens} < 1000 tokens) → Semantic chunking")
            return False
            
        if compression_ratio > 0.6:
            logger.info(f"Quick decision: Moderate compression ({compression_ratio:.1%} > 60%) → Semantic chunking")
            return False
        
        # Ask LLM to analyze complexity
        logger.info("Content requires analysis, asking LLM for complexity assessment...")
        
        prompt = f"""Analyze if this content requires advanced compression:

Content length: {current_tokens} tokens
Target length: {max_tokens} tokens  
Compression ratio: {compression_ratio:.0%}

Content preview:
{content[:500]}...

Is this content COMPLEX (technical, dense, requires preserving key details)?
Answer ONLY 'yes' or 'no'."""

        try:
            response = self._call_llm(
                messages=[
                    {"role": "user", "content": "You are an expert at analyzing text complexity. Answer only 'yes' or 'no'.\n\n" + prompt}
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            answer = response.strip().lower()
            logger.info(f"LLM complexity assessment: '{answer}'")
            
            decision = 'yes' in answer
            if decision:
                logger.info("✓ Decision: USE LLMLINGUA (content is complex)")
            else:
                logger.info("✓ Decision: USE SEMANTIC CHUNKING (content is simple)")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM complexity check failed: {e}")
            logger.info("Falling back to semantic chunking")
            return False
    
    def compress_with_llmlingua(self, content: str, max_tokens: int) -> str:
        """Compress content using LLMLingua"""
        logger.info("=== LLMLingua Compression Stage (Gemma) ===")
        
        if not self.llmlingua_compressor:
            logger.warning("LLMLingua compressor not available, falling back to semantic chunking")
            return self.truncate_to_tokens(content, max_tokens)
        
        try:
            current_tokens = self.count_tokens(content)
            compression_rate = max_tokens / current_tokens if current_tokens > 0 else 1.0
            
            logger.info(f"Starting LLMLingua compression: {current_tokens} → {max_tokens} tokens ({compression_rate:.1%} rate)")
            
            compressed = self.llmlingua_compressor.compress_prompt(
                content,
                target_token=max_tokens,
                rate=compression_rate
            )
            
            compressed_text = compressed['compressed_prompt']
            final_tokens = self.count_tokens(compressed_text)
            logger.info(f"✓ LLMLingua compression complete: {current_tokens} → {final_tokens} tokens")
            
            return compressed_text
            
        except Exception as e:
            logger.error(f"LLMLingua compression failed: {e}")
            logger.info("Falling back to semantic chunking")
            return self.truncate_to_tokens(content, max_tokens)
    
    def optimize_content(self, content: str, max_tokens: int = 500) -> str:
        """
        Optimize content using LLM-decided compression strategy
        
        Steps:
        1. Remove redundant whitespace
        2. Ask LLM if content is complex
        3. If complex: use LLMLingua
        4. Otherwise: use semantic chunking
        """
        logger.info("=== Content Optimization Stage (Gemma) ===")
        initial_tokens = self.count_tokens(content)
        logger.info(f"Initial content: {initial_tokens} tokens, target: {max_tokens} tokens")
        
        if initial_tokens <= max_tokens:
            logger.info("✓ Content already within limit, no optimization needed")
            return content
        
        # Step 1: Remove redundant whitespace
        logger.info("Step 1: Removing redundant whitespace...")
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        content = content.strip()
        
        tokens_after_cleanup = self.count_tokens(content)
        saved = initial_tokens - tokens_after_cleanup
        logger.info(f"✓ Whitespace cleanup: {initial_tokens} → {tokens_after_cleanup} tokens (saved {saved})")
        
        # Step 2: Check if optimization is enough
        if tokens_after_cleanup <= max_tokens:
            logger.info("✓ Content now within limit after cleanup")
            return content
        
        # Step 3: LLM decides if content is complex
        logger.info("Step 2: Determining compression method...")
        if self.should_use_llmlingua(content, max_tokens):
            result = self.compress_with_llmlingua(content, max_tokens)
            logger.info(f"=== Optimization Complete (LLMLingua) ===\n")
            return result
        
        # Step 4: Use semantic chunking for simple content
        logger.info("=== Semantic Chunking Stage (Gemma) ===")
        logger.info(f"Using semantic chunking: {tokens_after_cleanup} → {max_tokens} tokens")
        result = self.truncate_to_tokens(content, max_tokens)
        final_tokens = self.count_tokens(result)
        logger.info(f"✓ Semantic chunking complete: {tokens_after_cleanup} → {final_tokens} tokens")
        logger.info(f"=== Optimization Complete (Semantic) ===\n")
        return result
    
    def summarize_source(self, content: str, max_tokens: int = 500) -> str:
        """Backward compatibility wrapper for optimize_content"""
        return self.optimize_content(content, max_tokens)
    
    def generate_response(self, question: str, brand_name: str, brand_url: str, 
                         search_results: List[SearchResult], max_tokens: int = 1000) -> str:
        """
        Generate ChatGPT-style response using open-source model
        
        Implements response length management strategy.
        """
        # Build context from search results
        context_parts = []
        if search_results:
            for result in search_results[:3]:  # Limit to top 3 results
                # Use source summarization for each result
                summarized_content = self.summarize_source(
                    f"{result.title}\n{result.snippet}", 
                    max_tokens=200
                )
                context_parts.append(f"Source: {summarized_content}\nURL: {result.url}")
        
        # Build user message (no system message for Gemma)
        user_message = f"""You are a helpful assistant that provides accurate, well-sourced answers about {brand_name}.

Format your response in markdown and include relevant URLs when citing sources.

Question: {question}
Brand context: {brand_name} ({brand_url})"""
        
        if context_parts:
            user_message += f"\n\nAnswer the question based on the following context:\n" + "\n\n".join(context_parts)
        
        try:
            messages = [
                {"role": "user", "content": user_message}
            ]
            
            # Use default temperature (1.0) to mimic standard ChatGPT behavior
            response = self._call_llm(messages, max_tokens=max_tokens)
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_citations_and_mentions(self, response_text: str, brand_name: str) -> Tuple[List[Citation], List[Mention]]:
        """
        Use open-source model to analyze citations and mentions in the response
        
        This provides much better accuracy than regex patterns.
        """
        analysis_prompt = f"""Analyze the following text and extract:
1. Citations: Any mention of an entity (person, company, organization) that includes a URL
2. Brand mentions: Any mention of the brand "{brand_name}" (case-insensitive, including variations)

Text to analyze:
{response_text}

Brand to find: {brand_name}

Return your analysis in this exact JSON format:
{{
    "citations": [
        {{
            "text": "context around the citation",
            "url": "https://example.com",
            "entity": "entity name being cited"
        }}
    ],
    "mentions": [
        {{
            "text": "context around the mention",
            "position": 123,
            "exact_match": "Tesla",
            "variation": "tesla"
        }}
    ]
}}

Be thorough but accurate. Only include actual citations and mentions."""
        
        try:
            # Add system instructions to the prompt for Gemma
            full_prompt = f"""You are an expert at analyzing text for citations and brand mentions. Always return valid JSON.

{analysis_prompt}"""
            
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            # Note: Gemma may not support JSON response format, so we'll parse the response
            analysis_text = self._call_llm(messages, max_tokens=1500)
            
            # Parse JSON response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(analysis_text)
            
            # Convert to Pydantic models
            citations = [Citation(**citation) for citation in analysis_data.get("citations", [])]
            mentions = [Mention(**mention) for mention in analysis_data.get("mentions", [])]
            
            return citations, mentions
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            # Fallback to empty results
            return [], []
    
    def categorize_sources(self, urls: List[str], brand_url: str) -> Tuple[List[str], List[str]]:
        """
        Use open-source model to categorize sources as owned or external
        
        This provides better accuracy than simple domain matching.
        """
        if not urls:
            return [], []
        
        # Extract domain from brand URL
        brand_domain = urlparse(brand_url).netloc.lower()
        
        categorization_prompt = f"""Categorize the following URLs as either owned by the brand or external sources.

Brand domain: {brand_domain}
Brand URL: {brand_url}

URLs to categorize:
{chr(10).join(f"- {url}" for url in urls)}

A URL is "owned" if it belongs to the same organization as the brand domain.
This includes the exact domain, subdomains, and related domains.

Return your analysis in this exact JSON format:
{{
    "owned_sources": ["https://brand.com", "https://sub.brand.com"],
    "external_sources": ["https://external.com", "https://competitor.com"]
}}"""
        
        try:
            # Add system instructions to the prompt for Gemma
            full_prompt = f"""You are an expert at analyzing domain ownership and relationships. Always return valid JSON.

{categorization_prompt}"""
            
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            # Note: Gemma may not support JSON response format, so we'll parse the response
            analysis_text = self._call_llm(messages, max_tokens=1000)
            
            # Parse JSON response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(analysis_text)
            
            owned_sources = analysis_data.get("owned_sources", [])
            external_sources = analysis_data.get("external_sources", [])
            
            return owned_sources, external_sources
            
        except Exception as e:
            print(f"Error in source categorization: {e}")
            # Fallback to simple domain matching
            owned = []
            external = []
            for url in urls:
                try:
                    url_domain = urlparse(url).netloc.lower()
                    if brand_domain in url_domain or url_domain in brand_domain:
                        owned.append(url)
                    else:
                        external.append(url)
                except:
                    external.append(url)
            return owned, external
    
    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all calls"""
        return self.total_tokens_used