"""
LLM client for OpenAI integration with GPT-4
Supports both text generation and analysis tasks
"""
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import tiktoken
from models import SearchResult, Citation, Mention, AnalysisResult, Metadata

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class LLMClient:
    """
    OpenAI client wrapper for brand visibility analysis
    
    Supports both GPT-4 for analysis and text generation.
    Includes open-source model alternatives in comments.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o", enable_llmlingua: bool = False):
        # Use OpenRouter for LLM calls
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_tokens_used = 0
        self.enable_llmlingua = enable_llmlingua
        self.llmlingua_compressor = None
        
        # Lazy load LLMLingua only if enabled
        if enable_llmlingua:
            self._initialize_llmlingua()
        
        # Alternative open-source models (commented for reference):
        # For text generation: "gpt-3.5-turbo", "gpt-4-turbo-preview"
        # For analysis: "gpt-4", "gpt-4-1106-preview"
        # Open-source alternatives:
        # - Hugging Face: "meta-llama/Llama-2-70b-chat-hf", "microsoft/DialoGPT-large"
        # - Anthropic: "claude-3-opus-20240229", "claude-3-sonnet-20240229"
        # - Local: "ollama/llama2", "ollama/codellama"
    
    def _initialize_llmlingua(self):
        """Lazy initialization of LLMLingua compressor"""
        logger.info("Initializing LLMLingua compressor...")
        try:
            from llmlingua import PromptCompressor
            self.llmlingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                device_map="cpu"
            )
            logger.info("✓ LLMLingua compressor initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize LLMLingua: {e}")
            logger.info("Falling back to semantic chunking only")
            self.enable_llmlingua = False
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
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
            # Build up to max_tokens using complete paragraphs
            result_tokens = []
            result_text = []
            
            for para in paragraphs:
                para_tokens = self.encoding.encode(para)
                if len(result_tokens) + len(para_tokens) <= max_tokens:
                    result_tokens.extend(para_tokens)
                    result_text.append(para)
                else:
                    # If we have content, use it; otherwise include partial paragraph
                    if result_text:
                        return '\n\n'.join(result_text)
                    # First paragraph is too long, break it down further
                    break
        
        # Strategy 2: Sentence-level chunking (fallback)
        # Use period followed by space and capital letter as more reliable boundary
        import re
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
    
    def should_use_llmlingua(self, content: str, max_tokens: int) -> bool:
        """
        Use LLM to decide if content is complex enough to warrant LLMLingua compression
        
        Args:
            content: The content to analyze
            max_tokens: Target token count
            
        Returns:
            True if LLMLingua should be used, False for semantic chunking
        """
        logger.info("=== Compression Decision Stage ===")
        
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing text complexity. Answer only 'yes' or 'no'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().lower()
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
        logger.info("=== LLMLingua Compression Stage ===")
        
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
        logger.info("=== Content Optimization Stage ===")
        initial_tokens = self.count_tokens(content)
        logger.info(f"Initial content: {initial_tokens} tokens, target: {max_tokens} tokens")
        
        if initial_tokens <= max_tokens:
            logger.info("✓ Content already within limit, no optimization needed")
            return content
        
        # Step 1: Remove redundant whitespace
        logger.info("Step 1: Removing redundant whitespace...")
        import re
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
        logger.info("=== Semantic Chunking Stage ===")
        logger.info(f"Using semantic chunking: {tokens_after_cleanup} → {max_tokens} tokens")
        result = self.truncate_to_tokens(content, max_tokens)
        final_tokens = self.count_tokens(result)
        logger.info(f"✓ Semantic chunking complete: {tokens_after_cleanup} → {final_tokens} tokens")
        logger.info(f"=== Optimization Complete (Semantic) ===\n")
        return result
    
    def summarize_source(self, content: str, max_tokens: int = 500) -> str:
        """
        Backward compatibility wrapper for optimize_content
        
        Maintains the same interface while using new optimization strategies.
        """
        return self.optimize_content(content, max_tokens)
    
    def generate_response(self, question: str, brand_name: str, brand_url: str, 
                         search_results: List[SearchResult], max_tokens: int = 1000) -> str:
        """
        Generate ChatGPT-style response with web search as a tool
        
        Implements response length management strategy and uses web search as a tool.
        """
        logger.info("=== Response Generation Stage ===")
        logger.info(f"Question: {question}")
        logger.info(f"Brand: {brand_name} ({brand_url})")
        logger.info(f"Search results: {len(search_results)} sources")
        logger.info(f"Max response tokens: {max_tokens}")
        
        # Define web search tool
        web_search_tool = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information about a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        # Build system message with token-efficient language
        # Apply minimization: concise, clear instructions without repetition
        system_message = f"""Provide accurate, well-sourced answers about {brand_name}.

Use web_search tool for:
• Current {brand_name} information
• Details not in context
• Latest updates

Format: markdown with URLs."""
        
        # Build initial context from pre-fetched search results
        logger.info("Building context from search results...")
        context_parts = []
        if search_results:
            for idx, result in enumerate(search_results[:3], 1):  # Limit to top 3 results
                logger.info(f"Processing source {idx}/{min(len(search_results), 3)}: {result.title[:60]}...")
                # Use source summarization for each result
                summarized_content = self.summarize_source(
                    f"{result.title}\n{result.snippet}", 
                    max_tokens=400
                )
                context_parts.append(f"Source: {summarized_content}\nURL: {result.url}")
        logger.info(f"✓ Built context from {len(context_parts)} sources")
        
        # Build user message with efficient context provision
        # Structure: clear, organized, minimal tokens
        user_message = f"Q: {question}\nBrand: {brand_name} ({brand_url})"
        if context_parts:
            # Use structured format for efficiency
            user_message += f"\n\nContext:\n" + "\n".join(context_parts)
        
        try:
            # First call with tool
            # Note: Using default temperature (1.0) to mimic standard ChatGPT behavior
            logger.info("Calling LLM for response generation...")
            logger.info(f"Model: {self.model}, max_tokens: {max_tokens}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                tools=[web_search_tool],
                tool_choice="auto",
                max_tokens=max_tokens
                # temperature defaults to 1.0 (ChatGPT default)
            )
            
            logger.info(f"✓ LLM response received, tokens used: {response.usage.total_tokens}")
            
            message = response.choices[0].message
            
            # If the model wants to use web search, handle the tool call
            if message.tool_calls:
                logger.info("LLM requested web search tool")
                # Extract the search query from the tool call
                search_query = message.tool_calls[0].function.arguments
                import json
                try:
                    query_data = json.loads(search_query)
                    actual_query = query_data.get('query', f"{brand_name} {question}")
                except:
                    actual_query = f"{brand_name} {question}"
                
                logger.info(f"Search query: '{actual_query}'")
                
                # Use the pre-fetched search results as tool response
                if search_results:
                    tool_response = f"Web search results for '{actual_query}':\n\n"
                    for i, result in enumerate(search_results[:3], 1):
                        tool_response += f"{i}. {result.title}\n{result.snippet}\nURL: {result.url}\n\n"
                    logger.info(f"Providing {len(search_results[:3])} search results to LLM")
                else:
                    tool_response = f"No additional web search results found for '{actual_query}'. Using available context."
                    logger.warning("No search results available")
                
                # Second call with tool response
                # Note: Using default temperature (1.0) to mimic standard ChatGPT behavior
                logger.info("Making second LLM call with search results...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": None, "tool_calls": message.tool_calls},
                        {"role": "tool", "content": tool_response, "tool_call_id": message.tool_calls[0].id}
                    ],
                    max_tokens=max_tokens
                    # temperature defaults to 1.0 (ChatGPT default)
                )
                logger.info(f"✓ Second LLM response received, tokens used: {response.usage.total_tokens}")
            
            final_response = response.choices[0].message.content
            response_tokens = self.count_tokens(final_response)
            logger.info(f"✓ Final response: {response_tokens} tokens")
            logger.info(f"=== Response Generation Complete ===\n")
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def analyze_citations_and_mentions(self, response_text: str, brand_name: str) -> tuple[List[Citation], List[Mention]]:
        """
        Use LLM to analyze citations and mentions in the response
        
        This provides much better accuracy than regex patterns.
        """
        logger.info("=== Citation & Mention Analysis Stage ===")
        logger.info(f"Analyzing response for brand: {brand_name}")
        logger.info(f"Response length: {len(response_text)} characters, {self.count_tokens(response_text)} tokens")
        
        # Token-minimized analysis prompt
        # Removed redundancy, used clear structure
        analysis_prompt = f"""Extract from text:
1. Citations: Entity mentions with URLs
2. Brand mentions: "{brand_name}" (all variations)

Text:
{response_text}

Return JSON:
{{
  "citations": [{{"text": "context", "url": "https://...", "entity": "name"}}],
  "mentions": [{{"text": "context", "position": 0, "exact_match": "{brand_name}", "variation": "alt"}}]
}}

Include only actual citations and mentions."""
        
        try:
            logger.info("Calling LLM for citation and mention analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Analyze text for citations and brand mentions. Return valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            logger.info(f"✓ Analysis response received, tokens used: {response.usage.total_tokens}")
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            logger.info("Parsing JSON analysis...")
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(analysis_text)
            
            # Convert to Pydantic models
            citations = [Citation(**citation) for citation in analysis_data.get("citations", [])]
            mentions = [Mention(**mention) for mention in analysis_data.get("mentions", [])]
            
            logger.info(f"✓ Found {len(citations)} citations and {len(mentions)} mentions")
            logger.info(f"=== Analysis Complete ===\n")
            
            return citations, mentions
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            logger.info("Returning empty results")
            # Fallback to empty results
            return [], []
    
    def categorize_sources(self, urls: List[str], brand_url: str) -> tuple[List[str], List[str]]:
        """
        Use LLM to categorize sources as owned or external
        
        This provides better accuracy than simple domain matching.
        """
        logger.info("=== Source Categorization Stage ===")
        logger.info(f"Categorizing {len(urls)} URLs for brand: {brand_url}")
        
        if not urls:
            logger.info("No URLs to categorize")
            return [], []
        
        # Extract domain from brand URL
        from urllib.parse import urlparse
        brand_domain = urlparse(brand_url).netloc.lower()
        logger.info(f"Brand domain: {brand_domain}")
        
        categorization_prompt = f"""
        Categorize the following URLs as either owned by the brand or external sources.
        
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
        }}
        """
        
        try:
            logger.info("Calling LLM for source categorization...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing domain ownership and relationships. Always return valid JSON."},
                    {"role": "user", "content": categorization_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            logger.info(f"✓ Categorization response received, tokens used: {response.usage.total_tokens}")
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            logger.info("Parsing categorization JSON...")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                analysis_data = json.loads(analysis_text)
            
            owned_sources = analysis_data.get("owned_sources", [])
            external_sources = analysis_data.get("external_sources", [])
            
            logger.info(f"✓ Categorized: {len(owned_sources)} owned, {len(external_sources)} external")
            logger.info(f"=== Categorization Complete ===\n")
            
            return owned_sources, external_sources
            
        except Exception as e:
            logger.error(f"Error in source categorization: {e}")
            logger.info("Falling back to simple domain matching")
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
            logger.info(f"✓ Fallback categorization: {len(owned)} owned, {len(external)} external")
            return owned, external
    
