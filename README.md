# Brand Visibility Analyzer

A Python command-line application that analyzes brand visibility in LLM-generated answers. Given a brand, its website, and a question, it produces both a human-readable answer and a JSON summary of how the brand appears (citations, mentions, sources).

## Features

- **LLM Integration**: Uses OpenAI's GPT-4o for both text generation and content analysis
- **Open-Source Support**: Supports Gemma 3 4B and other open-source models via OpenRouter
- **Web Search**: Integrates with Tavily Search API for real-time information
- **Intelligent Compression**: LLM decides whether to use LLMLingua or semantic chunking
- **Token Optimization**: Implements semantic chunking and response length management
- **Comprehensive Logging**: Tracks every decision and stage of processing
- **Budget Constraints**: Enforces hard caps on searches and sources
- **CLI Interface**: Simple command-line interface with argparse

## Project Structure

```
Brand-Visibility-LLMs/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── .gitignore             # Git ignore rules
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── clients/           # LLM client implementations
│   │   ├── llm_client.py         # OpenAI/GPT-4o client
│   │   └── llm_client_gemma.py   # Open-source models client
│   ├── core/              # Core business logic
│   │   ├── models.py             # Pydantic data models
│   │   └── analyzer.py           # Main orchestrator
│   └── utils/             # Utility modules
│       └── web_search.py         # Web search engine
│
├── tests/                 # Test suite
│   ├── test_analyzer.py          # Unit tests
│   ├── test_integration.py       # Integration tests
│   └── demo.py                   # Demo script
│
└── docs/                  # Documentation
    ├── README.md                      # This file (symlink)
    ├── LLM_COMPRESSION_DECISION.md    # Compression routing guide
    ├── LOGGING_GUIDE.md               # Logging documentation
    ├── TOKEN_OPTIMIZATION_IMPROVEMENTS.md  # Token optimization details
    └── sample_output.json             # Example output
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Mehrads/Brand-Visibility-LLMs.git
cd Brand-Visibility-LLMs
```

2. **Set up virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your API keys
```

## Usage

### Basic Usage
```bash
python main.py --brand "Tesla" --url "https://www.tesla.com" --question "What is Tesla?"
```

### Advanced Usage
```bash
python main.py \
    --brand "Tesla" \
    --url "https://www.tesla.com" \
    --question "What is Tesla and what do they do?" \
    --max-searches 3 \
    --max-sources 8 \
    --model "gpt-4o" \
    --output "results.json" \
    --verbose
```

### With Intelligent Compression Routing
```bash
python main.py \
    --brand "Tesla" \
    --url "https://www.tesla.com" \
    --question "What is Tesla?" \
    --enable-llmlingua \
    --verbose
```

**LLM-Decided Compression**: `--enable-llmlingua` activates intelligent compression. The LLM analyzes content and automatically chooses:
- **LLMLingua**: Long, complex content → High-quality compression (2-20x)
- **Semantic chunking**: Well-structured content → Fast truncation

### Using Open-Source Models
```bash
python main.py \
    --brand "Tesla" \
    --url "https://www.tesla.com" \
    --question "What is Tesla?" \
    --model "google/gemma-3-4b-it:free"
```

## CLI Arguments

- `--brand`: Brand name to analyze (required)
- `--url`: Brand website URL (required)
- `--question`: Question to ask the LLM (required)
- `--max-searches`: Maximum web searches to perform (default: 5)
- `--max-sources`: Maximum sources to include (default: 10)
- `--model`: LLM model to use (default: gpt-4o)
  - OpenAI: `gpt-4o`, `gpt-3.5-turbo`, `gpt-4-turbo-preview`
  - Open-source: `google/gemma-3-4b-it:free`
- `--api-key`: API key (optional if env var is set)
- `--output`: Output file path (optional, defaults to stdout)
- `--verbose`: Enable verbose output
- `--enable-llmlingua`: Enable LLM-decided compression

## Token Optimization Strategies

The tool employs a comprehensive approach to token optimization:

### 1. Semantic Chunking (Input Optimization)
- **Paragraph-level**: Preserves logical sections
- **Sentence-level**: Falls back to complete sentences
- **Hard truncation**: Last resort for single long paragraphs
- **Zero token cost**: No LLM calls required
- **60-80% reduction**: Significant savings

### 2. Efficient Context Provision
- Removes redundant whitespace
- Normalizes formatting
- Structured information (bullets, lists)
- Concise instructions

### 3. Response Length Management (Output Optimization)
- Sets `max_tokens=1000` for responses
- Predictable costs
- Quality control

### 4. LLM-Decided Compression (Optional)
- LLM analyzes content complexity
- Chooses between LLMLingua or semantic chunking
- Optimal quality-speed balance

**Total savings: 70-80% token reduction**

## How Detection Works

All detection is performed by the LLM itself, eliminating brittle regex patterns:

- **Citations**: LLM identifies entity mentions with URLs
- **Mentions**: LLM finds all brand name occurrences (case-insensitive, variations)
- **Owned vs. External Sources**: LLM categorizes URLs based on domain ownership

## Web Search and Budget Strategy

- Uses **Tavily Search API** for real-time web search
- Strict budget enforcement:
  - `--max-searches`: Limits total web search queries
  - `--max-sources`: Limits sources included in LLM context
- If budget exhausted: stops fetching, relies on gathered context

## Logging

Comprehensive logging tracks every decision and stage:

```bash
# View all logs
python main.py --verbose ...

# View only key decisions
python main.py ... 2>&1 | grep -E "(===|✓|Decision)"
```

See [docs/LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md) for details.

## Documentation

- [LLM_COMPRESSION_DECISION.md](docs/LLM_COMPRESSION_DECISION.md) - How LLM-decided compression works
- [LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md) - Complete logging reference
- [TOKEN_OPTIMIZATION_IMPROVEMENTS.md](docs/TOKEN_OPTIMIZATION_IMPROVEMENTS.md) - Token optimization details
- [sample_output.json](docs/sample_output.json) - Example output

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyzer.py

# Run with coverage
pytest --cov=src
```

## Example Output

```json
{
  "human_response_markdown": "Tesla, Inc. is an American multinational...",
  "citations": [
    {
      "text": "Tesla's official website",
      "url": "https://www.tesla.com/",
      "entity": "Tesla"
    }
  ],
  "mentions": [
    {
      "text": "Tesla, Inc. is...",
      "position": 0,
      "exact_match": "Tesla",
      "variation": "Tesla, Inc."
    }
  ],
  "owned_sources": ["https://www.tesla.com/"],
  "sources": ["https://en.wikipedia.org/wiki/Tesla,_Inc."],
  "metadata": {
    "token_count": 258,
    "searches_performed": 1,
    "max_searches": 2,
    "sources_included": 6,
    "max_sources": 5,
    "citations_found": 2,
    "mentions_found": 5,
    "processing_time_seconds": 14.15,
    "model_used": "gpt-4o"
  }
}
```

## Requirements

- Python 3.8+
- OpenAI API key (or OpenRouter API key)
- Tavily API key (for web search)
- Optional: LLMLingua dependencies for advanced compression

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

https://github.com/Mehrads/Brand-Visibility-LLMs
