# Logging Guide

## Overview

Comprehensive logging has been added to both `llm_client.py` and `llm_client_gemma.py` to track every decision and stage of the process.

---

## What Gets Logged

### 1. **Initialization Stage**
```
✓ LLMLingua compressor initialized successfully
```
or
```
Warning: Could not initialize LLMLingua: [error]
Falling back to semantic chunking only
```

### 2. **Content Optimization Stage**
```
=== Content Optimization Stage ===
Initial content: 562 tokens, target: 400 tokens
Step 1: Removing redundant whitespace...
✓ Whitespace cleanup: 562 → 556 tokens (saved 6)
Step 2: Determining compression method...
```

### 3. **Compression Decision Stage**
```
=== Compression Decision Stage ===
Content analysis: 556 tokens → 400 tokens (71.9% compression)
Quick decision: Content is short (556 < 1000 tokens) → Semantic chunking
```

or for complex content:

```
=== Compression Decision Stage ===
Content analysis: 2500 tokens → 400 tokens (16.0% compression)
Content requires analysis, asking LLM for complexity assessment...
LLM complexity assessment: 'yes'
✓ Decision: USE LLMLINGUA (content is complex)
```

### 4. **LLMLingua Compression Stage** (if selected)
```
=== LLMLingua Compression Stage ===
Starting LLMLingua compression: 2500 → 400 tokens (16.0% rate)
✓ LLMLingua compression complete: 2500 → 398 tokens
=== Optimization Complete (LLMLingua) ===
```

### 5. **Semantic Chunking Stage** (if selected)
```
=== Semantic Chunking Stage ===
Using semantic chunking: 556 → 400 tokens
✓ Semantic chunking complete: 556 → 398 tokens
=== Optimization Complete (Semantic) ===
```

### 6. **Response Generation Stage**
```
=== Response Generation Stage ===
Question: What is Tesla?
Brand: Tesla (https://www.tesla.com)
Search results: 3 sources
Max response tokens: 1000
Building context from search results...
Processing source 1/3: Tesla, Inc. - Wikipedia...
Processing source 2/3: Tesla Official Site...
Processing source 3/3: Britannica - Tesla Motors...
✓ Built context from 3 sources
Calling LLM for response generation...
Model: gpt-4o, max_tokens: 1000
✓ LLM response received, tokens used: 361
✓ Final response: 154 tokens
=== Response Generation Complete ===
```

### 7. **Citation & Mention Analysis Stage**
```
=== Citation & Mention Analysis Stage ===
Analyzing response for brand: Tesla
Response length: 802 characters, 154 tokens
Calling LLM for citation and mention analysis...
✓ Analysis response received, tokens used: 586
Parsing JSON analysis...
✓ Found 2 citations and 4 mentions
=== Analysis Complete ===
```

### 8. **Source Categorization Stage**
```
=== Source Categorization Stage ===
Categorizing 3 URLs for brand: https://www.tesla.com
Brand domain: www.tesla.com
Calling LLM for source categorization...
✓ Categorization response received, tokens used: 248
Parsing categorization JSON...
✓ Categorized: 2 owned, 1 external
=== Categorization Complete ===
```

---

## Viewing Logs

### Full Logs
```bash
python main.py --brand "Tesla" --question "What is Tesla?" 2>&1
```

### Filtered Logs (Key Events Only)
```bash
python main.py --brand "Tesla" --question "What is Tesla?" 2>&1 | grep -E "(===|✓|Decision)"
```

### Stage-Specific Logs
```bash
# Compression decisions only
python main.py ... 2>&1 | grep "Compression Decision"

# Token counts only
python main.py ... 2>&1 | grep "tokens"

# LLM calls only
python main.py ... 2>&1 | grep "Calling LLM"
```

---

## Example Log Output

### Scenario 1: Simple Content (Semantic Chunking)

```
2025-10-01 10:15:23 - llm_client - INFO - === Content Optimization Stage ===
2025-10-01 10:15:23 - llm_client - INFO - Initial content: 800 tokens, target: 400 tokens
2025-10-01 10:15:23 - llm_client - INFO - Step 1: Removing redundant whitespace...
2025-10-01 10:15:23 - llm_client - INFO - ✓ Whitespace cleanup: 800 → 795 tokens (saved 5)
2025-10-01 10:15:23 - llm_client - INFO - Step 2: Determining compression method...
2025-10-01 10:15:23 - llm_client - INFO - === Compression Decision Stage ===
2025-10-01 10:15:23 - llm_client - INFO - Content analysis: 795 tokens → 400 tokens (50.3% compression)
2025-10-01 10:15:23 - llm_client - INFO - Quick decision: Content is short (795 < 1000 tokens) → Semantic chunking
2025-10-01 10:15:23 - llm_client - INFO - === Semantic Chunking Stage ===
2025-10-01 10:15:23 - llm_client - INFO - Using semantic chunking: 795 → 400 tokens
2025-10-01 10:15:23 - llm_client - INFO - ✓ Semantic chunking complete: 795 → 398 tokens
2025-10-01 10:15:23 - llm_client - INFO - === Optimization Complete (Semantic) ===
```

### Scenario 2: Complex Content (LLMLingua)

```
2025-10-01 10:20:15 - llm_client - INFO - === Content Optimization Stage ===
2025-10-01 10:20:15 - llm_client - INFO - Initial content: 2500 tokens, target: 400 tokens
2025-10-01 10:20:15 - llm_client - INFO - Step 1: Removing redundant whitespace...
2025-10-01 10:20:15 - llm_client - INFO - ✓ Whitespace cleanup: 2500 → 2485 tokens (saved 15)
2025-10-01 10:20:15 - llm_client - INFO - Step 2: Determining compression method...
2025-10-01 10:20:15 - llm_client - INFO - === Compression Decision Stage ===
2025-10-01 10:20:15 - llm_client - INFO - Content analysis: 2485 tokens → 400 tokens (16.1% compression)
2025-10-01 10:20:15 - llm_client - INFO - Content requires analysis, asking LLM for complexity assessment...
2025-10-01 10:20:16 - llm_client - INFO - LLM complexity assessment: 'yes'
2025-10-01 10:20:16 - llm_client - INFO - ✓ Decision: USE LLMLINGUA (content is complex)
2025-10-01 10:20:16 - llm_client - INFO - === LLMLingua Compression Stage ===
2025-10-01 10:20:16 - llm_client - INFO - Starting LLMLingua compression: 2485 → 400 tokens (16.1% rate)
2025-10-01 10:20:17 - llm_client - INFO - ✓ LLMLingua compression complete: 2485 → 398 tokens
2025-10-01 10:20:17 - llm_client - INFO - === Optimization Complete (LLMLingua) ===
```

---

## Log Levels

### INFO (Default)
- All stages and decisions
- Token counts
- Success messages
- Progress indicators

### WARNING
- LLMLingua initialization failures
- Fallback scenarios
- Missing data

### ERROR
- LLM API failures
- JSON parsing errors
- Compression failures

---

## Customizing Logging

### Change Log Level

```python
import logging

# Show only warnings and errors
logging.getLogger('llm_client').setLevel(logging.WARNING)

# Show debug information
logging.getLogger('llm_client').setLevel(logging.DEBUG)

# Disable logging
logging.getLogger('llm_client').setLevel(logging.ERROR)
```

### Save Logs to File

```python
import logging

# Add file handler
file_handler = logging.FileHandler('brand_analyzer.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger('llm_client').addHandler(file_handler)
```

### Custom Format

```python
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'  # Simpler format
)
```

---

## Monitoring Key Metrics

### Track All Decisions

```bash
python main.py --enable-llmlingua ... 2>&1 | grep "Decision:"
```

**Output:**
```
✓ Decision: USE SEMANTIC CHUNKING (content is simple)
✓ Decision: USE LLMLINGUA (content is complex)
```

### Track Token Usage

```bash
python main.py ... 2>&1 | grep "tokens used"
```

**Output:**
```
✓ LLM response received, tokens used: 361
✓ Analysis response received, tokens used: 586
✓ Categorization response received, tokens used: 248
```

### Track Compression Results

```bash
python main.py --enable-llmlingua ... 2>&1 | grep "complete:"
```

**Output:**
```
✓ Semantic chunking complete: 795 → 398 tokens
✓ LLMLingua compression complete: 2485 → 398 tokens
```

---

## Benefits

### 1. Transparency
✅ See every decision the system makes
✅ Understand why LLMLingua was/wasn't used
✅ Track token usage at each stage

### 2. Debugging
✅ Identify bottlenecks
✅ Find compression failures
✅ Trace errors to specific stages

### 3. Optimization
✅ Measure compression effectiveness
✅ Tune thresholds based on data
✅ Identify improvement opportunities

### 4. Monitoring
✅ Track LLM API calls
✅ Monitor token costs
✅ Measure processing time per stage

---

## Log Analysis Examples

### Find All LLM Calls

```bash
python main.py ... 2>&1 | grep "Calling LLM"
```

**Shows:**
- Response generation call
- Complexity assessment call
- Citation analysis call
- Source categorization call

### Calculate Total Token Usage

```bash
python main.py ... 2>&1 | grep "tokens used" | awk '{sum += $NF} END {print "Total:", sum}'
```

### Find Compression Decisions

```bash
python main.py --enable-llmlingua ... 2>&1 | grep -A2 "Compression Decision Stage"
```

---

## Summary

**What's logged:**
- ✅ Every stage of processing
- ✅ All LLM decisions
- ✅ Token counts (before/after)
- ✅ Compression methods used
- ✅ Success/failure status

**Benefits:**
- Complete transparency
- Easy debugging
- Performance monitoring
- Decision tracking

**Usage:**
Just run normally - logging is automatic! 🎉

