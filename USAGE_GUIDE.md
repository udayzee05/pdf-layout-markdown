# PDF to LLM Pipeline - Usage Guide

## Overview

This guide demonstrates how to use the `pdf-layout-markdown` library to convert PDF documents to Markdown and analyze them with Large Language Models (LLMs).

## Quick Start

### 1. Simple Example (Recommended for Beginners)

The simplest way to get started is with `simple_example.py`:

```bash
python simple_example.py
```

This script demonstrates the basic workflow:
1. Converts `invoice.pdf` to Markdown
2. Sends the Markdown to OpenAI's GPT-4o-mini
3. Saves the analysis results to `analysis_result.json`

### 2. Command Line Tool (Recommended for Production)

For more control and flexibility, use the `pdf_to_llm.py` CLI tool:

```bash
# Basic usage - processes a PDF and saves results
python pdf_to_llm.py invoice.pdf

# Specify custom output file
python pdf_to_llm.py invoice.pdf --output my_results.json

# Use a different OpenAI model
python pdf_to_llm.py invoice.pdf --model gpt-4o

# Get help
python pdf_to_llm.py --help
```

### 3. Python API (Recommended for Integration)

For integrating into your own applications:

```python
from pdf_to_llm import PDFToLLMPipeline

# Initialize the pipeline
pipeline = PDFToLLMPipeline(model="gpt-4o-mini")

# Process a single PDF
result = pipeline.process("invoice.pdf", output_path="analysis.json")

# Access the results
document_type = result["parsed_document"]["document_type"]
validation_status = result["parsed_document"]["validation"]["overall_status"]
cost = result["meta"]["cost_usd"]

print(f"Document Type: {document_type}")
print(f"Status: {validation_status}")
print(f"Cost: ${cost:.6f}")
```

## Step-by-Step Workflow

### Step 1: PDF to Markdown Conversion

```python
from pdf_layout_markdown import PDFLayoutAnalyzer

# Initialize analyzer
analyzer = PDFLayoutAnalyzer("path/to/document.pdf")

# Convert to Markdown
markdown_content = analyzer.convert()

# Save the markdown
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)
```

**What happens:**
- The library analyzes the PDF layout using OpenCV
- Detects text blocks, tables, and structural elements
- Preserves the spatial arrangement in Markdown format
- Generates debug images showing detected regions

### Step 2: LLM Analysis

```python
from openai import OpenAI
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a document analyst. Extract key information..."
        },
        {
            "role": "user",
            "content": markdown_content
        }
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

# Parse the result
result = json.loads(response.choices[0].message.content)
```

**What happens:**
- The Markdown is sent to the LLM
- The LLM analyzes the document structure and content
- Extracts structured information (parties, dates, amounts, etc.)
- Validates the data and flags any issues
- Returns a JSON object with all findings

### Step 3: Working with Results

The result contains:

```json
{
  "parsed_document": {
    "document_type": "Tax Invoice",
    "document_category": "IMPORT",
    "parties": {...},
    "reference_numbers": {...},
    "dates": {...},
    "route": {...},
    "goods": [...],
    "financials": {...},
    "validation": {
      "overall_status": "WARNINGS",
      "flags": [...]
    },
    "summary": "..."
  },
  "meta": {
    "tokens_used": 5081,
    "cost_usd": 0.001668,
    "processing_time_sec": 48.44,
    "model": "gpt-4o-mini"
  },
  "source": {
    "pdf_file": "...",
    "processed_at": "..."
  }
}
```

## Advanced Usage

### Custom System Prompt

You can customize the analysis by providing your own system prompt:

```python
from pdf_to_llm import PDFToLLMPipeline

custom_prompt = """
You are a financial document analyst.
Extract all monetary values, dates, and party information.
Return results as JSON with the following structure:
{
  "total_amount": "...",
  "currency": "...",
  "parties": [...],
  "dates": [...]
}
"""

pipeline = PDFToLLMPipeline(
    model="gpt-4o-mini",
    system_prompt=custom_prompt
)

result = pipeline.process("invoice.pdf")
```

### Batch Processing

Process multiple PDFs:

```python
from pdf_to_llm import PDFToLLMPipeline
from pathlib import Path
import json

pipeline = PDFToLLMPipeline()

pdf_files = Path("input").glob("*.pdf")
results = {}

for pdf_file in pdf_files:
    print(f"Processing {pdf_file.name}...")
    result = pipeline.process(str(pdf_file))
    results[pdf_file.name] = result

# Save all results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Cost Tracking

Track costs across multiple documents:

```python
from pdf_to_llm import PDFToLLMPipeline

pipeline = PDFToLLMPipeline()
total_cost = 0
total_tokens = 0

for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    result = pipeline.process(pdf_file)
    total_cost += result["meta"]["cost_usd"]
    total_tokens += result["meta"]["tokens_used"]

print(f"Total cost: ${total_cost:.6f}")
print(f"Total tokens: {total_tokens}")
```

## Environment Setup

### 1. Install Dependencies

```bash
# Using pip
pip install pdf-layout-markdown openai python-dotenv tiktoken

# Or using uv (recommended)
uv pip install pdf-layout-markdown openai python-dotenv tiktoken
```

### 2. Configure API Key

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-api-key-here
```

**Security Note:** Never commit your `.env` file to version control. Add it to `.gitignore`.

### 3. Verify Installation

```python
from pdf_layout_markdown import PDFLayoutAnalyzer
from openai import OpenAI

print("✅ All dependencies installed successfully!")
```

## Troubleshooting

### Issue: "No module named 'pdf_layout_markdown'"

**Solution:** Install the library:
```bash
pip install pdf-layout-markdown
```

### Issue: "OpenAI API key not found"

**Solution:** Create a `.env` file with your API key or set it as an environment variable:
```bash
export OPENAI_API_KEY=your-key-here  # Linux/Mac
set OPENAI_API_KEY=your-key-here     # Windows
```

### Issue: "PDF conversion produces poor quality Markdown"

**Solution:** The library works best with:
- Structured documents (invoices, forms, tables)
- Clear text (not scanned images)
- Standard layouts

For scanned PDFs, consider using OCR preprocessing.

### Issue: "LLM analysis is too expensive"

**Solution:**
- Use `gpt-4o-mini` instead of `gpt-4o` (much cheaper)
- Process only essential pages
- Use smaller system prompts
- Batch similar documents together

## Best Practices

1. **Start Small:** Test with a single PDF before batch processing
2. **Monitor Costs:** Track token usage and costs, especially with large documents
3. **Validate Results:** Always review LLM outputs for accuracy
4. **Save Intermediates:** Keep the Markdown files for debugging
5. **Version Control:** Track your system prompts and configurations
6. **Error Handling:** Wrap API calls in try-except blocks for production use

## Example Output

After running the pipeline, you'll get:

**Files Created:**
- `invoice.md` - The Markdown conversion
- `invoice_page1_debug.png` - Visual debug image
- `test_analysis.json` - Structured analysis results

**Console Output:**
```
📄 Converting PDF to Markdown: invoice.pdf
✅ Markdown saved to: invoice.md
🤖 Analyzing with gpt-4o-mini...
✅ Analysis complete in 48.44s
   Tokens: 5081 | Cost: $0.001668
💾 Results saved to: test_analysis.json

============================================================
📊 ANALYSIS SUMMARY
============================================================
Document Type: Tax Invoice
Validation Status: WARNINGS
Model: gpt-4o-mini
Tokens Used: 5081
Cost: $0.001668
Processing Time: 48.44s
============================================================
```

## Next Steps

- Customize the system prompt for your specific use case
- Integrate the pipeline into your application
- Set up batch processing for multiple documents
- Add custom validation rules
- Export results to your database or system

## Support

For issues or questions:
- Check the README.md
- Review the example scripts
- Examine the debug images to understand layout detection
- Test with different PDFs to understand capabilities
