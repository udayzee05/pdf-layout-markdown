# PDF Layout Markdown

## Overview

`pdf-layout-markdown` is a Python library designed to extract content from PDF documents while preserving the original layout structure as much as possible. It utilizes OpenCV for layout detection and PyMuPDF for text extraction, making it particularly useful for structured documents like invoices, forms, and tables.

## Features

- **Layout Preservation**: accurately detects and maintains the spatial arrangement of text blocks.
- **Table Detection**: Identifies table structures and cells using OpenCV contour detection.
- **Debug Visualization**: Generates annotated images showing detected text regions and layout boundaries.
- **Markdown Output**: Converts the PDF content into clean, structured Markdown.

## Installation

```bash
pip install pdf-layout-markdown
```

## Usage

```python
from pdf_layout_markdown import PDFLayoutAnalyzer

# Initialize the analyzer with your PDF file
analyzer = PDFLayoutAnalyzer("path/to/document.pdf")

# Convert the entire PDF to Markdown
# This will also generate debug images by default (e.g., document_page1_debug.png)
markdown_content = analyzer.convert()

# Print or save the markdown content
print(markdown_content)

# Alternatively, save directly to a file
with open("output.md", "w") as f:
    f.write(markdown_content)
```

## Requirements

- Python 3.12+
- OpenCV
- PyMuPDF
- NumPy
- OpenAI (for LLM integration)
- python-dotenv (for environment variables)

## PDF to LLM Pipeline

This library includes a complete pipeline for converting PDFs to Markdown and analyzing them with LLMs.

### Quick Start

```python
from pdf_to_llm import PDFToLLMPipeline

# Initialize the pipeline
pipeline = PDFToLLMPipeline(model="gpt-4o-mini")

# Process a PDF: converts to Markdown and analyzes with LLM
result = pipeline.process("invoice.pdf", output_path="analysis.json")

# Access the results
print(result["parsed_document"]["document_type"])
print(result["meta"]["cost_usd"])
```

### Command Line Usage

```bash
# Basic usage
python pdf_to_llm.py invoice.pdf

# Specify output file
python pdf_to_llm.py invoice.pdf --output results.json

# Use a different model
python pdf_to_llm.py invoice.pdf --model gpt-4o --output analysis.json
```

### Simple Example

For a minimal example, see `simple_example.py`:

```python
from pdf_layout_markdown import PDFLayoutAnalyzer
from openai import OpenAI

# Step 1: Convert PDF to Markdown
analyzer = PDFLayoutAnalyzer("invoice.pdf")
markdown_content = analyzer.convert()

# Step 2: Analyze with LLM
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract key information..."},
        {"role": "user", "content": markdown_content}
    ],
    response_format={"type": "json_object"}
)
```

### Environment Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```
