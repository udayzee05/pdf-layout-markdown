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
