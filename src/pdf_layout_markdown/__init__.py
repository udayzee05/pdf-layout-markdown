"""
PDF Layout Markdown - Extract PDF content preserving spatial layout.

A modular, object-oriented library for converting PDF documents to
Markdown while preserving the original layout structure.

Quick Start:
    from pdf_layout_markdown import PDFConverter
    
    converter = PDFConverter("document.pdf")
    markdown = converter.convert()

Modular Components:
    - models: TextBox, Rectangle data classes
    - extractors: Text extraction from PDFs
    - detectors: Rectangle and table detection
    - renderers: PDF page rendering
    - generators: Markdown output generation
    - postprocessors: Result refinement pipeline
    - analyzers: Page layout analysis
    - visualizers: Debug visualization
"""

__version__ = "0.1.0"

# Main converter
from .converter import PDFConverter

# Backward compatibility
from .layout import PDFLayoutAnalyzer

# Models
from .models import TextBox, Rectangle

# Analyzers
from .analyzers import PageAnalyzer

# Generators
from .generators import MarkdownGenerator

__all__ = [
    # Main classes
    "PDFConverter",
    "PDFLayoutAnalyzer",
    
    # Models
    "TextBox",
    "Rectangle",
    
    # Components
    "PageAnalyzer",
    "MarkdownGenerator",
]
