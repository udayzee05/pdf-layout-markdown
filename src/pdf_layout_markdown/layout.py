"""
PDF to Markdown with OpenCV Rectangle Detection

This module provides backward compatibility with the original API.
For new code, prefer using the modular components directly:

    from pdf_layout_markdown import PDFConverter
    from pdf_layout_markdown.extractors import TextExtractor
    from pdf_layout_markdown.detectors import RectangleDetector, TableDetector
    from pdf_layout_markdown.postprocessors import NMSProcessor, MergeProcessor

Approach:
1. Detect all rectangles/table cells using contour detection
2. Create annotated debug image
3. Map text blocks to detected regions
4. Generate markdown preserving layout
"""

import sys
from pathlib import Path

# Re-export models for backward compatibility
from .models import TextBox, Rectangle

# Re-export converter
from .converter import PDFConverter


class PDFLayoutAnalyzer:
    """
    PDF to Markdown converter using OpenCV rectangle detection.
    
    This is a backward-compatible wrapper around PDFConverter.
    For new code, prefer using PDFConverter directly.
    """
    
    def __init__(self, pdf_path: str, dpi: int = 300):
        self._converter = PDFConverter(pdf_path, dpi)
        self.pdf_path = Path(pdf_path)
        self.doc = self._converter.doc
        self.dpi = dpi
        self.zoom = dpi / 72
    
    def analyze_page(self, page_num: int = 0):
        """Analyze page layout and return structured information."""
        result = self._converter.analyze_page(page_num)
        # Return as dict for backward compatibility
        return result.to_dict()
    
    def generate_markdown(self, page_num: int = 0) -> str:
        """Generate markdown preserving spatial layout."""
        return self._converter.generate_markdown(page_num)
    
    def create_annotated_image(self, page_num: int = 0, output_path: str = None):
        """Create annotated image showing detected elements."""
        return self._converter.create_annotated_image(page_num, output_path)
    
    def convert(self, create_debug_image: bool = True) -> str:
        """Convert entire PDF to markdown."""
        return self._converter.convert(create_debug_image)
    
    def save(self, output_path: str = None, create_debug_image: bool = True):
        """Save markdown to file."""
        return self._converter.save(output_path, create_debug_image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python layout.py <pdf_file> [output_file]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = PDFLayoutAnalyzer(pdf_file)
    analyzer.save(output_file)
