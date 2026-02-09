"""
Main PDF to Markdown converter using modular components.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from .analyzers import PageAnalyzer
from .generators import MarkdownGenerator
from .renderers import PageRenderer
from .visualizers import PageAnnotator


class PDFConverter:
    """
    Main PDF to Markdown converter.
    
    Orchestrates all modular components to convert PDF documents
    to markdown while preserving layout.
    
    Example:
        converter = PDFConverter("document.pdf")
        markdown = converter.convert()
        converter.save("output.md")
    """
    
    def __init__(self, pdf_path: str, dpi: int = 300, output_dir: str = "output"):
        """
        Initialize the converter.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Rendering DPI (default: 300)
            output_dir: Directory for output files (default: "output")
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required")
        
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.doc = fitz.open(self.pdf_path)
        
        # Initialize components
        self.analyzer = PageAnalyzer(dpi)
        self.generator = MarkdownGenerator(dpi)
        self.renderer = PageRenderer(dpi)
        self.annotator = PageAnnotator()
    
    @property
    def page_count(self) -> int:
        """Number of pages in the PDF."""
        return len(self.doc)
    
    def analyze_page(self, page_num: int = 0):
        """Analyze a single page."""
        page = self.doc[page_num]
        return self.analyzer.analyze(page)
    
    def generate_markdown(self, page_num: int = 0) -> str:
        """Generate markdown for a single page."""
        analysis = self.analyze_page(page_num)
        return self.generator.generate(analysis.text_boxes)
    
    def create_annotated_image(self, page_num: int = 0, output_path: str = None) -> np.ndarray:
        """Create debug visualization for a page."""
        page = self.doc[page_num]
        image = self.renderer.render(page)
        analysis = self.analyzer.analyze(page)
        
        annotated = self.annotator.annotate(
            image,
            rectangles=analysis.rectangles,
            cells=analysis.cells,
            text_boxes=analysis.text_boxes,
            h_lines=analysis.h_lines,
            v_lines=analysis.v_lines
        )
        
        if output_path:
            self.annotator.save(annotated, output_path)
        
        return annotated
    
    def convert(self, create_debug_image: bool = True) -> str:
        """Convert entire PDF to markdown."""
        md_parts = []
        
        for i in range(self.page_count):
            if create_debug_image:
                debug_path = self.output_dir / f"{self.pdf_path.stem}_page{i+1}_debug.png"
                self.create_annotated_image(i, str(debug_path))
            
            md_parts.append(self.generate_markdown(i))
            md_parts.append("\n\n---\n\n")
        
        return "".join(md_parts)
    
    def save(self, output_path: str = None, create_debug_image: bool = True) -> str:
        """Save markdown to file in output directory."""
        if output_path is None:
            output_path = self.output_dir / f"{self.pdf_path.stem}.md"
        else:
            output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        markdown = self.convert(create_debug_image)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        print(f"✅ Saved markdown to: {output_path}")
        return markdown
    
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
