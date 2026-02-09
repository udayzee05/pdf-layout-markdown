"""
Page layout analysis combining all detection components.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field

from ..models import TextBox, Rectangle
from ..extractors import TextExtractor
from ..detectors import RectangleDetector, TableDetector
from ..renderers import PageRenderer
from ..postprocessors import PostProcessorPipeline, NMSProcessor, MergeProcessor, FilterProcessor


@dataclass
class PageAnalysisResult:
    """Container for page analysis results."""
    rectangles: List[Rectangle] = field(default_factory=list)
    cells: List[Rectangle] = field(default_factory=list)
    cell_text: Dict[int, List[TextBox]] = field(default_factory=dict)
    h_lines: List[int] = field(default_factory=list)
    v_lines: List[int] = field(default_factory=list)
    text_boxes: List[TextBox] = field(default_factory=list)
    uncategorized: List[TextBox] = field(default_factory=list)
    dimensions: tuple = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rectangles": self.rectangles,
            "cells": self.cells,
            "cell_text": self.cell_text,
            "h_lines": self.h_lines,
            "v_lines": self.v_lines,
            "text_boxes": self.text_boxes,
            "uncategorized": self.uncategorized,
            "dimensions": self.dimensions
        }


class PageAnalyzer:
    """Analyzes PDF page layout using all detection components."""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.renderer = PageRenderer(dpi)
        self.text_extractor = TextExtractor()
        self.rect_detector = RectangleDetector()
        self.table_detector = TableDetector()
        
        # Default post-processing pipeline
        self.pipeline = PostProcessorPipeline([
            MergeProcessor(max_gap=30, name="merge"),
            NMSProcessor(iou_threshold=0.6, name="nms"),
            FilterProcessor(min_area_ratio=0.01, name="filter")
        ])
    
    def analyze(self, page) -> PageAnalysisResult:
        """Analyze a PDF page and return structured results."""
        # Render page to image
        image = self.renderer.render(page)
        scale = self.renderer.scale
        
        # Extract text
        text_boxes = self.text_extractor.extract(page, scale)
        
        # Detect rectangles
        rectangles = self.rect_detector.detect(image)
        
        # Apply post-processing
        context = {"text_boxes": text_boxes, "dimensions": image.shape[:2][::-1]}
        rectangles = self.pipeline.process(rectangles, context)
        
        # Detect table grid
        h_lines, v_lines, cells = self.table_detector.get_table_structure(image)
        
        # Map text to cells
        cell_text = {}
        categorized = []
        for i, cell in enumerate(cells):
            cell_text[i] = [b for b in text_boxes if self._box_in_cell(b, cell)]
            categorized.extend(cell_text[i])
        
        uncategorized = [b for b in text_boxes if b not in categorized]
        
        return PageAnalysisResult(
            rectangles=rectangles,
            cells=cells,
            cell_text=cell_text,
            h_lines=h_lines,
            v_lines=v_lines,
            text_boxes=text_boxes,
            uncategorized=uncategorized,
            dimensions=(image.shape[1], image.shape[0])
        )
    
    def _box_in_cell(self, box: TextBox, cell: Rectangle) -> bool:
        return (box.x >= cell.x and box.y >= cell.y and
                box.x2 <= cell.x2 and box.y2 <= cell.y2)
