"""
Page annotation for debug visualization.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from ..models import TextBox, Rectangle


class PageAnnotator:
    """Creates annotated images for debugging layout detection."""
    
    # Default colors (BGR format)
    COLORS = {
        "rectangle": (255, 0, 0),    # Blue
        "cell": (0, 255, 0),          # Green
        "text": (0, 0, 255),          # Red
        "h_line": (255, 255, 0),      # Cyan
        "v_line": (255, 0, 255),      # Magenta
    }
    
    def __init__(self, colors: dict = None):
        self.colors = {**self.COLORS, **(colors or {})}
    
    def annotate(
        self,
        image: np.ndarray,
        rectangles: List[Rectangle] = None,
        cells: List[Rectangle] = None,
        text_boxes: List[TextBox] = None,
        h_lines: List[int] = None,
        v_lines: List[int] = None
    ) -> np.ndarray:
        """Create annotated image with all detected elements."""
        annotated = image.copy()
        
        if rectangles:
            self._draw_rectangles(annotated, rectangles, self.colors["rectangle"], 2)
        if cells:
            self._draw_rectangles(annotated, cells, self.colors["cell"], 1)
        if h_lines:
            self._draw_h_lines(annotated, h_lines, self.colors["h_line"])
        if v_lines:
            self._draw_v_lines(annotated, v_lines, self.colors["v_line"])
        if text_boxes:
            self._draw_text_boxes(annotated, text_boxes, self.colors["text"])
        
        return annotated
    
    def _draw_rectangles(self, img: np.ndarray, rects: List[Rectangle], color: Tuple, thickness: int):
        for rect in rects:
            cv2.rectangle(img, (rect.x, rect.y), (rect.x2, rect.y2), color, thickness)
    
    def _draw_text_boxes(self, img: np.ndarray, boxes: List[TextBox], color: Tuple):
        for box in boxes:
            cv2.rectangle(img, (box.x, box.y), (box.x2, box.y2), color, 1)
    
    def _draw_h_lines(self, img: np.ndarray, lines: List[int], color: Tuple):
        for y in lines:
            cv2.line(img, (0, y), (img.shape[1], y), color, 1)
    
    def _draw_v_lines(self, img: np.ndarray, lines: List[int], color: Tuple):
        for x in lines:
            cv2.line(img, (x, 0), (x, img.shape[0]), color, 1)
    
    def save(self, image: np.ndarray, path: str):
        """Save annotated image to file."""
        cv2.imwrite(path, image)
        print(f"✅ Annotated image saved: {path}")
