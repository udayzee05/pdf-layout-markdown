"""
Filter post-processor for removing invalid rectangles.
"""

from typing import List, Any, Dict
from .base import BasePostProcessor
from ..models import Rectangle


class FilterProcessor(BasePostProcessor):
    """Filters rectangles by size and other criteria."""
    
    def __init__(self, min_area: int = 0, max_area: int = None, 
                 min_area_ratio: float = None, name: str = None):
        super().__init__(name)
        self.min_area = min_area
        self.max_area = max_area
        self.min_area_ratio = min_area_ratio
    
    def process(self, items: List[Rectangle], context: Dict[str, Any] = None) -> List[Rectangle]:
        if not items:
            return []
        
        min_area = self.min_area
        max_area = self.max_area
        
        # Calculate dynamic thresholds from page dimensions
        if context and self.min_area_ratio:
            dims = context.get("dimensions")
            if dims:
                page_area = dims[0] * dims[1]
                min_area = int(page_area * self.min_area_ratio)
        
        filtered = []
        for rect in items:
            if rect.area < min_area:
                continue
            if max_area and rect.area > max_area:
                continue
            filtered.append(rect)
        
        return filtered
