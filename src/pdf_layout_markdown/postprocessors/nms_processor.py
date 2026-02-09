"""
Non-Maximum Suppression (NMS) post-processor.
"""

from typing import List, Any, Dict
from .base import BasePostProcessor
from ..models import Rectangle


class NMSProcessor(BasePostProcessor):
    """Applies NMS to remove overlapping rectangles."""
    
    def __init__(self, iou_threshold: float = 0.5, name: str = None):
        super().__init__(name)
        self.iou_threshold = iou_threshold
    
    def process(self, items: List[Rectangle], context: Dict[str, Any] = None) -> List[Rectangle]:
        if not items:
            return []
        
        sorted_rects = sorted(items, key=lambda r: r.area, reverse=True)
        keep = []
        
        for rect in sorted_rects:
            should_keep = True
            for kept_rect in keep:
                if rect.compute_iou(kept_rect) > self.iou_threshold:
                    should_keep = False
                    break
                if kept_rect.contains_rectangle(rect):
                    should_keep = False
                    break
            if should_keep:
                keep.append(rect)
        
        return keep
