"""
Merge nearby rectangles post-processor.
"""

from typing import List, Any, Dict
from .base import BasePostProcessor
from ..models import Rectangle, TextBox


class MergeProcessor(BasePostProcessor):
    """Merges nearby rectangles and expands to include text."""
    
    def __init__(self, max_gap: int = 20, padding: int = 10, name: str = None):
        super().__init__(name)
        self.max_gap = max_gap
        self.padding = padding
    
    def process(self, items: List[Rectangle], context: Dict[str, Any] = None) -> List[Rectangle]:
        if not items:
            return []
        
        text_boxes = context.get("text_boxes", []) if context else []
        
        # Expand rectangles to include nearby text
        expanded = self._expand_with_text(items, text_boxes)
        
        # Merge nearby rectangles
        return self._merge_nearby(expanded)
    
    def _expand_with_text(self, rects: List[Rectangle], boxes: List[TextBox]) -> List[Rectangle]:
        expanded = []
        for rect in rects:
            related = [b for b in boxes if self._is_related(b, rect)]
            if related:
                min_x = min(min(b.x for b in related), rect.x) - self.padding
                min_y = min(min(b.y for b in related), rect.y) - self.padding
                max_x = max(max(b.x2 for b in related), rect.x2) + self.padding
                max_y = max(max(b.y2 for b in related), rect.y2) + self.padding
                expanded.append(Rectangle(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y), rect.level))
            else:
                expanded.append(rect)
        return expanded
    
    def _is_related(self, box: TextBox, rect: Rectangle) -> bool:
        return (box.x >= rect.x - self.padding and box.x2 <= rect.x2 + self.padding and
                box.y >= rect.y - self.padding and box.y2 <= rect.y2 + self.padding)
    
    def _merge_nearby(self, rects: List[Rectangle]) -> List[Rectangle]:
        merged = []
        used = set()
        
        for i, rect1 in enumerate(rects):
            if i in used:
                continue
            
            group = [rect1]
            for j, rect2 in enumerate(rects):
                if i == j or j in used:
                    continue
                if self._should_merge(rect1, rect2):
                    group.append(rect2)
                    used.add(j)
            
            min_x = min(r.x for r in group)
            min_y = min(r.y for r in group)
            max_x = max(r.x2 for r in group)
            max_y = max(r.y2 for r in group)
            merged.append(Rectangle(min_x, min_y, max_x - min_x, max_y - min_y, rect1.level))
        
        return merged
    
    def _should_merge(self, r1: Rectangle, r2: Rectangle) -> bool:
        h_gap = max(r1.x, r2.x) - min(r1.x2, r2.x2)
        v_gap = max(r1.y, r2.y) - min(r1.y2, r2.y2)
        x_overlap = min(r1.x2, r2.x2) - max(r1.x, r2.x)
        y_overlap = min(r1.y2, r2.y2) - max(r1.y, r2.y)
        
        if x_overlap > min(r1.width, r2.width) * 0.5 and v_gap < self.max_gap:
            return True
        if y_overlap > min(r1.height, r2.height) * 0.5 and h_gap < self.max_gap:
            return True
        return False
