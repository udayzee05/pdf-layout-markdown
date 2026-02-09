"""
Markdown generator with layout preservation.
"""

from typing import List, Dict, Any
from collections import defaultdict
from .base import BaseGenerator
from ..models import TextBox


class MarkdownGenerator(BaseGenerator):
    """Generates markdown preserving spatial layout."""
    
    def __init__(self, dpi: int = 300, use_code_block: bool = True):
        self.dpi = dpi
        self.use_code_block = use_code_block
    
    def generate(self, text_boxes: List[TextBox], context: Dict[str, Any] = None) -> str:
        if not text_boxes:
            return ""
        
        char_width = self._calculate_char_width(text_boxes)
        lines = self._group_into_lines(text_boxes)
        
        result = []
        if self.use_code_block:
            result.append("```")
        
        for line_y in sorted(lines.keys()):
            line_boxes = sorted(lines[line_y], key=lambda b: b.x)
            line_text = self._build_line(line_boxes, char_width)
            if line_text:
                result.append(line_text)
        
        if self.use_code_block:
            result.append("```")
        
        return "\n".join(result)
    
    def _calculate_char_width(self, boxes: List[TextBox]) -> float:
        if not boxes:
            return self.dpi / 72 * 7.0
        
        total_width = sum(b.width for b in boxes[:100] if len(b.text) > 0)
        total_chars = sum(len(b.text) for b in boxes[:100] if len(b.text) > 0)
        
        return max(total_width / total_chars, 1.0) if total_chars else self.dpi / 72 * 7.0
    
    def _group_into_lines(self, boxes: List[TextBox]) -> Dict[int, List[TextBox]]:
        y_tolerance = int(self.dpi / 72 * 3)
        lines = defaultdict(list)
        
        for box in boxes:
            matched = False
            for line_y in list(lines.keys()):
                if abs(box.y - line_y) <= y_tolerance:
                    lines[line_y].append(box)
                    matched = True
                    break
            if not matched:
                lines[box.y].append(box)
        
        return lines
    
    def _build_line(self, boxes: List[TextBox], char_width: float) -> str:
        chars = []
        current_x = 0.0
        
        for box in boxes:
            gap = box.x - current_x
            spaces = max(0, int(round(gap / char_width)))
            chars.append(" " * spaces)
            chars.append(box.text)
            current_x = box.x + len(box.text) * char_width
        
        return "".join(chars).rstrip()
