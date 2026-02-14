"""
Structured markdown generator optimized for LLM understanding.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re
from .base import BaseGenerator
from ..models import TextBox, Rectangle


class StructuredMarkdownGenerator(BaseGenerator):
    """
    Generates structured markdown optimized for LLM parsing.
    
    Features:
    - Semantic sections (Header, Body, Tables, Footer)
    - Key-value pair extraction
    - Proper markdown tables for tabular data
    - Metadata frontmatter
    - Clear hierarchical structure
    """
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.char_width = None
    
    def generate(self, text_boxes: List[TextBox], context: Dict[str, Any] = None) -> str:
        """Generate structured markdown from text boxes."""
        if not text_boxes:
            return ""
        
        context = context or {}
        rectangles = context.get('rectangles', [])
        cells = context.get('cells', [])
        
        self.char_width = self._calculate_char_width(text_boxes)
        
        # Organize content into sections
        sections = self._organize_sections(text_boxes, rectangles)
        
        # Extract key-value pairs
        key_values = self._extract_key_values(text_boxes)
        
        # Build structured output
        result = []
        
        # Add metadata frontmatter
        result.append("---")
        result.append("document_type: invoice_or_delivery_order")
        if key_values:
            result.append("key_fields:")
            for key, value in list(key_values.items())[:10]:  # Top 10 key fields
                result.append(f"  {key}: {value}")
        result.append("---")
        result.append("")
        
        # Add main title
        result.append("# Document Content")
        result.append("")
        
        # Add sections
        for section_name, section_boxes in sections.items():
            if section_boxes:
                result.append(f"## {section_name}")
                result.append("")
                
                # Check if this looks like tabular data
                if self._is_tabular(section_boxes):
                    table_md = self._generate_table(section_boxes)
                    result.append(table_md)
                else:
                    # Generate as key-value pairs or structured text
                    structured_text = self._generate_structured_text(section_boxes)
                    result.append(structured_text)
                
                result.append("")
        
        # Add raw spatial layout as reference
        result.append("## Raw Spatial Layout")
        result.append("")
        result.append("```")
        result.append(self._generate_spatial_layout(text_boxes))
        result.append("```")
        result.append("")
        
        return "\n".join(result)
    
    def _organize_sections(self, text_boxes: List[TextBox], rectangles: List[Rectangle]) -> Dict[str, List[TextBox]]:
        """Organize text boxes into logical sections."""
        if not text_boxes:
            return {}
        
        # Sort by Y position
        sorted_boxes = sorted(text_boxes, key=lambda b: b.y)
        
        # Divide into sections based on Y position
        page_height = max(b.y2 for b in text_boxes) if text_boxes else 1000
        
        sections = {
            "Header": [],
            "Body": [],
            "Footer": []
        }
        
        for box in sorted_boxes:
            # Top 20% = Header
            if box.y < page_height * 0.2:
                sections["Header"].append(box)
            # Bottom 15% = Footer
            elif box.y > page_height * 0.85:
                sections["Footer"].append(box)
            # Middle = Body
            else:
                sections["Body"].append(box)
        
        return sections
    
    def _extract_key_values(self, text_boxes: List[TextBox]) -> Dict[str, str]:
        """Extract key-value pairs from text boxes."""
        key_values = {}
        
        # Common patterns for key-value pairs
        patterns = [
            r'^([A-Z][A-Za-z\s]+)\s*:\s*(.+)$',  # "Key : Value"
            r'^([A-Z][A-Za-z\s]+)\s*-\s*(.+)$',  # "Key - Value"
        ]
        
        for box in text_boxes:
            text = box.text.strip()
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    if len(key) < 50 and len(value) < 200:  # Reasonable lengths
                        key_values[key] = value
                    break
        
        return key_values
    
    def _is_tabular(self, text_boxes: List[TextBox]) -> bool:
        """Determine if text boxes represent tabular data."""
        if len(text_boxes) < 6:  # Need at least 2 rows x 3 cols
            return False
        
        # Group by Y position (rows)
        lines = self._group_into_lines(text_boxes)
        
        if len(lines) < 2:
            return False
        
        # Check if multiple rows have similar X positions (columns)
        x_positions = defaultdict(int)
        for line_boxes in lines.values():
            for box in line_boxes:
                # Round X to nearest 50 pixels to group columns
                x_bucket = round(box.x / 50) * 50
                x_positions[x_bucket] += 1
        
        # If we have at least 3 X positions that appear multiple times, it's likely a table
        repeated_x = sum(1 for count in x_positions.values() if count >= 2)
        return repeated_x >= 3
    
    def _generate_table(self, text_boxes: List[TextBox]) -> str:
        """Generate a markdown table from text boxes."""
        lines = self._group_into_lines(text_boxes)
        
        if not lines:
            return ""
        
        # Build table rows
        rows = []
        for line_y in sorted(lines.keys()):
            line_boxes = sorted(lines[line_y], key=lambda b: b.x)
            row_cells = [box.text.strip() for box in line_boxes]
            rows.append(row_cells)
        
        if not rows:
            return ""
        
        # Determine number of columns
        max_cols = max(len(row) for row in rows)
        
        # Pad rows to have same number of columns
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        
        # Build markdown table
        result = []
        
        # Header row (first row)
        if rows:
            result.append("| " + " | ".join(rows[0]) + " |")
            result.append("|" + "|".join(["---"] * max_cols) + "|")
        
        # Data rows
        for row in rows[1:]:
            result.append("| " + " | ".join(row) + " |")
        
        return "\n".join(result)
    
    def _generate_structured_text(self, text_boxes: List[TextBox]) -> str:
        """Generate structured text with key-value pairs highlighted."""
        lines = self._group_into_lines(text_boxes)
        
        result = []
        for line_y in sorted(lines.keys()):
            line_boxes = sorted(lines[line_y], key=lambda b: b.x)
            
            # Build line text
            line_parts = []
            for box in line_boxes:
                text = box.text.strip()
                
                # Check if it looks like a key-value pair
                if ':' in text or '-' in text:
                    # Bold the key part
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            line_parts.append(f"**{parts[0].strip()}:** {parts[1].strip()}")
                        else:
                            line_parts.append(text)
                    else:
                        line_parts.append(text)
                else:
                    line_parts.append(text)
            
            if line_parts:
                result.append("  ".join(line_parts))
        
        return "\n".join(result)
    
    def _generate_spatial_layout(self, text_boxes: List[TextBox]) -> str:
        """Generate spatial layout (original format)."""
        lines = self._group_into_lines(text_boxes)
        
        result = []
        for line_y in sorted(lines.keys()):
            line_boxes = sorted(lines[line_y], key=lambda b: b.x)
            line_text = self._build_line(line_boxes, self.char_width)
            if line_text:
                result.append(line_text)
        
        return "\n".join(result)
    
    def _calculate_char_width(self, boxes: List[TextBox]) -> float:
        """Calculate average character width."""
        if not boxes:
            return self.dpi / 72 * 7.0
        
        total_width = sum(b.width for b in boxes[:100] if len(b.text) > 0)
        total_chars = sum(len(b.text) for b in boxes[:100] if len(b.text) > 0)
        
        return max(total_width / total_chars, 1.0) if total_chars else self.dpi / 72 * 7.0
    
    def _group_into_lines(self, boxes: List[TextBox]) -> Dict[int, List[TextBox]]:
        """Group text boxes into lines based on Y position."""
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
        """Build a line with spatial positioning."""
        chars = []
        current_x = 0.0
        
        for box in boxes:
            gap = box.x - current_x
            spaces = max(0, int(round(gap / char_width)))
            chars.append(" " * spaces)
            chars.append(box.text)
            current_x = box.x + len(box.text) * char_width
        
        return "".join(chars).rstrip()
