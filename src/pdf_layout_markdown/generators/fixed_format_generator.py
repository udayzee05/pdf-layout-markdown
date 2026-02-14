"""
Fixed format generator producing output matching the user's specific requirements.
- HTML Tables
- Explicit Page Separators (handled by converter ideally, but we can ensure structure here)
- Semantic Headers
"""

from typing import List, Dict, Any
from .structured_markdown_generator import StructuredMarkdownGenerator
from ..models import TextBox

class FixedFormatGenerator(StructuredMarkdownGenerator):
    """
    Generates markdown with HTML tables and specific formatting.
    """
    
    def generate(self, text_boxes: List[TextBox], context: Dict[str, Any] = None) -> str:
        """
        Generate structured markdown without explicit section headers or metadata,
        matching the requested cleaner format.
        """
        if not text_boxes:
            return ""
        
        context = context or {}
        rectangles = context.get('rectangles', [])
        
        self.char_width = self._calculate_char_width(text_boxes)
        
        # Organize content into sections (Header, Body, Footer)
        sections = self._organize_sections(text_boxes, rectangles)
        
        result = []
        
        # We skip frontmatter and explicit section titles (e.g., ## Header)
        # to match the user's provided example which flows naturally.
        
        for section_name, section_boxes in sections.items():
            if not section_boxes:
                continue
            
            # Use table layout for Body if it looks tabular
            # For Header/Footer, we prefer text unless it's strongly tabular
            # However, _is_tabular is a heuristic.
            
            if self._is_tabular(section_boxes):
                result.append(self._generate_table(section_boxes))
            else:
                result.append(self._generate_structured_text(section_boxes))
            
            result.append("")
            
        return "\n".join(result)

    def _generate_table(self, text_boxes: List[TextBox]) -> str:
        """Generate an HTML table from text boxes."""
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
        
        # Build HTML table
        html = ["<table>"]
        
        # Header row (first row) if we have data
        if rows:
            html.append("  <thead>")
            html.append("    <tr>")
            for cell in rows[0]:
                html.append(f"      <th>{cell}</th>")
            html.append("    </tr>")
            html.append("  </thead>")
        
        # Data rows
        html.append("  <tbody>")
        for row in rows[1:]:
            html.append("    <tr>")
            for cell in row:
                html.append(f"      <td>{cell}</td>")
            html.append("    </tr>")
        html.append("  </tbody>")
        html.append("</table>")
        
        return "\n".join(html)

    def _generate_structured_text(self, text_boxes: List[TextBox]) -> str:
        # Use semantic bolding like the base class, maybe enhance if needed
        return super()._generate_structured_text(text_boxes)
