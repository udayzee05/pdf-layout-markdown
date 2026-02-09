"""
PDF to Markdown with OpenCV Rectangle Detection

Approach:
1. Detect all rectangles/table cells using contour detection
2. Create annotated debug image
3. Map text blocks to detected regions
4. Generate markdown preserving layout
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TextBox:
    """Text with position"""
    x: int
    y: int
    width: int
    height: int
    text: str
    font_size: float = 12.0
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


@dataclass
class Rectangle:
    """Detected rectangle region"""
    x: int
    y: int
    width: int
    height: int
    level: int = 0  # Nesting level
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def contains_point(self, px: int, py: int, margin: int = 5) -> bool:
        return (self.x - margin <= px <= self.x2 + margin and 
                self.y - margin <= py <= self.y2 + margin)
    
    def contains_box(self, box: TextBox) -> bool:
        return self.contains_point(box.center_x, box.center_y)


class PDFLayoutAnalyzer:
    """
    PDF to Markdown converter using OpenCV rectangle detection
    """
    
    def __init__(self, pdf_path: str, dpi: int = 300):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.doc = fitz.open(self.pdf_path)
        self.dpi = dpi
        self.zoom = dpi / 72
    
    def _render_page(self, page_num: int) -> np.ndarray:
        """Render PDF page as image"""
        page = self.doc[page_num]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def _extract_text_boxes(self, page, scale: float = 1.0) -> List[TextBox]:
        """Extract text with bounding boxes"""
        boxes = []
        
        dict_data = page.get_text("dict")
        
        for block in dict_data.get("blocks", []):
            if block.get("type") != 0:
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    
                    boxes.append(TextBox(
                        x=int(bbox[0] * scale),
                        y=int(bbox[1] * scale),
                        width=int((bbox[2] - bbox[0]) * scale),
                        height=int((bbox[3] - bbox[1]) * scale),
                        text=text,
                        font_size=span.get("size", 12)
                    ))
        
        return boxes
    
    def _non_max_suppression(self, rectangles: List[Rectangle], iou_threshold: float = 0.5) -> List[Rectangle]:
        """
        Apply Non-Maximum Suppression to remove overlapping rectangles.
        Keep larger, more significant rectangles and remove nested/overlapping ones.
        """
        if not rectangles:
            return []
        
        # Sort by area (larger first)
        sorted_rects = sorted(rectangles, key=lambda r: r.area, reverse=True)
        
        keep = []
        
        for rect in sorted_rects:
            # Check if this rectangle significantly overlaps with any kept rectangle
            should_keep = True
            
            for kept_rect in keep:
                # Calculate intersection over union (IoU)
                x_overlap = max(0, min(rect.x2, kept_rect.x2) - max(rect.x, kept_rect.x))
                y_overlap = max(0, min(rect.y2, kept_rect.y2) - max(rect.y, kept_rect.y))
                intersection = x_overlap * y_overlap
                
                union = rect.area + kept_rect.area - intersection
                iou = intersection / union if union > 0 else 0
                
                # If significant overlap, check containment
                if iou > iou_threshold:
                    # Keep the larger one
                    should_keep = False
                    break
                
                # Also remove if this rect is fully contained in a kept rect
                if (rect.x >= kept_rect.x and rect.y >= kept_rect.y and
                    rect.x2 <= kept_rect.x2 and rect.y2 <= kept_rect.y2):
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(rect)
        
        return keep
    
    def _merge_nearby_rectangles(self, rectangles: List[Rectangle], 
                                  text_boxes: List[TextBox],
                                  max_gap: int = 20) -> List[Rectangle]:
        """
        Merge rectangles that are close together and likely part of the same semantic block.
        Also expand rectangles to include nearby text with padding.
        """
        if not rectangles:
            return []
        
        # First, expand each rectangle to include text with padding
        expanded = []
        padding = 10  # pixels of padding around text
        
        for rect in rectangles:
            # Find all text boxes within or near this rectangle
            related_text = []
            for box in text_boxes:
                # Check if text is inside or very close to rectangle
                if (box.x >= rect.x - padding and box.x2 <= rect.x2 + padding and
                    box.y >= rect.y - padding and box.y2 <= rect.y2 + padding):
                    related_text.append(box)
            
            if related_text:
                # Expand rectangle to encompass all related text with padding
                min_x = min(min(b.x for b in related_text), rect.x) - padding
                min_y = min(min(b.y for b in related_text), rect.y) - padding
                max_x = max(max(b.x2 for b in related_text), rect.x2) + padding
                max_y = max(max(b.y2 for b in related_text), rect.y2) + padding
                
                expanded.append(Rectangle(
                    int(min_x), int(min_y), 
                    int(max_x - min_x), int(max_y - min_y),
                    rect.level
                ))
            else:
                expanded.append(rect)
        
        # Now merge rectangles that are very close
        merged = []
        used = set()
        
        for i, rect1 in enumerate(expanded):
            if i in used:
                continue
            
            # Start a merge group with this rectangle
            merge_group = [rect1]
            
            for j, rect2 in enumerate(expanded):
                if i == j or j in used:
                    continue
                
                # Check if rectangles are close enough to merge
                # (horizontally or vertically aligned with small gap)
                h_gap = max(rect1.x, rect2.x) - min(rect1.x2, rect2.x2)
                v_gap = max(rect1.y, rect2.y) - min(rect1.y2, rect2.y2)
                
                # Check for vertical alignment (same column)
                x_overlap = min(rect1.x2, rect2.x2) - max(rect1.x, rect2.x)
                y_overlap = min(rect1.y2, rect2.y2) - max(rect1.y, rect2.y)
                
                should_merge = False
                
                # Merge if vertically stacked with horizontal overlap
                if x_overlap > min(rect1.width, rect2.width) * 0.5 and v_gap < max_gap:
                    should_merge = True
                
                # Merge if horizontally adjacent with vertical overlap
                if y_overlap > min(rect1.height, rect2.height) * 0.5 and h_gap < max_gap:
                    should_merge = True
                
                if should_merge:
                    merge_group.append(rect2)
                    used.add(j)
            
            # Create merged rectangle from group
            if merge_group:
                min_x = min(r.x for r in merge_group)
                min_y = min(r.y for r in merge_group)
                max_x = max(r.x2 for r in merge_group)
                max_y = max(r.y2 for r in merge_group)
                
                merged.append(Rectangle(
                    min_x, min_y,
                    max_x - min_x, max_y - min_y,
                    rect1.level
                ))
        
        return merged
    
    def _detect_rectangles(self, image: np.ndarray) -> List[Rectangle]:
        """
        Detect ONLY structural layout rectangles (table cells, form sections).
        Filters out small rectangles around individual words.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # IMPORTANT: Size thresholds for structural regions
        # Minimum dimension: at least 5% of page dimension
        min_width = width // 20   # 5% of page width
        min_height = height // 40  # 2.5% of page height
        min_area = (width * height) // 50  # At least 2% of page area
        max_area = (width * height) * 0.95  # Max 95% of page
        
        rectangles = []
        
        # Method 1: Detect rectangles formed by lines (table structure)
        # Threshold to binary (invert: lines become white)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//8, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//8))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        # Combine lines to find table structure
        table_structure = cv2.add(h_lines, v_lines)
        
        # Dilate to connect nearby lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_structure = cv2.dilate(table_structure, kernel, iterations=2)
        
        # Find contours in the combined structure
        contours, hierarchy = cv2.findContours(
            table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size - ONLY keep large structural regions
            if (w >= min_width and h >= min_height and 
                min_area < area < max_area):
                
                # Additional check: reject very elongated horizontal shapes (likely text lines)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 15:  # Too wide for height - likely a text line
                    continue
                
                # Get hierarchy level
                level = 0
                if hierarchy is not None:
                    parent = hierarchy[0][i][3]
                    while parent != -1:
                        level += 1
                        parent = hierarchy[0][parent][3]
                
                rectangles.append(Rectangle(x, y, w, h, level))
        
        # Method 2: Edge-based detection for cleaner structures
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use larger kernels for structural lines only
        h_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (width//5, 1))
        v_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//5))
        
        h_lines2 = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel2)
        v_lines2 = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel2)
        
        table_mask = cv2.add(h_lines2, v_lines2)
        table_mask = cv2.dilate(table_mask, kernel, iterations=3)
        
        contours2, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours2:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size - ONLY keep large structural regions
            if (w >= min_width and h >= min_height and 
                min_area < area < max_area):
                
                # Check if not a duplicate
                is_duplicate = False
                for rect in rectangles:
                    # Consider duplicate if significantly overlapping
                    overlap_x = max(0, min(rect.x2, x + w) - max(rect.x, x))
                    overlap_y = max(0, min(rect.y2, y + h) - max(rect.y, y))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.5 * min(area, rect.area):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    rectangles.append(Rectangle(x, y, w, h, 0))
        
        # Sort by position (top to bottom, left to right)
        rectangles.sort(key=lambda r: (r.y, r.x))
        
        return rectangles
    
    def _detect_table_grid(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect horizontal and vertical lines forming tables
        Returns (h_lines, v_lines) as Y and X positions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # For horizontal lines: require significant width
        h_lines_detected = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=80,
            minLineLength=width//4,  # At least 25% of page width
            maxLineGap=10
        )
        
        # For vertical lines: require significant height
        v_lines_detected = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=80,
            minLineLength=height//3,  # At least 33% of page height
            maxLineGap=10
        )
        
        h_lines = []
        v_lines = []
        
        # Process horizontal lines
        if h_lines_detected is not None:
            for line in h_lines_detected:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # Nearly horizontal
                    h_lines.append((y1 + y2) // 2)
        
        # Process vertical lines
        if v_lines_detected is not None:
            for line in v_lines_detected:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 5:  # Nearly vertical
                    v_lines.append((x1 + x2) // 2)
        
        # Cluster nearby lines
        h_lines = self._cluster_positions(h_lines, threshold=15)
        v_lines = self._cluster_positions(v_lines, threshold=30)
        
        return h_lines, v_lines
    
    def _detect_table_cells(self, image: np.ndarray) -> Tuple[List[int], List[int], List[Rectangle]]:
        """
        Detect table grid lines and cells
        Returns: (h_lines, v_lines, cells)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//8, 1))
        h_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//8))
        v_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        # Find line positions
        h_proj = np.sum(h_lines_img, axis=1)
        v_proj = np.sum(v_lines_img, axis=0)
        
        # Get positions where lines are strong
        h_threshold = width * 0.1 * 255
        v_threshold = height * 0.1 * 255
        
        h_lines = [i for i, val in enumerate(h_proj) if val > h_threshold]
        v_lines = [i for i, val in enumerate(v_proj) if val > v_threshold]
        
        # Cluster nearby lines
        h_lines = self._cluster_positions(h_lines, threshold=15)
        v_lines = self._cluster_positions(v_lines, threshold=15)
        
        # Generate cells from grid
        cells = []
        
        if h_lines and v_lines:
            # Add boundaries
            if h_lines[0] > 20:
                h_lines = [0] + h_lines
            if h_lines[-1] < height - 20:
                h_lines = h_lines + [height]
            if v_lines[0] > 20:
                v_lines = [0] + v_lines
            if v_lines[-1] < width - 20:
                v_lines = v_lines + [width]
            
            for row_idx in range(len(h_lines) - 1):
                for col_idx in range(len(v_lines) - 1):
                    cell = Rectangle(
                        x=v_lines[col_idx],
                        y=h_lines[row_idx],
                        width=v_lines[col_idx + 1] - v_lines[col_idx],
                        height=h_lines[row_idx + 1] - h_lines[row_idx],
                        level=0
                    )
                    cells.append(cell)
        
        return h_lines, v_lines, cells
    
    def _cluster_positions(self, positions: List[int], threshold: int = 15) -> List[int]:
        """Cluster nearby positions"""
        if not positions:
            return []
        
        positions = sorted(positions)
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            if pos - clusters[-1][-1] < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        return [int(np.mean(c)) for c in clusters]
    
    def _detect_cells(self, h_lines: List[int], v_lines: List[int]) -> List[Rectangle]:
        """Detect table cells from grid lines"""
        cells = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x = v_lines[j]
                y = h_lines[i]
                w = v_lines[j + 1] - x
                h = h_lines[i + 1] - y
                cells.append(Rectangle(x, y, w, h, 0))
        return cells
    
    def _box_in_cell(self, box: TextBox, cell: Rectangle) -> bool:
        """Check if text box is inside cell"""
        return (box.x >= cell.x and box.y >= cell.y and
                box.x2 <= cell.x2 and box.y2 <= cell.y2)
    
    def create_annotated_image(self, page_num: int = 0, output_path: str = None) -> np.ndarray:
        """
        Create annotated image showing detected rectangles and text boxes
        """
        image = self._render_page(page_num)
        annotated = image.copy()
        
        page = self.doc[page_num]
        scale = self.dpi / 72
        text_boxes = self._extract_text_boxes(page, scale)
        rectangles = self._detect_rectangles(image)
        h_lines, v_lines, cells = self._detect_table_cells(image)
        
        # Draw detected rectangles (blue)
        for rect in rectangles:
            cv2.rectangle(
                annotated,
                (rect.x, rect.y),
                (rect.x2, rect.y2),
                (255, 0, 0),  # Blue
                2
            )
        
        # Draw table cells (green)
        for cell in cells:
            cv2.rectangle(
                annotated,
                (cell.x, cell.y),
                (cell.x2, cell.y2),
                (0, 255, 0),  # Green
                1
            )
        
        # Draw horizontal lines (cyan)
        for y in h_lines:
            cv2.line(annotated, (0, y), (annotated.shape[1], y), (255, 255, 0), 1)
        
        # Draw vertical lines (magenta)
        for x in v_lines:
            cv2.line(annotated, (x, 0), (x, annotated.shape[0]), (255, 0, 255), 1)
        
        # Draw text boxes (red)
        for box in text_boxes:
            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x2, box.y2),
                (0, 0, 255),  # Red
                1
            )
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"✅ Annotated image saved: {output_path}")
        
        return annotated
    
    def analyze_page(self, page_num: int = 0) -> Dict:
        """
        Analyze page layout and return structured information
        """
        page = self.doc[page_num]
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to OpenCV image
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA
            image = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Extract text boxes first
        text_boxes = self._extract_text_boxes(page, self.dpi / 72)
        
        # Detect rectangles
        rectangles = self._detect_rectangles(image)
        
        # Apply improvements:
        # 1. Merge nearby rectangles and expand to include text with padding
        rectangles = self._merge_nearby_rectangles(rectangles, text_boxes, max_gap=30)
        
        # 2. Apply Non-Maximum Suppression to remove overlaps
        rectangles = self._non_max_suppression(rectangles, iou_threshold=0.6)
        
        # 3. Filter out very small rectangles (likely noise)
        min_area = (pix.width * pix.height) // 100  # At least 1% of page
        rectangles = [r for r in rectangles if r.area >= min_area]
        
        # Detect table grid
        h_lines, v_lines = self._detect_table_grid(image)
        
        # Detect cells from grid
        cells = self._detect_cells(h_lines, v_lines)
        
        # Map text to cells
        cell_text = {}
        for i, cell in enumerate(cells):
            cell_text[i] = []
            for box in text_boxes:
                if self._box_in_cell(box, cell):
                    cell_text[i].append(box)
        
        # Find uncategorized text (not in any cell)
        categorized_boxes = []
        for boxes in cell_text.values():
            categorized_boxes.extend(boxes)
        
        uncategorized = [box for box in text_boxes if box not in categorized_boxes]
        
        return {
            "rectangles": rectangles,
            "cells": cells,
            "cell_text": cell_text,
            "h_lines": h_lines,
            "v_lines": v_lines,
            "text_boxes": text_boxes,
            "uncategorized": uncategorized,
            "dimensions": (pix.width, pix.height)
        }
    
    def calculate_char_width(self, boxes: List[TextBox]) -> float:
        """Calculate average character width with higher precision"""
        if not boxes:
            return self.dpi / 72 * 7.0
        
        total_width = 0
        total_chars = 0
        
        # Use more boxes for better average
        for box in boxes[:100]:
            if len(box.text) > 0:
                total_width += box.width
                total_chars += len(box.text)
        
        if total_chars == 0:
            return self.dpi / 72 * 7.0
        
        return max(total_width / total_chars, 1.0)
    
    def _group_into_lines(self, boxes: List[TextBox]) -> Dict[int, List[TextBox]]:
        """Group text boxes into horizontal lines"""
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

    def generate_markdown(self, page_num: int = 0) -> str:
        """
        Generate markdown preserving EXACT spatial layout from PDF
        """
        analysis = self.analyze_page(page_num)
        
        text_boxes = analysis["text_boxes"]
        
        if not text_boxes:
            return ""
        
        # Calculate character width for spacing (float precision)
        char_width = self.calculate_char_width(text_boxes)
        
        # Group text boxes into lines by Y position
        lines = self._group_into_lines(text_boxes)
        
        # Build output with exact positioning
        result = []
        result.append("```")  # Use code block for monospace font
        
        for line_y in sorted(lines.keys()):
            line_boxes = sorted(lines[line_y], key=lambda b: b.x)
            
            # Build line with exact spacing
            line_chars = []
            current_x = 0.0
            
            for box in line_boxes:
                # Calculate spaces needed to reach this box's position
                gap = box.x - current_x
                # Use round() for better precision with float char_width
                spaces = max(0, int(round(gap / char_width)))
                
                line_chars.append(" " * spaces)
                line_chars.append(box.text)
                current_x = box.x + len(box.text) * char_width
                
                # Adjust current_x to match actual end of box if needed
                # Ideally, current_x should be aligned to grid, but here we track actual consumption
                # Alternatively, we could reset current_x to box.x2 but that might double count spaces if we are inconsistent
                # Let's trust the calculation: gap is distance from end of last text to start of next text
            
            # Add the line
            line_text = "".join(line_chars).rstrip()
            if line_text:  # Only add non-empty lines
                result.append(line_text)
        
        result.append("```")
        
        return "\n".join(result)
    
    def convert(self, create_debug_image: bool = True) -> str:
        """
        Convert entire PDF to markdown
        """
        md_parts = []
        
        for i in range(len(self.doc)):
            if create_debug_image:
                base_name = self.pdf_path.stem
                self.create_annotated_image(i, f"{base_name}_page{i+1}_debug.png")
            
            md_parts.append(self.generate_markdown(i))
            md_parts.append("\n\n---\n\n")  # Page separator
        
        return "".join(md_parts)
    
    def save(self, output_path: str = None, create_debug_image: bool = True):
        """Save markdown to file"""
        if output_path is None:
            output_path = self.pdf_path.with_suffix(".md")
            
        markdown = self.convert(create_debug_image)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        return markdown

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python layout.py <pdf_file> [output_file]")
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = PDFLayoutAnalyzer(pdf_file)
    analyzer.save(output_file)
