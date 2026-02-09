"""
Table grid detection using OpenCV.
"""

from typing import List, Tuple
import cv2
import numpy as np

from .base import BaseDetector
from ..models import Rectangle


class TableDetector(BaseDetector):
    """
    Detects table grid structures (horizontal and vertical lines)
    and generates table cells from the grid intersections.
    """
    
    def __init__(
        self,
        min_line_length_ratio: float = 0.1,
        line_threshold: int = 80,
        line_gap: int = 10,
        cluster_threshold: int = 15
    ):
        """
        Initialize the table detector.
        
        Args:
            min_line_length_ratio: Minimum line length as ratio of dimension
            line_threshold: Hough transform threshold for line detection
            line_gap: Maximum gap between line segments
            cluster_threshold: Distance threshold for clustering nearby lines
        """
        self.min_line_length_ratio = min_line_length_ratio
        self.line_threshold = line_threshold
        self.line_gap = line_gap
        self.cluster_threshold = cluster_threshold
    
    def detect(self, image: np.ndarray) -> List[Rectangle]:
        """
        Detect table cells in an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detected table cell Rectangles
        """
        h_lines, v_lines = self.detect_grid_lines(image)
        cells = self.generate_cells(h_lines, v_lines, image.shape[:2])
        return cells
    
    def detect_grid_lines(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect horizontal and vertical grid lines.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines) as Y and X positions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using morphological operations
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 8, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 8))
        
        h_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        # Get line positions from projections
        h_proj = np.sum(h_lines_img, axis=1)
        v_proj = np.sum(v_lines_img, axis=0)
        
        h_threshold = width * 0.1 * 255
        v_threshold = height * 0.1 * 255
        
        h_lines = [i for i, val in enumerate(h_proj) if val > h_threshold]
        v_lines = [i for i, val in enumerate(v_proj) if val > v_threshold]
        
        # Cluster nearby lines
        h_lines = self.cluster_positions(h_lines, self.cluster_threshold)
        v_lines = self.cluster_positions(v_lines, self.cluster_threshold)
        
        return h_lines, v_lines
    
    def detect_grid_lines_hough(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect grid lines using Hough transform.
        Alternative method for more precise line detection.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines) as Y and X positions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect horizontal lines
        h_lines_detected = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.line_threshold,
            minLineLength=int(width * self.min_line_length_ratio),
            maxLineGap=self.line_gap
        )
        
        # Detect vertical lines
        v_lines_detected = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.line_threshold,
            minLineLength=int(height * self.min_line_length_ratio),
            maxLineGap=self.line_gap
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
        h_lines = self.cluster_positions(h_lines, self.cluster_threshold)
        v_lines = self.cluster_positions(v_lines, self.cluster_threshold * 2)
        
        return h_lines, v_lines
    
    def generate_cells(
        self,
        h_lines: List[int],
        v_lines: List[int],
        image_shape: Tuple[int, int]
    ) -> List[Rectangle]:
        """
        Generate table cells from grid lines.
        
        Args:
            h_lines: List of horizontal line Y positions
            v_lines: List of vertical line X positions
            image_shape: (height, width) of the image
            
        Returns:
            List of Rectangle objects representing cells
        """
        height, width = image_shape
        cells = []
        
        if not h_lines or not v_lines:
            return cells
        
        # Add boundaries if needed
        h_lines = self._add_boundaries(h_lines, 0, height, margin=20)
        v_lines = self._add_boundaries(v_lines, 0, width, margin=20)
        
        # Generate cells from grid intersections
        for row_idx in range(len(h_lines) - 1):
            for col_idx in range(len(v_lines) - 1):
                cell = Rectangle(
                    x=v_lines[col_idx],
                    y=h_lines[row_idx],
                    width=v_lines[col_idx + 1] - v_lines[col_idx],
                    height=h_lines[row_idx + 1] - h_lines[row_idx],
                    level=0,
                    rect_type="table_cell",
                    metadata={"row": row_idx, "col": col_idx}
                )
                cells.append(cell)
        
        return cells
    
    def _add_boundaries(
        self,
        lines: List[int],
        min_val: int,
        max_val: int,
        margin: int = 20
    ) -> List[int]:
        """Add boundary lines if they don't exist near the edges."""
        result = lines.copy()
        
        if result[0] > margin:
            result = [min_val] + result
        if result[-1] < max_val - margin:
            result = result + [max_val]
        
        return result
    
    def get_table_structure(
        self,
        image: np.ndarray
    ) -> Tuple[List[int], List[int], List[Rectangle]]:
        """
        Get complete table structure: lines and cells.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (h_lines, v_lines, cells)
        """
        h_lines, v_lines = self.detect_grid_lines(image)
        cells = self.generate_cells(h_lines, v_lines, image.shape[:2])
        return h_lines, v_lines, cells
