"""
Rectangle detection using OpenCV contour analysis.
"""

from typing import List, Tuple
import cv2
import numpy as np

from .base import BaseDetector
from ..models import Rectangle


class RectangleDetector(BaseDetector):
    """
    Detects structural layout rectangles (table cells, form sections)
    using OpenCV contour detection.
    
    This detector focuses on finding large structural regions,
    filtering out small rectangles around individual words.
    """
    
    def __init__(
        self,
        min_width_ratio: float = 0.05,
        min_height_ratio: float = 0.025,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.95,
        max_aspect_ratio: float = 15.0
    ):
        """
        Initialize the rectangle detector.
        
        Args:
            min_width_ratio: Minimum width as ratio of page width
            min_height_ratio: Minimum height as ratio of page height
            min_area_ratio: Minimum area as ratio of page area
            max_area_ratio: Maximum area as ratio of page area
            max_aspect_ratio: Maximum width/height ratio (filters text lines)
        """
        self.min_width_ratio = min_width_ratio
        self.min_height_ratio = min_height_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def detect(self, image: np.ndarray) -> List[Rectangle]:
        """
        Detect structural rectangles in an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detected Rectangle objects
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Calculate size thresholds
        min_width = int(width * self.min_width_ratio)
        min_height = int(height * self.min_height_ratio)
        min_area = int(width * height * self.min_area_ratio)
        max_area = int(width * height * self.max_area_ratio)
        
        rectangles = []
        
        # Method 1: Line-based detection (table structures)
        rects_from_lines = self._detect_from_lines(gray, min_width, min_height, min_area, max_area)
        rectangles.extend(rects_from_lines)
        
        # Method 2: Edge-based detection (cleaner structures)
        rects_from_edges = self._detect_from_edges(gray, min_width, min_height, min_area, max_area)
        
        # Add non-duplicate edge-based rectangles
        for rect in rects_from_edges:
            if not self._is_duplicate(rect, rectangles):
                rectangles.append(rect)
        
        # Sort by position (top to bottom, left to right)
        rectangles.sort(key=lambda r: (r.y, r.x))
        
        return self.filter_results(rectangles)
    
    def _detect_from_lines(
        self,
        gray: np.ndarray,
        min_width: int,
        min_height: int,
        min_area: int,
        max_area: int
    ) -> List[Rectangle]:
        """
        Detect rectangles formed by horizontal and vertical lines.
        """
        height, width = gray.shape
        rectangles = []
        
        # Threshold to binary (invert: lines become white)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 8, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 8))
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
            
            # Filter by size
            if not self._is_valid_size(w, h, area, min_width, min_height, min_area, max_area):
                continue
            
            # Get hierarchy level
            level = 0
            if hierarchy is not None:
                parent = hierarchy[0][i][3]
                while parent != -1:
                    level += 1
                    parent = hierarchy[0][parent][3]
            
            rectangles.append(Rectangle(x, y, w, h, level, rect_type="line_based"))
        
        return rectangles
    
    def _detect_from_edges(
        self,
        gray: np.ndarray,
        min_width: int,
        min_height: int,
        min_area: int,
        max_area: int
    ) -> List[Rectangle]:
        """
        Detect rectangles using edge detection.
        """
        height, width = gray.shape
        rectangles = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use larger kernels for structural lines only
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 5, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 5))
        
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        table_mask = cv2.add(h_lines, v_lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_mask = cv2.dilate(table_mask, kernel, iterations=3)
        
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if self._is_valid_size(w, h, area, min_width, min_height, min_area, max_area):
                rectangles.append(Rectangle(x, y, w, h, 0, rect_type="edge_based"))
        
        return rectangles
    
    def _is_valid_size(
        self,
        w: int,
        h: int,
        area: int,
        min_width: int,
        min_height: int,
        min_area: int,
        max_area: int
    ) -> bool:
        """Check if dimensions meet size requirements."""
        if w < min_width or h < min_height:
            return False
        if area < min_area or area > max_area:
            return False
        
        # Check aspect ratio (filter out text lines)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > self.max_aspect_ratio:
            return False
        
        return True
    
    def _is_duplicate(self, rect: Rectangle, existing: List[Rectangle], threshold: float = 0.5) -> bool:
        """Check if rectangle significantly overlaps with existing ones."""
        for existing_rect in existing:
            overlap_x = max(0, min(rect.x2, existing_rect.x2) - max(rect.x, existing_rect.x))
            overlap_y = max(0, min(rect.y2, existing_rect.y2) - max(rect.y, existing_rect.y))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > threshold * min(rect.area, existing_rect.area):
                return True
        
        return False
