"""
Layout detection components for rectangles and tables.
"""

from .base import BaseDetector
from .rectangle_detector import RectangleDetector
from .table_detector import TableDetector

__all__ = ["BaseDetector", "RectangleDetector", "TableDetector"]
