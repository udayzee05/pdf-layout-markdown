"""
Base class for layout detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Tuple
import numpy as np

from ..models import Rectangle


class BaseDetector(ABC):
    """
    Abstract base class for layout element detection.
    
    Subclasses should implement the detect method to identify
    specific structural elements in document images.
    """
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Rectangle]:
        """
        Detect layout elements in an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detected Rectangle objects
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection.
        Override in subclasses if needed.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return image
    
    def filter_results(self, rectangles: List[Rectangle]) -> List[Rectangle]:
        """
        Filter detection results.
        Override in subclasses if needed.
        
        Args:
            rectangles: Detected rectangles
            
        Returns:
            Filtered rectangles
        """
        return rectangles
    
    @staticmethod
    def cluster_positions(positions: List[int], threshold: int = 15) -> List[int]:
        """
        Cluster nearby positions into representative values.
        
        Args:
            positions: List of position values
            threshold: Maximum distance to cluster together
            
        Returns:
            List of clustered representative positions
        """
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
