"""
Base class for text extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Any

from ..models import TextBox


class BaseExtractor(ABC):
    """
    Abstract base class for text extraction from documents.
    
    Subclasses should implement the extract method to handle
    specific document formats or extraction strategies.
    """
    
    @abstractmethod
    def extract(self, page: Any, scale: float = 1.0) -> List[TextBox]:
        """
        Extract text boxes from a document page.
        
        Args:
            page: The page object to extract text from
            scale: Scaling factor for coordinates
            
        Returns:
            List of TextBox objects with position and text information
        """
        pass
    
    def preprocess(self, page: Any) -> Any:
        """
        Optional preprocessing step before extraction.
        Override in subclasses if needed.
        
        Args:
            page: The page object to preprocess
            
        Returns:
            Preprocessed page object
        """
        return page
    
    def postprocess(self, boxes: List[TextBox]) -> List[TextBox]:
        """
        Optional postprocessing step after extraction.
        Override in subclasses if needed.
        
        Args:
            boxes: Extracted text boxes
            
        Returns:
            Postprocessed text boxes
        """
        return boxes
