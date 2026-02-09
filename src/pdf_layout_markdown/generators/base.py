"""
Base class for output generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models import TextBox


class BaseGenerator(ABC):
    """Abstract base class for output generation."""
    
    @abstractmethod
    def generate(self, text_boxes: List[TextBox], context: Dict[str, Any] = None) -> str:
        """Generate output from text boxes."""
        pass
