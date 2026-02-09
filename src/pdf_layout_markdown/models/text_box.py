"""
TextBox model for text elements with position information.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TextBox:
    """
    Represents a text element with its bounding box position.
    
    Attributes:
        x: Left coordinate of the bounding box
        y: Top coordinate of the bounding box
        width: Width of the bounding box
        height: Height of the bounding box
        text: The actual text content
        font_size: Font size of the text (default: 12.0)
        metadata: Additional metadata for extensibility
    """
    x: int
    y: int
    width: int
    height: int
    text: str
    font_size: float = 12.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def x2(self) -> int:
        """Right coordinate of the bounding box."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom coordinate of the bounding box."""
        return self.y + self.height
    
    @property
    def center_x(self) -> int:
        """X coordinate of the center point."""
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        """Y coordinate of the center point."""
        return self.y + self.height // 2
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    def contains_point(self, px: int, py: int, margin: int = 0) -> bool:
        """Check if a point is within this text box."""
        return (self.x - margin <= px <= self.x2 + margin and 
                self.y - margin <= py <= self.y2 + margin)
    
    def overlaps_with(self, other: "TextBox", threshold: float = 0.5) -> bool:
        """
        Check if this text box overlaps significantly with another.
        
        Args:
            other: Another TextBox to compare with
            threshold: Minimum overlap ratio to consider as overlapping
            
        Returns:
            True if overlap exceeds threshold
        """
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        intersection = x_overlap * y_overlap
        
        min_area = min(self.area, other.area)
        if min_area == 0:
            return False
            
        return (intersection / min_area) > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "font_size": self.font_size,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBox":
        """Create TextBox from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            text=data["text"],
            font_size=data.get("font_size", 12.0),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"TextBox(x={self.x}, y={self.y}, text='{self.text[:20]}...')" if len(self.text) > 20 else f"TextBox(x={self.x}, y={self.y}, text='{self.text}')"
