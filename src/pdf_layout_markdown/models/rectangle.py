"""
Rectangle model for detected layout regions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .text_box import TextBox


@dataclass
class Rectangle:
    """
    Represents a detected rectangular region in the document.
    
    Attributes:
        x: Left coordinate of the rectangle
        y: Top coordinate of the rectangle
        width: Width of the rectangle
        height: Height of the rectangle
        level: Nesting level (for hierarchical structures)
        rect_type: Type of rectangle (table_cell, section, etc.)
        metadata: Additional metadata for extensibility
    """
    x: int
    y: int
    width: int
    height: int
    level: int = 0
    rect_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def x2(self) -> int:
        """Right coordinate of the rectangle."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom coordinate of the rectangle."""
        return self.y + self.height
    
    @property
    def area(self) -> int:
        """Area of the rectangle."""
        return self.width * self.height
    
    @property
    def center_x(self) -> int:
        """X coordinate of the center point."""
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        """Y coordinate of the center point."""
        return self.y + self.height // 2
    
    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio."""
        return self.width / self.height if self.height > 0 else 0
    
    def contains_point(self, px: int, py: int, margin: int = 5) -> bool:
        """Check if a point is within this rectangle."""
        return (self.x - margin <= px <= self.x2 + margin and 
                self.y - margin <= py <= self.y2 + margin)
    
    def contains_box(self, box: "TextBox") -> bool:
        """Check if a TextBox's center is within this rectangle."""
        return self.contains_point(box.center_x, box.center_y)
    
    def contains_rectangle(self, other: "Rectangle", margin: int = 0) -> bool:
        """Check if another rectangle is fully contained within this one."""
        return (other.x >= self.x - margin and 
                other.y >= self.y - margin and
                other.x2 <= self.x2 + margin and 
                other.y2 <= self.y2 + margin)
    
    def compute_iou(self, other: "Rectangle") -> float:
        """
        Compute Intersection over Union with another rectangle.
        
        Args:
            other: Another Rectangle to compare with
            
        Returns:
            IoU value between 0 and 1
        """
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        intersection = x_overlap * y_overlap
        
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0
    
    def merge_with(self, other: "Rectangle") -> "Rectangle":
        """
        Create a new rectangle that encompasses both rectangles.
        
        Args:
            other: Another Rectangle to merge with
            
        Returns:
            New Rectangle covering both
        """
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        max_x = max(self.x2, other.x2)
        max_y = max(self.y2, other.y2)
        
        return Rectangle(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            level=min(self.level, other.level),
            rect_type=self.rect_type,
            metadata={**self.metadata, **other.metadata}
        )
    
    def expand(self, padding: int) -> "Rectangle":
        """
        Create a new rectangle expanded by padding on all sides.
        
        Args:
            padding: Pixels to add on each side
            
        Returns:
            New expanded Rectangle
        """
        return Rectangle(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
            level=self.level,
            rect_type=self.rect_type,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "rect_type": self.rect_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rectangle":
        """Create Rectangle from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            level=data.get("level", 0),
            rect_type=data.get("rect_type", "generic"),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"Rectangle(x={self.x}, y={self.y}, w={self.width}, h={self.height}, type={self.rect_type})"
