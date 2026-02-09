"""
PDF text extraction using PyMuPDF.
"""

from typing import List, Any

from .base import BaseExtractor
from ..models import TextBox


class TextExtractor(BaseExtractor):
    """
    Extracts text with bounding boxes from PDF pages using PyMuPDF.
    
    This extractor parses the text dictionary from PyMuPDF to get
    accurate bounding boxes for each text span.
    """
    
    def __init__(self, min_text_length: int = 0):
        """
        Initialize the text extractor.
        
        Args:
            min_text_length: Minimum text length to include (default: 0, includes all)
        """
        self.min_text_length = min_text_length
    
    def extract(self, page: Any, scale: float = 1.0) -> List[TextBox]:
        """
        Extract text boxes from a PDF page.
        
        Args:
            page: PyMuPDF page object
            scale: Scaling factor for coordinates (typically DPI/72)
            
        Returns:
            List of TextBox objects with scaled positions
        """
        boxes = []
        
        # Get text dictionary with detailed position info
        dict_data = page.get_text("dict")
        
        for block in dict_data.get("blocks", []):
            # Skip non-text blocks (type 0 is text)
            if block.get("type") != 0:
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    
                    # Skip empty or too short text
                    if not text or len(text) < self.min_text_length:
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    
                    box = TextBox(
                        x=int(bbox[0] * scale),
                        y=int(bbox[1] * scale),
                        width=int((bbox[2] - bbox[0]) * scale),
                        height=int((bbox[3] - bbox[1]) * scale),
                        text=text,
                        font_size=span.get("size", 12),
                        metadata={
                            "font": span.get("font", ""),
                            "color": span.get("color", 0),
                            "flags": span.get("flags", 0),
                            "origin": span.get("origin", []),
                        }
                    )
                    boxes.append(box)
        
        return self.postprocess(boxes)
    
    def postprocess(self, boxes: List[TextBox]) -> List[TextBox]:
        """
        Post-process extracted boxes: sort by position.
        
        Args:
            boxes: Extracted text boxes
            
        Returns:
            Sorted text boxes (top to bottom, left to right)
        """
        return sorted(boxes, key=lambda b: (b.y, b.x))
    
    def extract_raw_text(self, page: Any) -> str:
        """
        Extract plain text from a page (for simple use cases).
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Plain text content
        """
        return page.get_text("text")
    
    def extract_blocks(self, page: Any, scale: float = 1.0) -> List[List[TextBox]]:
        """
        Extract text organized by blocks.
        
        Args:
            page: PyMuPDF page object
            scale: Scaling factor for coordinates
            
        Returns:
            List of blocks, where each block is a list of TextBox objects
        """
        blocks = []
        dict_data = page.get_text("dict")
        
        for block in dict_data.get("blocks", []):
            if block.get("type") != 0:
                continue
            
            block_boxes = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    box = TextBox(
                        x=int(bbox[0] * scale),
                        y=int(bbox[1] * scale),
                        width=int((bbox[2] - bbox[0]) * scale),
                        height=int((bbox[3] - bbox[1]) * scale),
                        text=text,
                        font_size=span.get("size", 12)
                    )
                    block_boxes.append(box)
            
            if block_boxes:
                blocks.append(block_boxes)
        
        return blocks
