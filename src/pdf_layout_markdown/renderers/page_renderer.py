"""
PDF page rendering to images.
"""

from typing import Tuple
import numpy as np
import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PageRenderer:
    """
    Renders PDF pages to images for visual analysis.
    
    Uses PyMuPDF to convert PDF pages to numpy arrays suitable
    for OpenCV processing.
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize the page renderer.
        
        Args:
            dpi: DPI resolution for rendering (default: 300)
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for page rendering")
        
        self.dpi = dpi
        self.zoom = dpi / 72.0  # PDF points to pixels
    
    @property
    def scale(self) -> float:
        """Get the scaling factor from PDF coordinates to pixels."""
        return self.zoom
    
    def render(self, page) -> np.ndarray:
        """
        Render a PDF page to a numpy array.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            OpenCV image (BGR format)
        """
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # Convert to BGR for OpenCV
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img
    
    def render_with_alpha(self, page) -> np.ndarray:
        """
        Render a PDF page with alpha channel.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            OpenCV image (BGRA format)
        """
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=True)
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
        return img
    
    def get_dimensions(self, page) -> Tuple[int, int]:
        """
        Get the dimensions of a rendered page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Tuple of (width, height) in pixels
        """
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.width, pix.height
    
    def render_region(
        self,
        page,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Render a specific region of a PDF page.
        
        Args:
            page: PyMuPDF page object
            x, y: Top-left corner in PDF coordinates
            width, height: Region size in PDF coordinates
            
        Returns:
            OpenCV image of the region (BGR format)
        """
        # Create clip rectangle in PDF coordinates
        clip = fitz.Rect(x, y, x + width, y + height)
        
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
