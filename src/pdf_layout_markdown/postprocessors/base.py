"""
Base classes for post-processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional


class BasePostProcessor(ABC):
    """
    Abstract base class for post-processors.
    
    Post-processors refine detection results by filtering, merging,
    or transforming detected elements. They can be chained together
    in a pipeline for complex processing workflows.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the post-processor.
        
        Args:
            name: Optional name for identification in pipelines
        """
        self.name = name or self.__class__.__name__
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        """Whether this processor is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable this processor."""
        self._enabled = value
    
    @abstractmethod
    def process(self, items: List[Any], context: Dict[str, Any] = None) -> List[Any]:
        """
        Process a list of items.
        
        Args:
            items: Items to process (typically Rectangle or TextBox objects)
            context: Optional context dictionary with additional information
                     (e.g., page dimensions, text boxes for reference)
            
        Returns:
            Processed list of items
        """
        pass
    
    def __call__(self, items: List[Any], context: Dict[str, Any] = None) -> List[Any]:
        """Allow processor to be called directly."""
        if not self.enabled:
            return items
        return self.process(items, context)
    
    def __repr__(self) -> str:
        return f"{self.name}(enabled={self.enabled})"


class PostProcessorPipeline:
    """
    A pipeline of post-processors that execute in sequence.
    
    Allows building complex processing workflows by chaining
    multiple processors together.
    
    Example:
        pipeline = PostProcessorPipeline([
            NMSProcessor(iou_threshold=0.5),
            MergeProcessor(max_gap=20),
            FilterProcessor(min_area=1000)
        ])
        processed = pipeline.process(rectangles, context)
    """
    
    def __init__(self, processors: List[BasePostProcessor] = None):
        """
        Initialize the pipeline.
        
        Args:
            processors: List of processors to include
        """
        self.processors: List[BasePostProcessor] = processors or []
    
    def add(self, processor: BasePostProcessor) -> "PostProcessorPipeline":
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor to add
            
        Returns:
            Self for method chaining
        """
        self.processors.append(processor)
        return self
    
    def remove(self, name: str) -> bool:
        """
        Remove a processor by name.
        
        Args:
            name: Name of processor to remove
            
        Returns:
            True if processor was found and removed
        """
        for i, proc in enumerate(self.processors):
            if proc.name == name:
                del self.processors[i]
                return True
        return False
    
    def get(self, name: str) -> Optional[BasePostProcessor]:
        """
        Get a processor by name.
        
        Args:
            name: Name of processor to find
            
        Returns:
            Processor if found, None otherwise
        """
        for proc in self.processors:
            if proc.name == name:
                return proc
        return None
    
    def process(self, items: List[Any], context: Dict[str, Any] = None) -> List[Any]:
        """
        Process items through all processors in sequence.
        
        Args:
            items: Items to process
            context: Optional context dictionary
            
        Returns:
            Processed items
        """
        result = items
        for processor in self.processors:
            if processor.enabled:
                result = processor.process(result, context)
        return result
    
    def __call__(self, items: List[Any], context: Dict[str, Any] = None) -> List[Any]:
        """Allow pipeline to be called directly."""
        return self.process(items, context)
    
    def __len__(self) -> int:
        return len(self.processors)
    
    def __repr__(self) -> str:
        proc_names = [p.name for p in self.processors]
        return f"PostProcessorPipeline({proc_names})"
