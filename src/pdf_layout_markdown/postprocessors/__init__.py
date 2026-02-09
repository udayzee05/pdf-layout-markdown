"""
Post-processing pipeline for layout analysis results.

This module provides extensible post-processors that can be chained
together to refine detection results. Common use cases:
- Non-Maximum Suppression (NMS) to remove overlapping detections
- Rectangle merging to combine related regions
- Noise filtering to remove small/invalid detections
"""

from .base import BasePostProcessor, PostProcessorPipeline
from .nms_processor import NMSProcessor
from .merge_processor import MergeProcessor
from .filter_processor import FilterProcessor

__all__ = [
    "BasePostProcessor",
    "PostProcessorPipeline",
    "NMSProcessor",
    "MergeProcessor",
    "FilterProcessor"
]
