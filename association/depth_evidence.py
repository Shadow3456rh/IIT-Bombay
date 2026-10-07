"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/depth_evidence.py

PURPOSE
-------
Interface for estimated image-space depth evidence.

This module provides a generic depth evidence representation so that we do not
tie the Layer 2 pipeline to a specific monocular depth model. In production, 
this could be populated by SAM3, a monocular depth model, or COLMAP dense stereo.

DEPTH CONVENTION
----------------
- Z > 0 is in front of the camera.
- depth_map values represent the camera-space Z distance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DepthEvidence:
    """
    Represents camera-space depth evidence for a 2D image.

    Attributes
    ----------
    depth_map : np.ndarray
        2D float array containing estimated Z depths.
    valid_mask : np.ndarray
        2D boolean array where True indicates the depth estimate is valid.
    confidence_map : np.ndarray, optional
        2D float array in [0, 1] containing confidence scores for the depths.
    """
    depth_map: np.ndarray
    valid_mask: np.ndarray
    confidence_map: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.depth_map, np.ndarray) or self.depth_map.ndim != 2:
            raise ValueError("depth_map must be a 2D NumPy array.")
        
        if not isinstance(self.valid_mask, np.ndarray) or self.valid_mask.ndim != 2:
            raise ValueError("valid_mask must be a 2D NumPy array.")
            
        if self.valid_mask.dtype != bool:
            raise TypeError("valid_mask must be boolean.")
            
        if self.depth_map.shape != self.valid_mask.shape:
            raise ValueError(f"depth_map shape {self.depth_map.shape} and valid_mask shape {self.valid_mask.shape} must match.")
        
        # Enforce all True valid_mask pixels are > 0 and finite.
        valid_depths = self.depth_map[self.valid_mask]
        if len(valid_depths) > 0:
            if not np.all(np.isfinite(valid_depths)):
                raise ValueError("Valid depth_map values must be finite (no NaN/inf).")
            if not np.all(valid_depths > 0):
                raise ValueError("Valid depth_map values must be > 0 (in front of camera).")
            
        if self.confidence_map is not None:
            if not isinstance(self.confidence_map, np.ndarray) or self.confidence_map.ndim != 2:
                raise ValueError("confidence_map must be a 2D NumPy array.")
            if self.confidence_map.shape != self.depth_map.shape:
                raise ValueError("confidence_map dimensions must match depth_map.")
                
            valid_confidences = self.confidence_map[self.valid_mask]
            if len(valid_confidences) > 0:
                if not np.all(np.isfinite(valid_confidences)):
                    raise ValueError("Valid confidences must be finite.")
                if not (np.all(valid_confidences >= 0.0) and np.all(valid_confidences <= 1.0)):
                    raise ValueError("Valid confidences must be within [0, 1].")
