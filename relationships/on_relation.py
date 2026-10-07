"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/on_relation.py

PURPOSE
-------
Implements the 3D 'ON' relationship extraction prototype.
Determines if Object A is geometrically 'ON' Support Object B based on 
vertical proximity and horizontal footprint overlap.

LIMITATIONS
-----------
- Relies purely on AABB bounding box geometry.
- Does not utilize exact surface normals or meshes.
- Operates under an assumed `up_axis` configuration.
- The 'on_score' is a prototype geometric heuristic, not a calibrated probability.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import numpy as np

from objects.object_3d import Object3D


@dataclass(frozen=True)
class ONRelationResult:
    """
    Structured output for the ON relationship evaluation.
    """
    subject_object_id: int
    subject_class_name: str
    support_object_id: int
    support_class_name: str
    relation: str = "ON"
    vertical_gap: float = 0.0
    overlap_ratio: float = 0.0
    on_score: float = 0.0
    is_on: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serialisable dictionary representation."""
        return {
            "subject_object_id": self.subject_object_id,
            "subject_class_name": self.subject_class_name,
            "support_object_id": self.support_object_id,
            "support_class_name": self.support_class_name,
            "relation": self.relation,
            "vertical_gap": round(self.vertical_gap, 6),
            "overlap_ratio": round(self.overlap_ratio, 6),
            "on_score": round(self.on_score, 6),
            "is_on": self.is_on,
        }


class ONRelationDetector:
    """
    Evaluates geometric 'ON' support constraints between Object3D AABBs.
    """
    def __init__(
        self,
        up_axis: Literal["x", "y", "z"] = "z",
        max_vertical_gap: float = 0.2,
        min_footprint_overlap: float = 0.3,
        on_score_threshold: float = 0.5
    ):
        """
        Initialize the prototype ON detector.

        Parameters
        ----------
        up_axis : str
            The assumed gravity/up direction in world coordinates ('x', 'y', or 'z').
        max_vertical_gap : float
            Maximum absolute vertical distance allowed between subject bottom and support top.
        min_footprint_overlap : float
            Minimum required intersection_area / subject_footprint_area in [0, 1].
        on_score_threshold : float
            Final combined score threshold required to flag `is_on = True`.
        """
        if up_axis not in ("x", "y", "z"):
            raise ValueError("up_axis must be 'x', 'y', or 'z'")
        if max_vertical_gap < 0:
            raise ValueError("max_vertical_gap must be >= 0")
        if not (0.0 <= min_footprint_overlap <= 1.0):
            raise ValueError("min_footprint_overlap must be in [0, 1]")
        if not (0.0 <= on_score_threshold <= 1.0):
            raise ValueError("on_score_threshold must be in [0, 1]")

        self.up_axis = up_axis
        self.max_vertical_gap = max_vertical_gap
        self.min_footprint_overlap = min_footprint_overlap
        self.on_score_threshold = on_score_threshold
        
        self.up_idx = {"x": 0, "y": 1, "z": 2}[self.up_axis]
        self.horiz_axes = [i for i in range(3) if i != self.up_idx]

    def _validate_object(self, obj: Object3D) -> None:
        if any(not np.isfinite(x) for x in obj.bbox_min) or any(not np.isfinite(x) for x in obj.bbox_max):
            raise ValueError(f"Object3D {obj.object_id} contains NaN/Inf coordinates.")
        if any(obj.bbox_min[i] > obj.bbox_max[i] for i in range(3)):
            raise ValueError(f"Object3D {obj.object_id} has reversed AABB min/max.")

    def detect(self, subject: Object3D, support: Object3D) -> ONRelationResult:
        """
        Evaluate if `subject` is ON `support`.

        Returns
        -------
        ONRelationResult
        """
        if subject.object_id == support.object_id:
            raise ValueError("Subject and support cannot be the identical Object3D.")
            
        self._validate_object(subject)
        self._validate_object(support)

        # 1. Vertical Gap Test
        # Subject bottom vs Support Top
        subj_bottom = subject.bbox_min[self.up_idx]
        supp_top = support.bbox_max[self.up_idx]
        
        vertical_gap = subj_bottom - supp_top
        
        # 2. Horizontal Footprint Overlap
        overlap_area = 1.0
        subj_area = 1.0
        
        for ax in self.horiz_axes:
            subj_min = subject.bbox_min[ax]
            subj_max = subject.bbox_max[ax]
            supp_min = support.bbox_min[ax]
            supp_max = support.bbox_max[ax]
            
            overlap_min = max(subj_min, supp_min)
            overlap_max = min(subj_max, supp_max)
            
            ax_overlap = max(0.0, overlap_max - overlap_min)
            subj_len = max(0.0, subj_max - subj_min)
            
            overlap_area *= ax_overlap
            subj_area *= subj_len

        if subj_area > 0:
            overlap_ratio = overlap_area / subj_area
        else:
            # Handle zero-area subject footprint edge case securely
            if overlap_area == 0.0 and subj_area == 0.0:
                # E.g. a 1D point dropping on a 1D point
                # Check point inclusion manually
                inside = True
                for ax in self.horiz_axes:
                    if not (support.bbox_min[ax] <= subject.bbox_min[ax] <= support.bbox_max[ax]):
                        inside = False
                overlap_ratio = 1.0 if inside else 0.0
            else:
                overlap_ratio = 0.0

        # 3. ON Score Formulation (Geometric Heuristic)
        # We penalize both positive gaps (floating) and large negative gaps (deep interpenetration).
        abs_gap = abs(vertical_gap)
        
        # Score linearly drops from 1.0 at gap=0 to 0.0 at gap=max_vertical_gap
        if self.max_vertical_gap > 0:
            gap_score = max(0.0, 1.0 - (abs_gap / self.max_vertical_gap))
        else:
            gap_score = 1.0 if abs_gap == 0.0 else 0.0
            
        on_score = gap_score * overlap_ratio

        # 4. Final Boolean Decision
        is_on = False
        if abs_gap <= self.max_vertical_gap:
            if overlap_ratio >= self.min_footprint_overlap:
                if on_score >= self.on_score_threshold:
                    is_on = True

        return ONRelationResult(
            subject_object_id=subject.object_id,
            subject_class_name=subject.class_name,
            support_object_id=support.object_id,
            support_class_name=support.class_name,
            relation="ON",
            vertical_gap=float(vertical_gap),
            overlap_ratio=float(overlap_ratio),
            on_score=float(on_score),
            is_on=is_on
        )
