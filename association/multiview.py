"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/multiview.py

PURPOSE
-------
Aggregates and fuses evidence for the same 3D Gaussian across multiple views/frames.
Reduces dependence on noisy single-frame associations.

LIMITATIONS
-----------
This is an EXPERIMENTAL PROTOTYPE for evidence fusion.
It is a confidence-weighted aggregation, NOT a final validated Bayesian model.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from collections import defaultdict


@dataclass(frozen=True)
class ViewEvidence:
    """
    Evidence for a specific Gaussian in a single frame/view.

    Attributes
    ----------
    frame_id : str
        Identifier for the view/frame.
    object_id : int
        The ID of the 2D object being associated.
    gaussian_id : int
        The 3D Gaussian ID.
    association_ratio : float
        The alpha-contribution footprint overlap [0, 1].
    relative_depth_error : Optional[float]
        The relative depth discrepancy from Step 7. None if depth missing.
    depth_valid : bool
        True if depth evidence was observed and valid.
    frame_weight : float
        Weight/importance of this frame. Default 1.0.
    """
    frame_id: str
    object_id: int
    gaussian_id: int
    association_ratio: float
    relative_depth_error: Optional[float]
    depth_valid: bool
    frame_weight: float = 1.0

    def __post_init__(self):
        import numpy as np
        if not np.isfinite(self.association_ratio):
            raise ValueError("association_ratio must be finite.")
        if not (0.0 <= self.association_ratio <= 1.0):
            raise ValueError(f"association_ratio must be in [0, 1], got {self.association_ratio}")
            
        if self.relative_depth_error is not None:
            if not np.isfinite(self.relative_depth_error):
                raise ValueError("relative_depth_error must be finite.")
            if self.relative_depth_error < 0.0:
                raise ValueError("relative_depth_error must be >= 0.")
                
        if not np.isfinite(self.frame_weight):
            raise ValueError("frame_weight must be finite.")
        if self.frame_weight < 0.0:
            raise ValueError("frame_weight must be >= 0.")


@dataclass(frozen=True)
class FusedGaussianResult:
    """
    Output of multi-view evidence fusion for a single Gaussian-Object pair.
    """
    object_id: int
    gaussian_id: int
    fused_score: float
    accepted: bool
    observation_count: int
    positive_observation_count: int
    depth_valid_observation_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "gaussian_id": self.gaussian_id,
            "fused_score": round(self.fused_score, 6),
            "accepted": self.accepted,
            "observation_count": self.observation_count,
            "positive_observation_count": self.positive_observation_count,
            "depth_valid_observation_count": self.depth_valid_observation_count,
        }


def fuse_multiview_evidence(
    evidences: List[ViewEvidence],
    depth_scale: float = 0.10,
    missing_depth_policy: Literal["zero", "one"] = "zero",
    fused_score_threshold: float = 0.5
) -> List[FusedGaussianResult]:
    """
    Fuse per-view evidence into a final robust confidence score.

    Parameters
    ----------
    evidences : List[ViewEvidence]
        The collection of observations across all frames.
    depth_scale : float
        Tuning parameter converting relative depth error to depth confidence.
        depth_confidence = exp(-error / depth_scale).
    missing_depth_policy : str
        Policy for assigning depth confidence when depth is missing.
        "zero": missing depth implies zero depth confidence.
        "one": missing depth implies perfect depth confidence.
    fused_score_threshold : float
        Threshold for marking the Gaussian as 'accepted'.

    Returns
    -------
    List[FusedGaussianResult]
        Sorted deterministically by object_id, then gaussian_id.
    """
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be > 0.")
    if missing_depth_policy not in ("zero", "one"):
        raise ValueError("missing_depth_policy must be 'zero' or 'one'.")
    if not (0.0 <= fused_score_threshold <= 1.0):
        raise ValueError("fused_score_threshold must be in [0, 1].")

    # Group by (object_id, gaussian_id)
    grouped = defaultdict(list)
    for ev in evidences:
        grouped[(ev.object_id, ev.gaussian_id)].append(ev)

    results = []

    for (obj_id, gauss_id), group_evidences in grouped.items():
        total_weight = 0.0
        sum_evidence = 0.0
        
        obs_count = 0
        pos_obs_count = 0
        depth_valid_count = 0

        for ev in group_evidences:
            obs_count += 1
            
            # Depth Confidence
            if ev.depth_valid and ev.relative_depth_error is not None:
                depth_confidence = math.exp(-ev.relative_depth_error / depth_scale)
                depth_valid_count += 1
            else:
                if missing_depth_policy == "zero":
                    depth_confidence = 0.0
                else:  # "one"
                    depth_confidence = 1.0

            # View Evidence Aggregation
            view_score = ev.association_ratio * depth_confidence
            
            # Track positives (any non-zero contribution to the numerator)
            if view_score > 0:
                pos_obs_count += 1

            weighted_score = view_score * ev.frame_weight
            
            sum_evidence += weighted_score
            total_weight += ev.frame_weight

        # Fusion
        if total_weight > 0:
            fused_score = sum_evidence / total_weight
        else:
            fused_score = 0.0

        accepted = bool(fused_score >= fused_score_threshold)

        results.append(FusedGaussianResult(
            object_id=obj_id,
            gaussian_id=gauss_id,
            fused_score=fused_score,
            accepted=accepted,
            observation_count=obs_count,
            positive_observation_count=pos_obs_count,
            depth_valid_observation_count=depth_valid_count
        ))

    # Deterministic sorting
    results.sort(key=lambda x: (x.object_id, x.gaussian_id))
    
    return results
