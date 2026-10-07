"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/objects/object_3d.py

PURPOSE
-------
Object-level 3D representation for a detected object in the Gaussian scene.
Constructed from the cleaned 3D Gaussians (post DBSCAN).

LIMITATIONS
-----------
- Bounding Box: Axis-Aligned Bounding Box (AABB) of Gaussian centers only.
- Centroid: Simple mean of Gaussian centers, not physically validated.
- No semantic relationship extraction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from gaussian.scene import GaussianScene


@dataclass(frozen=True)
class Object3D:
    """
    Object-level 3D representation.
    
    Attributes
    ----------
    object_id : int
        Persistent 2D object ID.
    semantic_class_id : int
        Semantic class ID.
    class_name : str
        Human-readable class name.
    gaussian_ids : List[int]
        Cleaned Gaussian IDs associated with this object.
    gaussian_scores : Dict[int, float]
        Fused confidence scores for each associated Gaussian.
    center_3d : List[float]
        Mean [X, Y, Z] position of all associated Gaussians.
    bbox_min : List[float]
        Minimum [X, Y, Z] coordinate bounds.
    bbox_max : List[float]
        Maximum [X, Y, Z] coordinate bounds.
    gaussian_count : int
        Total number of associated Gaussians.
    mean_confidence : float
        Mean of the fused scores.
    """
    object_id: int
    semantic_class_id: int
    class_name: str
    gaussian_ids: List[int]
    gaussian_scores: Dict[int, float]
    center_3d: List[float]
    bbox_min: List[float]
    bbox_max: List[float]
    gaussian_count: int
    mean_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serialisable dictionary representation."""
        return {
            "object_id": self.object_id,
            "semantic_class_id": self.semantic_class_id,
            "class_name": self.class_name,
            "gaussian_ids": self.gaussian_ids,
            "gaussian_scores": self.gaussian_scores,
            "gaussian_count": self.gaussian_count,
            "center_3d": self.center_3d,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "mean_confidence": self.mean_confidence,
        }


def build_object_3d(
    object_id: int,
    semantic_class_id: int,
    class_name: str,
    gaussian_ids: List[int],
    gaussian_scores: Dict[int, float],
    scene: GaussianScene
) -> Object3D:
    """
    Constructs an Object3D from a set of Gaussian IDs.

    Parameters
    ----------
    object_id : int
        The unique ID of the object.
    semantic_class_id : int
        The semantic classification ID.
    class_name : str
        The human readable class name.
    gaussian_ids : List[int]
        The cleaned Gaussian IDs belonging to this object.
    gaussian_scores : Dict[int, float]
        Dictionary mapping gaussian_id to its fused confidence score.
    scene : GaussianScene
        The world Gaussian scene containing XYZ data.

    Returns
    -------
    Object3D
    """
    if len(gaussian_ids) == 0:
        raise ValueError("Cannot create Object3D with zero Gaussians.")

    # Validate duplicates
    if len(gaussian_ids) != len(set(gaussian_ids)):
        raise ValueError("Duplicate Gaussian IDs are not allowed in object_3d construction.")

    id_to_idx = {gid: i for i, gid in enumerate(scene.ids)}
    
    xyz_list = []
    
    for gid in gaussian_ids:
        if gid not in id_to_idx:
            raise ValueError(f"Gaussian ID {gid} not found in scene.")
        if gid not in gaussian_scores:
            raise ValueError(f"Gaussian ID {gid} has no corresponding fused score.")
            
        pt = scene.xyz[id_to_idx[gid]]
        if not np.all(np.isfinite(pt)):
            raise ValueError(f"Gaussian {gid} has non-finite XYZ coordinates.")
        xyz_list.append(pt)

    xyz_arr = np.array(xyz_list, dtype=np.float64)
    
    # Prototype Centroid
    mean_xyz = np.mean(xyz_arr, axis=0)
    
    # AABB Bounding Box
    min_xyz = np.min(xyz_arr, axis=0)
    max_xyz = np.max(xyz_arr, axis=0)

    # Object-level confidence metric
    scores = [gaussian_scores[gid] for gid in gaussian_ids]
    mean_conf = float(np.mean(scores))
    
    # Ensure stable ordering
    sorted_ids = sorted(gaussian_ids)

    return Object3D(
        object_id=object_id,
        semantic_class_id=semantic_class_id,
        class_name=class_name,
        gaussian_ids=sorted_ids,
        gaussian_scores={gid: round(gaussian_scores[gid], 6) for gid in sorted_ids},
        center_3d=[round(float(c), 6) for c in mean_xyz],
        bbox_min=[round(float(c), 6) for c in min_xyz],
        bbox_max=[round(float(c), 6) for c in max_xyz],
        gaussian_count=len(sorted_ids),
        mean_confidence=round(mean_conf, 6)
    )
