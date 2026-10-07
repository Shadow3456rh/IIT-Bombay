"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/dbscan.py

PURPOSE
-------
Filters Gaussian candidates by applying a spatial coherence prior via DBSCAN.
Identifies and removes isolated noise points that mistakenly received high
confidence during multi-view fusion.

LIMITATIONS
-----------
The `eps` parameter is a prototype value that must be calibrated to the actual
Layer-1 3DGS coordinate scale.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from sklearn.cluster import DBSCAN

from gaussian.scene import GaussianScene


@dataclass(frozen=True)
class GaussianCandidate:
    """
    Input structure for DBSCAN cleanup, containing the Gaussian's multi-view score.
    """
    object_id: int
    gaussian_id: int
    fused_score: float

    def __post_init__(self):
        if not np.isfinite(self.fused_score):
            raise ValueError("fused_score must be finite.")


@dataclass(frozen=True)
class DBSCANCleanupResult:
    """
    Output structure for the spatial cleanup stage for a single object.
    """
    object_id: int
    input_gaussian_ids: List[int]
    cleaned_gaussian_ids: List[int]
    removed_gaussian_ids: List[int]
    cleaned_scores: Dict[int, float]
    cluster_labels: Dict[int, int]
    cluster_sizes: Dict[int, int]
    eps: float
    min_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "input_gaussian_ids": self.input_gaussian_ids,
            "cleaned_gaussian_ids": self.cleaned_gaussian_ids,
            "removed_gaussian_ids": self.removed_gaussian_ids,
            "cleaned_scores": self.cleaned_scores,
            "cluster_labels": self.cluster_labels,
            "cluster_sizes": self.cluster_sizes,
            "eps": self.eps,
            "min_samples": self.min_samples,
        }


def cleanup_with_dbscan(
    candidates: List[GaussianCandidate],
    scene: GaussianScene,
    eps: float = 0.05,
    min_samples: int = 3,
) -> List[DBSCANCleanupResult]:
    """
    Run DBSCAN clustering per-object to remove spatially isolated Gaussians.

    Parameters
    ----------
    candidates : List[GaussianCandidate]
        The Gaussians to be clustered.
    scene : GaussianScene
        The scene providing 3D XYZ positions for the Gaussians.
    eps : float
        The maximum distance between two samples for one to be considered as in
        the neighborhood of the other. Must be calibrated.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as
        a core point.

    Returns
    -------
    List[DBSCANCleanupResult]
        One result per unique object_id.
    """
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be a finite positive number.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    # Group candidates by object_id
    grouped_candidates = {}
    for c in candidates:
        if c.object_id not in grouped_candidates:
            grouped_candidates[c.object_id] = []
        grouped_candidates[c.object_id].append(c)

    # Build quick lookup for scene XYZ by ID
    # Since ids may not be just 0...N in arbitrary scenes, we map id -> index
    id_to_idx = {gid: i for i, gid in enumerate(scene.ids)}

    results = []

    # Sort object IDs for determinism
    for obj_id in sorted(grouped_candidates.keys()):
        obj_candidates = grouped_candidates[obj_id]
        
        # Check duplicates
        gids = [c.gaussian_id for c in obj_candidates]
        if len(gids) != len(set(gids)):
            raise ValueError(f"Duplicate Gaussian IDs found for object {obj_id}.")
            
        # Ensure deterministic ordering within object
        obj_candidates.sort(key=lambda c: c.gaussian_id)

        input_ids = []
        cleaned_ids = []
        removed_ids = []
        cleaned_scores = {}
        cluster_labels = {}
        
        xyz_list = []
        valid_candidates = []

        for c in obj_candidates:
            gid = c.gaussian_id
            if gid not in id_to_idx:
                raise ValueError(f"Gaussian ID {gid} not found in scene.")
            
            idx = id_to_idx[gid]
            pt = scene.xyz[idx]
            
            if not np.all(np.isfinite(pt)):
                raise ValueError(f"XYZ position for Gaussian {gid} is not finite.")
                
            input_ids.append(gid)
            xyz_list.append(pt)
            valid_candidates.append(c)

        if len(xyz_list) == 0:
            # Zero candidates -> trivially empty
            pass
            
        elif len(xyz_list) < min_samples:
            # Trivial reject everything as noise since it cannot form a cluster
            for c in valid_candidates:
                gid = c.gaussian_id
                removed_ids.append(gid)
                cluster_labels[gid] = -1
        else:
            X = np.array(xyz_list, dtype=np.float64)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            for c, label in zip(valid_candidates, labels):
                gid = c.gaussian_id
                cluster_labels[gid] = int(label)
                if label == -1:
                    removed_ids.append(gid)
                else:
                    cleaned_ids.append(gid)
                    cleaned_scores[gid] = float(c.fused_score)

        # Count cluster sizes
        cluster_sizes = {}
        for c_label in cluster_labels.values():
            if c_label >= 0:
                cluster_sizes[c_label] = cluster_sizes.get(c_label, 0) + 1

        res = DBSCANCleanupResult(
            object_id=obj_id,
            input_gaussian_ids=input_ids,
            cleaned_gaussian_ids=cleaned_ids,
            removed_gaussian_ids=removed_ids,
            cleaned_scores=cleaned_scores,
            cluster_labels=cluster_labels,
            cluster_sizes=cluster_sizes,
            eps=eps,
            min_samples=min_samples,
        )
        results.append(res)

    return results
