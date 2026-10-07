"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/depth_filter.py

PURPOSE
-------
Filters Gaussian associations based on depth evidence.

This step improves upon the 2D Alpha-Contribution Association by rejecting
Gaussians whose camera-space depth is inconsistent with the observed depth map.

LIMITATIONS
-----------
This is an EXPERIMENTAL PROTOTYPE.
- Depth maps are synthetic placeholders in tests.
- max_relative_depth_error (10%) is an unvalidated tunable parameter.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
import numpy as np

from gaussian.scene import GaussianScene
from cameras.camera import Camera
from cameras.projection import project_points
from objects.object import Object2D
from association.alpha_association import AlphaAssociationResult
from association.depth_evidence import DepthEvidence


@dataclass(frozen=True)
class DepthFilterResult:
    """
    Output of filter_by_depth().

    Attributes
    ----------
    object_id : int
        The object ID being processed.
    gaussian_ids : np.ndarray
        The IDs of Gaussians that passed the depth filter.
    depth_errors : Dict[int, Optional[float]]
        Absolute depth errors for each passed Gaussian (None if no evidence).
    relative_depth_errors : Dict[int, Optional[float]]
        Relative depth errors for each passed Gaussian (None if no evidence).
    observed_depths : Dict[int, Optional[float]]
        Median depth observed in the image footprint (None if no evidence).
    gaussian_depths : Dict[int, float]
        Camera-space depth Z of the Gaussian center.
    depth_confidences : Dict[int, Optional[float]]
        Median confidence of the depth evidence in the footprint.
    count : int
        Number of Gaussians remaining.
    """
    object_id: int
    gaussian_ids: np.ndarray
    depth_errors: Dict[int, Optional[float]]
    relative_depth_errors: Dict[int, Optional[float]]
    observed_depths: Dict[int, Optional[float]]
    gaussian_depths: Dict[int, float]
    depth_confidences: Dict[int, Optional[float]]
    count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the result."""
        return {
            "object_id": self.object_id,
            "gaussian_ids": self.gaussian_ids.tolist(),
            "depth_errors": self.depth_errors,
            "relative_depth_errors": self.relative_depth_errors,
            "observed_depths": self.observed_depths,
            "gaussian_depths": self.gaussian_depths,
            "depth_confidences": self.depth_confidences,
            "count": self.count,
        }


def filter_by_depth(
    alpha_result: AlphaAssociationResult,
    scene: GaussianScene,
    camera: Camera,
    depth_evidence: DepthEvidence,
    max_relative_depth_error: float = 0.10,
    insufficient_depth_policy: Literal["reject", "keep"] = "reject"
) -> DepthFilterResult:
    """
    Filter Gaussians by comparing their camera-space depth against an observed depth map.

    Parameters
    ----------
    alpha_result : AlphaAssociationResult
        The associated Gaussians from Step 6.
    scene : GaussianScene
        The 3DGS scene representation.
    camera : Camera
        The projection camera.
    depth_evidence : DepthEvidence
        Observed image-space depth map.
    max_relative_depth_error : float
        Maximum allowed relative depth error (default 0.10, i.e., 10%).
    insufficient_depth_policy : Literal["reject", "keep"]
        Whether to reject or keep Gaussians that land in regions with no valid depth.

    Returns
    -------
    DepthFilterResult
    """
    if not (0.0 <= max_relative_depth_error):
        raise ValueError("max_relative_depth_error must be >= 0.")
    if insufficient_depth_policy not in ("reject", "keep"):
        raise ValueError("insufficient_depth_policy must be 'reject' or 'keep'.")

    # Fast path if no Gaussians to process
    if alpha_result.count == 0:
        return DepthFilterResult(
            object_id=alpha_result.object_id,
            gaussian_ids=np.array([], dtype=np.int64),
            depth_errors={},
            relative_depth_errors={},
            observed_depths={},
            gaussian_depths={},
            depth_confidences={},
            count=0,
        )

    # Subset just the passing Gaussians
    id_to_idx = {gid: i for i, gid in enumerate(scene.ids)}
    indices = np.array([id_to_idx[gid] for gid in alpha_result.gaussian_ids])
    
    xyz_subset = scene.xyz[indices].astype(np.float64)
    proj = project_points(camera, xyz_subset)
    
    # Pre-extract physical scale for footprints
    scales_subset = scene.physical_scale()[indices]
    max_scales = np.max(scales_subset, axis=1)

    focal_length = max(camera.K[0, 0], camera.K[1, 1])
    W, H = camera.width, camera.height

    passed_ids = []
    
    d_errors = {}
    rel_d_errors = {}
    obs_depths = {}
    g_depths = {}
    confs = {}

    for k, i in enumerate(indices):
        gid = scene.ids[i]
        
        # Ensure it was a valid projection
        if not proj.valid[k]:
            continue
            
        u_c = proj.pixels[k, 0]
        v_c = proj.pixels[k, 1]
        z_c = proj.depth[k]
        
        # Save Gaussian depth
        g_depths[int(gid)] = float(z_c)
        
        # 1. Footprint bounds
        rad = (focal_length * max_scales[k]) / z_c
        rad = max(rad, 0.5)
        cutoff = 3.0 * rad
        
        u_min = max(0, int(np.floor(u_c - cutoff)))
        u_max = min(W - 1, int(np.ceil(u_c + cutoff)))
        v_min = max(0, int(np.floor(v_c - cutoff)))
        v_max = min(H - 1, int(np.ceil(v_c + cutoff)))
        
        if u_min > u_max or v_min > v_max:
            # Entirely outside image -> no depth evidence
            valid_depths = np.array([])
            valid_confs = np.array([])
        else:
            # 2. Sample Depth Evidence
            mask_slice = depth_evidence.valid_mask[v_min:v_max+1, u_min:u_max+1]
            depth_slice = depth_evidence.depth_map[v_min:v_max+1, u_min:u_max+1]
            
            valid_depths = depth_slice[mask_slice]
            
            if depth_evidence.confidence_map is not None:
                conf_slice = depth_evidence.confidence_map[v_min:v_max+1, u_min:u_max+1]
                valid_confs = conf_slice[mask_slice]
            else:
                valid_confs = np.array([])
        
        # 3. Decision
        if len(valid_depths) == 0:
            if insufficient_depth_policy == "keep":
                passed_ids.append(gid)
                d_errors[int(gid)] = None
                rel_d_errors[int(gid)] = None
                obs_depths[int(gid)] = None
                confs[int(gid)] = None
            else:
                pass # reject
        else:
            obs_z = float(np.median(valid_depths))
            abs_err = abs(float(z_c) - obs_z)
            rel_err = abs_err / obs_z if obs_z > 0 else float('inf')
            
            if rel_err <= max_relative_depth_error:
                passed_ids.append(gid)
                d_errors[int(gid)] = round(abs_err, 6)
                rel_d_errors[int(gid)] = round(rel_err, 6)
                obs_depths[int(gid)] = round(obs_z, 6)
                
                if len(valid_confs) > 0:
                    confs[int(gid)] = round(float(np.median(valid_confs)), 6)
                else:
                    confs[int(gid)] = None

    if passed_ids:
        passed_array = np.unique(passed_ids).astype(np.int64)
    else:
        passed_array = np.array([], dtype=np.int64)

    return DepthFilterResult(
        object_id=alpha_result.object_id,
        gaussian_ids=passed_array,
        depth_errors=d_errors,
        relative_depth_errors=rel_d_errors,
        observed_depths=obs_depths,
        gaussian_depths=g_depths,
        depth_confidences=confs,
        count=len(passed_array),
    )
