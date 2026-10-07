"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/alpha_association.py

PURPOSE
-------
Implements the PROTOTYPE Alpha-Contribution Association method.

This is a CPU reference approximation that improves upon the Step 5 baseline
by treating the Gaussian not as a single point, but as an image-space footprint
with a weighted contribution.

ALGORITHM
---------
1. Project Gaussian 3D centre to image coordinates (u, v).
2. Estimate the image-space footprint radius:
       radius = focal_length * max(physical_scale) / depth
3. Define a local bounding box of +/- 3 std devs (radius) around (u,v).
4. For each pixel in this bounding box, calculate the contribution:
       G(dx, dy) = exp(-0.5 * (dx^2 + dy^2) / radius^2)
       contribution = alpha * G(dx, dy)
5. Calculate the association ratio:
       ratio = sum(contribution inside Object2D mask) / sum(total footprint contribution)
6. Accept Gaussian if ratio >= threshold.

LIMITATIONS & PROTOTYPE ASSUMPTIONS
-----------------------------------
- This is an EXPERIMENTAL PROTOTYPE, not an exact 3DGS rasteriser.
- It ignores the full 3D covariance matrix (rotation and 3-axis scale).
- It assumes a symmetric 2D Gaussian footprint based on the maximum physical scale.
- It ignores depth sorting, transmittance (occlusion by other Gaussians), and
  spherical harmonics (colour).
- `threshold` is a tunable parameter (default 0.5), not an experimentally
  validated final value.
- To prevent numerical errors, footprints are clamped to a minimum radius of 0.5 pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from gaussian.scene import GaussianScene
from cameras.camera import Camera
from cameras.projection import project_points
from objects.object import Object2D


@dataclass(frozen=True)
class AlphaAssociationResult:
    """
    Output of associate_gaussians_alpha().

    Attributes
    ----------
    object_id : int
        The Object2D.object_id for which association was computed.
    gaussian_ids : np.ndarray, shape (K,), dtype int64
        IDs of Gaussians whose association ratio >= threshold. Sorted, unique.
    association_scores : Dict[int, float]
        Mapping from Gaussian ID to its association ratio [0.0, 1.0].
    count : int
        Number of associated Gaussians (len(gaussian_ids)).
    """
    object_id:          int
    gaussian_ids:       np.ndarray
    association_scores: Dict[int, float]
    count:              int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the result."""
        return {
            "object_id":          self.object_id,
            "gaussian_ids":       self.gaussian_ids.tolist(),
            "association_scores": self.association_scores,
            "count":              self.count,
        }


def _compute_footprint_contribution(
    u_c: float,
    v_c: float,
    rad: float,
    alpha: float,
    W: int,
    H: int,
    mask: np.ndarray
) -> tuple[float, float]:
    """
    Internal helper to compute the object and total contribution of a single
    Gaussian footprint.

    Returns
    -------
    (obj_contrib, total_contrib)
    """
    cutoff = 3.0 * rad
    u_min = max(0, int(np.floor(u_c - cutoff)))
    u_max = min(W - 1, int(np.ceil(u_c + cutoff)))
    v_min = max(0, int(np.floor(v_c - cutoff)))
    v_max = min(H - 1, int(np.ceil(v_c + cutoff)))

    if u_min > u_max or v_min > v_max:
        return 0.0, 0.0

    uu, vv = np.meshgrid(np.arange(u_min, u_max + 1), np.arange(v_min, v_max + 1))
    dx = uu - u_c
    dy = vv - v_c

    # G(dx, dy)
    G = np.exp(-0.5 * (dx**2 + dy**2) / (rad**2))
    contrib = alpha * G

    total_contrib = float(np.sum(contrib))

    mask_slice = mask[v_min:v_max+1, u_min:u_max+1]
    obj_contrib = float(np.sum(contrib[mask_slice]))

    return obj_contrib, total_contrib


def associate_gaussians_alpha(
    scene:     GaussianScene,
    camera:    Camera,
    obj:       Object2D,
    threshold: float = 0.5,
) -> AlphaAssociationResult:
    """
    Prototype alpha-contribution based Gaussian association.

    Parameters
    ----------
    scene : GaussianScene
        The 3DGS scene.
    camera : Camera
        The projection camera.
    obj : Object2D
        The 2D object with a binary mask.
    threshold : float
        Association ratio cutoff in [0, 1]. Gaussians with an object-overlap
        ratio >= threshold are associated. Default is 0.5.

    Returns
    -------
    AlphaAssociationResult
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    if obj.mask.shape != (camera.height, camera.width):
        raise ValueError(
            f"Mask shape {obj.mask.shape} does not match camera "
            f"dimensions ({camera.height}, {camera.width})."
        )

    N = scene.n_gaussians
    if N == 0:
        return AlphaAssociationResult(
            object_id=obj.object_id,
            gaussian_ids=np.array([], dtype=np.int64),
            association_scores={},
            count=0,
        )

    # 1. Project points
    xyz = scene.xyz.astype(np.float64)
    proj = project_points(camera, xyz)

    # 2. Prepare physical properties
    alphas = scene.physical_opacity()
    scales = scene.physical_scale()
    
    # Use max scale as the scalar size approximation
    max_scales = np.max(scales, axis=1)
    
    focal_length = max(camera.K[0, 0], camera.K[1, 1])
    
    W, H = camera.width, camera.height
    
    associated_ids = []
    scores = {}

    valid_idx = np.where(proj.valid)[0]

    for i in valid_idx:
        u_c = proj.pixels[i, 0]
        v_c = proj.pixels[i, 1]
        z_c = proj.depth[i]

        # 3. Footprint radius approximation
        rad = (focal_length * max_scales[i]) / z_c
        rad = max(rad, 0.5)  # safe minimum radius

        # 4. Compute contributions
        obj_contrib, total_contrib = _compute_footprint_contribution(
            u_c, v_c, rad, alphas[i], W, H, obj.mask
        )

        # 5. Association ratio
        if total_contrib > 1e-9:
            ratio = obj_contrib / total_contrib
        else:
            ratio = 0.0

        if ratio >= threshold and obj_contrib > 1e-9:
            gid = scene.ids[i]
            associated_ids.append(gid)
            scores[int(gid)] = round(float(ratio), 6)

    # 6. Ensure determinism (sorted unique IDs)
    if associated_ids:
        associated_array = np.unique(associated_ids).astype(np.int64)
    else:
        associated_array = np.array([], dtype=np.int64)

    return AlphaAssociationResult(
        object_id          = obj.object_id,
        gaussian_ids       = associated_array,
        association_scores = scores,
        count              = len(associated_array),
    )
