"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/association/projection_association.py

PURPOSE
-------
Implements the BASELINE 2D->3D Gaussian association method.

This is the simplest possible working association:

    3D Gaussian centre (world coords)
          |
          v  Camera extrinsic transform  (W2C)
    3D point in camera space
          |
          v  Perspective projection
    2D pixel (u, v)  [float]
          |
          v  Nearest-pixel rounding  -> (u_int, v_int)  [int]
          |
          v  Integer bounds check: 0 <= u_int < W,  0 <= v_int < H
          |
          v  Object2D binary mask lookup:  mask[v_int, u_int]
          |
          v  Gaussian associated if mask is True

==============================================================================
IMPORTANT LIMITATIONS  (BASELINE METHOD)
==============================================================================

This method is a PROTOTYPE BASELINE.  It has known limitations that will be
addressed in later milestones:

1.  CENTRE-ONLY PROJECTION
    Only the Gaussian's 3D centre is projected.  A real Gaussian has a
    non-zero spatial extent (governed by scale and rotation).  Large Gaussians
    near the image boundary may be missed or incorrectly included.

2.  NO ALPHA CONTRIBUTION
    3DGS rendering uses alpha compositing to determine each Gaussian's actual
    contribution to a pixel.  This baseline ignores alpha entirely.
    (Milestone 6 will address this.)

3.  NO OCCLUSION / DEPTH BUFFERING
    Multiple Gaussians may project to the same pixel.  The baseline associates
    all of them equally; it does not reason about which Gaussian is actually
    visible at that pixel.

4.  NO DEPTH-AWARE FILTERING
    A Gaussian behind the visible surface (but with Z > 0) still passes the
    depth check.  Depth-aware filtering is planned for Milestone 7.

5.  SINGLE-VIEW ONLY
    Evidence from multiple camera views is not combined here.
    Multi-view fusion is planned for Milestone 8.

6.  NO DBSCAN CLEANUP
    Isolated floating Gaussians that happen to project inside the mask are
    not removed.  DBSCAN spatial cleanup is planned for Milestone 9.

==============================================================================
PIXEL-INDEXING CONVENTION  (must not be confused)
==============================================================================

NumPy image arrays follow row-major ordering:

    array[row, col]  ==  array[y, x]  ==  array[v, u]

Therefore the mask lookup is ALWAYS:

    mask[v_int, u_int]    <- CORRECT
    mask[u_int, v_int]    <- WRONG (swaps x and y)

This is explicitly enforced in this module.

==============================================================================
ROUNDING CONVENTION
==============================================================================

Float pixel coordinates are converted to integer pixel indices using:

    u_int = round(u_float)   (nearest-integer / "round half to even")
    v_int = round(v_float)

After rounding, a second integer bounds check is performed.  A pixel that
passes the float-level bounds check can fall outside after rounding
(e.g. u_float = 639.6  ->  u_int = 640  ->  out of bounds for W=640).
Such pixels are rejected.  They are NOT clamped to the boundary.

Clamping would silently associate a Gaussian at the boundary with a pixel
that is not actually where it projects, which would be misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from gaussian.scene import GaussianScene
from cameras.camera import Camera
from cameras.projection import project_points
from objects.object import Object2D


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssociationResult:
    """
    Output of associate_gaussians_with_object().

    Attributes
    ----------
    object_id : int
        The Object2D.object_id for which association was computed.
        Copied directly from the input Object2D.

    gaussian_ids : np.ndarray, shape (K,), dtype int64
        IDs of Gaussians whose projected centre falls inside the object mask.
        Sorted in ascending order.  Unique (no duplicates).
        K may be 0 if no Gaussians are associated.

    count : int
        len(gaussian_ids).  Provided for convenience.

    Notes
    -----
    gaussian_ids uses the IDs from GaussianScene.ids, which in the current
    implementation equal the PLY vertex index (0 .. N-1).
    """
    object_id:    int
    gaussian_ids: np.ndarray   # int64, sorted, unique
    count:        int

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly dict.

        The gaussian_ids array is converted to a Python list so the result
        can be passed to json.dumps() without further conversion.
        """
        return {
            "object_id":    self.object_id,
            "gaussian_ids": self.gaussian_ids.tolist(),
            "count":        self.count,
        }


# ---------------------------------------------------------------------------
# Core association function
# ---------------------------------------------------------------------------

def associate_gaussians_with_object(
    scene:  GaussianScene,
    camera: Camera,
    obj:    Object2D,
) -> AssociationResult:
    """
    Baseline projection association: associate Gaussian IDs with a 2D object.

    Algorithm
    ---------
    1.  Extract every Gaussian's 3D world-space centre from `scene.xyz`.
    2.  Project all centres into the camera image plane.
    3.  Reject projections that are:
          - behind or on the camera image plane  (Z <= 0)
          - outside the image after nearest-pixel rounding
    4.  For each remaining (valid) projected Gaussian, look up the pixel in
        the object mask:  obj.mask[v_int, u_int]
    5.  Collect the IDs of Gaussians whose pixel is True in the mask.

    Parameters
    ----------
    scene : GaussianScene
        The 3DGS scene loaded via GaussianScene.from_ply() or constructed
        synthetically.

    camera : Camera
        A fully initialised Camera with matching intrinsics and extrinsics.
        The coordinate convention stored in the Camera object is used as-is;
        no silent inversion is performed here.

    obj : Object2D
        The 2D object whose mask defines which pixels belong to the object.
        obj.mask.shape must be (camera.height, camera.width).

    Returns
    -------
    AssociationResult
        object_id, gaussian_ids (sorted, unique), count.

    Raises
    ------
    ValueError
        If obj.mask shape does not match the camera's image dimensions.
    """
    # -- Sanity check: mask must match camera image dimensions -------------
    if obj.mask.shape != (camera.height, camera.width):
        raise ValueError(
            f"Object2D mask shape {obj.mask.shape} does not match "
            f"camera image dimensions (height={camera.height}, "
            f"width={camera.width}).  "
            f"The mask must be (camera.height, camera.width) = "
            f"({camera.height}, {camera.width})."
        )

    N   = scene.n_gaussians
    ids = scene.ids                          # (N,) int64
    xyz = scene.xyz.astype(np.float64)      # (N, 3) -- promote to float64 for projection

    # ------------------------------------------------------------------
    # STEP 1: Project all Gaussian centres
    # ------------------------------------------------------------------
    proj = project_points(camera, xyz)
    # proj.pixels : (N, 2) float64  -- NaN for invalid (behind camera)
    # proj.valid  : (N,)   bool     -- True where Z > 0

    # ------------------------------------------------------------------
    # STEP 2: Nearest-pixel rounding for valid projections
    #
    # We only round pixels where valid=True (NaN rounding is undefined).
    # We use a sentinel of -1 for invalid slots so the bounds check below
    # naturally rejects them.
    # ------------------------------------------------------------------
    u_int = np.full(N, -1, dtype=np.int64)
    v_int = np.full(N, -1, dtype=np.int64)

    in_front = proj.valid   # (N,) bool

    if np.any(in_front):
        # np.round uses "round half to even" (banker's rounding)
        # This is deterministic and is the documented convention for this module.
        u_int[in_front] = np.round(proj.pixels[in_front, 0]).astype(np.int64)
        v_int[in_front] = np.round(proj.pixels[in_front, 1]).astype(np.int64)

    # ------------------------------------------------------------------
    # STEP 3: Integer bounds check after rounding
    #
    # Note: a float pixel that passed a float bounds check can still fall
    # outside after rounding (e.g. 639.6 -> 640 for width=640).
    # We reject rather than clamp such pixels.
    #
    # Valid integer pixel range:
    #   0 <= u_int < camera.width
    #   0 <= v_int < camera.height
    # ------------------------------------------------------------------
    int_in_bounds = (
        (u_int >= 0) & (u_int < camera.width) &
        (v_int >= 0) & (v_int < camera.height)
    )

    # A Gaussian must be both in-front-of-camera AND have a valid integer pixel
    candidate = in_front & int_in_bounds   # (N,) bool

    # ------------------------------------------------------------------
    # STEP 4: Mask lookup
    #
    # NumPy array indexing convention:
    #     mask[row, col]  =  mask[y, x]  =  mask[v, u]
    #
    # Using mask[v_int, u_int] is CORRECT.
    # Using mask[u_int, v_int] would silently transpose x and y -- WRONG.
    # ------------------------------------------------------------------
    associated = np.zeros(N, dtype=bool)

    if np.any(candidate):
        cand_idx  = np.where(candidate)[0]
        in_mask   = obj.mask[v_int[candidate], u_int[candidate]]   # mask[v, u]
        associated[cand_idx[in_mask]] = True

    # ------------------------------------------------------------------
    # STEP 5: Collect IDs
    #
    # np.unique returns a sorted, deduplicated array.
    # Deduplication is a safety measure; in practice each Gaussian appears
    # at most once since each index in scene.ids is unique.
    # ------------------------------------------------------------------
    associated_ids = np.unique(ids[associated]).astype(np.int64)

    return AssociationResult(
        object_id    = obj.object_id,
        gaussian_ids = associated_ids,
        count        = len(associated_ids),
    )
