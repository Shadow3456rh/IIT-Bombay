"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/cameras/projection.py

PURPOSE
-------
Perspective projection utilities for the pinhole Camera model.

These are standalone functions that take a Camera object as an argument so
they remain independent of internal Camera state while sharing a clear API.

PROJECTION EQUATIONS
--------------------
Given a 3D point in camera space:  X_cam = [X, Y, Z]

1. Depth check:
       Z <= 0  ->  point is behind (or on) the camera image plane.
                   Projection is undefined; the point is marked invalid.

2. Perspective division (normalised image coordinates):
       x_n = X / Z
       y_n = Y / Z

3. Pixel coordinates:
       u = fx * x_n + cx
       v = fy * y_n + cy

VISIBILITY vs IMAGE BOUNDS
---------------------------
"Visible" (in front of camera):  Z > 0
"In bounds" (inside image):      0 <= u < width  AND  0 <= v < height

These are separate conditions.  A point can be:
  - in front of camera but projected outside the image frame
  - marked invalid (Z <= 0) regardless of where its pixel would fall

PROTOTYPE STATUS
----------------
This module uses only the basic centre-of-Gaussian projection.
It does NOT account for the full Gaussian footprint / splatting radius.
That is a later-milestone concern.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Import Camera via relative path so the module is usable both from src/ and
# from tests/ (sys.path is handled by the caller).
from cameras.camera import Camera


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class ProjectionResult(NamedTuple):
    """
    Output of project_points().

    pixels : np.ndarray, shape (N, 2), dtype float64
        Projected pixel coordinates (u, v).
        Undefined (set to NaN) for invalid points.

    depth : np.ndarray, shape (N,), dtype float64
        Camera-space Z depth for each point.
        Negative or zero for points behind/on the image plane.

    valid : np.ndarray, shape (N,), dtype bool
        True  -> point is in front of the camera (Z > 0).
        False -> point is behind or on the image plane.

    Note: `valid` does NOT indicate whether the pixel is inside the image.
    Use in_image_bounds() for that check.
    """
    pixels: np.ndarray
    depth:  np.ndarray
    valid:  np.ndarray


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_points(
    camera:       Camera,
    points_world: np.ndarray,
) -> ProjectionResult:
    """
    Project 3D world points onto the 2D image plane of a pinhole camera.

    Steps
    -----
    1. world_to_camera  -- apply W2C extrinsics
    2. depth check      -- flag points with Z <= 0 as invalid
    3. perspective div  -- (X/Z, Y/Z)
    4. pixel mapping    -- u = fx*(X/Z)+cx, v = fy*(Y/Z)+cy

    Parameters
    ----------
    camera : Camera
        Fully initialised Camera object.
    points_world : array-like, shape (N, 3)
        3D points in world coordinates.

    Returns
    -------
    ProjectionResult
        See class docstring.
    """
    pts_world = np.atleast_2d(np.array(points_world, dtype=np.float64))
    if pts_world.shape[1] != 3:
        raise ValueError(
            f"points_world must be shape (N, 3), got {pts_world.shape}"
        )
    N = pts_world.shape[0]

    # 1. World -> camera
    pts_cam = camera.world_to_camera(pts_world)   # (N, 3)

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    # 2. Depth and validity
    depth = Z.copy()
    valid = Z > 0.0   # only points strictly in front of camera are valid

    # 3 & 4. Perspective projection (safe: only divide where valid)
    pixels = np.full((N, 2), np.nan, dtype=np.float64)

    if np.any(valid):
        Z_safe        = np.where(valid, Z, 1.0)   # avoid div/0 for invalid pts
        u             = camera.fx * (X / Z_safe) + camera.cx
        v             = camera.fy * (Y / Z_safe) + camera.cy
        pixels[valid, 0] = u[valid]
        pixels[valid, 1] = v[valid]

    return ProjectionResult(pixels=pixels, depth=depth, valid=valid)


# ---------------------------------------------------------------------------
# Image bounds check
# ---------------------------------------------------------------------------

def in_image_bounds(
    camera: Camera,
    pixels: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask indicating which projected pixels lie inside the
    image frame.

    NOTE: This is a separate check from validity (Z > 0).
    A point can be valid (in front of camera) but still outside the image.

    Parameters
    ----------
    camera : Camera
    pixels : np.ndarray, shape (N, 2)
        Projected (u, v) pixel coordinates, as returned by project_points().
        NaN pixels (invalid projections) will return False.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True if 0 <= u < width AND 0 <= v < height.
    """
    pixels = np.atleast_2d(np.array(pixels, dtype=np.float64))

    u = pixels[:, 0]
    v = pixels[:, 1]

    in_u = (u >= 0.0) & (u < camera.width)
    in_v = (v >= 0.0) & (v < camera.height)

    # NaN comparisons always return False, so NaN pixels are automatically excluded
    return in_u & in_v
