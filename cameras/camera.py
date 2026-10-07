"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/cameras/camera.py

PURPOSE
-------
Provides the Camera class: a complete pinhole camera model with explicit,
unambiguous extrinsic convention documentation.

==============================================================================
COORDINATE SYSTEM CONVENTION  (read before modifying)
==============================================================================

WORLD SPACE
-----------
A right-handed coordinate system used by the 3DGS reconstruction.
Axis orientation depends on the reconstruction tool (COLMAP / Nerfstudio).
This module does NOT assume a specific world-space up direction.

CAMERA SPACE
------------
The camera's own coordinate system follows the standard OpenCV convention:

    +X  -> right   (along image columns, increasing u)
    +Y  -> down    (along image rows, increasing v)
    +Z  -> forward (into the scene, the "look" direction)

The camera looks along +Z.  A point with Z > 0 is in front of the camera.

EXTRINSICS -- stored as WORLD-TO-CAMERA (W2C)
----------------------------------------------
We store R_w2c (3x3) and t_w2c (3,) such that:

    X_cam = R_w2c @ X_world + t_w2c

This is the convention used by OpenCV, COLMAP, and most SfM tools.

T_w2c (4x4 homogeneous):
    | R_w2c   t_w2c |
    | 0  0  0   1   |

T_c2w (4x4 homogeneous) is the inverse, i.e. camera-to-world:
    T_c2w = inv(T_w2c)

IMPORTANT: When loading real IITB / Nerfstudio camera data, verify whether
the stored transform is C2W or W2C before assigning.  Nerfstudio and many
NeRF frameworks store C2W by default; COLMAP stores W2C.  The constructor
accepts either convention and converts internally.

INTRINSICS
----------
Standard pinhole intrinsic matrix K:

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

Projection (from camera space to pixel space):
    u = fx * (X_cam / Z_cam) + cx
    v = fy * (Y_cam / Z_cam) + cy

PROTOTYPE STATUS
----------------
The Camera class is currently tested with a synthetic camera only.
Real IITB camera parameters will come from the Layer-1 / COLMAP /
Nerfstudio output and must NOT be invented here.
==============================================================================
"""

from __future__ import annotations

from typing import Literal
import numpy as np


ExtrinsicConvention = Literal["w2c", "c2w"]


class Camera:
    """
    Pinhole camera model with explicit extrinsic convention.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    R : array-like, shape (3, 3)
        Rotation matrix.  Interpretation depends on `convention`.
    t : array-like, shape (3,)
        Translation vector.  Interpretation depends on `convention`.
    convention : "w2c" or "c2w"
        Declares the meaning of (R, t):

        "w2c"  (default, COLMAP / OpenCV style)
            X_cam = R @ X_world + t
            T_w2c = | R  t |
                    | 0  1 |

        "c2w"  (Nerfstudio / NeRF style)
            X_world = R @ X_cam + t
            Stored as T_c2w; W2C is derived by inversion.

    Notes
    -----
    Internally, we always store R_w2c and t_w2c regardless of the input
    convention.  The original C2W form is always accessible via `T_c2w`.
    """

    def __init__(
        self,
        width:      int,
        height:     int,
        fx:         float,
        fy:         float,
        cx:         float,
        cy:         float,
        R:          "np.ndarray | list",
        t:          "np.ndarray | list",
        convention: ExtrinsicConvention = "w2c",
    ) -> None:

        # -- Intrinsics ---------------------------------------------------
        self.width  = int(width)
        self.height = int(height)
        self.fx     = float(fx)
        self.fy     = float(fy)
        self.cx     = float(cx)
        self.cy     = float(cy)

        # -- Extrinsics ---------------------------------------------------
        R = np.array(R, dtype=np.float64)
        t = np.array(t, dtype=np.float64).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"R must be shape (3,3), got {R.shape}")

        self._convention = convention

        if convention == "w2c":
            # Provided directly in world-to-camera form
            self._R_w2c = R.copy()
            self._t_w2c = t.copy()
        elif convention == "c2w":
            # Provided in camera-to-world form; derive W2C by inversion
            # T_w2c = inv(T_c2w)
            # For a rigid transform: R_w2c = R_c2w^T,  t_w2c = -R_c2w^T @ t_c2w
            self._R_w2c = R.T.copy()
            self._t_w2c = (-R.T @ t).copy()
        else:
            raise ValueError(
                f"convention must be 'w2c' or 'c2w', got '{convention}'"
            )

    # ------------------------------------------------------------------
    # Intrinsics
    # ------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """
        Intrinsic matrix K, shape (3, 3).

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        return np.array([
            [self.fx,      0.0, self.cx],
            [    0.0, self.fy,  self.cy],
            [    0.0,      0.0,     1.0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Extrinsics -- W2C form (always valid)
    # ------------------------------------------------------------------

    @property
    def R_w2c(self) -> np.ndarray:
        """
        Rotation matrix for world-to-camera transform, shape (3, 3).
        X_cam = R_w2c @ X_world + t_w2c
        """
        return self._R_w2c.copy()

    @property
    def t_w2c(self) -> np.ndarray:
        """
        Translation vector for world-to-camera transform, shape (3,).
        X_cam = R_w2c @ X_world + t_w2c
        """
        return self._t_w2c.copy()

    @property
    def T_w2c(self) -> np.ndarray:
        """
        4x4 homogeneous world-to-camera transform.

            T_w2c = | R_w2c  t_w2c |
                    |  0  0  0   1 |

        Meaning:  X_cam_h = T_w2c @ X_world_h
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._R_w2c
        T[:3,  3] = self._t_w2c
        return T

    # ------------------------------------------------------------------
    # Extrinsics -- C2W form (derived from W2C)
    # ------------------------------------------------------------------

    @property
    def R_c2w(self) -> np.ndarray:
        """
        Rotation matrix for camera-to-world transform, shape (3, 3).
        R_c2w = R_w2c^T
        """
        return self._R_w2c.T.copy()

    @property
    def t_c2w(self) -> np.ndarray:
        """
        Camera centre in world coordinates, shape (3,).
        t_c2w = -R_w2c^T @ t_w2c
        """
        return (-self._R_w2c.T @ self._t_w2c).copy()

    @property
    def T_c2w(self) -> np.ndarray:
        """
        4x4 homogeneous camera-to-world transform.

            T_c2w = | R_c2w  t_c2w |
                    |  0  0  0   1 |

        Meaning:  X_world_h = T_c2w @ X_cam_h
        T_c2w = inv(T_w2c)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R_c2w
        T[:3,  3] = self.t_c2w
        return T

    # ------------------------------------------------------------------
    # Coordinate transformation
    # ------------------------------------------------------------------

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """
        Transform 3D world points into the camera coordinate frame.

        Convention (W2C, OpenCV):
            X_cam = R_w2c @ X_world + t_w2c

        Parameters
        ----------
        points_world : array-like, shape (N, 3) or (3,)
            Points in world coordinates.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Points in camera coordinates.
            +X = right, +Y = down, +Z = forward (into scene).
            Points with Z > 0 are in front of the camera.
        """
        pts = np.atleast_2d(np.array(points_world, dtype=np.float64))
        if pts.shape[1] != 3:
            raise ValueError(
                f"points_world must have shape (N,3), got {pts.shape}"
            )
        # X_cam = R_w2c @ X_world^T  +  t_w2c  (broadcast)
        return (pts @ self._R_w2c.T + self._t_w2c).astype(np.float64)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def camera_centre_world(self) -> np.ndarray:
        """Camera centre in world coordinates (= t_c2w)."""
        return self.t_c2w

    def __repr__(self) -> str:
        return (
            f"Camera(W={self.width}, H={self.height}, "
            f"fx={self.fx}, fy={self.fy}, "
            f"cx={self.cx}, cy={self.cy}, "
            f"convention='{self._convention}')"
        )
