"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/gaussian/scene.py

PURPOSE
-------
Provides GaussianScene: a memory-efficient NumPy-backed representation of a
3D Gaussian Splatting scene loaded from a PLY file.

DESIGN DECISIONS
----------------
* All Gaussian attributes are stored as NumPy arrays, NOT as a list of
  individual Python objects.  This keeps memory proportional to the number
  of Gaussians and is compatible with the ~4 GB real IITB PLY.

* RAW values from the PLY are stored unchanged:
    - opacity  : logit space (apply sigmoid  -> physical opacity in [0, 1])
    - scale    : log   space (apply exp      -> physical scale in metres)
    - rotation : raw quaternion coefficients (rot_0, rot_1, rot_2, rot_3)
    - xyz      : metric world coordinates (stored directly)

* No sigmoid / exp conversion is performed at load time so that the stored
  arrays exactly mirror the PLY bytes (important for reproducibility and for
  future renderer compatibility).

* Physical conversions are provided as helper methods that return NEW arrays
  and never mutate the stored raw arrays.

PROTOTYPE STATUS
----------------
* Gaussian IDs are simply PLY vertex indices (0 .. N-1).
* object_id and semantic_class_id are NOT present in any standard 3DGS PLY.
  They will be assigned by the Layer-2 association pipeline in a later step.
* Spherical Harmonic (SH) coefficients are loaded into a single optional
  array but are not required by the current milestone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# plyfile is listed in requirements.txt
try:
    from plyfile import PlyData
except ImportError:
    print("[ERROR] 'plyfile' is not installed.  Run:  pip install plyfile")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Required PLY property names (standard 3DGS convention)
# ---------------------------------------------------------------------------

_REQUIRED_PROPS = [
    "x", "y", "z",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

# SH / color properties -- optional, loaded if present
_DC_PROPS   = ["f_dc_0", "f_dc_1", "f_dc_2"]
_REST_PREFIX = "f_rest_"


class GaussianScene:
    """
    Memory-efficient representation of a 3DGS Gaussian Splatting scene.

    Attributes
    ----------
    ids : np.ndarray, shape (N,), dtype int64
        Gaussian IDs.  Currently equal to the PLY vertex index (0 .. N-1).
        Will be used as stable keys when Layer-2 assigns object memberships.

    xyz : np.ndarray, shape (N, 3), dtype float32
        World-space 3D centre position of each Gaussian (x, y, z).

    opacity_raw : np.ndarray, shape (N,), dtype float32
        Opacity stored in logit space, exactly as loaded from the PLY.
        Use `physical_opacity()` to obtain values in [0, 1].

    scale_raw : np.ndarray, shape (N, 3), dtype float32
        Scale stored in log space (log of metres), exactly as loaded.
        Use `physical_scale()` to obtain actual scale values.

    rotation : np.ndarray, shape (N, 4), dtype float32
        Rotation quaternion (rot_0, rot_1, rot_2, rot_3) as stored in the PLY.
        Quaternion ordering is NOT remapped here; it matches the PLY convention.
        The correct mapping (scalar-first vs. scalar-last) must be confirmed
        against the renderer before use in Step 3+.

    ply_vertex_index : np.ndarray, shape (N,), dtype int64
        Original PLY vertex index.  Currently identical to `ids` but preserved
        as a separate field so that future sub-sampling does not break the link
        back to the source PLY.

    sh_dc : np.ndarray or None, shape (N, 3), dtype float32
        DC spherical harmonic coefficients (f_dc_0/1/2), if present in PLY.

    sh_rest : np.ndarray or None, shape (N, K), dtype float32
        Higher-order SH coefficients (f_rest_0 .. f_rest_K-1), if present.
        K depends on the SH degree used during training.

    n_gaussians : int
        Total number of Gaussians.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        ids:               np.ndarray,
        xyz:               np.ndarray,
        opacity_raw:       np.ndarray,
        scale_raw:         np.ndarray,
        rotation:          np.ndarray,
        ply_vertex_index:  np.ndarray,
        sh_dc:             Optional[np.ndarray] = None,
        sh_rest:           Optional[np.ndarray] = None,
    ) -> None:
        self.ids              = ids
        self.xyz              = xyz
        self.opacity_raw      = opacity_raw
        self.scale_raw        = scale_raw
        self.rotation         = rotation
        self.ply_vertex_index = ply_vertex_index
        self.sh_dc            = sh_dc
        self.sh_rest          = sh_rest

    # ------------------------------------------------------------------
    # Class-method factory: load from PLY
    # ------------------------------------------------------------------

    @classmethod
    def from_ply(cls, path: str | Path) -> "GaussianScene":
        """
        Load a 3DGS PLY file and return a GaussianScene.

        Parameters
        ----------
        path : str or Path
            Absolute or relative path to the .ply file.

        Returns
        -------
        GaussianScene

        Raises
        ------
        FileNotFoundError
            If the PLY file does not exist.
        KeyError / ValueError
            If any required PLY property is missing.
        """
        path = Path(path)

        # 1. File existence
        if not path.exists():
            raise FileNotFoundError(
                f"PLY file not found: {path}\n"
                f"Place the demo PLY at: data/ply/3dgs_sr_demo.ply"
            )

        # 2. Load PLY
        ply_data = PlyData.read(str(path))

        # 3. Locate vertex element
        element_names = [e.name for e in ply_data.elements]
        if "vertex" in element_names:
            vertex_el = ply_data["vertex"]
        else:
            vertex_el = ply_data.elements[0]

        data = vertex_el.data
        n    = len(data)

        # 4. Validate required properties
        available = {p.name for p in vertex_el.properties}
        missing   = [p for p in _REQUIRED_PROPS if p not in available]
        if missing:
            raise ValueError(
                f"PLY is missing required properties: {missing}\n"
                f"Available properties: {sorted(available)}"
            )

        # 5. Gaussian IDs = vertex indices
        ply_vertex_index = np.arange(n, dtype=np.int64)
        ids              = ply_vertex_index.copy()

        # 6. XYZ  -- shape (N, 3)
        xyz = np.column_stack([
            data["x"].astype(np.float32),
            data["y"].astype(np.float32),
            data["z"].astype(np.float32),
        ])

        # 7. Opacity -- shape (N,)  raw logit space
        opacity_raw = data["opacity"].astype(np.float32)

        # 8. Scale -- shape (N, 3)  raw log space
        scale_raw = np.column_stack([
            data["scale_0"].astype(np.float32),
            data["scale_1"].astype(np.float32),
            data["scale_2"].astype(np.float32),
        ])

        # 9. Rotation -- shape (N, 4)  raw quaternion (PLY convention)
        rotation = np.column_stack([
            data["rot_0"].astype(np.float32),
            data["rot_1"].astype(np.float32),
            data["rot_2"].astype(np.float32),
            data["rot_3"].astype(np.float32),
        ])

        # 10. Optional SH / color
        sh_dc   = None
        sh_rest = None

        if all(p in available for p in _DC_PROPS):
            sh_dc = np.column_stack([
                data["f_dc_0"].astype(np.float32),
                data["f_dc_1"].astype(np.float32),
                data["f_dc_2"].astype(np.float32),
            ])

        rest_props = sorted(
            [p for p in available if p.startswith(_REST_PREFIX)],
            key=lambda s: int(s[len(_REST_PREFIX):])   # numeric sort
        )
        if rest_props:
            sh_rest = np.column_stack(
                [data[p].astype(np.float32) for p in rest_props]
            )

        return cls(
            ids=ids,
            xyz=xyz,
            opacity_raw=opacity_raw,
            scale_raw=scale_raw,
            rotation=rotation,
            ply_vertex_index=ply_vertex_index,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )

    # ------------------------------------------------------------------
    # Physical conversion helpers  (return new arrays, never mutate self)
    # ------------------------------------------------------------------

    def physical_opacity(self) -> np.ndarray:
        """
        Convert raw logit opacity to physical opacity in [0, 1].

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            sigmoid(opacity_raw) = 1 / (1 + exp(-opacity_raw))

        Note
        ----
        The PLY stores opacity as the logit (pre-sigmoid) value.  Very
        negative values correspond to near-transparent Gaussians; values
        near 0 give opacity ~0.5; large positive values give ~1.0.
        """
        return (1.0 / (1.0 + np.exp(-self.opacity_raw))).astype(np.float32)

    def physical_scale(self) -> np.ndarray:
        """
        Convert raw log-space scale to physical scale (in scene units).

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
            exp(scale_raw)

        Note
        ----
        The PLY stores scale as log(metres).  The physical value represents
        the standard deviation of the Gaussian along each axis before the
        rotation is applied.  Scene units depend on the reconstruction
        coordinate frame.
        """
        return np.exp(self.scale_raw).astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_gaussians(self) -> int:
        """Total number of Gaussians."""
        return len(self.ids)

    def __repr__(self) -> str:
        sh_info = (
            f"sh_dc={self.sh_dc.shape if self.sh_dc is not None else None}, "
            f"sh_rest={self.sh_rest.shape if self.sh_rest is not None else None}"
        )
        return (
            f"GaussianScene("
            f"n={self.n_gaussians:,}, "
            f"xyz={self.xyz.shape}, "
            f"opacity_raw={self.opacity_raw.shape}, "
            f"scale_raw={self.scale_raw.shape}, "
            f"rotation={self.rotation.shape}, "
            f"{sh_info})"
        )
