"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/objects/object.py

PURPOSE
-------
Provides Object2D: a clean, validated representation of one detected and
segmented object instance inside a single drone image frame.

Object2D holds ONLY 2D information.  It does NOT contain:
  - Gaussian IDs
  - 3D position / 3D bounding box
  - 3D relationships
  - layer classification
  - motion state
  - tracking state beyond the object_id

Those belong to later milestones.

==============================================================================
PIPELINE CONTEXT
==============================================================================

The intended full pipeline (not yet implemented) is:

    Drone frame
      -> YOLO26-L    [detects class, bounding box, confidence]
      -> SAM3        [produces precise instance mask]
      -> Object2D    [this class]
      -> 2D->3D Gaussian Association  [later step]

For this milestone (Step 4), YOLO26-L and SAM3 are NOT executed.
Object2D can be constructed manually (for testing) or from real detector
output (later).  It has NO internal dependency on either model.

==============================================================================
SEMANTIC CLASS NOTE
==============================================================================

`semantic_class_id` and `class_name` are stored as provided.  The caller is
responsible for ensuring they agree.

IMPORTANT: VisDrone, UAVid, and the actual IITB campus dataset may use
different numeric class IDs for the same category.  A dataset-to-project
semantic mapping layer will be added later (Milestone 14).  Do NOT assume
that a `semantic_class_id` from one dataset matches another.

==============================================================================
BOUNDING BOX FORMAT
==============================================================================

bbox is [x1, y1, x2, y2] in pixel coordinates (not normalised).

    x1, y1 = top-left corner
    x2, y2 = bottom-right corner

    Constraint:  x1 < x2  and  y1 < y2

If the upstream YOLO26-L output uses normalised [0,1] coordinates, convert
to pixel coordinates before constructing Object2D.

==============================================================================
MASK FORMAT
==============================================================================

mask is a NumPy boolean array of shape (image_height, image_width).

    True  = pixel belongs to this object instance
    False = pixel does not belong to this object instance

When SAM3 is integrated later, its output mask must be converted to this
format before being stored here.

==============================================================================
PROTOTYPE STATUS
==============================================================================

- object_id is treated as a persistent instance identifier, but the final
  tracking/re-identification mechanism will be connected in a later step.
- Class IDs are demo values; a dataset mapping layer is planned for M14.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


class Object2D:
    """
    Representation of one detected + segmented object instance in one image.

    Parameters
    ----------
    object_id : int
        Persistent instance identifier.  The same physical object should carry
        the same object_id across multiple frames.
        (Tracking mechanism to be connected later.)

    semantic_class_id : int
        Numeric class index from the upstream detector.
        NOTE: IDs are dataset-specific; mapping to a project-wide ontology
        is planned for a later milestone.

    class_name : str
        Human-readable class label (e.g. "car", "pole").

    confidence : float
        Detector/segmenter confidence in [0.0, 1.0].
        Raises ValueError if outside range.

    bbox : list or array-like of 4 numbers  [x1, y1, x2, y2]
        Bounding box in pixel coordinates.  Must satisfy x1 < x2, y1 < y2.
        Values must be finite.

    mask : np.ndarray, shape (H, W), dtype bool
        Binary instance mask.  True where the pixel belongs to this object.
        Must be 2-D with positive dimensions.

    source_frame : str
        Identifier / filename of the source image frame
        (e.g. "frame_001.jpg").  The file does not need to exist on disk for
        this milestone.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        object_id:         int,
        semantic_class_id: int,
        class_name:        str,
        confidence:        float,
        bbox:              "list | np.ndarray",
        mask:              np.ndarray,
        source_frame:      str,
    ) -> None:

        # --- object_id -------------------------------------------------
        if not isinstance(object_id, (int, np.integer)):
            raise TypeError(
                f"object_id must be an integer, got {type(object_id).__name__}"
            )
        self.object_id = int(object_id)

        # --- semantic_class_id -----------------------------------------
        if not isinstance(semantic_class_id, (int, np.integer)):
            raise TypeError(
                f"semantic_class_id must be an integer, "
                f"got {type(semantic_class_id).__name__}"
            )
        self.semantic_class_id = int(semantic_class_id)

        # --- class_name ------------------------------------------------
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        self.class_name = class_name.strip()

        # --- confidence ------------------------------------------------
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {confidence}"
            )
        self.confidence = confidence

        # --- bbox [x1, y1, x2, y2] ------------------------------------
        bbox = list(bbox)
        if len(bbox) != 4:
            raise ValueError(
                f"bbox must have exactly 4 values [x1, y1, x2, y2], "
                f"got {len(bbox)}"
            )
        x1, y1, x2, y2 = [float(v) for v in bbox]
        for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x2, y2]):
            if not math.isfinite(val):
                raise ValueError(
                    f"bbox value '{name}' must be finite, got {val}"
                )
        if x1 >= x2:
            raise ValueError(
                f"bbox must satisfy x1 < x2, got x1={x1}, x2={x2}"
            )
        if y1 >= y2:
            raise ValueError(
                f"bbox must satisfy y1 < y2, got y1={y1}, y2={y2}"
            )
        self.bbox: List[float] = [x1, y1, x2, y2]

        # --- mask ------------------------------------------------------
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"mask must be a numpy ndarray, got {type(mask).__name__}"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"mask must be 2-D (height, width), got ndim={mask.ndim}"
            )
        if mask.dtype != bool:
            raise ValueError(
                f"mask dtype must be bool, got {mask.dtype}.  "
                f"Convert with mask.astype(bool) before passing."
            )
        if mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError(
                f"mask dimensions must be positive, got shape {mask.shape}"
            )
        self.mask: np.ndarray = mask   # shape (H, W), bool

        # --- source_frame ----------------------------------------------
        if not isinstance(source_frame, str) or not source_frame.strip():
            raise ValueError("source_frame must be a non-empty string")
        self.source_frame = source_frame.strip()

    # ------------------------------------------------------------------
    # Mask properties
    # ------------------------------------------------------------------

    @property
    def image_height(self) -> int:
        """Image height inferred from mask shape."""
        return self.mask.shape[0]

    @property
    def image_width(self) -> int:
        """Image width inferred from mask shape."""
        return self.mask.shape[1]

    @property
    def mask_area(self) -> int:
        """Number of True pixels in the mask (object pixel count)."""
        return int(np.count_nonzero(self.mask))

    @property
    def mask_coverage(self) -> float:
        """
        Fraction of the image covered by this object.

            mask_coverage = mask_area / (image_width * image_height)

        Returns 0.0 for an empty mask.
        """
        total = self.image_height * self.image_width
        if total == 0:
            return 0.0
        return self.mask_area / total

    @property
    def mask_bbox(self) -> Optional[List[int]]:
        """
        Tightest axis-aligned bounding box around True pixels in the mask.

        Returns
        -------
        [x_min, y_min, x_max, y_max] as integers (pixel coordinates)
        or None if the mask is entirely False (empty object).

        Note: mask rows = Y axis, mask columns = X axis.
        """
        rows_any = np.any(self.mask, axis=1)   # shape (H,) -- any True in row?
        cols_any = np.any(self.mask, axis=0)   # shape (W,) -- any True in col?

        if not rows_any.any():
            return None   # empty mask

        y_min = int(np.argmax(rows_any))
        y_max = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
        x_min = int(np.argmax(cols_any))
        x_max = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))

        return [x_min, y_min, x_max, y_max]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly dictionary of this object's metadata.

        The full NumPy mask is NOT included (it can be large and is not
        JSON-serialisable).  mask_shape and mask_area are stored instead.

        Example output::

            {
                "object_id": 17,
                "semantic_class_id": 2,
                "class_name": "car",
                "confidence": 0.94,
                "bbox": [200.0, 180.0, 350.0, 320.0],
                "source_frame": "demo_frame_001.jpg",
                "mask_shape": [480, 640],
                "mask_area": 21141,
                "mask_coverage": 0.06876
            }
        """
        return {
            "object_id":         self.object_id,
            "semantic_class_id": self.semantic_class_id,
            "class_name":        self.class_name,
            "confidence":        round(self.confidence, 6),
            "bbox":              [round(v, 2) for v in self.bbox],
            "source_frame":      self.source_frame,
            "mask_shape":        list(self.mask.shape),   # [H, W]
            "mask_area":         self.mask_area,
            "mask_coverage":     round(self.mask_coverage, 6),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Object2D("
            f"id={self.object_id}, "
            f"class='{self.class_name}'({self.semantic_class_id}), "
            f"conf={self.confidence:.2f}, "
            f"bbox={[round(v) for v in self.bbox]}, "
            f"mask={self.mask.shape}, "
            f"area={self.mask_area}px, "
            f"frame='{self.source_frame}')"
        )
