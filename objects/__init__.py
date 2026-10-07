"""
Layer 2 -- Objects sub-package.

Exports Object2D so callers can write:

    from objects import Object2D, Object3D, build_object_3d
"""

from .object import Object2D
from .object_3d import Object3D, build_object_3d

__all__ = ["Object2D", "Object3D", "build_object_3d"]
