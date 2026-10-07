"""
Layer 2 -- Cameras sub-package.

Exports the primary Camera class and projection utilities so callers can do:

    from cameras import Camera
    from cameras.projection import project_points, in_image_bounds
"""

from .camera import Camera

__all__ = ["Camera"]
