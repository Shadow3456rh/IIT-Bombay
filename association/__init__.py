"""
Layer 2 -- Association sub-package.

Exports the projection-based association function and result type.

    from association import associate_gaussians_with_object, AssociationResult
    from association import associate_gaussians_alpha, AlphaAssociationResult
    from association import filter_by_depth, DepthEvidence, DepthFilterResult
    from association import fuse_multiview_evidence, ViewEvidence, FusedGaussianResult
    from association import cleanup_with_dbscan, GaussianCandidate, DBSCANCleanupResult
"""

from .projection_association import associate_gaussians_with_object, AssociationResult
from .alpha_association import associate_gaussians_alpha, AlphaAssociationResult
from .depth_evidence import DepthEvidence
from .depth_filter import filter_by_depth, DepthFilterResult
from .multiview import fuse_multiview_evidence, ViewEvidence, FusedGaussianResult
from .dbscan import cleanup_with_dbscan, GaussianCandidate, DBSCANCleanupResult

__all__ = [
    "associate_gaussians_with_object", "AssociationResult",
    "associate_gaussians_alpha", "AlphaAssociationResult",
    "DepthEvidence", "filter_by_depth", "DepthFilterResult",
    "fuse_multiview_evidence", "ViewEvidence", "FusedGaussianResult",
    "cleanup_with_dbscan", "GaussianCandidate", "DBSCANCleanupResult"
]
