"""
Layer 2 -- Relationships sub-package.
"""

from .on_relation import ONRelationDetector, ONRelationResult
from .graph_builder import (
    SceneGraph,
    SceneGraphBuilder,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
)
from .gnn_model import (
    RelationshipGNN,
    BatchSceneGraph,
    RelationshipClass,
    RELATIONSHIP_NAMES,
    NUM_RELATIONSHIP_CLASSES,
)
from .gnn_dataset import (
    RelationshipGraphSample,
    RelationshipDataset,
    collate_relationship_graphs,
    split_dataset_by_scene,
    generate_synthetic_dataset,
)
from .train_gnn import (
    TrainingConfig,
    train_gnn_model,
    compute_relationship_metrics,
    evaluate_dataset,
)
from .relationship_pipeline import (
    RelationshipPipeline,
    RelationshipPrediction,
    PipelineResult,
)

__all__ = [
    "ONRelationDetector",
    "ONRelationResult",
    "SceneGraph",
    "SceneGraphBuilder",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "NODE_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "RelationshipGNN",
    "BatchSceneGraph",
    "RelationshipClass",
    "RELATIONSHIP_NAMES",
    "NUM_RELATIONSHIP_CLASSES",
    "RelationshipGraphSample",
    "RelationshipDataset",
    "collate_relationship_graphs",
    "split_dataset_by_scene",
    "generate_synthetic_dataset",
    "TrainingConfig",
    "train_gnn_model",
    "compute_relationship_metrics",
    "evaluate_dataset",
    "RelationshipPipeline",
    "RelationshipPrediction",
    "PipelineResult",
]
