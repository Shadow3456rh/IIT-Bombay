"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/relationship_pipeline.py

PURPOSE
-------
Inference orchestrator for Layer 2 3D spatial relationship extraction.
Accepts Object3D instances and produces structured relationship predictions
under two explicit operational modes:

Mode A: 'baseline'
    - Relies purely on geometric constraints via ONRelationDetector.
    - Deterministic, requires no model checkpoint.
    - Detects 'ON' support relationships.

Mode B: 'gnn'
    - Runs a trained RelationshipGNN checkpoint.
    - Predicts directed relationships across 7 classes:
      NONE, ON, UNDER, INSIDE, NEXT_TO, IN_FRONT_OF, BEHIND.
    - Requires an explicit loadable checkpoint; rejects uncalibrated random weights.
    - Softmax outputs are documented as uncalibrated model scores, NOT real-world probabilities.

COMPATIBILITY
-------------
Supports serialization to Layer2Output (output/layer2_json.py) both in the
original ON-only JSON format (version 0.1) and an extended multi-relation schema (version 0.2).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from output.layer2_json import Layer2Output

import numpy as np
import torch

from objects.object_3d import Object3D
from relationships.graph_builder import SceneGraph, SceneGraphBuilder
from relationships.gnn_model import (
    NUM_RELATIONSHIP_CLASSES,
    RELATIONSHIP_NAMES,
    RelationshipClass,
    RelationshipGNN,
)
from relationships.on_relation import ONRelationDetector, ONRelationResult



PipelineMode = Literal["baseline", "gnn"]


# ---------------------------------------------------------------------------
# Prediction Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationshipPrediction:
    """
    Structured relationship prediction for a directed pair of objects (subject -> target).

    Attributes
    ----------
    subject_object_id : int
    subject_class_name : str
    target_object_id : int
    target_class_name : str
    relation : str
        Predicted relationship name (e.g., 'ON', 'NEXT_TO', 'UNDER').
    method : str
        Provenance: 'geometric_baseline' or 'gnn'.
    score : Optional[float]
        Confidence metric. For geometric baseline, this is the heuristic on_score in [0, 1].
        For GNN, this is the softmax score (uncalibrated heuristic score, not a probability).
    details : Dict[str, Any]
        Additional geometric or model metadata.
    """

    subject_object_id: int
    subject_class_name: str
    target_object_id: int
    target_class_name: str
    relation: str
    method: Literal["geometric_baseline", "gnn"]
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serialisable dictionary representation."""
        return {
            "subject_object_id": self.subject_object_id,
            "subject_class_name": self.subject_class_name,
            "target_object_id": self.target_object_id,
            "target_class_name": self.target_class_name,
            "relation": self.relation,
            "method": self.method,
            "score": round(self.score, 6) if self.score is not None else None,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """
    Complete output payload of the RelationshipPipeline.
    """

    mode: PipelineMode
    predictions: List[RelationshipPrediction]
    num_objects: int
    num_candidate_edges: int
    checkpoint_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "num_objects": self.num_objects,
            "num_candidate_edges": self.num_candidate_edges,
            "num_predictions": len(self.predictions),
            "checkpoint_path": self.checkpoint_path,
            "predictions": [p.to_dict() for p in self.predictions],
            "metadata": self.metadata,
        }

    def to_layer2_output(
        self,
        objects: List[Object3D],
        include_all_relations: bool = False,
    ) -> Layer2Output:
        """
        Converts the pipeline predictions into a standard Layer2Output.

        Parameters
        ----------
        objects : List[Object3D]
            The list of 3D objects evaluated.
        include_all_relations : bool
            If False (default), preserves strict backward compatibility with version 0.1
            by filtering only 'ON' relationships and calling build_layer2_output.
            If True, uses the extended multi-class schema (version 0.2).
        """
        from output.layer2_json import (
            Layer2Output,
            build_layer2_output,
            build_extended_layer2_output,
        )

        if not include_all_relations:
            # Backward-compatible ON-only serialization
            on_results: List[ONRelationResult] = []
            for p in self.predictions:
                if p.relation == "ON":
                    on_res = ONRelationResult(
                        subject_object_id=p.subject_object_id,
                        subject_class_name=p.subject_class_name,
                        support_object_id=p.target_object_id,
                        support_class_name=p.target_class_name,
                        relation="ON",
                        vertical_gap=float(p.details.get("vertical_gap", 0.0)),
                        overlap_ratio=float(p.details.get("overlap_ratio", 1.0)),
                        on_score=float(p.score if p.score is not None else 1.0),
                        is_on=True,
                    )
                    on_results.append(on_res)
            return build_layer2_output(objects, on_results)
        else:
            # Extended multi-class serialization
            rel_dicts = [p.to_dict() for p in self.predictions]
            return build_extended_layer2_output(objects, rel_dicts, version="0.2")



# ---------------------------------------------------------------------------
# Relationship Pipeline
# ---------------------------------------------------------------------------


class RelationshipPipeline:
    """
    Main orchestration class for Layer 2 spatial relationship extraction.
    """

    def __init__(
        self,
        mode: PipelineMode = "baseline",
        checkpoint_path: Optional[Union[str, Path]] = None,
        up_axis: Literal["x", "y", "z"] = "z",
        confidence_threshold: float = 0.0,
        filter_none: bool = True,
        max_edge_distance: Optional[float] = None,
        on_detector: Optional[ONRelationDetector] = None,
    ):
        """
        Initialize the inference pipeline.

        Parameters
        ----------
        mode : 'baseline' or 'gnn'
            Operational mode.
        checkpoint_path : Optional[str, Path]
            Path to trained RelationshipGNN checkpoint. Required when mode='gnn'.
        up_axis : 'x', 'y', or 'z'
            Vertical axis orientation.
        confidence_threshold : float
            Minimum score threshold for GNN predictions.
        filter_none : bool
            If True, filters out predictions categorized as 'NONE' (class 0).
        max_edge_distance : Optional[float]
            Maximum Euclidean distance for candidate edges.
        on_detector : Optional[ONRelationDetector]
            Custom detector for geometric baseline.
        """
        if mode not in ("baseline", "gnn"):
            raise ValueError(f"mode must be 'baseline' or 'gnn', got {mode!r}")
        if up_axis not in ("x", "y", "z"):
            raise ValueError(f"up_axis must be 'x', 'y', or 'z', got {up_axis!r}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        self.mode: PipelineMode = mode
        self.up_axis = up_axis
        self.confidence_threshold = confidence_threshold
        self.filter_none = filter_none
        self.max_edge_distance = max_edge_distance

        self.on_detector = on_detector or ONRelationDetector(up_axis=self.up_axis)
        self.checkpoint_path: Optional[Path] = None
        self.model: Optional[RelationshipGNN] = None

        if self.mode == "gnn":
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when mode='gnn'. "
                    "Use mode='baseline' when no trained checkpoint is available."
                )
            ckpt_p = Path(checkpoint_path)
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Trained GNN checkpoint not found at: {ckpt_p}")

            self.checkpoint_path = ckpt_p
            self.model = RelationshipGNN.load_checkpoint(self.checkpoint_path)
            self.model.eval()

    def _validate_objects(self, objects: List[Object3D]) -> None:
        """Validates numerical integrity of input objects."""
        seen = set()
        for obj in objects:
            if obj.object_id in seen:
                raise ValueError(f"Duplicate object ID {obj.object_id} detected in input.")
            seen.add(obj.object_id)

            coords = list(obj.center_3d) + list(obj.bbox_min) + list(obj.bbox_max)
            if any(not np.isfinite(c) for c in coords):
                raise ValueError(f"Object {obj.object_id} contains non-finite coordinates.")
            if any(obj.bbox_min[i] > obj.bbox_max[i] for i in range(3)):
                raise ValueError(f"Object {obj.object_id} has bbox_min > bbox_max.")

    def run(self, objects: List[Object3D]) -> PipelineResult:
        """
        Extracts spatial relationships between the given Object3D instances.

        Parameters
        ----------
        objects : List[Object3D]

        Returns
        -------
        PipelineResult
        """
        self._validate_objects(objects)
        num_objects = len(objects)

        # Empty or single object boundary cases
        if num_objects < 2:
            return PipelineResult(
                mode=self.mode,
                predictions=[],
                num_objects=num_objects,
                num_candidate_edges=0,
                checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
                metadata={"status": "Trivially empty: less than 2 objects."},
            )

        # Build candidate graph
        builder = SceneGraphBuilder(
            up_axis=self.up_axis,
            max_edge_distance=self.max_edge_distance,
            on_detector=self.on_detector,
        )
        graph = builder.build(objects)
        num_edges = graph.num_edges

        obj_by_id = {obj.object_id: obj for obj in objects}
        predictions: List[RelationshipPrediction] = []

        if self.mode == "baseline":
            # Mode A: Geometric baseline heuristic
            for pair, on_res in graph.heuristic_on_results.items():
                if on_res.is_on:
                    pred = RelationshipPrediction(
                        subject_object_id=on_res.subject_object_id,
                        subject_class_name=on_res.subject_class_name,
                        target_object_id=on_res.support_object_id,
                        target_class_name=on_res.support_class_name,
                        relation="ON",
                        method="geometric_baseline",
                        score=round(float(on_res.on_score), 6),
                        details={
                            "vertical_gap": round(float(on_res.vertical_gap), 6),
                            "overlap_ratio": round(float(on_res.overlap_ratio), 6),
                            "description": "Satisfies geometric vertical gap and footprint overlap.",
                        },
                    )
                    predictions.append(pred)

        elif self.mode == "gnn":
            # Mode B: Trained GNN inference
            assert self.model is not None

            if num_edges > 0:
                with torch.no_grad():
                    logits = self.model.forward_graph(graph)  # (E, 7)
                    probs = torch.softmax(logits, dim=-1)     # (E, 7)
                    pred_classes = torch.argmax(logits, dim=-1)  # (E,)

                for idx in range(num_edges):
                    src_id, dst_id = graph.edge_pairs_object_ids[idx]
                    cls_id = int(pred_classes[idx].item())
                    score = float(probs[idx, cls_id].item())

                    # Optional filter for 'NONE'
                    if self.filter_none and cls_id == RelationshipClass.NONE:
                        continue

                    # Confidence threshold check
                    if score < self.confidence_threshold:
                        continue

                    src_obj = obj_by_id[src_id]
                    dst_obj = obj_by_id[dst_id]

                    pred = RelationshipPrediction(
                        subject_object_id=src_id,
                        subject_class_name=src_obj.class_name,
                        target_object_id=dst_id,
                        target_class_name=dst_obj.class_name,
                        relation=RELATIONSHIP_NAMES[cls_id],
                        method="gnn",
                        score=round(score, 6),
                        details={
                            "class_id": cls_id,
                            "raw_logits": [round(float(v), 4) for v in logits[idx].tolist()],
                            "confidence_type": "softmax_score",
                            "note": "Uncalibrated softmax score; not a calibrated real-world probability.",
                        },
                    )
                    predictions.append(pred)

        return PipelineResult(
            mode=self.mode,
            predictions=predictions,
            num_objects=num_objects,
            num_candidate_edges=num_edges,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            metadata={
                "up_axis": self.up_axis,
                "confidence_threshold": self.confidence_threshold,
                "filter_none": self.filter_none,
            },
        )
