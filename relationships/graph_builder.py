"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/graph_builder.py

PURPOSE
-------
Constructs a structured 3D Scene Graph representation from a collection of
Object3D instances. Computes node features, directed candidate edges, and relative
geometric edge features. Evaluates baseline geometric 'ON' relationship evidence
using ONRelationDetector while keeping heuristic evidence distinct from learned
representations.

LIMITATIONS
-----------
- Geometric edge features rely on AABB extents and centroids, not fine 3D meshes.
- Candidate edge selection uses distance gating, not learned edge filtering.
- Does not assign learned relationship predictions (deferred to GNN model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from objects.object_3d import Object3D
from relationships.on_relation import ONRelationDetector, ONRelationResult


NODE_FEATURE_DIM = 9
EDGE_FEATURE_DIM = 10

NODE_FEATURE_NAMES = [
    "center_x",
    "center_y",
    "center_z",
    "extent_x",
    "extent_y",
    "extent_z",
    "log_gaussian_count",
    "mean_confidence",
    "semantic_class_id",
]

EDGE_FEATURE_NAMES = [
    "disp_x",
    "disp_y",
    "disp_z",
    "distance_3d",
    "distance_horizontal",
    "vertical_offset",
    "clearance_x",
    "clearance_y",
    "clearance_z",
    "log_volume_ratio",
]


@dataclass(frozen=True)
class SceneGraph:
    """
    Structured representation of a 3D scene graph.

    Attributes
    ----------
    node_features : np.ndarray, shape (N, 9), dtype float32
        Per-object feature matrix.
    edge_index : np.ndarray, shape (2, E), dtype int64
        Directed edges as source and destination node indices [0, ..., N-1].
        Row 0 is source node indices, row 1 is destination node indices.
    edge_features : np.ndarray, shape (E, 10), dtype float32
        Relative geometric feature matrix for each candidate edge.
    node_object_ids : List[int]
        Maps node index 0..N-1 to the persistent Object3D.object_id.
    object_id_to_idx : Dict[int, int]
        Maps persistent Object3D.object_id to node index 0..N-1.
    edge_pairs_object_ids : List[Tuple[int, int]]
        List of (src_object_id, dst_object_id) for each edge 0..E-1.
    heuristic_on_results : Dict[Tuple[int, int], ONRelationResult]
        Baseline geometric ON heuristic results indexed by (src_object_id, dst_object_id).
    up_axis : str
        The vertical axis convention used ('x', 'y', or 'z').
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_object_ids: List[int]
    object_id_to_idx: Dict[int, int]
    edge_pairs_object_ids: List[Tuple[int, int]]
    heuristic_on_results: Dict[Tuple[int, int], ONRelationResult]
    up_axis: str = "z"

    @property
    def num_nodes(self) -> int:
        return len(self.node_object_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_pairs_object_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph topology, feature shapes, and heuristic evidence to a dictionary."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "up_axis": self.up_axis,
            "node_object_ids": list(self.node_object_ids),
            "edge_pairs_object_ids": [list(pair) for pair in self.edge_pairs_object_ids],
            "node_features_shape": list(self.node_features.shape),
            "edge_features_shape": list(self.edge_features.shape),
            "heuristic_on_count": sum(1 for r in self.heuristic_on_results.values() if r.is_on),
        }


class SceneGraphBuilder:
    """
    Constructs a SceneGraph from Object3D instances.
    """

    def __init__(
        self,
        up_axis: Literal["x", "y", "z"] = "z",
        max_edge_distance: Optional[float] = None,
        on_detector: Optional[ONRelationDetector] = None,
    ):
        """
        Initialize the SceneGraphBuilder.

        Parameters
        ----------
        up_axis : Literal["x", "y", "z"]
            Vertical axis for directional features. Default 'z'.
        max_edge_distance : Optional[float]
            Maximum Euclidean distance between centroids to create an edge.
            If None, all directed pairs (i, j) with i != j are connected.
        on_detector : Optional[ONRelationDetector]
            Custom detector for baseline geometric ON relationship extraction.
            If None, a default ONRelationDetector(up_axis=up_axis) is instantiated.
        """
        if up_axis not in ("x", "y", "z"):
            raise ValueError(f"up_axis must be 'x', 'y', or 'z', got {up_axis!r}")
        if max_edge_distance is not None and max_edge_distance < 0:
            raise ValueError(f"max_edge_distance must be >= 0, got {max_edge_distance}")

        self.up_axis = up_axis
        self.max_edge_distance = max_edge_distance
        self.up_idx = {"x": 0, "y": 1, "z": 2}[self.up_axis]
        self.horiz_axes = [i for i in range(3) if i != self.up_idx]

        if on_detector is not None:
            self.on_detector = on_detector
        else:
            self.on_detector = ONRelationDetector(up_axis=self.up_axis)

    def _validate_object(self, obj: Object3D) -> None:
        """Validates numerical integrity of Object3D geometry."""
        coords = list(obj.center_3d) + list(obj.bbox_min) + list(obj.bbox_max)
        if any(not np.isfinite(c) for c in coords):
            raise ValueError(
                f"Object3D ID {obj.object_id} contains non-finite coordinates in center_3d or bbox."
            )
        if any(obj.bbox_min[i] > obj.bbox_max[i] for i in range(3)):
            raise ValueError(
                f"Object3D ID {obj.object_id} has invalid AABB bounds (bbox_min > bbox_max)."
            )

    def _compute_node_feature(self, obj: Object3D) -> np.ndarray:
        """
        Computes the 9-dimensional node feature vector:
        [center_x, center_y, center_z, extent_x, extent_y, extent_z,
         log_gaussian_count, mean_confidence, semantic_class_id]
        """
        extent_x = obj.bbox_max[0] - obj.bbox_min[0]
        extent_y = obj.bbox_max[1] - obj.bbox_min[1]
        extent_z = obj.bbox_max[2] - obj.bbox_min[2]
        log_count = math.log(1.0 + max(0, obj.gaussian_count))
        mean_conf = float(obj.mean_confidence) if obj.mean_confidence is not None else 0.0

        return np.array(
            [
                float(obj.center_3d[0]),
                float(obj.center_3d[1]),
                float(obj.center_3d[2]),
                float(extent_x),
                float(extent_y),
                float(extent_z),
                float(log_count),
                float(mean_conf),
                float(obj.semantic_class_id),
            ],
            dtype=np.float32,
        )

    def _compute_edge_feature(self, src: Object3D, dst: Object3D) -> np.ndarray:
        """
        Computes the 10-dimensional directed edge feature vector:
        1-3: Relative displacement (dst - src)
        4:   3D Euclidean distance
        5:   Horizontal distance
        6:   Signed vertical offset along up_axis (dst - src)
        7-9: AABB clearance / penetration depth along each axis
        10:  Log volume ratio log((V_dst + 1e-6) / (V_src + 1e-6))
        """
        c_src = np.array(src.center_3d, dtype=np.float64)
        c_dst = np.array(dst.center_3d, dtype=np.float64)
        disp = c_dst - c_src

        dist_3d = float(np.linalg.norm(disp))
        horiz_disp = disp[self.horiz_axes]
        dist_horiz = float(np.linalg.norm(horiz_disp))
        vert_offset = float(disp[self.up_idx])

        # AABB clearance along each axis:
        # If separated, positive gap (overlap_min - overlap_max > 0)
        # If overlapping, negative penetration (overlap_min - overlap_max < 0)
        clearances = []
        for k in range(3):
            overlap_min = max(src.bbox_min[k], dst.bbox_min[k])
            overlap_max = min(src.bbox_max[k], dst.bbox_max[k])
            clearance_k = overlap_min - overlap_max
            clearances.append(float(clearance_k))

        # Volumes
        extent_src = np.array(src.bbox_max) - np.array(src.bbox_min)
        extent_dst = np.array(dst.bbox_max) - np.array(dst.bbox_min)
        vol_src = float(np.prod(np.maximum(extent_src, 0.0)))
        vol_dst = float(np.prod(np.maximum(extent_dst, 0.0)))
        log_vol_ratio = float(math.log((vol_dst + 1e-6) / (vol_src + 1e-6)))

        return np.array(
            [
                float(disp[0]),
                float(disp[1]),
                float(disp[2]),
                float(dist_3d),
                float(dist_horiz),
                float(vert_offset),
                clearances[0],
                clearances[1],
                clearances[2],
                log_vol_ratio,
            ],
            dtype=np.float32,
        )

    def build(self, objects: List[Object3D]) -> SceneGraph:
        """
        Builds a SceneGraph from the given list of Object3D instances.

        Parameters
        ----------
        objects : List[Object3D]
            Collection of 3D objects.

        Returns
        -------
        SceneGraph
        """
        # Validate unique IDs and numerical sanity
        seen_ids = set()
        for obj in objects:
            if obj.object_id in seen_ids:
                raise ValueError(f"Duplicate Object3D ID detected: {obj.object_id}")
            seen_ids.add(obj.object_id)
            self._validate_object(obj)

        # Deterministic sorting by object_id
        sorted_objects = sorted(objects, key=lambda o: o.object_id)
        num_nodes = len(sorted_objects)

        node_object_ids = [obj.object_id for obj in sorted_objects]
        object_id_to_idx = {obj.object_id: idx for idx, obj in enumerate(sorted_objects)}

        # Edge case: empty scene
        if num_nodes == 0:
            return SceneGraph(
                node_features=np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_features=np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
                node_object_ids=[],
                object_id_to_idx={},
                edge_pairs_object_ids=[],
                heuristic_on_results={},
                up_axis=self.up_axis,
            )

        # Compute node features
        node_features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        for idx, obj in enumerate(sorted_objects):
            node_features[idx] = self._compute_node_feature(obj)

        # Edge case: single object (no candidate edges possible without self-loops)
        if num_nodes == 1:
            return SceneGraph(
                node_features=node_features,
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_features=np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
                node_object_ids=node_object_ids,
                object_id_to_idx=object_id_to_idx,
                edge_pairs_object_ids=[],
                heuristic_on_results={},
                up_axis=self.up_axis,
            )

        # Build candidate directed edges
        src_indices: List[int] = []
        dst_indices: List[int] = []
        edge_feat_list: List[np.ndarray] = []
        edge_pairs_object_ids: List[Tuple[int, int]] = []
        heuristic_on_results: Dict[Tuple[int, int], ONRelationResult] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # strictly avoid self-loops

                src_obj = sorted_objects[i]
                dst_obj = sorted_objects[j]

                # Distance threshold filtering
                disp = np.array(dst_obj.center_3d) - np.array(src_obj.center_3d)
                dist_3d = float(np.linalg.norm(disp))

                if self.max_edge_distance is not None and dist_3d > self.max_edge_distance:
                    continue

                feat = self._compute_edge_feature(src_obj, dst_obj)

                src_indices.append(i)
                dst_indices.append(j)
                edge_feat_list.append(feat)
                pair = (src_obj.object_id, dst_obj.object_id)
                edge_pairs_object_ids.append(pair)

                # Baseline ON relationship heuristic calculation
                on_res = self.on_detector.detect(subject=src_obj, support=dst_obj)
                heuristic_on_results[pair] = on_res

        num_edges = len(src_indices)
        if num_edges > 0:
            edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
            edge_features = np.stack(edge_feat_list, axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

        return SceneGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_object_ids=node_object_ids,
            object_id_to_idx=object_id_to_idx,
            edge_pairs_object_ids=edge_pairs_object_ids,
            heuristic_on_results=heuristic_on_results,
            up_axis=self.up_axis,
        )
