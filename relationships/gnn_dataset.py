"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/gnn_dataset.py

PURPOSE
-------
Provides dataset and batch loading infrastructure for training and validating the
RelationshipGNN model. Manages graph samples with directed edge relationship labels,
enforces strict shape and class ID validation, and prevents data leakage across
training/validation/test splits via scene-level grouping.

DATA INTEGRITY NOTICE
---------------------
No real annotated relationship data exists yet in the IIT-Bombay Layer-2 repository.
This module includes a clearly marked synthetic data generator (`generate_synthetic_dataset`)
strictly for unit testing, pipeline validation, and training loop verification.
Synthetic labels MUST NOT be claimed as real-world ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from objects.object_3d import Object3D
from relationships.graph_builder import SceneGraph, SceneGraphBuilder
from relationships.gnn_model import (
    BatchSceneGraph,
    RelationshipClass,
    RELATIONSHIP_NAMES,
    NUM_RELATIONSHIP_CLASSES,
)


# ---------------------------------------------------------------------------
# Relationship Graph Sample
# ---------------------------------------------------------------------------


@dataclass
class RelationshipGraphSample:
    """
    Container for a single 3D scene graph with optional edge-level relationship labels.

    Attributes
    ----------
    graph : SceneGraph
        The 3D SceneGraph built from Object3D instances.
    labels : Optional[torch.Tensor]
        1D int64 tensor of length `graph.num_edges` containing ground-truth
        class IDs (0..6). None for unlabeled inference graphs.
    scene_id : Optional[str]
        Identifier of the parent scene/flight to prevent cross-split leakage.
    sample_id : Optional[str]
        Unique identifier for this graph sample.
    is_synthetic : bool
        Flag explicitly declaring whether this sample is synthetically generated.
    """

    graph: SceneGraph
    labels: Optional[torch.Tensor] = None
    scene_id: Optional[str] = None
    sample_id: Optional[str] = None
    is_synthetic: bool = False

    def __post_init__(self):
        if self.labels is not None:
            if not isinstance(self.labels, torch.Tensor):
                self.labels = torch.as_tensor(self.labels, dtype=torch.long)
            else:
                self.labels = self.labels.to(dtype=torch.long)

            if self.labels.dim() != 1:
                raise ValueError(
                    f"Relationship labels must be a 1D tensor, got dim={self.labels.dim()} with shape {list(self.labels.shape)}"
                )

            num_edges = self.graph.num_edges
            if self.labels.shape[0] != num_edges:
                raise ValueError(
                    f"Label count ({self.labels.shape[0]}) does not match graph edge count ({num_edges})."
                )

            if num_edges > 0:
                min_cls = int(self.labels.min())
                max_cls = int(self.labels.max())
                if min_cls < 0 or max_cls >= NUM_RELATIONSHIP_CLASSES:
                    raise ValueError(
                        f"Relationship label values must be in [0, {NUM_RELATIONSHIP_CLASSES - 1}], "
                        f"got range [{min_cls}, {max_cls}]."
                    )

    @property
    def is_labeled(self) -> bool:
        return self.labels is not None

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        return self.graph.num_edges


# ---------------------------------------------------------------------------
# Relationship Dataset
# ---------------------------------------------------------------------------


class RelationshipDataset(Dataset):
    """
    PyTorch Dataset of RelationshipGraphSample instances.
    """

    def __init__(self, samples: List[RelationshipGraphSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RelationshipGraphSample:
        return self.samples[idx]

    @property
    def total_nodes(self) -> int:
        return sum(s.num_nodes for s in self.samples)

    @property
    def total_edges(self) -> int:
        return sum(s.num_edges for s in self.samples)

    @property
    def total_labeled_edges(self) -> int:
        return sum(s.num_edges for s in self.samples if s.is_labeled)

    def get_class_counts(self) -> Dict[int, int]:
        """Returns distribution of relationship classes across all labeled edges."""
        counts = {cls_id: 0 for cls_id in range(NUM_RELATIONSHIP_CLASSES)}
        for s in self.samples:
            if s.labels is not None and s.labels.numel() > 0:
                unique, cnts = torch.unique(s.labels, return_counts=True)
                for u, c in zip(unique.tolist(), cnts.tolist()):
                    counts[u] += c
        return counts

    def compute_class_weights(self, beta: float = 0.999) -> torch.Tensor:
        """
        Computes effective number of samples class weights for imbalanced cross-entropy loss.
        """
        counts = self.get_class_counts()
        weights = []
        for cls_id in range(NUM_RELATIONSHIP_CLASSES):
            n = counts[cls_id]
            if n > 0:
                eff = (1.0 - beta) / (1.0 - math.pow(beta, n) + 1e-8)
                weights.append(eff)
            else:
                weights.append(0.0)

        w_tensor = torch.tensor(weights, dtype=torch.float32)
        total = w_tensor.sum()
        if total > 0:
            w_tensor = (w_tensor / total) * NUM_RELATIONSHIP_CLASSES
        return w_tensor


# ---------------------------------------------------------------------------
# Collate Function for DataLoaders
# ---------------------------------------------------------------------------


def collate_relationship_graphs(
    samples: List[RelationshipGraphSample],
    device: Optional[torch.device] = None,
) -> Tuple[BatchSceneGraph, Optional[torch.Tensor]]:
    """
    Collates a list of RelationshipGraphSample instances into a BatchSceneGraph
    and a concatenated labels tensor.

    Returns
    -------
    batched_graph : BatchSceneGraph
    batched_labels : Optional[torch.Tensor], shape (E_total,) or None
    """
    if len(samples) == 0:
        empty_batch = BatchSceneGraph.from_scene_graphs([], device=device)
        return empty_batch, None

    scene_graphs = [s.graph for s in samples]
    batch_graph = BatchSceneGraph.from_scene_graphs(scene_graphs, device=device)

    all_labeled = all(s.is_labeled for s in samples)
    if all_labeled:
        label_list = []
        for s in samples:
            if s.labels is not None and s.labels.numel() > 0:
                label_list.append(s.labels.to(device=device))
        if label_list:
            batched_labels = torch.cat(label_list, dim=0)
        else:
            batched_labels = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        batched_labels = None

    return batch_graph, batched_labels


# ---------------------------------------------------------------------------
# Scene-Grouped Dataset Splitter (Zero Data Leakage)
# ---------------------------------------------------------------------------


def split_dataset_by_scene(
    dataset: RelationshipDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[RelationshipDataset, RelationshipDataset, RelationshipDataset]:
    """
    Splits samples into Train, Val, and Test sets grouped by scene_id to ensure
    no graphs from the same scene appear in both train and validation/test.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
        )

    # Group sample indices by scene_id
    scene_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        # Fallback to unique sample index if scene_id is omitted
        sid = sample.scene_id if sample.scene_id is not None else f"sample_{idx}"
        scene_to_indices.setdefault(sid, []).append(idx)

    unique_scenes = sorted(scene_to_indices.keys())
    rng = np.random.RandomState(seed)
    shuffled_scenes = list(rng.permutation(unique_scenes))

    n_scenes = len(shuffled_scenes)
    n_train = max(1, int(round(n_scenes * train_ratio))) if n_scenes > 1 else 1
    n_val = int(round(n_scenes * val_ratio)) if n_scenes > 2 else (1 if n_scenes == 2 else 0)

    train_scenes = set(shuffled_scenes[:n_train])
    val_scenes = set(shuffled_scenes[n_train : n_train + n_val])
    test_scenes = set(shuffled_scenes[n_train + n_val :])

    # Reconcile empty splits for small scene counts
    if len(train_scenes) == 0 and n_scenes > 0:
        train_scenes.add(shuffled_scenes[0])
    if len(val_scenes) == 0 and len(shuffled_scenes) > 1 and val_ratio > 0:
        val_scenes.add(shuffled_scenes[1])
        train_scenes.discard(shuffled_scenes[1])

    train_indices = [idx for sid in train_scenes for idx in scene_to_indices[sid]]
    val_indices = [idx for sid in val_scenes for idx in scene_to_indices[sid]]
    test_indices = [idx for sid in test_scenes for idx in scene_to_indices[sid]]

    train_set = RelationshipDataset([dataset.samples[i] for i in train_indices])
    val_set = RelationshipDataset([dataset.samples[i] for i in val_indices])
    test_set = RelationshipDataset([dataset.samples[i] for i in test_indices])

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator (FOR PIPELINE & UNIT TESTING ONLY)
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    num_scenes: int = 5,
    samples_per_scene: int = 2,
    seed: int = 42,
) -> RelationshipDataset:
    """
    Generates a deterministic synthetic relationship dataset.

    DISCLAIMER
    ----------
    This function generates SYNTHETIC 3D geometries and relationship labels for
    unit testing and training pipeline validation only. It does NOT represent
    real ground-truth data from the IIT-Bombay drone dataset.
    """
    rng = np.random.RandomState(seed)
    builder = SceneGraphBuilder(up_axis="z")
    samples: List[RelationshipGraphSample] = []

    for s_idx in range(num_scenes):
        scene_id = f"synthetic_scene_{s_idx:02d}"

        for sub_idx in range(samples_per_scene):
            sample_id = f"{scene_id}_sample_{sub_idx:02d}"

            # Create synthetic objects:
            # 1. Road (large flat support)
            road = Object3D(
                object_id=1,
                semantic_class_id=1,
                class_name="road",
                gaussian_ids=list(range(50)),
                gaussian_scores={i: 0.99 for i in range(50)},
                center_3d=[0.0, 0.0, 0.1],
                bbox_min=[-10.0, -10.0, 0.0],
                bbox_max=[10.0, 10.0, 0.2],
                gaussian_count=50,
                mean_confidence=0.99,
            )

            # 2. Car 1 (on road)
            car1_x = float(rng.uniform(-2.0, 2.0))
            car1 = Object3D(
                object_id=2,
                semantic_class_id=2,
                class_name="car",
                gaussian_ids=list(range(20)),
                gaussian_scores={i: 0.95 for i in range(20)},
                center_3d=[car1_x, 0.0, 1.0],
                bbox_min=[car1_x - 1.0, -1.0, 0.2],
                bbox_max=[car1_x + 1.0, 1.0, 1.8],
                gaussian_count=20,
                mean_confidence=0.95,
            )

            # 3. Car 2 (next to Car 1 on road)
            car2_x = car1_x + float(rng.choice([-3.0, 3.0]))
            car2 = Object3D(
                object_id=3,
                semantic_class_id=2,
                class_name="car",
                gaussian_ids=list(range(20)),
                gaussian_scores={i: 0.94 for i in range(20)},
                center_3d=[car2_x, 0.0, 1.0],
                bbox_min=[car2_x - 1.0, -1.0, 0.2],
                bbox_max=[car2_x + 1.0, 1.0, 1.8],
                gaussian_count=20,
                mean_confidence=0.94,
            )

            graph = builder.build([road, car1, car2])
            # Directed candidate pairs in sorted order:
            # (1, 2): road -> car1  (UNDER)
            # (1, 3): road -> car2  (UNDER)
            # (2, 1): car1 -> road  (ON)
            # (2, 3): car1 -> car2  (NEXT_TO)
            # (3, 1): car2 -> road  (ON)
            # (3, 2): car2 -> car1  (NEXT_TO)
            labels = torch.tensor(
                [
                    RelationshipClass.UNDER,
                    RelationshipClass.UNDER,
                    RelationshipClass.ON,
                    RelationshipClass.NEXT_TO,
                    RelationshipClass.ON,
                    RelationshipClass.NEXT_TO,
                ],
                dtype=torch.long,
            )

            sample = RelationshipGraphSample(
                graph=graph,
                labels=labels,
                scene_id=scene_id,
                sample_id=sample_id,
                is_synthetic=True,
            )
            samples.append(sample)

    return RelationshipDataset(samples)
