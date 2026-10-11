"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/gnn_model.py

PURPOSE
-------
Implements a self-contained PyTorch Message-Passing Graph Neural Network (MPNN)
for predicting directed 3D spatial and semantic relationships between pairs of
Object3D instances. Consumes node and edge features produced by `SceneGraphBuilder`.

RELATIONSHIP CLASSES
--------------------
- NONE       = 0 (no significant spatial relationship)
- ON         = 1 (subject is supported on top of target)
- UNDER      = 2 (subject is positioned beneath target)
- INSIDE     = 3 (subject is contained within target)
- NEXT_TO    = 4 (subject is laterally adjacent to target)
- IN_FRONT_OF = 5 (subject is in front of target)
- BEHIND     = 6 (subject is behind target)

IMPORTANT DISCLAIMER
--------------------
An untrained or synthetically initialized instance of this network produces
uncalibrated logits. Raw model outputs MUST NOT be treated as verified predictions
until trained on validated ground-truth annotations.
"""

from __future__ import annotations

import os
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from relationships.graph_builder import (
    SceneGraph,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Relationship Classes Specification
# ---------------------------------------------------------------------------


class RelationshipClass(IntEnum):
    NONE = 0
    ON = 1
    UNDER = 2
    INSIDE = 3
    NEXT_TO = 4
    IN_FRONT_OF = 5
    BEHIND = 6


RELATIONSHIP_NAMES: List[str] = [
    "NONE",
    "ON",
    "UNDER",
    "INSIDE",
    "NEXT_TO",
    "IN_FRONT_OF",
    "BEHIND",
]

NUM_RELATIONSHIP_CLASSES: int = len(RELATIONSHIP_NAMES)


# ---------------------------------------------------------------------------
# Batching Container
# ---------------------------------------------------------------------------


class BatchSceneGraph:
    """
    Collation container for batching multiple SceneGraph instances with variable
    numbers of nodes and candidate edges into a single disjoint graph.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_offsets: List[int],
        edge_offsets: List[int],
        num_nodes_per_graph: List[int],
        num_edges_per_graph: List[int],
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_offsets = node_offsets
        self.edge_offsets = edge_offsets
        self.num_nodes_per_graph = num_nodes_per_graph
        self.num_edges_per_graph = num_edges_per_graph

    @property
    def num_graphs(self) -> int:
        return len(self.num_nodes_per_graph)

    @property
    def total_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def total_edges(self) -> int:
        return int(self.edge_features.shape[0])

    @classmethod
    def from_scene_graphs(
        cls,
        graphs: List[SceneGraph],
        device: Optional[torch.device] = None,
    ) -> BatchSceneGraph:
        """
        Collates a list of SceneGraph instances into a single BatchSceneGraph.

        Disjoint node indexing is achieved by shifting edge_index for each graph
        by the cumulative number of preceding nodes.
        """
        if len(graphs) == 0:
            return cls(
                node_features=torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32, device=device),
                edge_index=torch.zeros((2, 0), dtype=torch.int64, device=device),
                edge_features=torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device),
                node_offsets=[0],
                edge_offsets=[0],
                num_nodes_per_graph=[],
                num_edges_per_graph=[],
            )

        node_feat_list = []
        edge_feat_list = []
        edge_index_list = []
        node_offsets = [0]
        edge_offsets = [0]
        num_nodes_per_graph = []
        num_edges_per_graph = []

        running_node_offset = 0
        running_edge_offset = 0

        for g in graphs:
            n_nodes = g.num_nodes
            n_edges = g.num_edges
            num_nodes_per_graph.append(n_nodes)
            num_edges_per_graph.append(n_edges)

            if n_nodes > 0:
                t_nf = torch.from_numpy(g.node_features).to(dtype=torch.float32, device=device)
                node_feat_list.append(t_nf)

            if n_edges > 0:
                t_ef = torch.from_numpy(g.edge_features).to(dtype=torch.float32, device=device)
                edge_feat_list.append(t_ef)

                t_ei = torch.from_numpy(g.edge_index).to(dtype=torch.int64, device=device)
                # Shift node indices by running offset
                t_ei_shifted = t_ei + running_node_offset
                edge_index_list.append(t_ei_shifted)

            running_node_offset += n_nodes
            running_edge_offset += n_edges
            node_offsets.append(running_node_offset)
            edge_offsets.append(running_edge_offset)

        if node_feat_list:
            batched_node_features = torch.cat(node_feat_list, dim=0)
        else:
            batched_node_features = torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32, device=device)

        if edge_feat_list:
            batched_edge_features = torch.cat(edge_feat_list, dim=0)
            batched_edge_index = torch.cat(edge_index_list, dim=1)
        else:
            batched_edge_features = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)
            batched_edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)

        return cls(
            node_features=batched_node_features,
            edge_index=batched_edge_index,
            edge_features=batched_edge_features,
            node_offsets=node_offsets,
            edge_offsets=edge_offsets,
            num_nodes_per_graph=num_nodes_per_graph,
            num_edges_per_graph=num_edges_per_graph,
        )

    def split_edge_logits(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """Splits batched edge logits of shape (E_total, C) back into per-graph tensors."""
        per_graph_logits = []
        for i in range(self.num_graphs):
            start = self.edge_offsets[i]
            end = self.edge_offsets[i + 1]
            per_graph_logits.append(logits[start:end])
        return per_graph_logits


# ---------------------------------------------------------------------------
# Message-Passing Layer
# ---------------------------------------------------------------------------


class MessagePassingLayer(nn.Module):
    """
    Single directed message-passing step using native PyTorch operations.

    For directed edge (u, v):
      u: source node (subject)
      v: target node (support/object)
      message_{u -> v} = MLP_msg([h_u, h_v, e_{u -> v}])
      aggregate at v:  m_v = sum_{u in N_in(v)} message_{u -> v}
      update node v:   h_v' = MLP_upd([h_v, m_v]) + residual(h_v)
      update edge:     e_{u -> v}' = MLP_edge([h_u', h_v', e_{u -> v}])
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generator from source, target, and edge features
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update from old node state and aggregated incoming messages
        self.node_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge update from updated node states and existing edge state
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.edge_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        h_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        h_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = h_nodes.shape[0]
        num_edges = h_edges.shape[0]

        if num_edges == 0:
            return h_nodes, h_edges

        src_indices = edge_index[0]  # shape (E,)
        dst_indices = edge_index[1]  # shape (E,)

        h_src = h_nodes[src_indices]  # (E, D)
        h_dst = h_nodes[dst_indices]  # (E, D)

        # Directed message: [subject, support, edge]
        msg_input = torch.cat([h_src, h_dst, h_edges], dim=-1)  # (E, 3*D)
        messages = self.msg_mlp(msg_input)  # (E, D)

        # Aggregate incoming messages at target node dst_indices
        aggregated = torch.zeros(
            (num_nodes, self.hidden_dim),
            dtype=h_nodes.dtype,
            device=h_nodes.device,
        )
        aggregated.index_add_(0, dst_indices, messages)

        # Update node representations
        node_input = torch.cat([h_nodes, aggregated], dim=-1)
        h_nodes_new = h_nodes + self.node_update_mlp(node_input)
        h_nodes_new = self.node_norm(h_nodes_new)

        # Update edge representations using updated endpoints
        h_src_new = h_nodes_new[src_indices]
        h_dst_new = h_nodes_new[dst_indices]
        edge_input = torch.cat([h_src_new, h_dst_new, h_edges], dim=-1)
        h_edges_new = h_edges + self.edge_update_mlp(edge_input)
        h_edges_new = self.edge_norm(h_edges_new)

        return h_nodes_new, h_edges_new


# ---------------------------------------------------------------------------
# GNN Relationship Model
# ---------------------------------------------------------------------------


class RelationshipGNN(nn.Module):
    """
    Message-Passing Neural Network for predicting directed 3D spatial relationships.
    """

    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = NUM_RELATIONSHIP_CLASSES,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if node_in_dim <= 0:
            raise ValueError(f"node_in_dim must be positive, got {node_in_dim}")
        if edge_in_dim <= 0:
            raise ValueError(f"edge_in_dim must be positive, got {edge_in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.config: Dict[str, Any] = {
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
            "use_layer_norm": use_layer_norm,
        }

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Initial Feature Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message Passing Layers
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Relationship Prediction Head (per directed edge)
        # Input: [h_src, h_dst, h_edge]
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _validate_inputs(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> None:
        """Validates tensor shapes, dimensions, and indexing validity."""
        if not isinstance(node_features, torch.Tensor):
            raise TypeError(f"node_features must be a torch.Tensor, got {type(node_features).__name__}")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"edge_index must be a torch.Tensor, got {type(edge_index).__name__}")
        if not isinstance(edge_features, torch.Tensor):
            raise TypeError(f"edge_features must be a torch.Tensor, got {type(edge_features).__name__}")

        if node_features.dim() != 2:
            raise ValueError(f"node_features must be 2D of shape (N, {self.node_in_dim}), got dim={node_features.dim()}")
        if node_features.shape[1] != self.node_in_dim:
            raise ValueError(
                f"node_features feature dimension must be {self.node_in_dim}, got {node_features.shape[1]}"
            )

        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {list(edge_index.shape)}")

        if edge_features.dim() != 2:
            raise ValueError(f"edge_features must be 2D of shape (E, {self.edge_in_dim}), got dim={edge_features.dim()}")
        if edge_features.shape[1] != self.edge_in_dim:
            raise ValueError(
                f"edge_features feature dimension must be {self.edge_in_dim}, got {edge_features.shape[1]}"
            )

        num_edges = edge_features.shape[0]
        if edge_index.shape[1] != num_edges:
            raise ValueError(
                f"Mismatch between edge_index count ({edge_index.shape[1]}) and edge_features count ({num_edges})"
            )

        num_nodes = node_features.shape[0]
        if num_edges > 0:
            min_idx = int(edge_index.min())
            max_idx = int(edge_index.max())
            if min_idx < 0 or max_idx >= num_nodes:
                raise ValueError(
                    f"edge_index out of bounds: contains index range [{min_idx}, {max_idx}], "
                    f"but graph only has {num_nodes} nodes."
                )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single graph or batched graph.

        Parameters
        ----------
        node_features : torch.Tensor, shape (N, node_in_dim)
        edge_index : torch.Tensor, shape (2, E)
        edge_features : torch.Tensor, shape (E, edge_in_dim)

        Returns
        -------
        logits : torch.Tensor, shape (E, num_classes)
        """
        self._validate_inputs(node_features, edge_index, edge_features)

        num_nodes = node_features.shape[0]
        num_edges = edge_features.shape[0]

        # Handle zero edges (empty graph or single/unconnected nodes)
        if num_edges == 0:
            return torch.zeros(
                (0, self.num_classes),
                dtype=node_features.dtype,
                device=node_features.device,
            )

        # 1. Initial Encodings
        h_nodes = self.node_encoder(node_features)
        h_edges = self.edge_encoder(edge_features)

        # 2. Message Passing
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, edge_index, h_edges)

        # 3. Directed Edge Classification Head
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        h_src = h_nodes[src_idx]
        h_dst = h_nodes[dst_idx]

        # Explicit directed ordering: [subject, target, edge]
        pair_repr = torch.cat([h_src, h_dst, h_edges], dim=-1)
        logits = self.classifier(pair_repr)  # (E, num_classes)

        return logits

    def forward_graph(
        self,
        graph: SceneGraph,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Helper to run inference directly on a SceneGraph instance."""
        node_features = torch.from_numpy(graph.node_features).to(dtype=torch.float32, device=device)
        edge_index = torch.from_numpy(graph.edge_index).to(dtype=torch.int64, device=device)
        edge_features = torch.from_numpy(graph.edge_features).to(dtype=torch.float32, device=device)

        return self.forward(node_features, edge_index, edge_features)

    def forward_batch(self, batch: BatchSceneGraph) -> torch.Tensor:
        """Helper to run inference on a BatchSceneGraph."""
        return self.forward(batch.node_features, batch.edge_index, batch.edge_features)

    # -----------------------------------------------------------------------
    # Checkpoint Serialization Helpers
    # -----------------------------------------------------------------------

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """
        Saves model weights and full configuration to a checkpoint file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_config": self.config,
            "state_dict": self.state_dict(),
            "relationship_names": RELATIONSHIP_NAMES,
            "version": "0.1.0",
        }
        torch.save(payload, str(path))

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> RelationshipGNN:
        """
        Loads a checkpoint and instantiates the configured RelationshipGNN model.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        if "model_config" not in payload or "state_dict" not in payload:
            raise KeyError("Invalid checkpoint format: missing 'model_config' or 'state_dict'.")

        config = payload["model_config"]
        model = cls(**config)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model
