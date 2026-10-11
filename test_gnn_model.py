"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : test_gnn_model.py

PURPOSE
-------
Validation test suite for Stage C: RelationshipGNN Model.
Tests model initialization, forward pass shapes, directed edge sensitivity,
zero-edge / empty graphs, batched graphs of varying sizes, input validation,
gradient backpropagation, and checkpoint serialization round-trip.
"""

import sys
import os
import math
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from objects.object_3d import Object3D
from relationships.graph_builder import SceneGraph, SceneGraphBuilder
from relationships.gnn_model import (
    RelationshipGNN,
    BatchSceneGraph,
    RelationshipClass,
    RELATIONSHIP_NAMES,
    NUM_RELATIONSHIP_CLASSES,
)


_failures = 0


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"  [PASS]  {label}")
        if detail:
            print(f"         {detail}")
    else:
        _failures += 1
        print(f"  [FAIL]  {label}")
        if detail:
            print(f"         DETAIL: {detail}")


def expect_error(label: str, fn, expected_exception=ValueError) -> None:
    global _failures
    try:
        fn()
        _failures += 1
        print(f"  [FAIL]  {label} (did not raise {expected_exception.__name__})")
    except expected_exception as e:
        print(f"  [PASS]  {label}")
        print(f"         {type(e).__name__}: {e}")
    except Exception as e:
        _failures += 1
        print(f"  [FAIL]  {label} (raised unexpected {type(e).__name__}: {e})")


def make_mock_obj(
    object_id: int,
    class_name: str,
    semantic_class_id: int,
    center_3d: list,
    bbox_min: list,
    bbox_max: list,
    gaussian_count: int = 10,
    mean_confidence: float = 0.95,
) -> Object3D:
    return Object3D(
        object_id=object_id,
        semantic_class_id=semantic_class_id,
        class_name=class_name,
        gaussian_ids=list(range(gaussian_count)),
        gaussian_scores={i: mean_confidence for i in range(gaussian_count)},
        center_3d=[float(c) for c in center_3d],
        bbox_min=[float(c) for c in bbox_min],
        bbox_max=[float(c) for c in bbox_max],
        gaussian_count=gaussian_count,
        mean_confidence=float(mean_confidence),
    )


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


def test_1_model_initialization_and_config() -> None:
    section("1. Model Initialization and Configuration")

    model = RelationshipGNN(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=32,
        num_layers=3,
        num_classes=7,
        dropout=0.05,
    )

    check("Model instantiated successfully", isinstance(model, nn.Module))
    check("Config node_in_dim == 9", model.config["node_in_dim"] == 9)
    check("Config edge_in_dim == 10", model.config["edge_in_dim"] == 10)
    check("Config hidden_dim == 32", model.config["hidden_dim"] == 32)
    check("Config num_layers == 3", model.config["num_layers"] == 3)
    check("Config num_classes == 7", model.config["num_classes"] == 7)
    check("Number of layers in ModuleList == 3", len(model.layers) == 3)
    check("NUM_RELATIONSHIP_CLASSES constant == 7", NUM_RELATIONSHIP_CLASSES == 7)

    # Invalid configurations
    expect_error("Negative node_in_dim raises ValueError", lambda: RelationshipGNN(node_in_dim=-1))
    expect_error("Negative edge_in_dim raises ValueError", lambda: RelationshipGNN(edge_in_dim=-1))
    expect_error("Zero hidden_dim raises ValueError", lambda: RelationshipGNN(hidden_dim=0))
    expect_error("Invalid dropout >= 1.0 raises ValueError", lambda: RelationshipGNN(dropout=1.0))


def test_2_single_graph_forward_pass() -> None:
    section("2. Single Graph Forward Pass and Tensor Shapes")

    builder = SceneGraphBuilder()
    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    obj2 = make_mock_obj(2, "car", 2, [1, 1, 0.9], [0, 0, 0.2], [2, 2, 1.6])
    obj3 = make_mock_obj(3, "pole", 3, [4, 4, 2.0], [3.9, 3.9, 0.0], [4.1, 4.1, 4.0])

    graph = builder.build([obj1, obj2, obj3])
    # 3 nodes -> 6 directed edges
    check("Graph has 3 nodes and 6 edges", graph.num_nodes == 3 and graph.num_edges == 6)

    model = RelationshipGNN(hidden_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        logits = model.forward_graph(graph)

    check("Logits is a Tensor", isinstance(logits, torch.Tensor))
    check("Logits shape is (6, 7)", logits.shape == (6, 7))
    check("All logits are finite (no NaN or Inf)", bool(torch.all(torch.isfinite(logits))))


def test_3_directed_edge_sensitivity() -> None:
    section("3. Directed Edge Semantic Sensitivity")

    # In a directed graph, the prediction for (car -> road) must not be forced to match (road -> car).
    builder = SceneGraphBuilder()
    road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

    graph = builder.build([road, car])
    # Edge 0: road -> car (1 -> 2)
    # Edge 1: car -> road (2 -> 1)
    check("Graph edge pairs: (1, 2) and (2, 1)", graph.edge_pairs_object_ids == [(1, 2), (2, 1)])

    torch.manual_seed(42)
    model = RelationshipGNN(hidden_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        logits = model.forward_graph(graph)

    logits_road_to_car = logits[0]
    logits_car_to_road = logits[1]

    # They should not be identical because edge and node representations are asymmetric
    diff = torch.norm(logits_road_to_car - logits_car_to_road).item()
    check(
        f"Directed predictions are distinct (diff = {diff:.4f})",
        diff > 1e-4,
        "Ensures (subject, target) is not treated as undirected",
    )


def test_4_empty_and_zero_edge_graphs() -> None:
    section("4. Empty Graphs and Zero-Edge Handling")

    builder = SceneGraphBuilder()
    model = RelationshipGNN(hidden_dim=32, num_layers=2)
    model.eval()

    # Case A: Empty graph (0 nodes, 0 edges)
    g_empty = builder.build([])
    with torch.no_grad():
        logits_empty = model.forward_graph(g_empty)
    check("Empty graph produces shape (0, 7)", logits_empty.shape == (0, 7))

    # Case B: Single node graph (1 node, 0 edges)
    obj1 = make_mock_obj(1, "solo", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1])
    g_single = builder.build([obj1])
    with torch.no_grad():
        logits_single = model.forward_graph(g_single)
    check("Single node graph produces shape (0, 7)", logits_single.shape == (0, 7))

    # Case C: Multi-node graph with zero edges due to distance threshold
    obj2 = make_mock_obj(2, "far", 1, [100, 100, 100], [99, 99, 99], [101, 101, 101])
    builder_tight = SceneGraphBuilder(max_edge_distance=1.0)
    g_disconnected = builder_tight.build([obj1, obj2])
    check("Gated graph has 2 nodes and 0 edges", g_disconnected.num_nodes == 2 and g_disconnected.num_edges == 0)
    with torch.no_grad():
        logits_disc = model.forward_graph(g_disconnected)
    check("Disconnected graph produces shape (0, 7)", logits_disc.shape == (0, 7))


def test_5_batched_graphs_varying_sizes() -> None:
    section("5. Multiple Graphs in a Batch with Variable Sizes")

    builder = SceneGraphBuilder()
    # Graph A: 3 nodes (6 edges)
    g_a = builder.build([
        make_mock_obj(1, "a1", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1]),
        make_mock_obj(2, "a2", 2, [2, 0, 0], [1, -1, -1], [3, 1, 1]),
        make_mock_obj(3, "a3", 3, [0, 2, 0], [-1, 1, -1], [1, 3, 1]),
    ])
    # Graph B: 2 nodes (2 edges)
    g_b = builder.build([
        make_mock_obj(10, "b1", 1, [5, 5, 5], [4, 4, 4], [6, 6, 6]),
        make_mock_obj(20, "b2", 2, [7, 7, 7], [6, 6, 6], [8, 8, 8]),
    ])
    # Graph C: 1 node (0 edges)
    g_c = builder.build([
        make_mock_obj(100, "c1", 1, [0, 0, 10], [-1, -1, 9], [1, 1, 11]),
    ])

    batch = BatchSceneGraph.from_scene_graphs([g_a, g_b, g_c])

    check("Batch has 3 subgraphs", batch.num_graphs == 3)
    check("Batch total nodes == 6 (3 + 2 + 1)", batch.total_nodes == 6)
    check("Batch total edges == 8 (6 + 2 + 0)", batch.total_edges == 8)
    check("Batch edge_index shape is (2, 8)", batch.edge_index.shape == (2, 8))

    model = RelationshipGNN(hidden_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        batch_logits = model.forward_batch(batch)

    check("Batch logits shape is (8, 7)", batch_logits.shape == (8, 7))

    # Split logits
    split_logits = batch.split_edge_logits(batch_logits)
    check("Split returns 3 tensors", len(split_logits) == 3)
    check("Split 0 (Graph A) shape is (6, 7)", split_logits[0].shape == (6, 7))
    check("Split 1 (Graph B) shape is (2, 7)", split_logits[1].shape == (2, 7))
    check("Split 2 (Graph C) shape is (0, 7)", split_logits[2].shape == (0, 7))


def test_6_invalid_input_validations() -> None:
    section("6. Invalid Input Shapes and Index Validations")

    model = RelationshipGNN(node_in_dim=9, edge_in_dim=10, hidden_dim=32)

    # Valid base shapes
    nf = torch.zeros((3, 9))
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ef = torch.zeros((2, 10))

    # Non-tensor input
    expect_error("Non-tensor node_features raises TypeError", lambda: model(np.zeros((3, 9)), ei, ef), TypeError)  # type: ignore

    # Wrong node_features dimension
    expect_error("Wrong node_features dim (N, 8) raises ValueError", lambda: model(torch.zeros((3, 8)), ei, ef))

    # Wrong edge_features dimension
    expect_error("Wrong edge_features dim (E, 12) raises ValueError", lambda: model(nf, ei, torch.zeros((2, 12))))

    # Edge count mismatch
    expect_error(
        "Edge count mismatch (E=2 vs E=3) raises ValueError",
        lambda: model(nf, torch.tensor([[0, 1, 2], [1, 2, 0]]), ef),
    )

    # Out of bounds edge index (node index 99 >= 3)
    expect_error(
        "Out of bounds edge index raises ValueError",
        lambda: model(nf, torch.tensor([[0, 99], [1, 0]]), ef),
    )


def test_7_gradient_and_backprop_checks() -> None:
    section("7. Finite Outputs and Gradient Backpropagation")

    builder = SceneGraphBuilder()
    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    obj2 = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])
    graph = builder.build([road, car]) if "road" in locals() else builder.build([obj1, obj2])

    torch.manual_seed(101)
    model = RelationshipGNN(hidden_dim=32, num_layers=2)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Forward
    logits = model.forward_graph(graph)
    # Synthetic target: edge 0 -> NONE (0), edge 1 -> ON (1)
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss = criterion(logits, targets)
    check("Loss is scalar and finite", bool(torch.isfinite(loss)))

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist and are finite
    grads_valid = True
    non_zero_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                grads_valid = False
                break
            if not torch.all(torch.isfinite(param.grad)):
                grads_valid = False
                break
            if torch.any(param.grad != 0):
                non_zero_grads = True

    check("All trainable parameters received finite gradients", grads_valid)
    check("Non-zero gradients computed across layers", non_zero_grads)


def test_8_checkpoint_save_and_load() -> None:
    section("8. Checkpoint Save/Load Round-Trip")

    builder = SceneGraphBuilder()
    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0], [-2, -2, -0.1], [2, 2, 0.1])
    obj2 = make_mock_obj(2, "car", 2, [0, 0, 0.5], [-1, -1, 0.1], [1, 1, 0.9])
    graph = builder.build([obj1, obj2])

    model1 = RelationshipGNN(hidden_dim=48, num_layers=3, dropout=0.2)
    model1.eval()

    with torch.no_grad():
        logits1 = model1.forward_graph(graph)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")

        # Save
        model1.save_checkpoint(ckpt_path)
        check("Checkpoint file created", os.path.exists(ckpt_path))

        # Load
        model2 = RelationshipGNN.load_checkpoint(ckpt_path)
        check("Loaded model is RelationshipGNN instance", isinstance(model2, RelationshipGNN))
        check("Loaded model preserved config hidden_dim == 48", model2.config["hidden_dim"] == 48)
        check("Loaded model preserved config num_layers == 3", model2.config["num_layers"] == 3)

        with torch.no_grad():
            logits2 = model2.forward_graph(graph)

        check(
            "Loaded model produces identical logits",
            bool(torch.allclose(logits1, logits2, atol=1e-6)),
        )

    # Missing file raises FileNotFoundError
    expect_error("Non-existent checkpoint raises FileNotFoundError", lambda: RelationshipGNN.load_checkpoint("non_existent.pt"), FileNotFoundError)


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- Stage C: RelationshipGNN Model Unit Tests")
    print("=" * 60)

    test_1_model_initialization_and_config()
    test_2_single_graph_forward_pass()
    test_3_directed_edge_sensitivity()
    test_4_empty_and_zero_edge_graphs()
    test_5_batched_graphs_varying_sizes()
    test_6_invalid_input_validations()
    test_7_gradient_and_backprop_checks()
    test_8_checkpoint_save_and_load()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Stage C -- RelationshipGNN Model: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Stage C -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
