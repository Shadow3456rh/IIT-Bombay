"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : test_gnn_dataset.py

PURPOSE
-------
Validation test suite for Stage D: RelationshipDataset and Data Collation.
Tests RelationshipGraphSample validation, label bounds, empty/zero-edge graphs,
scene-grouped split leakage prevention, class distribution computation, and
the synthetic dataset generator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from objects.object_3d import Object3D
from relationships.graph_builder import SceneGraph, SceneGraphBuilder
from relationships.gnn_model import RelationshipClass, NUM_RELATIONSHIP_CLASSES
from relationships.gnn_dataset import (
    RelationshipGraphSample,
    RelationshipDataset,
    collate_relationship_graphs,
    split_dataset_by_scene,
    generate_synthetic_dataset,
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
) -> Object3D:
    return Object3D(
        object_id=object_id,
        semantic_class_id=semantic_class_id,
        class_name=class_name,
        gaussian_ids=[0, 1],
        gaussian_scores={0: 0.9, 1: 0.9},
        center_3d=[float(c) for c in center_3d],
        bbox_min=[float(c) for c in bbox_min],
        bbox_max=[float(c) for c in bbox_max],
        gaussian_count=2,
        mean_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


def test_1_sample_validation_and_labels() -> None:
    section("1. RelationshipGraphSample Validation and Label Verification")

    builder = SceneGraphBuilder()
    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0], [-2, -2, -0.1], [2, 2, 0.1])
    obj2 = make_mock_obj(2, "car", 2, [0, 0, 0.5], [-1, -1, 0.1], [1, 1, 0.9])
    graph = builder.build([obj1, obj2])  # 2 nodes, 2 edges

    # Valid labeled sample
    labels = torch.tensor([RelationshipClass.UNDER, RelationshipClass.ON], dtype=torch.long)
    sample = RelationshipGraphSample(graph=graph, labels=labels, scene_id="scene_01")

    check("Sample is labeled", sample.is_labeled == True)
    check("num_nodes == 2", sample.num_nodes == 2)
    check("num_edges == 2", sample.num_edges == 2)

    # Valid unlabeled sample
    unlabeled = RelationshipGraphSample(graph=graph, labels=None)
    check("Unlabeled sample is_labeled == False", unlabeled.is_labeled == False)

    # Label count mismatch (3 labels for 2 edges)
    expect_error(
        "Label count mismatch raises ValueError",
        lambda: RelationshipGraphSample(graph=graph, labels=torch.tensor([0, 1, 2])),
    )

    # Invalid class ID (label 99 out of range 0..6)
    expect_error(
        "Class ID out of bounds raises ValueError",
        lambda: RelationshipGraphSample(graph=graph, labels=torch.tensor([0, 99])),
    )

    # Negative class ID (-1)
    expect_error(
        "Negative class ID raises ValueError",
        lambda: RelationshipGraphSample(graph=graph, labels=torch.tensor([-1, 0])),
    )

    # 2D labels tensor
    expect_error(
        "Multi-dimensional labels raise ValueError",
        lambda: RelationshipGraphSample(graph=graph, labels=torch.zeros((2, 1), dtype=torch.long)),
    )


def test_2_empty_and_zero_edge_samples() -> None:
    section("2. Empty and Zero-Edge Samples")

    builder = SceneGraphBuilder()

    # Empty graph
    g_empty = builder.build([])
    sample_empty = RelationshipGraphSample(graph=g_empty, labels=torch.zeros(0, dtype=torch.long))
    check("Empty sample num_nodes == 0", sample_empty.num_nodes == 0)
    check("Empty sample num_edges == 0", sample_empty.num_edges == 0)

    # Single node graph (0 candidate edges)
    obj = make_mock_obj(1, "solo", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1])
    g_single = builder.build([obj])
    sample_single = RelationshipGraphSample(graph=g_single, labels=torch.zeros(0, dtype=torch.long))
    check("Single node sample num_nodes == 1", sample_single.num_nodes == 1)
    check("Single node sample num_edges == 0", sample_single.num_edges == 0)


def test_3_collation_and_batching() -> None:
    section("3. Variable Graph Sizes and Collation")

    builder = SceneGraphBuilder()
    # Sample A: 2 nodes, 2 edges
    g_a = builder.build([
        make_mock_obj(1, "a1", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1]),
        make_mock_obj(2, "a2", 2, [2, 0, 0], [1, -1, -1], [3, 1, 1]),
    ])
    s_a = RelationshipGraphSample(graph=g_a, labels=torch.tensor([0, 1]))

    # Sample B: 3 nodes, 6 edges
    g_b = builder.build([
        make_mock_obj(10, "b1", 1, [5, 5, 5], [4, 4, 4], [6, 6, 6]),
        make_mock_obj(20, "b2", 2, [7, 7, 7], [6, 6, 6], [8, 8, 8]),
        make_mock_obj(30, "b3", 3, [5, 7, 5], [4, 6, 4], [6, 8, 6]),
    ])
    s_b = RelationshipGraphSample(graph=g_b, labels=torch.tensor([0, 1, 2, 3, 4, 5]))

    # Sample C: 1 node, 0 edges
    g_c = builder.build([make_mock_obj(100, "c1", 1, [0, 0, 10], [-1, -1, 9], [1, 1, 11])])
    s_c = RelationshipGraphSample(graph=g_c, labels=torch.zeros(0, dtype=torch.long))

    batch_graph, batched_labels = collate_relationship_graphs([s_a, s_b, s_c])

    check("Batched graph total_nodes == 6", batch_graph.total_nodes == 6)
    check("Batched graph total_edges == 8", batch_graph.total_edges == 8)
    check("Batched labels is not None", batched_labels is not None)
    check("Batched labels shape is (8,)", batched_labels.shape == (8,))
    expected_labels = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5], dtype=torch.long)
    check("Batched labels match order", bool(torch.equal(batched_labels, expected_labels)))


def test_4_scene_grouped_split_leakage_prevention() -> None:
    section("4. Scene-Grouped Splitting and Zero Data Leakage")

    builder = SceneGraphBuilder()
    samples = []
    # 6 scenes, 2 samples per scene = 12 samples
    for s_id in range(6):
        scene_name = f"flight_{s_id}"
        for sample_sub in range(2):
            g = builder.build([
                make_mock_obj(1, "a", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1]),
                make_mock_obj(2, "b", 2, [2, 0, 0], [1, -1, -1], [3, 1, 1]),
            ])
            samples.append(
                RelationshipGraphSample(
                    graph=g,
                    labels=torch.tensor([0, 1]),
                    scene_id=scene_name,
                    sample_id=f"{scene_name}_frame_{sample_sub}",
                )
            )

    dataset = RelationshipDataset(samples)
    train_set, val_set, test_set = split_dataset_by_scene(
        dataset,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
    )

    train_scenes = {s.scene_id for s in train_set}
    val_scenes = {s.scene_id for s in val_set}
    test_scenes = {s.scene_id for s in test_set}

    # Verify zero leakage across splits
    overlap_train_val = train_scenes.intersection(val_scenes)
    overlap_train_test = train_scenes.intersection(test_scenes)
    overlap_val_test = val_scenes.intersection(test_scenes)

    check("Zero overlap between Train and Val scenes", len(overlap_train_val) == 0)
    check("Zero overlap between Train and Test scenes", len(overlap_train_test) == 0)
    check("Zero overlap between Val and Test scenes", len(overlap_val_test) == 0)
    check(
        "All scenes accounted for across splits",
        len(train_scenes) + len(val_scenes) + len(test_scenes) == 6,
    )
    check("Total samples preserved (12)", len(train_set) + len(val_set) + len(test_set) == 12)


def test_5_class_distribution_and_weights() -> None:
    section("5. Class Distribution and Weights")

    builder = SceneGraphBuilder()
    g = builder.build([
        make_mock_obj(1, "a", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1]),
        make_mock_obj(2, "b", 2, [2, 0, 0], [1, -1, -1], [3, 1, 1]),
    ])
    # 2 edges: class 1 (ON) and class 4 (NEXT_TO)
    s = RelationshipGraphSample(graph=g, labels=torch.tensor([1, 4]))
    dataset = RelationshipDataset([s])

    counts = dataset.get_class_counts()
    check("Class 1 count == 1", counts[1] == 1)
    check("Class 4 count == 1", counts[4] == 1)
    check("Class 0 count == 0", counts[0] == 0)

    weights = dataset.compute_class_weights()
    check("Weights tensor shape is (7,)", weights.shape == (NUM_RELATIONSHIP_CLASSES,))
    check("Weights are non-negative and finite", bool(torch.all(weights >= 0) and torch.all(torch.isfinite(weights))))


def test_6_synthetic_dataset_generator() -> None:
    section("6. Synthetic Dataset Generator Verification")

    dataset = generate_synthetic_dataset(num_scenes=4, samples_per_scene=2, seed=99)

    check("Synthetic dataset length is 8", len(dataset) == 8)
    check("All samples marked as is_synthetic == True", all(s.is_synthetic for s in dataset))
    check("Total labeled edges > 0", dataset.total_labeled_edges == 8 * 6)  # 6 edges per sample

    sample0 = dataset[0]
    check("Sample 0 has scene_id set", sample0.scene_id is not None)
    check("Sample 0 labels shape is (6,)", sample0.labels.shape == (6,))


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- Stage D: Relationship Dataset Unit Tests")
    print("=" * 60)

    test_1_sample_validation_and_labels()
    test_2_empty_and_zero_edge_samples()
    test_3_collation_and_batching()
    test_4_scene_grouped_split_leakage_prevention()
    test_5_class_distribution_and_weights()
    test_6_synthetic_dataset_generator()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Stage D -- Relationship Dataset: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Stage D -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
