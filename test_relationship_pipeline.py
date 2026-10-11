"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : test_relationship_pipeline.py

PURPOSE
-------
Validation test suite for Stage E: Relationship Inference Pipeline.
Tests Mode A (geometric baseline), Mode B (trained GNN inference),
missing checkpoint rejection, empty/single-object boundary handling,
confidence and NONE thresholding, serialization compatibility with
Layer2Output (v0.1 ON-only and v0.2 multi-class), and input validations.
"""

import sys
import os
import json
import math
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from objects.object_3d import Object3D
from relationships.gnn_model import (
    RelationshipGNN,
    RelationshipClass,
    RELATIONSHIP_NAMES,
)
from relationships.gnn_dataset import generate_synthetic_dataset
from relationships.train_gnn import TrainingConfig, train_gnn_model
from relationships.relationship_pipeline import (
    RelationshipPipeline,
    RelationshipPrediction,
    PipelineResult,
)
from output.layer2_json import Layer2Output


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
        gaussian_scores={0: 0.95, 1: 0.95},
        center_3d=[float(c) for c in center_3d],
        bbox_min=[float(c) for c in bbox_min],
        bbox_max=[float(c) for c in bbox_max],
        gaussian_count=2,
        mean_confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


def test_1_baseline_mode_inference() -> None:
    section("1. Mode A: Baseline Geometric Inference")

    pipe = RelationshipPipeline(mode="baseline", up_axis="z")

    road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

    result = pipe.run([road, car])

    check("Mode is 'baseline'", result.mode == "baseline")
    check("num_objects == 2", result.num_objects == 2)
    check("num_candidate_edges == 2", result.num_candidate_edges == 2)
    check("Exactly 1 relationship detected (car ON road)", len(result.predictions) == 1)

    pred = result.predictions[0]
    check("Subject is car (ID 2)", pred.subject_object_id == 2)
    check("Target is road (ID 1)", pred.target_object_id == 1)
    check("Relation is 'ON'", pred.relation == "ON")
    check("Method is 'geometric_baseline'", pred.method == "geometric_baseline")
    check("Score is positive float in [0, 1]", 0.0 < pred.score <= 1.0)  # type: ignore


def test_2_empty_and_single_object_scenes() -> None:
    section("2. Boundary Cases: Empty and Single-Object Scenes")

    pipe = RelationshipPipeline(mode="baseline")

    # Empty list
    res_empty = pipe.run([])
    check("Empty scene returns 0 predictions", len(res_empty.predictions) == 0)
    check("Empty scene num_objects == 0", res_empty.num_objects == 0)

    # Single object
    obj = make_mock_obj(1, "solo", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1])
    res_single = pipe.run([obj])
    check("Single object returns 0 predictions", len(res_single.predictions) == 0)
    check("Single object num_objects == 1", res_single.num_objects == 1)


def test_3_missing_or_invalid_gnn_checkpoint() -> None:
    section("3. Checkpoint Validation and Rejection of Missing Models")

    # Mode 'gnn' without checkpoint_path must raise ValueError
    expect_error(
        "Missing checkpoint_path in GNN mode raises ValueError",
        lambda: RelationshipPipeline(mode="gnn", checkpoint_path=None),
    )

    # Mode 'gnn' with non-existent checkpoint must raise FileNotFoundError
    expect_error(
        "Non-existent checkpoint file raises FileNotFoundError",
        lambda: RelationshipPipeline(mode="gnn", checkpoint_path="non_existent_model.pt"),
        FileNotFoundError,
    )


def test_4_gnn_mode_inference_and_determinism() -> None:
    section("4. Mode B: Trained GNN Inference and Determinism")

    # Train a fast synthetic model fixture in a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_data = generate_synthetic_dataset(num_scenes=4, samples_per_scene=2, seed=42)

        train_result = train_gnn_model(
            train_dataset=synth_data,
            config=TrainingConfig(epochs=3, batch_size=2, checkpoint_dir=tmpdir, seed=42),
        )

        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        check("Synthetic checkpoint generated for testing", os.path.exists(ckpt_path))


        # Instantiate pipeline with trained checkpoint
        pipe_gnn = RelationshipPipeline(
            mode="gnn",
            checkpoint_path=ckpt_path,
            filter_none=True,
            confidence_threshold=0.0,
        )

        road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-10, -10, 0.0], [10, 10, 0.2])
        car1 = make_mock_obj(2, "car", 2, [0, 0, 1.0], [-1, -1, 0.2], [1, 1, 1.8])
        car2 = make_mock_obj(3, "car", 2, [3, 0, 1.0], [2, -1, 0.2], [4, 1, 1.8])

        # Run 1
        res1 = pipe_gnn.run([road, car1, car2])
        check("GNN mode produced predictions", len(res1.predictions) > 0)
        check("All predictions have method='gnn'", all(p.method == "gnn" for p in res1.predictions))
        check(
            "All predicted relations belong to defined 7 classes",
            all(p.relation in RELATIONSHIP_NAMES for p in res1.predictions),
        )

        # Run 2 (Determinism check)
        res2 = pipe_gnn.run([road, car1, car2])
        check("Deterministic prediction count", len(res1.predictions) == len(res2.predictions))
        same_predictions = all(
            p1.relation == p2.relation and math.isclose(p1.score, p2.score, rel_tol=1e-5)  # type: ignore
            for p1, p2 in zip(res1.predictions, res2.predictions)
        )
        check("Deterministic predictions and scores", same_predictions)


def test_5_none_filtering_and_confidence_threshold() -> None:
    section("5. NONE Class Filtering and Confidence Thresholding")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        model = RelationshipGNN(hidden_dim=32, num_layers=2)
        model.save_checkpoint(ckpt_path)

        road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
        car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

        # Pipeline with filter_none=False (retains all candidate edges including NONE)
        pipe_all = RelationshipPipeline(
            mode="gnn",
            checkpoint_path=ckpt_path,
            filter_none=False,
            confidence_threshold=0.0,
        )
        res_all = pipe_all.run([road, car])
        # 2 objects -> 2 candidate edges -> exactly 2 predictions (even if class is NONE)
        check("filter_none=False preserves all 2 candidate edges", len(res_all.predictions) == 2)

        # Pipeline with an impossible threshold 1.0 -> should yield 0 predictions
        pipe_strict = RelationshipPipeline(
            mode="gnn",
            checkpoint_path=ckpt_path,
            filter_none=False,
            confidence_threshold=1.0,
        )
        res_strict = pipe_strict.run([road, car])
        check("High confidence threshold filters out uncertain predictions", len(res_strict.predictions) == 0)


def test_6_serialization_and_backward_compatibility() -> None:
    section("6. Output Serialization and Backward Compatibility")

    pipe = RelationshipPipeline(mode="baseline")

    road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

    result = pipe.run([road, car])

    # 1. PipelineResult.to_dict() JSON serializability
    d = result.to_dict()
    check("PipelineResult.to_dict() returns dict", isinstance(d, dict))
    try:
        s = json.dumps(d)
        check("PipelineResult dict is JSON serialisable", isinstance(s, str))
    except Exception as e:
        check("PipelineResult dict is JSON serialisable", False, str(e))

    # 2. Strict backward compatibility with Layer2Output (v0.1)
    l2_v1 = result.to_layer2_output([road, car], include_all_relations=False)
    check("Backward-compatible Layer2Output returned", isinstance(l2_v1, Layer2Output))
    check("Layer2Output version is 0.1", l2_v1.version == "0.1")
    d_v1 = l2_v1.to_dict()
    check("Layer2Output has objects list", len(d_v1["objects"]) == 2)
    # Car (object 2) has relationship "ON"
    car_record = next(o for o in d_v1["objects"] if o["object_id"] == 2)
    check("Car object record has ON relationship", len(car_record["relationships"]) == 1)
    check("Car relation is ON", car_record["relationships"][0]["relation"] == "ON")

    # 3. Extended multi-class serialization (v0.2)
    l2_v2 = result.to_layer2_output([road, car], include_all_relations=True)
    check("Extended Layer2Output version is 0.2", l2_v2.version == "0.2")
    d_v2 = l2_v2.to_dict()
    car_record_v2 = next(o for o in d_v2["objects"] if o["object_id"] == 2)
    check("Extended relationship record contains method", "method" in car_record_v2["relationships"][0])


def test_7_input_validations_and_non_finite() -> None:
    section("7. Input Validation and Non-Finite Coords")

    pipe = RelationshipPipeline(mode="baseline")
    valid_obj = make_mock_obj(1, "valid", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1])

    # Duplicate IDs
    expect_error("Duplicate object IDs raise ValueError", lambda: pipe.run([valid_obj, valid_obj]))

    # NaN coordinates
    nan_obj = make_mock_obj(2, "nan", 2, [np.nan, 0, 0], [-1, -1, -1], [1, 1, 1])
    expect_error("NaN coordinates raise ValueError", lambda: pipe.run([valid_obj, nan_obj]))

    # Inverted bounding box
    rev_obj = make_mock_obj(3, "rev", 3, [0, 0, 0], [2, 1, 1], [1, 1, 1])
    expect_error("Reversed AABB bounds raise ValueError", lambda: pipe.run([valid_obj, rev_obj]))


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- Stage E: Relationship Pipeline Unit Tests")
    print("=" * 60)

    test_1_baseline_mode_inference()
    test_2_empty_and_single_object_scenes()
    test_3_missing_or_invalid_gnn_checkpoint()
    test_4_gnn_mode_inference_and_determinism()
    test_5_none_filtering_and_confidence_threshold()
    test_6_serialization_and_backward_compatibility()
    test_7_input_validations_and_non_finite()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Stage E -- Relationship Pipeline: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Stage E -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
