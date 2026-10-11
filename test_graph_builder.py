"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : test_graph_builder.py

PURPOSE
-------
Validation test suite for Stage B: 3D Scene Graph Builder.
Tests node feature extraction, directed edge construction, relative geometric
edge features, baseline ON heuristic integration, edge cases (empty, single node),
input validations, determinism, and configurable up-axis / distance threshold.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from objects.object_3d import Object3D
from relationships.graph_builder import (
    SceneGraph,
    SceneGraphBuilder,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from relationships.on_relation import ONRelationDetector


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


def test_1_empty_and_single_node() -> None:
    section("1. Empty Scene and Single Node Graph")

    builder = SceneGraphBuilder()

    # Empty graph
    g_empty = builder.build([])
    check("Empty graph num_nodes == 0", g_empty.num_nodes == 0)
    check("Empty graph num_edges == 0", g_empty.num_edges == 0)
    check(
        "Empty node_features shape is (0, 9)",
        g_empty.node_features.shape == (0, NODE_FEATURE_DIM),
    )
    check("Empty edge_index shape is (2, 0)", g_empty.edge_index.shape == (2, 0))
    check(
        "Empty edge_features shape is (0, 10)",
        g_empty.edge_features.shape == (0, EDGE_FEATURE_DIM),
    )
    check("Empty node_object_ids is empty list", g_empty.node_object_ids == [])
    check("Empty heuristic_on_results is empty dict", g_empty.heuristic_on_results == {})

    # Single node graph
    obj1 = make_mock_obj(
        1, "car", 3, [1.0, 2.0, 0.5], [0.0, 1.0, 0.0], [2.0, 3.0, 1.0], gaussian_count=15
    )
    g_single = builder.build([obj1])
    check("Single node num_nodes == 1", g_single.num_nodes == 1)
    check("Single node num_edges == 0", g_single.num_edges == 0)
    check(
        "Single node_features shape is (1, 9)",
        g_single.node_features.shape == (1, NODE_FEATURE_DIM),
    )
    check("Single edge_index shape is (2, 0)", g_single.edge_index.shape == (2, 0))
    check(
        "Single edge_features shape is (0, 10)",
        g_single.edge_features.shape == (0, EDGE_FEATURE_DIM),
    )
    check("Single node_object_ids contains obj1", g_single.node_object_ids == [1])
    check("Single object_id_to_idx correctly maps 1 -> 0", g_single.object_id_to_idx == {1: 0})


def test_2_node_feature_correctness() -> None:
    section("2. Node Feature Vector Correctness (9 Dimensions)")

    builder = SceneGraphBuilder()
    obj = make_mock_obj(
        object_id=42,
        class_name="truck",
        semantic_class_id=5,
        center_3d=[10.0, 20.0, 30.0],
        bbox_min=[8.0, 17.0, 28.0],
        bbox_max=[12.0, 23.0, 32.0],
        gaussian_count=99,
        mean_confidence=0.876,
    )

    g = builder.build([obj])
    feat = g.node_features[0]

    check("Feature vector length is 9", len(feat) == NODE_FEATURE_DIM)
    check("feat[0..2] is center_3d [10, 20, 30]", np.allclose(feat[0:3], [10.0, 20.0, 30.0]))
    check("feat[3..5] is extent [4, 6, 4]", np.allclose(feat[3:6], [4.0, 6.0, 4.0]))
    expected_log_count = math.log(1.0 + 99)
    check("feat[6] is log(1 + count)", math.isclose(feat[6], expected_log_count, rel_tol=1e-5))
    check("feat[7] is mean_confidence (0.876)", math.isclose(feat[7], 0.876, abs_tol=1e-4))
    check("feat[8] is semantic_class_id (5.0)", math.isclose(feat[8], 5.0))


def test_3_directed_edges_and_no_self_loops() -> None:
    section("3. Candidate Directed Edges and No Self-Loops")

    builder = SceneGraphBuilder()
    # 3 objects -> 3 * 2 = 6 directed edges
    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0], [-5, -5, -0.1], [5, 5, 0.1])
    obj2 = make_mock_obj(2, "car", 2, [1, 1, 0.5], [0, 0, 0.1], [2, 2, 0.9])
    obj3 = make_mock_obj(3, "pole", 3, [4, 4, 2.0], [3.9, 3.9, 0.0], [4.1, 4.1, 4.0])

    g = builder.build([obj1, obj2, obj3])

    check("num_nodes == 3", g.num_nodes == 3)
    check("num_edges == 6 (3 * 2)", g.num_edges == 6)
    check("edge_index shape is (2, 6)", g.edge_index.shape == (2, 6))
    check("edge_features shape is (6, 10)", g.edge_features.shape == (6, EDGE_FEATURE_DIM))

    # Verify no self-loops (src != dst for all edges)
    srcs = g.edge_index[0]
    dsts = g.edge_index[1]
    has_self_loop = any(s == d for s, d in zip(srcs, dsts))
    check("Strictly no self-loops in edge_index", not has_self_loop)

    # Verify all directed pairs exist
    expected_pairs = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    check("All 6 directed pairs represented in order", g.edge_pairs_object_ids == expected_pairs)


def test_4_edge_features_and_directional_semantics() -> None:
    section("4. Edge Feature Correctness and Directional Semantics")

    builder = SceneGraphBuilder(up_axis="z")

    # Road flat on XY, top at Z = 0.2
    road = make_mock_obj(10, "road", 1, [0.0, 0.0, 0.1], [-5.0, -5.0, 0.0], [5.0, 5.0, 0.2])
    # Car resting on road at Z in [0.2, 1.8], center at Z = 1.0, X in [1, 3], Y in [1, 4]
    car = make_mock_obj(20, "car", 2, [2.0, 2.5, 1.0], [1.0, 1.0, 0.2], [3.0, 4.0, 1.8])

    g = builder.build([road, car])

    check("num_nodes == 2, num_edges == 2", g.num_nodes == 2 and g.num_edges == 2)

    # Edge 0: road -> car (idx 0 -> 1)
    # Edge 1: car -> road (idx 1 -> 0)
    feat_road_to_car = g.edge_features[0]
    feat_car_to_road = g.edge_features[1]

    # 1-3: Displacement
    disp_road_to_car = feat_road_to_car[0:3]
    disp_car_to_road = feat_car_to_road[0:3]
    check("Displacement road->car is [2.0, 2.5, 0.9]", np.allclose(disp_road_to_car, [2.0, 2.5, 0.9]))
    check(
        "Displacement is anti-symmetric (car->road == -(road->car))",
        np.allclose(disp_car_to_road, -disp_road_to_car),
    )

    # 4: 3D distance
    dist_3d = math.sqrt(2.0**2 + 2.5**2 + 0.9**2)
    check("Distance 3D is symmetric and accurate", math.isclose(feat_road_to_car[3], dist_3d, rel_tol=1e-5))
    check("Distance 3D road->car == car->road", math.isclose(feat_road_to_car[3], feat_car_to_road[3]))

    # 5: Horizontal distance
    dist_horiz = math.sqrt(2.0**2 + 2.5**2)
    check("Distance horizontal is accurate", math.isclose(feat_road_to_car[4], dist_horiz, rel_tol=1e-5))

    # 6: Signed vertical offset along z
    check("Vertical offset road->car is +0.9", math.isclose(feat_road_to_car[5], 0.9, rel_tol=1e-5))
    check("Vertical offset car->road is -0.9", math.isclose(feat_car_to_road[5], -0.9, rel_tol=1e-5))

    # 7-9: Clearance / penetration along axes
    # Along Z: road Z is [0.0, 0.2], car Z is [0.2, 1.8]. Touching at Z=0.2 -> clearance = max(0, 0.2) - min(0.2, 1.8) = 0.2 - 0.2 = 0.0
    check("Z clearance is 0.0 (touching faces)", math.isclose(feat_road_to_car[8], 0.0, abs_tol=1e-5))
    # Along X: road [-5, 5], car [1, 3] -> overlap [1, 3], clearance = 1 - 3 = -2.0 (overlap)
    check("X clearance is -2.0 (horizontal overlap)", math.isclose(feat_road_to_car[6], -2.0, abs_tol=1e-5))

    # 10: Log volume ratio
    # road vol = 10 * 10 * 0.2 = 20
    # car vol = 2 * 3 * 1.6 = 9.6
    # road -> car: log(9.6 / 20) < 0
    # car -> road: log(20 / 9.6) > 0
    check("Volume ratio road->car is negative", feat_road_to_car[9] < 0)
    check("Volume ratio car->road is positive", feat_car_to_road[9] > 0)
    check(
        "Volume ratio is anti-symmetric",
        math.isclose(feat_road_to_car[9], -feat_car_to_road[9], rel_tol=1e-4),
    )


def test_5_baseline_on_heuristic_integration() -> None:
    section("5. Baseline ON Heuristic Evidence Separation")

    builder = SceneGraphBuilder(up_axis="z")

    road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

    g = builder.build([road, car])

    # Check that heuristic_on_results exists and evaluates both directions
    check("(car, road) in heuristic_on_results", (2, 1) in g.heuristic_on_results)
    check("(road, car) in heuristic_on_results", (1, 2) in g.heuristic_on_results)

    car_on_road = g.heuristic_on_results[(2, 1)]
    road_on_car = g.heuristic_on_results[(1, 2)]

    check("car ON road is True", car_on_road.is_on == True)
    check("car ON road vertical_gap == 0.0", math.isclose(car_on_road.vertical_gap, 0.0))
    check("road ON car is False (road is below car)", road_on_car.is_on == False)

    # Confirm heuristic results are separate and don't pollute node or edge features
    check("Node features shape untouched (2, 9)", g.node_features.shape == (2, NODE_FEATURE_DIM))
    check("Edge features shape untouched (2, 10)", g.edge_features.shape == (2, EDGE_FEATURE_DIM))


def test_6_distance_threshold_filtering() -> None:
    section("6. Configurable Distance Threshold")

    # obj1 at (0, 0, 0), obj2 at (3, 0, 0) -> dist = 3.0
    # obj3 at (10, 0, 0) -> dist to obj1 = 10, dist to obj2 = 7
    obj1 = make_mock_obj(1, "a", 1, [0, 0, 0], [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    obj2 = make_mock_obj(2, "b", 2, [3, 0, 0], [2.5, -0.5, -0.5], [3.5, 0.5, 0.5])
    obj3 = make_mock_obj(3, "c", 3, [10, 0, 0], [9.5, -0.5, -0.5], [10.5, 0.5, 0.5])

    # Threshold = 5.0 -> only edges between obj1 and obj2 should exist (distance 3 <= 5)
    builder_gated = SceneGraphBuilder(max_edge_distance=5.0)
    g = builder_gated.build([obj1, obj2, obj3])

    check("Gated num_nodes == 3", g.num_nodes == 3)
    check("Gated num_edges == 2 (only (1,2) and (2,1))", g.num_edges == 2)
    check("Gated edge pairs are (1,2) and (2,1)", g.edge_pairs_object_ids == [(1, 2), (2, 1)])

    # Threshold = 1.0 -> no edges should exist
    builder_tight = SceneGraphBuilder(max_edge_distance=1.0)
    g_none = builder_tight.build([obj1, obj2, obj3])
    check("Tight threshold num_edges == 0", g_none.num_edges == 0)
    check("Tight edge_index shape is (2, 0)", g_none.edge_index.shape == (2, 0))
    check("Tight edge_features shape is (0, 10)", g_none.edge_features.shape == (0, EDGE_FEATURE_DIM))


def test_7_determinism_and_sorting() -> None:
    section("7. Deterministic Construction Regardless of Input Order")

    builder = SceneGraphBuilder()

    obj1 = make_mock_obj(1, "road", 1, [0, 0, 0], [-2, -2, -0.1], [2, 2, 0.1])
    obj2 = make_mock_obj(2, "car", 2, [0, 0, 0.5], [-1, -1, 0.1], [1, 1, 0.9])
    obj3 = make_mock_obj(3, "person", 3, [1, 1, 0.8], [0.8, 0.8, 0.1], [1.2, 1.2, 1.5])

    # Run 1: [obj1, obj2, obj3]
    g1 = builder.build([obj1, obj2, obj3])
    # Run 2: [obj3, obj1, obj2] (shuffled input)
    g2 = builder.build([obj3, obj1, obj2])

    check("Node order is identical", g1.node_object_ids == g2.node_object_ids == [1, 2, 3])
    check("Node features match exactly", np.allclose(g1.node_features, g2.node_features))
    check("Edge indices match exactly", np.array_equal(g1.edge_index, g2.edge_index))
    check("Edge features match exactly", np.allclose(g1.edge_features, g2.edge_features))
    check("Edge pairs list matches exactly", g1.edge_pairs_object_ids == g2.edge_pairs_object_ids)


def test_8_configurable_up_axis() -> None:
    section("8. Configurable Up-Axis ('y')")

    # Road flat on XZ, Y is up
    road = make_mock_obj(1, "road", 1, [0, 0.1, 0], [-5, 0.0, -5], [5, 0.2, 5])
    car = make_mock_obj(2, "car", 2, [0, 1.0, 0], [-1, 0.2, -1], [1, 1.8, 1])

    builder_y = SceneGraphBuilder(up_axis="y")
    g = builder_y.build([road, car])

    check("Graph up_axis is 'y'", g.up_axis == "y")
    # Edge road -> car (0 -> 1)
    feat = g.edge_features[0]
    # In 'y' up_axis: vertical offset is feature 5 (disp along y = 1.0 - 0.1 = 0.9)
    check("Vertical offset along y is +0.9", math.isclose(feat[5], 0.9, rel_tol=1e-5))
    # Horizontal distance in XZ plane (both at X=0, Z=0) -> 0.0
    check("Horizontal distance in XZ plane is 0.0", math.isclose(feat[4], 0.0, abs_tol=1e-5))

    # Heuristic detection under 'y' up-axis
    check("car ON road under 'y' is True", g.heuristic_on_results[(2, 1)].is_on == True)


def test_9_errors_and_validations() -> None:
    section("9. Errors and Input Validations")

    # Builder constructor validations
    expect_error("Invalid up_axis 'w' raises ValueError", lambda: SceneGraphBuilder(up_axis="w"))  # type: ignore
    expect_error("Negative max_edge_distance raises ValueError", lambda: SceneGraphBuilder(max_edge_distance=-2.0))

    builder = SceneGraphBuilder()
    valid_obj = make_mock_obj(1, "valid", 1, [0, 0, 0], [-1, -1, -1], [1, 1, 1])

    # Duplicate object ID
    expect_error(
        "Duplicate object ID raises ValueError",
        lambda: builder.build([valid_obj, valid_obj]),
    )

    # Non-finite coordinates (NaN)
    nan_obj = make_mock_obj(2, "nan_obj", 2, [np.nan, 0, 0], [-1, -1, -1], [1, 1, 1])
    expect_error("NaN centroid raises ValueError", lambda: builder.build([valid_obj, nan_obj]))

    # Non-finite coordinates (Inf)
    inf_obj = make_mock_obj(3, "inf_obj", 3, [0, 0, 0], [-1, -np.inf, -1], [1, 1, 1])
    expect_error("Inf bbox raises ValueError", lambda: builder.build([valid_obj, inf_obj]))

    # Reversed AABB bounds (bbox_min > bbox_max)
    rev_obj = make_mock_obj(4, "rev_obj", 4, [0, 0, 0], [2, 1, 1], [1, 1, 1])
    expect_error("Reversed AABB bounds raises ValueError", lambda: builder.build([valid_obj, rev_obj]))


def test_10_serialization() -> None:
    section("10. Serialization and Dict Export")

    builder = SceneGraphBuilder()
    road = make_mock_obj(1, "road", 1, [0, 0, 0.1], [-5, -5, 0.0], [5, 5, 0.2])
    car = make_mock_obj(2, "car", 2, [0, 0, 0.9], [-1, -1, 0.2], [1, 1, 1.6])

    g = builder.build([road, car])
    d = g.to_dict()

    check("to_dict() returns dictionary", isinstance(d, dict))
    check("num_nodes == 2 in dict", d["num_nodes"] == 2)
    check("num_edges == 2 in dict", d["num_edges"] == 2)
    check("up_axis in dict", d["up_axis"] == "z")
    check("node_object_ids in dict", d["node_object_ids"] == [1, 2])
    check("heuristic_on_count == 1 in dict", d["heuristic_on_count"] == 1)

    try:
        s = json.dumps(d)
        check("Dict is JSON serialisable", isinstance(s, str))
    except Exception as e:
        check("Dict is JSON serialisable", False, str(e))


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- Stage B: Scene Graph Builder Unit Tests")
    print("=" * 60)

    test_1_empty_and_single_node()
    test_2_node_feature_correctness()
    test_3_directed_edges_and_no_self_loops()
    test_4_edge_features_and_directional_semantics()
    test_5_baseline_on_heuristic_integration()
    test_6_distance_threshold_filtering()
    test_7_determinism_and_sorting()
    test_8_configurable_up_axis()
    test_9_errors_and_validations()
    test_10_serialization()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Stage B -- Scene Graph Builder: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Stage B -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
