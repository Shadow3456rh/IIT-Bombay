"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_object_3d.py

PURPOSE
-------
Validation script for STEP 10: 3D Object Representation.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from gaussian.scene import GaussianScene
from objects.object_3d import Object3D, build_object_3d

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
_failures = 0

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    tag = "[PASS]" if condition else "[FAIL]"
    print(f"  {tag}  {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        _failures += 1

def expect_error(label: str, fn, exc=(ValueError, TypeError)) -> None:
    global _failures
    try:
        fn()
        print(f"  [FAIL]  {label}  -- no exception raised")
        _failures += 1
    except exc as e:
        print(f"  [PASS]  {label}")
        print(f"         {type(e).__name__}: {e}")

def make_scene(xyz_data: list) -> GaussianScene:
    n = len(xyz_data)
    xyz = np.array(xyz_data, dtype=np.float32)
    opacity_raw = np.zeros(n, dtype=np.float32)
    scale_raw = np.zeros((n, 3), dtype=np.float32)
    rotation = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    ids = np.arange(n, dtype=np.int64)

    return GaussianScene(
        ids=ids,
        xyz=xyz,
        opacity_raw=opacity_raw,
        scale_raw=scale_raw,
        rotation=rotation,
        ply_vertex_index=ids.copy(),
    )


# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_to_8_construction_and_bounds() -> None:
    section("1-8. Object3D Construction, AABB, Centroid")

    scene = make_scene([
        [1.0, 2.0, 3.0], # 0
        [3.0, 0.0, 1.0], # 1
        [2.0, 4.0, 2.0]  # 2
    ])
    
    g_ids = [0, 1, 2]
    scores = {0: 1.0, 1: 0.8, 2: 0.6}
    
    obj = build_object_3d(17, 2, "car", g_ids, scores, scene)
    
    check("Object3D construction successful", isinstance(obj, Object3D))
    check("Required fields present", hasattr(obj, "object_id") and hasattr(obj, "semantic_class_id"))
    check("Gaussian ID preservation", obj.gaussian_ids == [0, 1, 2])
    check("Correct Gaussian count", obj.gaussian_count == 3)
    
    check("Centroid calculation (mean X)", math.isclose(obj.center_3d[0], 2.0))
    check("Centroid calculation (mean Y)", math.isclose(obj.center_3d[1], 2.0))
    check("Centroid calculation (mean Z)", math.isclose(obj.center_3d[2], 2.0))
    
    check("bbox minimum calculation", obj.bbox_min == [1.0, 0.0, 1.0])
    check("bbox maximum calculation", obj.bbox_max == [3.0, 4.0, 3.0])
    check("AABB correctness (min <= max)", all(a <= b for a,b in zip(obj.bbox_min, obj.bbox_max)))


def test_9_to_14_multiple_objects_and_scores() -> None:
    section("9-14. Cardinalities and Fused Scores")
    
    scene = make_scene([
        [0.0, 0.0, 0.0], # 0 (single)
        [1.0, 1.0, 1.0], # 1 (multi obj A)
        [2.0, 2.0, 2.0], # 2 (multi obj A)
        [3.0, 3.0, 3.0], # 3 (multi obj B)
    ])
    
    # Single-Gaussian object
    obj_single = build_object_3d(1, 1, "pole", [0], {0: 0.9}, scene)
    check("Single-Gaussian object supported", obj_single.gaussian_count == 1)
    
    # Multiple-Gaussian object
    obj_multi = build_object_3d(2, 2, "car", [1, 2], {1: 0.8, 2: 0.6}, scene)
    check("Multiple-Gaussian object supported", obj_multi.gaussian_count == 2)
    
    # Multiple independent objects
    obj_indep = build_object_3d(3, 3, "road", [3], {3: 0.95}, scene)
    check("Multiple independent objects separation", obj_multi.object_id != obj_indep.object_id)
    
    check("Object identity preservation", obj_multi.object_id == 2)
    check("Fused score preservation", obj_multi.gaussian_scores[1] == 0.8)
    
    # Mean confidence = (0.8 + 0.6) / 2 = 0.7
    check("Object-level score calculation (mean_confidence)", math.isclose(obj_multi.mean_confidence, 0.7))


def test_15_to_22_errors_and_edge_cases() -> None:
    section("15-22. Errors and Edge Cases")
    
    scene = make_scene([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]])
    
    expect_error("Empty input / zero Gaussians rejected", lambda: build_object_3d(1, 1, "car", [], {}, scene))
    
    # Invalid Gaussian ID
    expect_error("Invalid Gaussian ID (not in scene)", lambda: build_object_3d(1, 1, "car", [99], {99: 1.0}, scene))
    
    # Missing Gaussian ID in scores
    expect_error("Missing Gaussian ID score", lambda: build_object_3d(1, 1, "car", [0], {}, scene))
    
    # NaN/Inf XYZ
    expect_error("NaN XYZ rejection", lambda: build_object_3d(1, 1, "car", [1], {1: 1.0}, scene))
    expect_error("Inf XYZ rejection", lambda: build_object_3d(1, 1, "car", [2], {2: 1.0}, scene))
    
    # Duplicate IDs
    expect_error("Duplicate Gaussian ID within one object rejected", lambda: build_object_3d(1, 1, "car", [0, 0], {0: 1.0}, scene))


def test_23_24_serialization() -> None:
    section("23-24. Serialization")
    
    scene = make_scene([[1.23, 2.45, 0.87], [1.0, 2.1, 0.7], [1.5, 2.8, 1.1]])
    obj = build_object_3d(17, 2, "car", [0, 1, 2], {0: 0.9, 1: 0.8, 2: 0.7}, scene)
    
    d = obj.to_dict()
    check("to_dict() output is dictionary", isinstance(d, dict))
    check("to_dict() contains required fields", "center_3d" in d and "gaussian_count" in d)
    
    try:
        j = json.dumps(d)
        check("JSON-serializable output", isinstance(j, str))
    except Exception as e:
        check("JSON-serializable output", False, str(e))


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 10: 3D Object Representation")
    print("=" * 60)
    print()

    test_1_to_8_construction_and_bounds()
    test_9_to_14_multiple_objects_and_scores()
    test_15_to_22_errors_and_edge_cases()
    test_23_24_serialization()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 10 -- 3D Object Representation: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 10 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
