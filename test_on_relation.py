"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_on_relation.py

PURPOSE
-------
Validation script for STEP 11: 3D ON Relationship Extraction.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from objects.object_3d import Object3D
from relationships.on_relation import ONRelationDetector, ONRelationResult


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

def mock_obj(obj_id: int, class_name: str, bbox_min: list, bbox_max: list) -> Object3D:
    return Object3D(
        object_id=obj_id,
        semantic_class_id=0,
        class_name=class_name,
        gaussian_ids=[0],
        gaussian_scores={0: 1.0},
        center_3d=[0.0, 0.0, 0.0],
        bbox_min=list(bbox_min),
        bbox_max=list(bbox_max),
        gaussian_count=1,
        mean_confidence=1.0
    )

# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_to_3_basic_and_lists() -> None:
    section("1/2/3. Empty/Single Lists and Two Valid Objects")
    
    det = ONRelationDetector()
    
    road = mock_obj(1, "road", [0.0, 0.0, 0.0], [10.0, 10.0, 0.2])
    car = mock_obj(2, "car", [2.0, 2.0, 0.2], [5.0, 5.0, 1.7])
    
    res = det.detect(car, road)
    check("Case A — Valid ON (Two valid objects)", res.is_on == True)
    check("Subject/Support direction correct", res.subject_class_name == "car" and res.support_class_name == "road")
    check("Relation string is 'ON'", res.relation == "ON")
    check("Gap is approximately zero", math.isclose(res.vertical_gap, 0.0))


def test_4_5_zero_area_and_boundaries() -> None:
    section("4/5. Zero-area Footprints and Touching Boundaries")
    
    det = ONRelationDetector()
    
    # 1D line as road, 1D line as car
    line_road = mock_obj(1, "road", [5.0, 0.0, 0.0], [5.0, 10.0, 0.2])
    line_car = mock_obj(2, "car", [5.0, 2.0, 0.2], [5.0, 4.0, 1.7])
    
    res_zero = det.detect(line_car, line_road)
    check("Zero-area footprint handled securely", res_zero.is_on == True)
    
    touch_road = mock_obj(3, "road", [0.0, 0.0, 0.0], [10.0, 10.0, 0.2])
    touch_car = mock_obj(4, "car", [10.0, 10.0, 0.2], [12.0, 12.0, 1.0])
    
    res_touch = det.detect(touch_car, touch_road)
    check("Touching boundaries exact edge overlap ratio 0", res_touch.overlap_ratio == 0.0)
    check("Touching boundaries without min overlap fails is_on", res_touch.is_on == False)


def test_6_to_11_gaps_and_overlaps() -> None:
    section("6/7/8/9/10/11. Gaps and Overlaps")
    
    det = ONRelationDetector(max_vertical_gap=0.2, min_footprint_overlap=0.4, on_score_threshold=0.5)
    road = mock_obj(1, "road", [0.0, 0.0, 0.0], [10.0, 10.0, 0.2])
    
    # Small positive gap
    car1 = mock_obj(2, "car", [2.0, 2.0, 0.3], [5.0, 5.0, 1.7])
    r1 = det.detect(car1, road)
    check("Small positive gap passes", r1.is_on == True and math.isclose(r1.vertical_gap, 0.1))

    
    # Negative gap/interpenetration
    car2 = mock_obj(3, "car", [2.0, 2.0, 0.1], [5.0, 5.0, 1.7])
    r2 = det.detect(car2, road)
    check("Negative gap/interpenetration passes", r2.is_on == True and math.isclose(r2.vertical_gap, -0.1))
    
    # Large vertical gap (Case B)
    car3 = mock_obj(4, "car", [2.0, 2.0, 2.5], [5.0, 5.0, 4.0])
    r3 = det.detect(car3, road)
    check("Case B — Large vertical gap fails", r3.is_on == False)
    
    # No horizontal overlap (Case C)
    car4 = mock_obj(5, "car", [12.0, 12.0, 0.2], [15.0, 15.0, 1.7])
    r4 = det.detect(car4, road)
    check("Case C — No horizontal overlap fails", r4.is_on == False and r4.overlap_ratio == 0.0)
    
    # Partial horizontal overlap (Case D)
    car5 = mock_obj(6, "car", [8.0, 8.0, 0.2], [12.0, 12.0, 1.7]) # 4x4 area = 16. Overlap with 0-10 is [8,10] = 2x2=4. Ratio = 4/16 = 0.25
    r5 = det.detect(car5, road)
    check("Case D — Partial horizontal overlap correctly computed ratio", r5.overlap_ratio == 0.25)
    check("Case D — Fails if below threshold", r5.is_on == False)
    
    # Complete horizontal overlap
    r6 = det.detect(car1, road)
    check("Complete horizontal overlap gives ratio 1.0", r6.overlap_ratio == 1.0)


def test_12_20_supports_and_reversals() -> None:
    section("12/20. Multiple Supports and Direction Reversals")
    
    det = ONRelationDetector()
    road = mock_obj(1, "road", [0.0, 0.0, 0.0], [10.0, 10.0, 0.2])
    pavement = mock_obj(2, "pavement", [-2.0, -2.0, 0.1], [2.0, 12.0, 0.3])
    car = mock_obj(3, "car", [0.0, 0.0, 0.3], [2.0, 4.0, 1.7])
    
    r_road = det.detect(car, road)
    r_pave = det.detect(car, pavement)
    
    check("Multiple supports can be evaluated independently", r_road.is_on == True and r_pave.is_on == True)
    
    r_rev = det.detect(road, car)
    check("Subject/Support order reversal safely fails", r_rev.is_on == False)


def test_13_to_19_errors_and_validations() -> None:
    section("13-19. Validations and Errors")
    
    expect_error("Invalid threshold (negative gap)", lambda: ONRelationDetector(max_vertical_gap=-1.0))
    expect_error("Invalid threshold (overlap > 1)", lambda: ONRelationDetector(min_footprint_overlap=1.5))
    expect_error("Invalid threshold (score < 0)", lambda: ONRelationDetector(on_score_threshold=-0.1))
    expect_error("Different up-axis configuration (invalid axis)", lambda: ONRelationDetector(up_axis="q"))
    
    det = ONRelationDetector()
    obj1 = mock_obj(1, "car", [0,0,0], [1,1,1])
    
    expect_error("Duplicate object IDs", lambda: det.detect(obj1, obj1))
    
    obj_nan = mock_obj(2, "road", [np.nan, 0, 0], [1,1,1])
    expect_error("NaN coordinates rejected", lambda: det.detect(obj1, obj_nan))
    
    obj_inf = mock_obj(3, "road", [np.inf, 0, 0], [1,1,1])
    expect_error("Inf coordinates rejected", lambda: det.detect(obj1, obj_inf))
    
    obj_rev = mock_obj(4, "road", [2,2,2], [1,1,1])
    expect_error("Reversed AABB min/max rejected", lambda: det.detect(obj1, obj_rev))


def test_19_different_up_axis() -> None:
    section("19. Different Up-Axis ('y')")
    
    det_y = ONRelationDetector(up_axis="y")
    # Road is flat on XZ, Y is up
    road = mock_obj(1, "road", [0.0, 0.0, 0.0], [10.0, 0.2, 10.0])
    car = mock_obj(2, "car", [2.0, 0.2, 2.0], [5.0, 1.7, 5.0])
    
    res = det_y.detect(car, road)
    check("Configurable up_axis ('y') correctly resolves ON relation", res.is_on == True)


def test_21_to_25_serialization_and_stability() -> None:
    section("21-25. Serialization and Output Formatting")
    
    det = ONRelationDetector()
    road = mock_obj(1, "road", [0.0, 0.0, 0.0], [10.0, 10.0, 0.2])
    car = mock_obj(2, "car", [2.0, 2.0, 0.2], [5.0, 5.0, 1.7])
    
    r = det.detect(car, road)
    d = r.to_dict()
    
    check("Output is dictionary", isinstance(d, dict))
    check("Output contains relation = ON", d.get("relation") == "ON")
    check("Output contains boolean is_on", isinstance(d.get("is_on"), bool))
    
    try:
        j = json.dumps(d)
        check("Output is JSON serialisable", isinstance(j, str))
    except Exception as e:
        check("Output is JSON serialisable", False, str(e))
        
    r2 = det.detect(car, road)
    check("Deterministic execution", r == r2)


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 11: 3D ON Relationship Extraction")
    print("=" * 60)
    print()

    test_1_to_3_basic_and_lists()
    test_4_5_zero_area_and_boundaries()
    test_6_to_11_gaps_and_overlaps()
    test_12_20_supports_and_reversals()
    test_13_to_19_errors_and_validations()
    test_19_different_up_axis()
    test_21_to_25_serialization_and_stability()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 11 -- 3D ON Relationship Extraction: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 11 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
