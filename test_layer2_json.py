"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_layer2_json.py

PURPOSE
-------
Validation script for STEP 12: Final Layer-2 JSON Output.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from objects.object_3d import Object3D
from relationships.on_relation import ONRelationResult
from output.layer2_json import build_layer2_output, Layer2Output, Layer2ObjectRecord


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

def mock_obj(obj_id: int, class_name: str, center: list, bbox_min: list, bbox_max: list, mean_conf: float = 1.0) -> Object3D:
    return Object3D(
        object_id=obj_id,
        semantic_class_id=0,
        class_name=class_name,
        gaussian_ids=[0, 1],
        gaussian_scores={0: 1.0, 1: 1.0},
        center_3d=list(center),
        bbox_min=list(bbox_min),
        bbox_max=list(bbox_max),
        gaussian_count=2,
        mean_confidence=mean_conf
    )

def mock_rel(sub: int, sub_cls: str, sup: int, sup_cls: str, is_on: bool) -> ONRelationResult:
    return ONRelationResult(
        subject_object_id=sub,
        subject_class_name=sub_cls,
        support_object_id=sup,
        support_class_name=sup_cls,
        relation="ON",
        vertical_gap=0.012,
        overlap_ratio=0.82,
        on_score=0.89,
        is_on=is_on
    )

# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_to_3_counts() -> None:
    section("1/2/3. Output Object Counts")
    
    # Empty
    out_empty = build_layer2_output([], [])
    check("Empty Layer-2 output works", len(out_empty.objects) == 0)
    
    # One Object
    o1 = mock_obj(1, "road", [0,0,0], [0,0,0], [1,1,1])
    out_one = build_layer2_output([o1], [])
    check("One object output works", len(out_one.objects) == 1)
    
    # Multiple Objects
    o2 = mock_obj(2, "car", [0,0,0], [0,0,0], [1,1,1])
    out_multi = build_layer2_output([o1, o2], [])
    check("Multiple objects output works", len(out_multi.objects) == 2)


def test_4_to_11_21_object_fields() -> None:
    section("4-11/21. Object Field Preservation")
    
    o = mock_obj(17, "car", [1.2, 2.4, 0.8], [1.0, 2.1, 0.7], [1.5, 2.8, 1.1], 0.84)
    # We change semantic_class_id to 2 for check
    o = Object3D(17, 2, "car", [102,103,104], {}, [1.2, 2.4, 0.8], [1.0, 2.1, 0.7], [1.5, 2.8, 1.1], 3, 0.84)
    
    out = build_layer2_output([o], [])
    rec = out.objects[0]
    
    check("Object ID preservation", rec.object_id == 17)
    check("Semantic class preservation", rec.semantic_class_id == 2)
    check("Class name preservation", rec.class_name == "car")
    check("Gaussian ID preservation", rec.gaussian_ids == [102, 103, 104])
    check("Gaussian count preservation", rec.gaussian_count == 3)
    check("Center 3D preservation", rec.center_3d == [1.2, 2.4, 0.8])
    check("BBox min/max preservation", rec.bbox_3d["min"] == [1.0, 2.1, 0.7] and rec.bbox_3d["max"] == [1.5, 2.8, 1.1])
    check("Confidence preservation", math.isclose(rec.mean_confidence, 0.84))
    
    check("Missing layer metadata correctly mapped to null (None)", rec.layer is None)
    
    d = rec.to_dict()
    check("Empty relationships defaults to empty list []", d["relationships"] == [])


def test_12_to_20_relationships() -> None:
    section("12-20. Relationships")
    
    o1 = mock_obj(3, "road", [0,0,0], [0,0,0], [1,1,1])
    o2 = mock_obj(17, "car", [0,0,0], [0,0,0], [1,1,1])
    o3 = mock_obj(99, "pavement", [0,0,0], [0,0,0], [1,1,1])
    
    r_on = mock_rel(17, "car", 3, "road", True)
    r_off = mock_rel(17, "car", 99, "pavement", False)
    
    out = build_layer2_output([o1, o2, o3], [r_on, r_off])
    
    rec_road = out.objects[0] # ID 3
    rec_car = out.objects[1] # ID 17
    
    check("Support object has empty relationships", len(rec_road.relationships) == 0)
    check("ON relationship serialization attached to subject", len(rec_car.relationships) == 1)
    
    rel_dict = rec_car.relationships[0]
    check("Target object ID preservation", rel_dict["target_object_id"] == 3)
    check("Target class preservation", rel_dict["target_class_name"] == "road")
    check("Vertical gap preservation", rel_dict["vertical_gap"] == 0.012)
    check("Overlap ratio preservation", rel_dict["overlap_ratio"] == 0.82)
    check("on_score preservation", rel_dict["on_score"] == 0.89)
    check("Relationship direction strictly preserved (Car -> ON -> Road)", rel_dict["target_object_id"] == 3)
    
    r_on_2 = mock_rel(17, "car", 99, "pavement", True)
    out_multi = build_layer2_output([o1, o2, o3], [r_on, r_on_2])
    check("Multiple relationships preserved", len(out_multi.objects[1].relationships) == 2)


def test_22_to_24_validations() -> None:
    section("22-24. Validations")
    
    o1 = mock_obj(1, "road", [0,0,0], [0,0,0], [1,1,1])
    o1_dup = mock_obj(1, "road", [0,0,0], [0,0,0], [1,1,1])
    
    expect_error("Duplicate object ID raises Error", lambda: build_layer2_output([o1, o1_dup], []))
    
    r_inv = mock_rel(1, "road", 999, "void", True)
    expect_error("Invalid target ID raises Error", lambda: build_layer2_output([o1], [r_inv]))
    
    o_nan = mock_obj(2, "car", [np.nan, 0, 0], [0,0,0], [1,1,1])
    expect_error("Invalid XYZ (NaN center) raises Error", lambda: build_layer2_output([o1, o_nan], []))
    
    o_rev = mock_obj(3, "car", [0,0,0], [2,2,2], [1,1,1])
    expect_error("Invalid XYZ (Bbox Min > Max) raises Error", lambda: build_layer2_output([o1, o_rev], []))
    
    r_bad_rel = ONRelationResult(1, "road", 1, "road", "NEAR", 0, 0, 0, True)
    expect_error("Invalid relation string raises Error", lambda: build_layer2_output([o1], [r_bad_rel]))


def test_25_json_serialization() -> None:
    section("25. JSON Serialization Round-Trip")
    
    o1 = mock_obj(3, "road", [0,0,0], [0,0,0], [1,1,1])
    o2 = mock_obj(17, "car", [0,0,0], [0,0,0], [1,1,1])
    r = mock_rel(17, "car", 3, "road", True)
    
    out = build_layer2_output([o1, o2], [r])
    d = out.to_dict()
    
    try:
        j = json.dumps(d)
        d2 = json.loads(j)
        check("JSON dump/load successful", True)
        
        check("Top-level metadata preserved", d2["layer"] == "Layer2" and d2["coordinate_system"] == "Layer-1 world coordinates")
        check("Object serialization robust", len(d2["objects"]) == 2)
        
        car_obj = d2["objects"][1]
        check("JSON float extraction stable", math.isclose(car_obj["mean_confidence"], 1.0))
        check("JSON relationships loaded correctly", car_obj["relationships"][0]["relation"] == "ON")
        
    except Exception as e:
        check("JSON serialization round-trip", False, str(e))


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 12: Final Layer-2 JSON Output")
    print("=" * 60)
    print()

    test_1_to_3_counts()
    test_4_to_11_21_object_fields()
    test_12_to_20_relationships()
    test_22_to_24_validations()
    test_25_json_serialization()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 12 -- Final Layer-2 JSON Output: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 12 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
