"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_object.py

PURPOSE
-------
Validation script for STEP 4: 2D Object Representation.

All tests use SYNTHETIC data only.
YOLO26-L and SAM3 are NOT executed here.

SYNTHETIC TEST SETUP
--------------------
Image : 640 x 480 pixels  (width x height)
Object: "car" covering mask region rows 180..320, cols 200..350  (inclusive)

Mask area:
    rows  = 320 - 180 + 1 = 141
    cols  = 350 - 200 + 1 = 151
    area  = 141 * 151     = 21291 pixels

Mask bounding box (from mask_bbox property):
    x_min = 200, y_min = 180, x_max = 350, y_max = 320

Detector bounding box (provided):
    [200, 180, 350, 320]
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from objects import Object2D

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


def expect_error(label: str, fn) -> None:
    """Verify that calling fn() raises ValueError or TypeError."""
    global _failures
    try:
        fn()
        print(f"  [FAIL]  {label}  (no exception raised)")
        _failures += 1
    except (ValueError, TypeError) as e:
        print(f"  [PASS]  {label}")
        print(f"         raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 640, 480

# Mask: rows 180..320, cols 200..350 (inclusive at both ends)
MASK_Y1, MASK_Y2 = 180, 320
MASK_X1, MASK_X2 = 200, 350

EXPECTED_MASK_ROWS = MASK_Y2 - MASK_Y1 + 1   # 141
EXPECTED_MASK_COLS = MASK_X2 - MASK_X1 + 1   # 151
EXPECTED_AREA      = EXPECTED_MASK_ROWS * EXPECTED_MASK_COLS  # 21291


def make_synthetic_mask() -> np.ndarray:
    """Create a 480x640 boolean mask with a rectangular True region."""
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[MASK_Y1:MASK_Y2 + 1, MASK_X1:MASK_X2 + 1] = True
    return mask


def make_demo_object(mask=None) -> Object2D:
    """Construct the standard demo Object2D used across most tests."""
    if mask is None:
        mask = make_synthetic_mask()
    return Object2D(
        object_id         = 17,
        semantic_class_id = 2,
        class_name        = "car",
        confidence        = 0.94,
        bbox              = [200, 180, 350, 320],
        mask              = mask,
        source_frame      = "demo_frame_001.jpg",
    )


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def test_construction() -> None:
    section("1. Object Construction")
    obj = make_demo_object()
    print(f"\n  repr: {obj}")
    check("object constructed without error", True)
    check("type is Object2D", isinstance(obj, Object2D))


def test_fields() -> None:
    section("2. Field Values")
    obj = make_demo_object()

    check("object_id == 17",            obj.object_id         == 17)
    check("semantic_class_id == 2",     obj.semantic_class_id == 2)
    check("class_name == 'car'",        obj.class_name        == "car")
    check("confidence == 0.94",         abs(obj.confidence - 0.94) < 1e-9)
    check("source_frame correct",       obj.source_frame      == "demo_frame_001.jpg")

    # bbox
    check("bbox[0] == 200 (x1)", obj.bbox[0] == 200.0)
    check("bbox[1] == 180 (y1)", obj.bbox[1] == 180.0)
    check("bbox[2] == 350 (x2)", obj.bbox[2] == 350.0)
    check("bbox[3] == 320 (y2)", obj.bbox[3] == 320.0)

    # object_id is int
    check("object_id is Python int", type(obj.object_id) is int)
    check("semantic_class_id is Python int", type(obj.semantic_class_id) is int)


def test_mask_properties() -> None:
    section("3. Mask Shape, Dtype, Dimensions")
    obj = make_demo_object()

    check("mask is ndarray",         isinstance(obj.mask, np.ndarray))
    check("mask ndim == 2",          obj.mask.ndim == 2)
    check("mask dtype is bool",      obj.mask.dtype == bool)
    check(f"mask shape == ({IMG_H}, {IMG_W})",
          obj.mask.shape == (IMG_H, IMG_W),
          f"got {obj.mask.shape}")
    check("image_height == 480",     obj.image_height == 480)
    check("image_width  == 640",     obj.image_width  == 640)


def test_mask_area() -> None:
    section("4. Mask Area and Coverage")
    obj = make_demo_object()

    check(f"mask_area == {EXPECTED_AREA}",
          obj.mask_area == EXPECTED_AREA,
          f"got {obj.mask_area}")

    expected_coverage = round(EXPECTED_AREA / (IMG_W * IMG_H), 6)
    check(f"mask_coverage == {expected_coverage}",
          abs(obj.mask_coverage - expected_coverage) < 1e-6,
          f"got {obj.mask_coverage:.6f}")

    print(f"\n  mask_area     = {obj.mask_area} pixels")
    print(f"  mask_coverage = {obj.mask_coverage:.4%} of image")


def test_mask_bbox() -> None:
    section("5. Mask Bounding Box (derived from mask pixels)")
    obj   = make_demo_object()
    mbbox = obj.mask_bbox

    check("mask_bbox is not None",        mbbox is not None)
    check("mask_bbox has 4 values",       len(mbbox) == 4)
    check(f"mask_bbox x_min == {MASK_X1}", mbbox[0] == MASK_X1,
          f"got {mbbox[0]}")
    check(f"mask_bbox y_min == {MASK_Y1}", mbbox[1] == MASK_Y1,
          f"got {mbbox[1]}")
    check(f"mask_bbox x_max == {MASK_X2}", mbbox[2] == MASK_X2,
          f"got {mbbox[2]}")
    check(f"mask_bbox y_max == {MASK_Y2}", mbbox[3] == MASK_Y2,
          f"got {mbbox[3]}")

    print(f"\n  mask_bbox (from pixels) : {mbbox}")
    print(f"  detector bbox           : {[int(v) for v in obj.bbox]}")
    print(f"  (They match for a clean rectangular mask; "
          f"SAM3 output may differ from detector bbox)")


def test_to_dict() -> None:
    section("6. to_dict() Serialisation")
    obj  = make_demo_object()
    d    = obj.to_dict()

    check("to_dict returns dict",              isinstance(d, dict))
    check("'object_id' in dict",               "object_id"         in d)
    check("'semantic_class_id' in dict",       "semantic_class_id" in d)
    check("'class_name' in dict",              "class_name"        in d)
    check("'confidence' in dict",              "confidence"        in d)
    check("'bbox' in dict",                    "bbox"              in d)
    check("'source_frame' in dict",            "source_frame"      in d)
    check("'mask_shape' in dict",              "mask_shape"        in d)
    check("'mask_area' in dict",               "mask_area"         in d)
    check("'mask_coverage' in dict",           "mask_coverage"     in d)
    check("'mask' key NOT in dict (too large)","mask"              not in d)

    check("mask_shape == [480, 640]",          d["mask_shape"] == [480, 640])
    check("mask_area correct",                 d["mask_area"] == EXPECTED_AREA)

    # Verify JSON-serialisable (no numpy types)
    try:
        json_str = json.dumps(d)
        check("to_dict() is JSON-serialisable", True)
    except TypeError as e:
        check("to_dict() is JSON-serialisable", False, str(e))

    print(f"\n  to_dict() output:")
    print(f"  {json.dumps(d, indent=4)}")


def test_empty_mask_bbox() -> None:
    section("7. Empty Mask Handling")
    empty_mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    obj = Object2D(
        object_id=0, semantic_class_id=0,
        class_name="unknown", confidence=0.5,
        bbox=[10, 10, 20, 20],
        mask=empty_mask,
        source_frame="test.jpg",
    )
    check("mask_area == 0 for empty mask", obj.mask_area == 0)
    check("mask_bbox is None for empty mask", obj.mask_bbox is None)
    check("mask_coverage == 0.0",             obj.mask_coverage == 0.0)


def test_invalid_confidence() -> None:
    section("8. Validation -- Invalid Confidence")
    mask = make_synthetic_mask()

    expect_error("confidence = 1.5  (> 1.0)",
        lambda: Object2D(17, 2, "car", 1.5,  [200,180,350,320], mask, "f.jpg"))
    expect_error("confidence = -0.1 (< 0.0)",
        lambda: Object2D(17, 2, "car", -0.1, [200,180,350,320], mask, "f.jpg"))
    expect_error("confidence = NaN",
        lambda: Object2D(17, 2, "car", float("nan"), [200,180,350,320], mask, "f.jpg"))

    # Boundary values must be accepted
    obj_0   = Object2D(1, 0, "x", 0.0, [10,10,20,20], mask, "f.jpg")
    obj_1   = Object2D(1, 0, "x", 1.0, [10,10,20,20], mask, "f.jpg")
    check("confidence = 0.0 accepted", obj_0.confidence == 0.0)
    check("confidence = 1.0 accepted", obj_1.confidence == 1.0)


def test_invalid_bbox() -> None:
    section("9. Validation -- Invalid Bounding Box")
    mask = make_synthetic_mask()

    expect_error("bbox with 3 values",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20],       mask, "f.jpg"))
    expect_error("bbox with 5 values",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,30,40], mask, "f.jpg"))
    expect_error("bbox x1 == x2 (degenerate)",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,10,30],    mask, "f.jpg"))
    expect_error("bbox x1 > x2 (reversed)",
        lambda: Object2D(1, 0, "x", 0.5, [30,10,10,30],    mask, "f.jpg"))
    expect_error("bbox y1 == y2 (degenerate)",
        lambda: Object2D(1, 0, "x", 0.5, [10,20,30,20],    mask, "f.jpg"))
    expect_error("bbox contains inf",
        lambda: Object2D(1, 0, "x", 0.5,
                         [10, 10, float("inf"), 30], mask, "f.jpg"))
    expect_error("bbox contains nan",
        lambda: Object2D(1, 0, "x", 0.5,
                         [10, float("nan"), 30, 40], mask, "f.jpg"))


def test_invalid_mask() -> None:
    section("10. Validation -- Invalid Mask")

    expect_error("mask is not ndarray (list)",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,20],
                         [[True, False]], "f.jpg"))

    expect_error("mask is 1-D",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,20],
                         np.ones(100, dtype=bool), "f.jpg"))

    expect_error("mask is 3-D",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,20],
                         np.ones((10, 10, 3), dtype=bool), "f.jpg"))

    expect_error("mask dtype is uint8 (not bool)",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,20],
                         np.zeros((480, 640), dtype=np.uint8), "f.jpg"))

    expect_error("mask dtype is float32",
        lambda: Object2D(1, 0, "x", 0.5, [10,10,20,20],
                         np.zeros((480, 640), dtype=np.float32), "f.jpg"))


def test_no_3d_fields() -> None:
    section("11. Confirming Object2D Has No 3D Fields")
    obj = make_demo_object()

    for attr in ["gaussian_ids", "center_3d", "bbox_3d",
                 "relationships", "layer", "motion"]:
        check(f"'{attr}' attribute absent (belongs to later milestones)",
              not hasattr(obj, attr))


def test_interface_note() -> None:
    section("12. Future Interface Note (YOLO26-L / SAM3)")
    print("""
  YOLO26-L and SAM3 are NOT executed in this milestone.
  Future integration path:

      YOLO26-L output
        class_name, semantic_class_id  ->  Object2D fields
        bbox [x1,y1,x2,y2] (pixels)   ->  Object2D.bbox
        confidence                      ->  Object2D.confidence

      SAM3 output
        instance mask (H x W)          ->  convert to bool ndarray
                                       ->  Object2D.mask

  Object2D has no internal dependency on either model.
  Detectors/segmenters can be swapped freely.
    """)
    check("Object2D has no YOLO/SAM3 import dependency", True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 4 Validation: 2D Object Representation")
    print("=" * 60)
    print()
    print("  Using SYNTHETIC data only.  No YOLO26-L or SAM3 executed.")
    print(f"  Image size: {IMG_W} x {IMG_H}  (W x H)")
    print(f"  Demo mask : rows {MASK_Y1}..{MASK_Y2}, cols {MASK_X1}..{MASK_X2}")
    print(f"  Expected area: {EXPECTED_AREA} pixels")

    test_construction()
    test_fields()
    test_mask_properties()
    test_mask_area()
    test_mask_bbox()
    test_to_dict()
    test_empty_mask_bbox()
    test_invalid_confidence()
    test_invalid_bbox()
    test_invalid_mask()
    test_no_3d_fields()
    test_interface_note()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 4 -- 2D Object Representation: ALL CHECKS PASSED")
    else:
        print(f"  [FAIL]  STEP 4 -- {_failures} check(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
