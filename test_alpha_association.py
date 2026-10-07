"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_alpha_association.py

PURPOSE
-------
Validation script for STEP 6: Alpha-Contribution Association Prototype.

All tests use SYNTHETIC data only.
No PLY file is loaded.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from gaussian.scene import GaussianScene
from cameras.camera import Camera
from objects.object import Object2D
from association.alpha_association import (
    associate_gaussians_alpha,
    AlphaAssociationResult,
    _compute_footprint_contribution
)
from association.projection_association import associate_gaussians_with_object

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


# ---------------------------------------------------------------------------
# Synthetic constructors
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480

def make_camera() -> Camera:
    return Camera(
        width=IMG_W, height=IMG_H,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        R=np.eye(3), t=np.zeros(3),
        convention="w2c",
    )

def alpha_to_raw(a: float) -> float:
    a = max(1e-6, min(1.0 - 1e-6, a))
    return float(np.log(a / (1.0 - a)))

def scale_to_raw(s: float) -> float:
    return float(np.log(s))

def make_scene_alpha(gaussians: list) -> GaussianScene:
    """
    gaussians: list of [x, y, z, alpha, scale]
    """
    n = len(gaussians)
    xyz = np.zeros((n, 3), dtype=np.float32)
    opacity_raw = np.zeros(n, dtype=np.float32)
    scale_raw = np.zeros((n, 3), dtype=np.float32)
    rotation = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    ids = np.arange(n, dtype=np.int64)

    for i, g in enumerate(gaussians):
        xyz[i] = [g[0], g[1], g[2]]
        opacity_raw[i] = alpha_to_raw(g[3])
        scale_raw[i, :] = scale_to_raw(g[4])

    return GaussianScene(
        ids=ids,
        xyz=xyz,
        opacity_raw=opacity_raw,
        scale_raw=scale_raw,
        rotation=rotation,
        ply_vertex_index=ids.copy(),
    )

def make_mask_half() -> np.ndarray:
    """Mask covers left half of the image (cols 0 to 319)."""
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[:, :IMG_W // 2] = True
    return mask

def make_object(mask: np.ndarray, object_id: int = 17) -> Object2D:
    return Object2D(
        object_id=object_id,
        semantic_class_id=1,
        class_name="demo",
        confidence=0.9,
        bbox=[0, 0, IMG_W-1, IMG_H-1],
        mask=mask,
        source_frame="synthetic.jpg",
    )


# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_2_3_overlap_ratios() -> None:
    section("1/2/3. Overlap Ratios (Inside, Outside, Crossing Boundary)")

    cam = make_camera()
    mask = make_mask_half()  # Left half is True
    obj = make_object(mask)

    # Point 0: Deep inside left half (u=120)
    # Point 1: Deep inside right half (u=520, outside mask)
    # Point 2: Exactly on the boundary (u=320)
    # Z=5, fx=500, cx=320 -> u = 500*(X/5)+320 = 100*X + 320
    # X=-2 -> u=120. X=2 -> u=520. X=0 -> u=320.
    
    scene = make_scene_alpha([
        [-2.0, 0.0, 5.0, 0.8, 0.1],   # 0: INSIDE
        [ 2.0, 0.0, 5.0, 0.8, 0.1],   # 1: OUTSIDE
        [ 0.0, 0.0, 5.0, 0.8, 0.1],   # 2: CROSSING
    ])

    # With threshold 0.0, we get the exact scores for everything
    r = associate_gaussians_alpha(scene, cam, obj, threshold=0.0)

    s0 = r.association_scores.get(0, 0.0)
    s1 = r.association_scores.get(1, 0.0)
    s2 = r.association_scores.get(2, 0.0)

    check("Gaussian entirely inside mask -> ratio near 1.0", s0 > 0.99, f"score={s0}")
    check("Gaussian entirely outside mask -> ratio near 0.0", s1 < 0.01, f"score={s1}")
    check("Gaussian crossing boundary -> ratio near 0.5", 0.4 < s2 < 0.6, f"score={s2}")


def test_4_5_opacity_scaling() -> None:
    section("4/5. Opacity Scaling and Absolute Contribution")

    # Test the internal helper to verify absolute contribution increases
    mask = make_mask_half()
    
    # Base alpha
    obj_c1, tot_c1 = _compute_footprint_contribution(
        u_c=320.0, v_c=240.0, rad=10.0, alpha=0.5, W=IMG_W, H=IMG_H, mask=mask
    )
    # Double alpha
    obj_c2, tot_c2 = _compute_footprint_contribution(
        u_c=320.0, v_c=240.0, rad=10.0, alpha=1.0, W=IMG_W, H=IMG_H, mask=mask
    )

    check("Higher opacity increases absolute contribution",
          tot_c2 > tot_c1 and obj_c2 > obj_c1,
          f"tot_c1={tot_c1:.2f}, tot_c2={tot_c2:.2f}")

    ratio1 = obj_c1 / tot_c1
    ratio2 = obj_c2 / tot_c2
    
    check("Association ratio is independent of uniform opacity scaling",
          abs(ratio1 - ratio2) < 1e-6,
          f"ratio1={ratio1:.4f}, ratio2={ratio2:.4f}")


def test_6_larger_footprint() -> None:
    section("6. Larger Gaussian Footprint Produces Partial Overlap")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)

    # Both Gaussians at X=-0.5 (u=270, well inside the left half)
    # But Gaussian 1 has a huge scale, so its footprint spills over the boundary.
    scene = make_scene_alpha([
        [-0.5, 0.0, 5.0, 0.9, 0.05],  # 0: small radius
        [-0.5, 0.0, 5.0, 0.9, 1.00],  # 1: massive radius
    ])

    r = associate_gaussians_alpha(scene, cam, obj, threshold=0.0)
    s0 = r.association_scores.get(0, 0.0)
    s1 = r.association_scores.get(1, 0.0)

    check("Small footprint is fully inside (ratio ~1)", s0 > 0.99, f"score={s0}")
    check("Large footprint spills out (partial overlap)", s1 < 0.9, f"score={s1}")


def test_7_8_9_camera_rejections() -> None:
    section("7/8/9. Invalid Camera Projections Handled Safely")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)

    scene = make_scene_alpha([
        [ 0.0, 0.0, -5.0, 0.5, 0.1],  # 0: Z < 0 (behind)
        [ 0.0, 0.0,  0.0, 0.5, 0.1],  # 1: Z = 0 (invalid)
        [10.0, 0.0,  5.0, 0.5, 0.1],  # 2: u = 1320 (outside image entirely)
    ])

    r = associate_gaussians_alpha(scene, cam, obj, threshold=0.0)
    check("Behind camera rejected", 0 not in r.gaussian_ids)
    check("Z=0 rejected", 1 not in r.gaussian_ids)
    check("Outside image rejected safely", 2 not in r.gaussian_ids)


def test_10_11_mask_extremes() -> None:
    section("10/11. Empty and All-True Masks")

    cam = make_camera()
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.5, 0.1]])

    obj_empty = make_object(np.zeros((IMG_H, IMG_W), dtype=bool))
    r_empty = associate_gaussians_alpha(scene, cam, obj_empty, threshold=0.0)
    check("Empty mask produces zero association", r_empty.count == 0)

    obj_full = make_object(np.ones((IMG_H, IMG_W), dtype=bool))
    r_full = associate_gaussians_alpha(scene, cam, obj_full, threshold=0.99)
    check("All-True mask associates in-image Gaussian", r_full.count == 1)
    
    score = r_full.association_scores.get(0, 0.0)
    check("All-True mask ratio is 1.0", score >= 0.999)


def test_12_13_thresholds() -> None:
    section("12/13. Configurable Threshold and Determinism")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)

    # Point at u=320 (exactly on boundary). Ratio will be exactly ~0.5.
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.5, 0.1]])

    # Get the exact score
    r0 = associate_gaussians_alpha(scene, cam, obj, threshold=0.0)
    score = r0.association_scores[0]
    
    # Check slightly below threshold
    r_below = associate_gaussians_alpha(scene, cam, obj, threshold=score - 0.01)
    check(f"Accepted when threshold < {score}", r_below.count == 1)

    # Check slightly above threshold
    r_above = associate_gaussians_alpha(scene, cam, obj, threshold=score + 0.01)
    check(f"Rejected when threshold > {score}", r_above.count == 0)


def test_14_15_16_17_result_format() -> None:
    section("14/15/16/17. Result Format Validations")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)

    scene = make_scene_alpha([
        [-1.0, 0.0, 5.0, 0.5, 0.1],
        [-0.5, 0.0, 5.0, 0.5, 0.1],
    ])

    r = associate_gaussians_alpha(scene, cam, obj, threshold=0.1)

    ids = r.gaussian_ids.tolist()
    check("No duplicate IDs", len(ids) == len(set(ids)))
    check("Deterministic sorting", ids == sorted(ids))
    check("Result count matches length", r.count == len(ids))

    all_valid = all(0.0 <= s <= 1.0 for s in r.association_scores.values())
    check("All association_scores in [0, 1]", all_valid)


def test_18_json() -> None:
    section("18. to_dict() is JSON serialisable")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)
    scene = make_scene_alpha([[-1.0, 0.0, 5.0, 0.5, 0.1]])

    r = associate_gaussians_alpha(scene, cam, obj)
    d = r.to_dict()

    try:
        json_str = json.dumps(d)
        check("to_dict() serialises correctly", True)
    except Exception as e:
        check("to_dict() serialises correctly", False, str(e))


def test_19_20_errors() -> None:
    section("19/20. Invalid Inputs Raise Errors")

    cam = make_camera()
    mask = make_mask_half()
    obj = make_object(mask)
    scene = make_scene_alpha([])

    expect_error("threshold = 1.5", lambda: associate_gaussians_alpha(scene, cam, obj, threshold=1.5))
    expect_error("threshold = -0.1", lambda: associate_gaussians_alpha(scene, cam, obj, threshold=-0.1))

    bad_mask = np.zeros((100, 100), dtype=bool)
    obj_bad = make_object(bad_mask)
    expect_error("mask shape mismatch", lambda: associate_gaussians_alpha(scene, cam, obj_bad))


def test_21_comparison_with_step5() -> None:
    section("21. Comparison: Step 5 vs Step 6 on Boundary")

    cam = make_camera()
    mask = make_mask_half() # cols < 320 are True
    obj = make_object(mask)

    # Point projects slightly outside the mask (e.g. u = 322).
    # Step 5 (centre only) will completely REJECT it.
    # Step 6 (footprint) will register a partial overlap.
    
    # X=0.02 -> u = 500*(0.02/5) + 320 = 2 + 320 = 322
    # radius is large enough to spill back into u<320.
    scene = make_scene_alpha([[0.02, 0.0, 5.0, 0.9, 0.1]])

    # Run Step 5
    r5 = associate_gaussians_with_object(scene, cam, obj)
    
    # Run Step 6 (low threshold just to detect overlap)
    r6 = associate_gaussians_alpha(scene, cam, obj, threshold=0.1)

    print(f"  Step 5 associated count: {r5.count}")
    print(f"  Step 6 associated count: {r6.count}")
    
    check("Step 5 rejects centre outside mask", r5.count == 0)
    check("Step 6 detects partial footprint overlap", r6.count == 1)
    if r6.count == 1:
        score = r6.association_scores[0]
        print(f"         Step 6 overlap score = {score:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 6: Alpha-Contribution Association Prototype")
    print("=" * 60)
    print()

    test_1_2_3_overlap_ratios()
    test_4_5_opacity_scaling()
    test_6_larger_footprint()
    test_7_8_9_camera_rejections()
    test_10_11_mask_extremes()
    test_12_13_thresholds()
    test_14_15_16_17_result_format()
    test_18_json()
    test_19_20_errors()
    test_21_comparison_with_step5()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 6 -- Alpha-Contribution Association: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 6 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
