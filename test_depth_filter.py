"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_depth_filter.py

PURPOSE
-------
Validation script for STEP 7: Depth-Aware Filtering.

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
from association.alpha_association import AlphaAssociationResult
from association.depth_evidence import DepthEvidence
from association.depth_filter import filter_by_depth


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


# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_to_5_depth_differences() -> None:
    section("1/2/3/4/5. Depth Differences and Thresholds")
    
    cam = make_camera()
    
    # 5 points projecting to the same center
    # Target depth map uniformly 5.0
    # Point 0: Z=5.0 (perfect)
    # Point 1: Z=5.4 (within 10%)
    # Point 2: Z=5.6 (above 10% threshold)
    # Point 3: Z=1.0 (significantly closer)
    # Point 4: Z=10.0 (significantly farther)
    
    scene = make_scene_alpha([
        [0.0, 0.0, 5.0, 0.9, 0.1],
        [0.0, 0.0, 5.4, 0.9, 0.1],
        [0.0, 0.0, 5.6, 0.9, 0.1],
        [0.0, 0.0, 1.0, 0.9, 0.1],
        [0.0, 0.0, 10.0, 0.9, 0.1],
    ])
    
    alpha_res = AlphaAssociationResult(
        object_id=17,
        gaussian_ids=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        association_scores={i: 1.0 for i in range(5)},
        count=5
    )
    
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    evidence = DepthEvidence(depth_map, valid_mask)
    
    r = filter_by_depth(alpha_res, scene, cam, evidence, max_relative_depth_error=0.10)
    
    ids = r.gaussian_ids.tolist()
    check("Perfect depth match passes", 0 in ids)
    check("Small difference within threshold passes", 1 in ids)
    check("Difference above threshold rejected", 2 not in ids)
    check("Significantly closer rejected", 3 not in ids)
    check("Significantly farther rejected", 4 not in ids)


def test_6_7_missing_evidence_policies() -> None:
    section("6/7. Missing Depth Evidence Policies")
    
    cam = make_camera()
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.9, 0.1]])
    
    alpha_res = AlphaAssociationResult(17, np.array([0], dtype=np.int64), {0: 1.0}, 1)
    
    # All invalid mask
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    valid_mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    evidence = DepthEvidence(depth_map, valid_mask)
    
    r_reject = filter_by_depth(alpha_res, scene, cam, evidence, insufficient_depth_policy="reject")
    check("Default 'reject' policy behaves correctly", r_reject.count == 0)
    
    r_keep = filter_by_depth(alpha_res, scene, cam, evidence, insufficient_depth_policy="keep")
    check("'keep' policy behaves correctly", r_keep.count == 1)
    check("Missing depth puts None in observed_depths", r_keep.observed_depths[0] is None)


def test_8_9_10_evidence_validation() -> None:
    section("8/9/10/19/20. Evidence Validation & Invalid Data")
    
    depth_map = np.ones((IMG_H, IMG_W), dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    
    # Non-positive
    bad_depth = depth_map.copy()
    bad_depth[0, 0] = -1.0
    expect_error("Non-positive depth with valid_mask=True", lambda: DepthEvidence(bad_depth, valid_mask))
    
    # NaN
    bad_depth2 = depth_map.copy()
    bad_depth2[0, 0] = np.nan
    expect_error("NaN depth with valid_mask=True", lambda: DepthEvidence(bad_depth2, valid_mask))
    
    # Shape mismatch
    expect_error("Invalid depth map shape", lambda: DepthEvidence(np.ones((100, 100)), valid_mask))
    
    # Validation passes if invalid values are masked out
    bad_depth3 = depth_map.copy()
    bad_depth3[0, 0] = np.nan
    safe_mask = valid_mask.copy()
    safe_mask[0, 0] = False
    
    try:
        DepthEvidence(bad_depth3, safe_mask)
        check("NaN depths are ignored if valid_mask is False", True)
    except Exception as e:
        check("NaN depths are ignored if valid_mask is False", False, str(e))


def test_11_median_robustness() -> None:
    section("11. Median Depth Calculation Robustness")
    
    cam = make_camera()
    # Point at u=320, v=240, rad is small
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.9, 0.05]])
    alpha_res = AlphaAssociationResult(17, np.array([0], dtype=np.int64), {0: 1.0}, 1)
    
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    # Add one massive outlier
    depth_map[240, 320] = 1000.0
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    
    evidence = DepthEvidence(depth_map, valid_mask)
    r = filter_by_depth(alpha_res, scene, cam, evidence)
    
    check("Median calculation robust to one outlier", r.count == 1)
    check("Observed depth is 5.0 despite outlier", r.observed_depths[0] == 5.0)


def test_12_13_14_config_and_boundary() -> None:
    section("12/13/14. Depth Units, Configs, Boundary Pixels")
    
    cam = make_camera()
    scene = make_scene_alpha([
        [-2.0, 0.0, 5.0, 0.9, 0.1],  # boundary
    ])
    alpha_res = AlphaAssociationResult(17, np.array([0], dtype=np.int64), {0: 1.0}, 1)
    
    # depth differs by 19%
    depth_map = np.full((IMG_H, IMG_W), 4.2, dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    evidence = DepthEvidence(depth_map, valid_mask)
    
    r_tight = filter_by_depth(alpha_res, scene, cam, evidence, max_relative_depth_error=0.10)
    check("Rejected by 10% threshold", r_tight.count == 0)
    
    r_loose = filter_by_depth(alpha_res, scene, cam, evidence, max_relative_depth_error=0.20)
    check("Accepted by 20% threshold", r_loose.count == 1)


def test_16_17_18_confidence() -> None:
    section("16/17/18. Confidence Handling")
    
    cam = make_camera()
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.9, 0.1]])
    alpha_res = AlphaAssociationResult(17, np.array([0], dtype=np.int64), {0: 1.0}, 1)
    
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    
    evidence_none = DepthEvidence(depth_map, valid_mask)
    r_none = filter_by_depth(alpha_res, scene, cam, evidence_none)
    check("Confidence absent -> None", r_none.depth_confidences[0] is None)
    
    conf_map = np.full((IMG_H, IMG_W), 0.8, dtype=np.float32)
    evidence_conf = DepthEvidence(depth_map, valid_mask, confidence_map=conf_map)
    r_conf = filter_by_depth(alpha_res, scene, cam, evidence_conf)
    check("Confidence present -> Calculated", r_conf.depth_confidences[0] == 0.8)
    
    bad_conf = conf_map.copy()
    bad_conf[0, 0] = 1.5
    expect_error("Invalid confidence raises error", lambda: DepthEvidence(depth_map, valid_mask, confidence_map=bad_conf))


def test_21_to_24_formatting() -> None:
    section("21/22/23/24. Format and JSON")
    
    cam = make_camera()
    scene = make_scene_alpha([[0.0, 0.0, 5.0, 0.9, 0.1], [0.1, 0.0, 5.0, 0.9, 0.1]])
    alpha_res = AlphaAssociationResult(17, np.array([0, 1], dtype=np.int64), {0: 1.0, 1: 1.0}, 2)
    
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    evidence = DepthEvidence(depth_map, valid_mask)
    
    r = filter_by_depth(alpha_res, scene, cam, evidence)
    
    ids = r.gaussian_ids.tolist()
    check("Deterministic ordering", ids == sorted(ids))
    check("No duplicates", len(ids) == len(set(ids)))
    check("Count matches", r.count == len(ids))
    
    try:
        json.dumps(r.to_dict())
        check("JSON serialisation works", True)
    except Exception as e:
        check("JSON serialisation works", False, str(e))


def test_25_key_benefit_demo() -> None:
    section("25. Key Benefit Demonstration")
    
    # 2 Gaussians project to exactly the same region.
    # Object depth is 5.0
    # Gaussian 0 is the true object (depth 5.0)
    # Gaussian 1 is the background (depth 15.0)
    
    cam = make_camera()
    scene = make_scene_alpha([
        [0.0, 0.0, 5.0, 0.9, 0.1],    # Target
        [0.0, 0.0, 15.0, 0.9, 0.3],   # Background
    ])
    
    # Step 6 accepts both
    alpha_res = AlphaAssociationResult(17, np.array([0, 1], dtype=np.int64), {0: 1.0, 1: 1.0}, 2)
    
    # Step 7 filters
    depth_map = np.full((IMG_H, IMG_W), 5.0, dtype=np.float32)
    valid_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    evidence = DepthEvidence(depth_map, valid_mask)
    
    r = filter_by_depth(alpha_res, scene, cam, evidence)
    
    check("True object Gaussian is retained", 0 in r.gaussian_ids)
    check("Background Gaussian is rejected", 1 not in r.gaussian_ids)
    
    if r.count == 1 and 0 in r.gaussian_ids:
        print(f"         Only Gaussian 0 (depth 5.0) survived filtering.")


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 7: Depth-Aware Filtering")
    print("=" * 60)
    print()
    
    test_1_to_5_depth_differences()
    test_6_7_missing_evidence_policies()
    test_8_9_10_evidence_validation()
    test_11_median_robustness()
    test_12_13_14_config_and_boundary()
    test_16_17_18_confidence()
    test_21_to_24_formatting()
    test_25_key_benefit_demo()
    
    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 7 -- Depth-Aware Filtering: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 7 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
