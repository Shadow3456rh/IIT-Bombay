"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_projection_association.py

PURPOSE
-------
Validation script for STEP 5: Projection-Based 2D->3D Association Baseline.

ALL tests use SYNTHETIC data only.
The 17,020-Gaussian demo PLY is NOT loaded here.
YOLO26-L and SAM3 are NOT executed.

==============================================================================
SYNTHETIC SETUP
==============================================================================

Camera (identity W2C -- camera at world origin, looking along +Z):
    width=640, height=480, fx=fy=500, cx=320, cy=240
    R_w2c = I,  t_w2c = [0,0,0]

For the identity camera:
    X_cam = X_world,  Y_cam = Y_world,  Z_cam = Z_world
    u = 500 * (X/Z) + 320
    v = 500 * (Y/Z) + 240

Standard test mask (mask_A):
    480 x 640 bool array
    True in rows [150, 350) and cols [150, 490)
    i.e. mask[150:350, 150:490] = True

HAND-CALCULATED PROJECTIONS FOR STANDARD GAUSSIAN SET
------------------------------------------------------
All Gaussians use Z=5 unless stated otherwise.

Gaussian  | World XYZ           | u (float) | v (float) | u_int | v_int | In mask_A?
----------|---------------------|-----------|-----------|-------|-------|------------
0         | [0,      0,    5]   | 320.0     | 240.0     | 320   | 240   | YES (150<=320<490, 150<=240<350)
1         | [0.2,   -0.1,  5]   | 340.0     | 230.0     | 340   | 230   | YES
2         | [1.7,    0,    5]   | 490.0     | 240.0     | 490   | 240   | NO  (490 not < 490)
3         | [5,      0,    5]   | 820.0     | 240.0     | 820   | 240   | OUTSIDE IMAGE (820>=640)
4         | [0,      0,   -3]   | (behind)  | (behind)  | N/A   | N/A   | BEHIND CAMERA
5         | [0,      0,    0]   | (Z=0)     | (Z=0)     | N/A   | N/A   | Z=0 -> INVALID
6         | [-0.34,  0.2,  5]   | 286.0     | 260.0     | 286   | 260   | YES
7         | [-0.68,  0,    5]   | 252.0     | 240.0     | 252   | 240   | YES

Expected associated Gaussian IDs from mask_A: {0, 1, 6, 7}

Verification:
    Gaussian 0: mask[240, 320]  -> row 240 in [150,350)? YES. col 320 in [150,490)? YES. -> True
    Gaussian 1: mask[230, 340]  -> row 230 in [150,350)? YES. col 340 in [150,490)? YES. -> True
    Gaussian 2: mask[240, 490]  -> col 490 not in [150,490) (490 not < 490)             -> False
    Gaussian 3: u=820 >= 640 -> outside image -> rejected
    Gaussian 4: Z=-3 -> behind camera -> rejected
    Gaussian 5: Z=0  -> invalid -> rejected
    Gaussian 6: mask[260, 286]  -> row 260 in [150,350)? YES. col 286 in [150,490)? YES. -> True
    Gaussian 7: mask[240, 252]  -> row 240 in [150,350)? YES. col 252 in [150,490)? YES. -> True
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from gaussian.scene import GaussianScene
from cameras.camera import Camera
from objects.object import Object2D
from association import associate_gaussians_with_object, AssociationResult

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
    """Identity W2C camera -- camera at world origin, looking along +Z."""
    return Camera(
        width=IMG_W, height=IMG_H,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        R=np.eye(3), t=np.zeros(3),
        convention="w2c",
    )


def make_scene(xyz_list: list) -> GaussianScene:
    """
    Construct a tiny GaussianScene directly from a list of 3D points.
    Does NOT load any PLY file.
    Gaussian IDs = 0, 1, 2, ... (same as PLY vertex indices in the real scene).
    """
    n   = len(xyz_list)
    xyz = np.array(xyz_list, dtype=np.float32)
    ids = np.arange(n, dtype=np.int64)
    return GaussianScene(
        ids              = ids,
        xyz              = xyz,
        opacity_raw      = np.zeros(n, dtype=np.float32),
        scale_raw        = np.zeros((n, 3), dtype=np.float32),
        rotation         = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        ply_vertex_index = ids.copy(),
    )


def make_mask_A() -> np.ndarray:
    """
    Standard test mask:  True in rows [150,350) and cols [150,490).
    All tests using this mask must match the hand-calculated table in the
    module docstring above.
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[150:350, 150:490] = True
    return mask


def make_object(mask: np.ndarray, object_id: int = 17) -> Object2D:
    return Object2D(
        object_id         = object_id,
        semantic_class_id = 2,
        class_name        = "car",
        confidence        = 0.94,
        bbox              = [150.0, 150.0, 489.0, 349.0],
        mask              = mask,
        source_frame      = "synthetic_frame.jpg",
    )


# ---------------------------------------------------------------------------
# STANDARD GAUSSIAN SET
# Hand-calculated pixel projections in module docstring above.
# ---------------------------------------------------------------------------

STANDARD_XYZ = [
    [ 0.0,   0.0,  5.0],   # Gaussian 0 -> (320, 240) -> IN  mask_A
    [ 0.2,  -0.1,  5.0],   # Gaussian 1 -> (340, 230) -> IN  mask_A
    [ 1.7,   0.0,  5.0],   # Gaussian 2 -> (490, 240) -> NOT in mask_A (col 490 == boundary)
    [ 5.0,   0.0,  5.0],   # Gaussian 3 -> (820, 240) -> OUTSIDE IMAGE
    [ 0.0,   0.0, -3.0],   # Gaussian 4 -> behind camera
    [ 0.0,   0.0,  0.0],   # Gaussian 5 -> Z=0, invalid
    [-0.34,  0.2,  5.0],   # Gaussian 6 -> (286, 260) -> IN  mask_A
    [-0.68,  0.0,  5.0],   # Gaussian 7 -> (252, 240) -> IN  mask_A
]

EXPECTED_ASSOCIATED_IDS = {0, 1, 6, 7}   # Gaussians in mask_A


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_association() -> None:
    section("1. Basic Association -- Standard Gaussian Set")

    cam    = make_camera()
    scene  = make_scene(STANDARD_XYZ)
    mask   = make_mask_A()
    obj    = make_object(mask)

    result = associate_gaussians_with_object(scene, cam, obj)

    check("result is AssociationResult",  isinstance(result, AssociationResult))
    check("object_id matches obj.object_id",
          result.object_id == obj.object_id,
          f"expected {obj.object_id}, got {result.object_id}")
    check("count == len(gaussian_ids)",
          result.count == len(result.gaussian_ids))

    got_ids = set(result.gaussian_ids.tolist())
    check(f"associated IDs == {sorted(EXPECTED_ASSOCIATED_IDS)}",
          got_ids == EXPECTED_ASSOCIATED_IDS,
          f"got {sorted(got_ids)}")

    print(f"\n  gaussian_ids: {sorted(got_ids)}")
    print(f"  count       : {result.count}")


def test_in_mask_gaussians() -> None:
    section("2. Gaussians Inside Mask are Associated")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    got = set(r.gaussian_ids.tolist())

    for gid, label in [(0, "[0,0,5]->col320,row240"),
                       (1, "[0.2,-0.1,5]->col340,row230"),
                       (6, "[-0.34,0.2,5]->col286,row260"),
                       (7, "[-0.68,0,5]->col252,row240")]:
        check(f"Gaussian {gid} ({label}) is associated", gid in got)


def test_outside_mask_not_associated() -> None:
    section("3. Gaussian Outside Mask is NOT Associated")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    got = set(r.gaussian_ids.tolist())

    # Gaussian 2 projects to col=490, which is NOT < 490 (boundary excluded)
    check("Gaussian 2 ([1.7,0,5] -> col=490 == boundary) NOT associated",
          2 not in got)


def test_outside_image_rejected() -> None:
    section("4. Gaussian Outside Image is Rejected")

    cam   = make_camera()
    # Only add point 3 (projects to u=820, outside image)
    scene = make_scene([[ 5.0,  0.0, 5.0]])   # u=820
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    check("Gaussian outside image: count == 0", r.count == 0)
    check("gaussian_ids is empty",              len(r.gaussian_ids) == 0)


def test_behind_camera_rejected() -> None:
    section("5. Gaussian Behind Camera is Rejected (Z <= 0)")

    cam = make_camera()
    mask = make_mask_A()
    obj  = make_object(mask)

    # Behind camera (Z < 0)
    scene_neg = make_scene([[0.0, 0.0, -3.0]])
    r_neg = associate_gaussians_with_object(scene_neg, cam, obj)
    check("Z=-3: count == 0 (behind camera)",  r_neg.count == 0)

    # Exactly on image plane (Z = 0)
    scene_z0 = make_scene([[0.0, 0.0, 0.0]])
    r_z0 = associate_gaussians_with_object(scene_z0, cam, obj)
    check("Z=0 : count == 0 (on image plane)", r_z0.count == 0)

    # No crash on Z=0
    check("Z=0 : no exception raised", True)


def test_boundary_pixel_top_left() -> None:
    section("6. Boundary Pixel (0, 0) is Valid")

    # Gaussian that projects exactly to (u=0, v=0):
    #   u = 500*(X/Z)+320 = 0  ->  X/Z = -0.640  ->  X = -3.2  (Z=5)
    #   v = 500*(Y/Z)+240 = 0  ->  Y/Z = -0.480  ->  Y = -2.4
    cam   = make_camera()
    scene = make_scene([[-3.2, -2.4, 5.0]])

    # Mask with ONLY pixel (0,0) = True
    mask_tl = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask_tl[0, 0] = True
    obj = make_object(mask_tl)

    r = associate_gaussians_with_object(scene, cam, obj)

    check("Gaussian at (u=0,v=0) boundary pixel is associated",
          r.count == 1,
          f"count={r.count}, ids={r.gaussian_ids.tolist()}")


def test_boundary_pixel_bottom_right() -> None:
    section("7. Boundary Pixel (W-1, H-1) is Valid")

    # Gaussian that projects exactly to (u=639, v=479):
    #   u = 500*(X/Z)+320 = 639  ->  X/Z = 319/500 = 0.638  ->  X=3.19  (Z=5)
    #   v = 500*(Y/Z)+240 = 479  ->  Y/Z = 239/500 = 0.478  ->  Y=2.39
    cam   = make_camera()
    scene = make_scene([[3.19, 2.39, 5.0]])

    # Mask with ONLY pixel (479, 639) = True
    mask_br = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask_br[IMG_H - 1, IMG_W - 1] = True
    obj = make_object(mask_br)

    r = associate_gaussians_with_object(scene, cam, obj)

    check("Gaussian at (u=639,v=479) bottom-right boundary is associated",
          r.count == 1,
          f"count={r.count}, ids={r.gaussian_ids.tolist()}")


def test_mask_indexing_convention() -> None:
    section("8. Correct NumPy Indexing -- mask[v, u] NOT mask[u, v]")

    # Construct a mask that is True ONLY at (row=200, col=300) -- i.e. v=200, u=300.
    # A point projecting to u=300, v=200 should be associated.
    # The same point checked with mask[u,v] = mask[300, 200] would be False
    # (because we only set mask[200, 300] = True).
    # If the test passes, indexing is correct.

    cam = make_camera()

    # Point projecting to u=300, v=200:
    #   u = 500*(X/Z)+320 = 300  -> X/Z = (300-320)/500 = -0.04 -> X=-0.04*5=-0.20
    #   v = 500*(Y/Z)+240 = 200  -> Y/Z = (200-240)/500 = -0.08 -> Y=-0.08*5=-0.40
    scene = make_scene([[-0.2, -0.4, 5.0]])

    mask_single = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask_single[200, 300] = True   # row=200 (v=200), col=300 (u=300)

    obj = make_object(mask_single)
    r   = associate_gaussians_with_object(scene, cam, obj)

    check("Gaussian projecting to (u=300, v=200) is associated via mask[200, 300]",
          r.count == 1,
          f"count={r.count} (should be 1 if mask[v,u] is used correctly)")

    # Verify: if indexing were mask[u,v], mask[300, 200] would be accessed.
    # mask[300, 200] is False in our mask, so count would be 0.
    mask_transposed_val = mask_single[300, 200]
    check("mask[u, v] = mask[300, 200] is False (confirms swap would give wrong answer)",
          mask_transposed_val == False,
          "This confirms that mask[v,u] and mask[u,v] give different results here")


def test_no_duplicate_ids() -> None:
    section("9. No Duplicate Gaussian IDs in Result")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    ids_list   = r.gaussian_ids.tolist()
    ids_unique = list(set(ids_list))
    check("gaussian_ids has no duplicates",
          len(ids_list) == len(ids_unique),
          f"len={len(ids_list)}, unique={len(ids_unique)}")


def test_deterministic_ordering() -> None:
    section("10. Deterministic Ordering (sorted ascending)")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)

    r1 = associate_gaussians_with_object(scene, cam, obj)
    r2 = associate_gaussians_with_object(scene, cam, obj)

    check("Two runs produce identical gaussian_ids",
          np.array_equal(r1.gaussian_ids, r2.gaussian_ids))
    check("gaussian_ids are sorted ascending",
          np.all(r1.gaussian_ids[:-1] <= r1.gaussian_ids[1:]),
          f"ids={r1.gaussian_ids.tolist()}")


def test_empty_mask() -> None:
    section("11. Empty Mask Returns Zero Associations")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)

    empty_mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    obj = make_object(empty_mask)

    r = associate_gaussians_with_object(scene, cam, obj)

    check("count == 0 for all-False mask",        r.count == 0)
    check("gaussian_ids is empty array",          len(r.gaussian_ids) == 0)
    check("gaussian_ids dtype is int64",          r.gaussian_ids.dtype == np.int64)


def test_object_id_in_result() -> None:
    section("12. Object ID in Result Matches Object2D.object_id")

    cam = make_camera()
    mask = make_mask_A()

    for oid in [0, 17, 999]:
        obj = make_object(mask, object_id=oid)
        scene = make_scene(STANDARD_XYZ)
        r = associate_gaussians_with_object(scene, cam, obj)
        check(f"result.object_id == {oid}", r.object_id == oid)


def test_count_matches_ids() -> None:
    section("13. result.count == len(result.gaussian_ids)")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    check("count == len(gaussian_ids)",
          r.count == len(r.gaussian_ids),
          f"count={r.count}, len={len(r.gaussian_ids)}")


def test_large_batch() -> None:
    section("14. Large Batch of Synthetic Gaussians")

    # Create 2000 Gaussians on a grid in front of the camera (Z=5)
    # All should project into the image; some will land in the mask, others not.
    cam = make_camera()

    N = 2000
    rng = np.random.default_rng(seed=42)   # fixed seed -> deterministic
    # X in [-3, 3], Y in [-2, 2], Z fixed at 5
    X = rng.uniform(-3.0, 3.0, N)
    Y = rng.uniform(-2.0, 2.0, N)
    Z = np.full(N, 5.0)
    xyz_batch = np.column_stack([X, Y, Z]).tolist()

    scene_large = make_scene(xyz_batch)
    mask  = make_mask_A()
    obj   = make_object(mask)

    r = associate_gaussians_with_object(scene_large, cam, obj)

    check("batch result is AssociationResult", isinstance(r, AssociationResult))
    check("count >= 0",                        r.count >= 0)
    check("count <= N",                        r.count <= N)
    check("count == len(gaussian_ids)",        r.count == len(r.gaussian_ids))
    check("no duplicates in batch result",
          len(r.gaussian_ids) == len(np.unique(r.gaussian_ids)))
    check("gaussian_ids sorted",
          np.all(r.gaussian_ids[:-1] <= r.gaussian_ids[1:]) if r.count > 1 else True)

    print(f"\n  {r.count}/{N} Gaussians associated  "
          f"({100*r.count/N:.1f}%)")


def test_to_dict_serialisation() -> None:
    section("15. AssociationResult.to_dict() is JSON-Serialisable")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)
    mask  = make_mask_A()
    obj   = make_object(mask)
    r     = associate_gaussians_with_object(scene, cam, obj)

    d = r.to_dict()

    check("to_dict returns dict",             isinstance(d, dict))
    check("'object_id' in dict",              "object_id"    in d)
    check("'gaussian_ids' in dict",           "gaussian_ids" in d)
    check("'count' in dict",                  "count"        in d)
    check("gaussian_ids is a list (not ndarray)", isinstance(d["gaussian_ids"], list))

    try:
        json_str = json.dumps(d)
        check("to_dict() is JSON-serialisable", True)
    except TypeError as e:
        check("to_dict() is JSON-serialisable", False, str(e))

    print(f"\n  to_dict(): {json.dumps(d)}")


def test_mask_shape_mismatch_raises() -> None:
    section("16. Mask/Camera Shape Mismatch Raises ValueError")

    cam   = make_camera()           # 640 x 480
    scene = make_scene(STANDARD_XYZ)

    # Mask with wrong size
    wrong_mask = np.zeros((200, 300), dtype=bool)
    obj_wrong = Object2D(
        object_id=1, semantic_class_id=0, class_name="x",
        confidence=0.5, bbox=[0, 0, 299, 199],
        mask=wrong_mask, source_frame="f.jpg",
    )

    expect_error(
        "mask shape (200,300) != camera (480,640) raises ValueError",
        lambda: associate_gaussians_with_object(scene, cam, obj_wrong),
    )


def test_full_mask() -> None:
    section("17. All-True Mask Associates All Valid In-Image Gaussians")

    cam   = make_camera()
    scene = make_scene(STANDARD_XYZ)

    full_mask = np.ones((IMG_H, IMG_W), dtype=bool)
    obj = make_object(full_mask)

    r = associate_gaussians_with_object(scene, cam, obj)

    # Gaussians 0,1,2,6,7 project inside the image (valid + in-bounds)
    # Gaussian 3 (u=820) is outside image
    # Gaussian 4 (Z=-3) is behind camera
    # Gaussian 5 (Z=0) is invalid
    # So expected: {0, 1, 2, 6, 7}
    expected_all_valid = {0, 1, 2, 6, 7}
    got = set(r.gaussian_ids.tolist())

    check(f"full mask associates all valid-in-image Gaussians {sorted(expected_all_valid)}",
          got == expected_all_valid,
          f"got {sorted(got)}")


def test_rounding_boundary_case() -> None:
    section("18. Rounding -- Pixel Just Inside vs Just Outside")

    cam = make_camera()
    mask = make_mask_A()   # True in cols [150, 490)
    obj  = make_object(mask)

    # A Gaussian that projects to u=489.4 -> rounds to 489 -> col 489 < 490 -> IN mask
    # u = 500*(X/Z)+320 = 489.4  ->  X/Z = (489.4-320)/500 = 0.3388 -> X=0.3388*5=1.694
    scene_in = make_scene([[1.694, 0.0, 5.0]])
    r_in = associate_gaussians_with_object(scene_in, cam, obj)
    check("u=489.4 rounds to 489, col 489 < 490 -> IN mask",
          r_in.count == 1,
          f"u_float=489.4, u_int=489, count={r_in.count}")

    # A Gaussian that projects to u=489.6 -> rounds to 490 -> col 490 not < 490 -> NOT
    # u=489.6 -> X/Z = (489.6-320)/500 = 0.3392 -> X=1.696
    scene_out = make_scene([[1.696, 0.0, 5.0]])
    r_out = associate_gaussians_with_object(scene_out, cam, obj)
    check("u=489.6 rounds to 490, col 490 NOT < 490 -> NOT in mask",
          r_out.count == 0,
          f"u_float~=489.6, u_int=490, count={r_out.count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 5: Projection Association Baseline")
    print("=" * 60)
    print()
    print("  SYNTHETIC DATA ONLY. No PLY file loaded. No YOLO/SAM3.")
    print(f"  Camera: {IMG_W}x{IMG_H}, fx=fy=500, cx=320, cy=240, identity W2C")
    print(f"  Standard mask: True in rows [150,350), cols [150,490)")

    test_basic_association()
    test_in_mask_gaussians()
    test_outside_mask_not_associated()
    test_outside_image_rejected()
    test_behind_camera_rejected()
    test_boundary_pixel_top_left()
    test_boundary_pixel_bottom_right()
    test_mask_indexing_convention()
    test_no_duplicate_ids()
    test_deterministic_ordering()
    test_empty_mask()
    test_object_id_in_result()
    test_count_matches_ids()
    test_large_batch()
    test_to_dict_serialisation()
    test_mask_shape_mismatch_raises()
    test_full_mask()
    test_rounding_boundary_case()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 5 -- Projection Association Baseline: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 5 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
