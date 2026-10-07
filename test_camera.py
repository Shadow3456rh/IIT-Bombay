"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_camera.py

PURPOSE
-------
Validation script for STEP 3: Camera / Coordinate System Module.

Tests are performed entirely with a SYNTHETIC camera.

IMPORTANT: No real IITB camera parameters are used here.
The actual IITB camera data will come from the Layer-1 / COLMAP / Nerfstudio
pipeline and must be verified against the correct coordinate convention before
use.

SYNTHETIC CAMERA SETUP
-----------------------
    Width  = 640 px,  Height = 480 px
    fx = fy = 500 px
    cx = 320 px,  cy = 240 px

    Extrinsics: identity rotation, zero translation  (W2C convention)
    -> R_w2c = I(3x3),  t_w2c = [0, 0, 0]
    -> Camera sits at the world origin, looking along +Z.

EXPECTED PROJECTION RESULTS (by hand)
--------------------------------------
Convention: X_cam = R_w2c @ X_world + t_w2c
            u = fx*(X/Z) + cx,  v = fy*(Y/Z) + cy

    World [0, 0, 5]  -> cam [0, 0, 5]  -> u=320,      v=240   (dead centre)
    World [1, 0, 5]  -> cam [1, 0, 5]  -> u=420,      v=240   (right)
    World [0, 1, 5]  -> cam [0, 1, 5]  -> u=320,      v=340   (down, Y-down convention)
    World [-1, 0, 5] -> cam [-1, 0, 5] -> u=220,      v=240   (left)
    World [0, 0,-5]  -> cam [0, 0,-5]  -> INVALID     (behind camera, Z<0)
    World [0, 0, 0]  -> cam [0, 0, 0]  -> INVALID     (Z=0, on image plane)
    World [100,0, 5] -> cam [100,0, 5] -> u=10320     -> in front but OUT OF BOUNDS
"""

import sys
import math
from pathlib import Path

# Allow running from project root:  python src/test_camera.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from cameras.camera import Camera
from cameras.projection import project_points, in_image_bounds, ProjectionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        _failures += 1


def approx_eq(a: float, b: float, tol: float = 0.5) -> bool:
    """True if |a - b| <= tol (pixel-level tolerance by default)."""
    return abs(a - b) <= tol


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# SYNTHETIC CAMERA
# ---------------------------------------------------------------------------

def make_synthetic_camera() -> Camera:
    """
    Create a simple test camera.

    W2C convention:
        R_w2c = I  (no rotation)
        t_w2c = 0  (camera at world origin)
    The camera looks along world +Z.
    Image: 640 x 480, fx=fy=500, cx=320, cy=240.
    """
    return Camera(
        width=640, height=480,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        R=np.eye(3),
        t=np.zeros(3),
        convention="w2c",
    )


# ---------------------------------------------------------------------------
# TEST SUITE
# ---------------------------------------------------------------------------

def test_intrinsics(cam: Camera) -> None:
    section("1. Intrinsic Matrix K")

    K = cam.K
    check("K shape is (3,3)",       K.shape == (3, 3))
    check("K[0,0] == fx (500)",     K[0, 0] == 500.0)
    check("K[1,1] == fy (500)",     K[1, 1] == 500.0)
    check("K[0,2] == cx (320)",     K[0, 2] == 320.0)
    check("K[1,2] == cy (240)",     K[1, 2] == 240.0)
    check("K[2,2] == 1",            K[2, 2] == 1.0)
    check("K[0,1] == 0 (no skew)", K[0, 1] == 0.0)
    check("K[1,0] == 0",            K[1, 0] == 0.0)
    check("K[2,0] == 0",            K[2, 0] == 0.0)
    check("K[2,1] == 0",            K[2, 1] == 0.0)

    print(f"\n  K =\n{K}")


def test_extrinsics(cam: Camera) -> None:
    section("2. Extrinsic Matrices (W2C and C2W)")

    R_w2c = cam.R_w2c
    t_w2c = cam.t_w2c
    T_w2c = cam.T_w2c
    T_c2w = cam.T_c2w

    # Shapes
    check("R_w2c shape is (3,3)",   R_w2c.shape == (3, 3))
    check("t_w2c shape is (3,)",    t_w2c.shape == (3,))
    check("T_w2c shape is (4,4)",   T_w2c.shape == (4, 4))
    check("T_c2w shape is (4,4)",   T_c2w.shape == (4, 4))

    # For identity camera: R_w2c = I, t_w2c = 0
    check("R_w2c == I (synthetic)", np.allclose(R_w2c, np.eye(3)))
    check("t_w2c == 0 (synthetic)", np.allclose(t_w2c, np.zeros(3)))

    # T_w2c @ T_c2w should be identity
    product = T_w2c @ T_c2w
    check("T_w2c @ T_c2w == I4",
          np.allclose(product, np.eye(4), atol=1e-10),
          f"max deviation: {np.max(np.abs(product - np.eye(4))):.2e}")

    # Camera centre in world = t_c2w = 0 for identity camera
    centre = cam.camera_centre_world
    check("camera_centre_world == [0,0,0] (synthetic)",
          np.allclose(centre, np.zeros(3)))

    print(f"\n  T_w2c =\n{T_w2c}")
    print(f"\n  T_c2w =\n{T_c2w}")


def test_world_to_camera(cam: Camera) -> None:
    section("3. world_to_camera() Transformation")

    # For identity W2C: cam coords == world coords
    pts_world = np.array([
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
    ])
    pts_cam = cam.world_to_camera(pts_world)

    check("output shape (3,3)",     pts_cam.shape == (3, 3))
    check("[0,0,5] -> cam [0,0,5]", np.allclose(pts_cam[0], [0, 0, 5]))
    check("[1,0,5] -> cam [1,0,5]", np.allclose(pts_cam[1], [1, 0, 5]))
    check("[0,1,5] -> cam [0,1,5]", np.allclose(pts_cam[2], [0, 1, 5]))

    # Single-point input
    single = cam.world_to_camera([3.0, 4.0, 7.0])
    check("single point shape (1,3)", single.shape == (1, 3))
    check("single point value correct", np.allclose(single[0], [3, 4, 7]))


def test_projection_basic(cam: Camera) -> None:
    section("4. project_points() -- Basic Projection")

    test_pts = np.array([
        [0.0,  0.0, 5.0],   # centre
        [1.0,  0.0, 5.0],   # right
        [0.0,  1.0, 5.0],   # down
        [-1.0, 0.0, 5.0],   # left
    ])

    # Hand-calculated expected pixels:
    # u = 500*(X/Z)+320,  v = 500*(Y/Z)+240
    expected = np.array([
        [320.0, 240.0],   # [0,0,5]
        [420.0, 240.0],   # [1,0,5]  -> +100 u
        [320.0, 340.0],   # [0,1,5]  -> +100 v  (Y-down)
        [220.0, 240.0],   # [-1,0,5] -> -100 u
    ])

    result: ProjectionResult = project_points(cam, test_pts)

    check("pixels shape (4,2)",         result.pixels.shape == (4, 2))
    check("depth shape (4,)",           result.depth.shape  == (4,))
    check("valid shape (4,)",           result.valid.shape  == (4,))
    check("all 4 points valid",         np.all(result.valid))

    for i, (exp, label) in enumerate(zip(expected, [
        "[0,0,5] -> (320,240)",
        "[1,0,5] -> (420,240) right",
        "[0,1,5] -> (320,340) down (Y-down convention)",
        "[-1,0,5]-> (220,240) left",
    ])):
        u_ok = approx_eq(result.pixels[i, 0], exp[0])
        v_ok = approx_eq(result.pixels[i, 1], exp[1])
        check(label, u_ok and v_ok,
              f"got u={result.pixels[i,0]:.2f}, v={result.pixels[i,1]:.2f}; "
              f"expected u={exp[0]}, v={exp[1]}")

    # Depth must equal Z in camera space (for identity camera = world Z)
    for i, z_exp in enumerate([5.0, 5.0, 5.0, 5.0]):
        check(f"depth[{i}] == {z_exp}",
              approx_eq(result.depth[i], z_exp, tol=1e-4))

    # +X moves pixel RIGHT (u increases)
    check("positive X -> larger u",
          result.pixels[1, 0] > result.pixels[0, 0])

    # +Y moves pixel DOWN (v increases) -- Y-down camera convention
    check("positive Y -> larger v (Y-down convention)",
          result.pixels[2, 1] > result.pixels[0, 1])


def test_projection_behind_camera(cam: Camera) -> None:
    section("5. project_points() -- Points Behind Camera (Z <= 0)")

    pts = np.array([
        [0.0, 0.0, -5.0],   # clearly behind
        [0.0, 0.0,  0.0],   # exactly on image plane
        [1.0, 2.0, -1.0],   # behind and off-axis
    ])

    result = project_points(cam, pts)

    check("all 3 points marked invalid",      np.all(~result.valid))
    check("pixels[0] are NaN (behind)",       np.all(np.isnan(result.pixels[0])))
    check("pixels[1] are NaN (Z=0)",          np.all(np.isnan(result.pixels[1])))
    check("pixels[2] are NaN (behind+off)",   np.all(np.isnan(result.pixels[2])))
    check("no ZeroDivisionError for Z=0",     True)   # reaching here proves it

    # Depth should still be set correctly (negative Z)
    check("depth[0] == -5",   approx_eq(result.depth[0], -5.0, tol=1e-4))
    check("depth[1] ==  0",   approx_eq(result.depth[1],  0.0, tol=1e-4))

    print(f"\n  pixels (should be NaN):")
    for i, row in enumerate(result.pixels):
        print(f"    [{i}] u={row[0]}, v={row[1]}   valid={result.valid[i]}")


def test_image_bounds(cam: Camera) -> None:
    section("6. in_image_bounds() -- Image Frame Check")

    # Mix of valid projections + one out-of-bounds
    pts = np.array([
        [0.0,   0.0,  5.0],    # centre -> (320, 240) -> IN
        [1.0,   0.0,  5.0],    # -> (420, 240) -> IN
        [100.0, 0.0,  5.0],    # -> u=10320, OUT of bounds (right)
        [0.0, 100.0,  5.0],    # -> v=10240, OUT of bounds (below)
        [0.0,   0.0, -5.0],    # behind -> NaN -> OUT
    ])

    result = project_points(cam, pts)
    bounds = in_image_bounds(cam, result.pixels)

    check("bounds shape (5,)",           bounds.shape == (5,))
    check("centre (320,240) IN bounds",  bool(bounds[0]))
    check("(420,240) IN bounds",         bool(bounds[1]))
    check("u=10320 OUT of bounds",       not bool(bounds[2]))
    check("v=10240 OUT of bounds",       not bool(bounds[3]))
    check("NaN pixel (behind) OUT",      not bool(bounds[4]))

    # Demonstrate visibility != bounds: point can be in-front but out-of-frame
    check("point[2] valid (in-front) but NOT in bounds",
          bool(result.valid[2]) and not bool(bounds[2]))

    print(f"\n  valid mask : {result.valid.tolist()}")
    print(f"  bounds mask: {bounds.tolist()}")
    print(f"  pixels:")
    for i, (px, v, b) in enumerate(
            zip(result.pixels, result.valid, bounds)):
        print(f"    [{i}] u={px[0]:>10.1f}  v={px[1]:>10.1f} "
              f"| valid={v}  in_bounds={b}")


def test_c2w_convention(cam: Camera) -> None:
    section("7. C2W Input Convention")

    # Create an IDENTICAL camera but supply R and t in C2W convention.
    # For identity camera: R_c2w = I, t_c2w = [0,0,0]
    cam_c2w = Camera(
        width=640, height=480,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        R=np.eye(3),
        t=np.zeros(3),
        convention="c2w",       # <-- C2W input
    )

    # Both cameras should produce the same projections
    pts = np.array([[1.0, 2.0, 5.0]])
    res_w2c = project_points(cam,      pts)
    res_c2w = project_points(cam_c2w,  pts)

    check("C2W-input cam produces same pixels as W2C-input cam",
          np.allclose(res_w2c.pixels, res_c2w.pixels, equal_nan=True))
    check("C2W-input cam T_w2c @ T_c2w == I4",
          np.allclose(cam_c2w.T_w2c @ cam_c2w.T_c2w, np.eye(4), atol=1e-10))


def test_non_identity_rotation(cam: Camera) -> None:
    section("8. Non-Identity Rotation (round-trip consistency)")

    # Camera rotated 90 degrees around world Y axis (looking along -X)
    # R_w2c such that world +X becomes camera -Z  (camera looks along world -X)
    # Rotation: angle=90 deg around Y
    theta = math.pi / 2
    R_y90 = np.array([
        [ math.cos(theta), 0, math.sin(theta)],
        [              0,  1,              0],
        [-math.sin(theta), 0, math.cos(theta)],
    ])
    # Camera at world origin, looking along -X (world coords)
    cam_rot = Camera(
        width=640, height=480,
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        R=R_y90,
        t=np.zeros(3),
        convention="w2c",
    )

    # World point [-5, 0, 0] should be directly in front of this camera
    # Camera-space: X_cam = R_y90 @ [-5,0,0] = [-5*cos90, 0, -5*(-sin90)]
    #                                         = [0, 0, 5]  -> Z=5, in front
    pt_world = np.array([[-5.0, 0.0, 0.0]])
    pt_cam   = cam_rot.world_to_camera(pt_world)

    check("rotated cam: [-5,0,0] world -> Z_cam > 0 (in front)",
          float(pt_cam[0, 2]) > 0,
          f"Z_cam = {float(pt_cam[0,2]):.4f}")

    result = project_points(cam_rot, pt_world)
    check("rotated cam: [-5,0,0] projects to centre (approx)",
          bool(result.valid[0]) and
          approx_eq(result.pixels[0, 0], 320.0) and
          approx_eq(result.pixels[0, 1], 240.0),
          f"u={result.pixels[0,0]:.2f}, v={result.pixels[0,1]:.2f}")

    # Consistency: T_w2c @ T_c2w == I
    check("T_w2c @ T_c2w == I4 for rotated camera",
          np.allclose(cam_rot.T_w2c @ cam_rot.T_c2w, np.eye(4), atol=1e-10))


def test_single_point_input(cam: Camera) -> None:
    section("9. Single-Point and Edge-Case Input")

    # Single-point list input
    result = project_points(cam, [0.0, 0.0, 5.0])
    check("single list input: shape (1,2)",  result.pixels.shape == (1, 2))
    check("single list input: valid",        bool(result.valid[0]))
    check("single list input: u==320",       approx_eq(result.pixels[0, 0], 320.0))

    # Large batch: 1000 points along Z=5 plane
    N = 1000
    pts = np.zeros((N, 3))
    pts[:, 2] = 5.0
    result_batch = project_points(cam, pts)
    check(f"batch {N} points: pixels shape ({N},2)",
          result_batch.pixels.shape == (N, 2))
    check(f"batch {N} points: all valid",
          np.all(result_batch.valid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 3 Validation: Camera Module")
    print("=" * 60)
    print()
    print("  SYNTHETIC CAMERA ONLY -- no real IITB data used.")
    print("  Coordinate convention: W2C (OpenCV/COLMAP style)")
    print("  Camera looks along +Z in camera space.")
    print("  +X = right, +Y = down in image.")

    cam = make_synthetic_camera()
    print(f"\n  Camera: {cam}")

    test_intrinsics(cam)
    test_extrinsics(cam)
    test_world_to_camera(cam)
    test_projection_basic(cam)
    test_projection_behind_camera(cam)
    test_image_bounds(cam)
    test_c2w_convention(cam)
    test_non_identity_rotation(cam)
    test_single_point_input(cam)

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 3 -- Camera Module: ALL CHECKS PASSED")
    else:
        print(f"  [FAIL]  STEP 3 -- {_failures} check(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
