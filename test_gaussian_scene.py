"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_gaussian_scene.py

PURPOSE
-------
Validation script for STEP 2: Gaussian Data Model.

Loads the demo PLY and verifies that GaussianScene correctly:
  - counts Gaussians
  - assigns IDs from PLY vertex indices
  - stores raw opacity (logit space) and converts to physical via sigmoid
  - stores raw scale (log space) and converts to physical via exp
  - stores rotation quaternion unchanged
  - exposes correct array shapes

This script uses ONLY the small demo PLY (17,020 Gaussians).
It does NOT use the actual 4 GB IITB PLY.

PROTOTYPE STATUS
----------------
No camera projection, object IDs, semantic classes, or rendering are tested
here.  Those belong to later steps.
"""

import sys
import os
from pathlib import Path

# Allow running from project root:  python src/test_gaussian_scene.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from gaussian import GaussianScene

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent      # .../Layer2/src
PROJECT_ROOT = SCRIPT_DIR.parent                     # .../Layer2
PLY_PATH     = PROJECT_ROOT / "data" / "ply" / "3dgs_sr_demo.ply"

EXPECTED_N          = 17_020
EXPECTED_FIRST_ID   = 0
EXPECTED_LAST_ID    = 17_019


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, got, expected=None, *, warn_only: bool = False) -> bool:
    if expected is None:
        print(f"  [INFO]  {label}: {got}")
        return True
    ok = got == expected
    status = "[PASS]" if ok else ("[WARN]" if warn_only else "[FAIL]")
    print(f"  {status}  {label}")
    print(f"          expected : {expected}")
    print(f"          got      : {got}")
    return ok


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> int:
    failures = 0

    section("Layer 2 -- STEP 2 Validation: GaussianScene")

    # --- Load -----------------------------------------------------------
    print(f"\n  Loading: {PLY_PATH}")
    try:
        scene = GaussianScene.from_ply(PLY_PATH)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        return 1
    except ValueError as e:
        print(f"\n  [ERROR] {e}")
        return 1
    print(f"  repr   : {scene}")

    # --- Count ----------------------------------------------------------
    section("1. Gaussian count")
    if not check("n_gaussians", scene.n_gaussians, EXPECTED_N):
        failures += 1

    # --- Array shapes ---------------------------------------------------
    section("2. Array shapes")
    check("ids.shape",              scene.ids.shape,              (EXPECTED_N,))
    check("xyz.shape",              scene.xyz.shape,              (EXPECTED_N, 3))
    check("opacity_raw.shape",      scene.opacity_raw.shape,      (EXPECTED_N,))
    check("scale_raw.shape",        scene.scale_raw.shape,        (EXPECTED_N, 3))
    check("rotation.shape",         scene.rotation.shape,         (EXPECTED_N, 4))
    check("ply_vertex_index.shape", scene.ply_vertex_index.shape, (EXPECTED_N,))

    # Check dtypes
    check("ids dtype",              str(scene.ids.dtype),              "int64")
    check("xyz dtype",              str(scene.xyz.dtype),              "float32")
    check("opacity_raw dtype",      str(scene.opacity_raw.dtype),      "float32")
    check("scale_raw dtype",        str(scene.scale_raw.dtype),        "float32")
    check("rotation dtype",         str(scene.rotation.dtype),         "float32")

    # --- Gaussian IDs ---------------------------------------------------
    section("3. Gaussian IDs (must equal PLY vertex indices)")
    if not check("first Gaussian ID", int(scene.ids[0]),  EXPECTED_FIRST_ID):
        failures += 1
    if not check("last  Gaussian ID", int(scene.ids[-1]), EXPECTED_LAST_ID):
        failures += 1
    ids_match_ply = np.array_equal(scene.ids, scene.ply_vertex_index)
    if not check("ids == ply_vertex_index", ids_match_ply, True):
        failures += 1

    # --- First Gaussian XYZ --------------------------------------------
    section("4. First Gaussian position (XYZ)")
    x0, y0, z0 = scene.xyz[0]
    check("x[0]", float(f"{x0:.6f}"))
    check("y[0]", float(f"{y0:.6f}"))
    check("z[0]", float(f"{z0:.6f}"))

    # --- Opacity (raw vs physical) -------------------------------------
    section("5. Opacity -- raw (logit) vs. physical (sigmoid)")
    raw_op = float(scene.opacity_raw[0])
    phys_op_arr = scene.physical_opacity()
    phys_op = float(phys_op_arr[0])

    check("opacity_raw[0] (logit)",          round(raw_op, 4))
    check("physical_opacity[0] in [0,1]",
          0.0 <= phys_op <= 1.0, True)
    check("physical_opacity shape",          phys_op_arr.shape, (EXPECTED_N,))

    # Verify sigmoid manually
    expected_phys = float(1.0 / (1.0 + np.exp(-raw_op)))
    close = abs(phys_op - expected_phys) < 1e-5
    if not check("sigmoid(opacity_raw[0]) matches helper", close, True):
        failures += 1

    # Verify raw is NOT overwritten by calling physical_opacity()
    raw_after = float(scene.opacity_raw[0])
    if not check("raw opacity_raw unchanged after helper call",
                 round(raw_after, 4), round(raw_op, 4)):
        failures += 1

    print(f"\n  opacity_raw[0]      = {raw_op:.6f}  (logit)")
    print(f"  physical_opacity[0] = {phys_op:.6f}  (sigmoid, [0,1])")

    # --- Scale (raw vs physical) ---------------------------------------
    section("6. Scale -- raw (log) vs. physical (exp)")
    raw_sc  = scene.scale_raw[0]
    phys_sc_arr = scene.physical_scale()
    phys_sc = phys_sc_arr[0]

    check("scale_raw[0] (log space)",   [round(float(v), 4) for v in raw_sc])
    check("physical_scale shape",       phys_sc_arr.shape, (EXPECTED_N, 3))
    check("physical_scale[0] all > 0",  bool(np.all(phys_sc > 0)), True)

    # Verify exp manually
    expected_phys_sc = np.exp(raw_sc)
    close_sc = np.allclose(phys_sc, expected_phys_sc, atol=1e-5)
    if not check("exp(scale_raw[0]) matches helper", close_sc, True):
        failures += 1

    # Verify raw is NOT overwritten
    raw_sc_after = scene.scale_raw[0].copy()
    if not check("raw scale_raw unchanged after helper call",
                 np.allclose(raw_sc_after, raw_sc), True):
        failures += 1

    print(f"\n  scale_raw[0]      = {[round(float(v),4) for v in raw_sc]}  (log)")
    print(f"  physical_scale[0] = {[round(float(v),6) for v in phys_sc]}  (exp)")

    # --- Rotation ------------------------------------------------------
    section("7. Rotation quaternion (raw, no conversion)")
    rot0 = scene.rotation[0]
    check("rotation[0] shape", rot0.shape, (4,))
    print(f"\n  rotation[0] = {[round(float(v), 6) for v in rot0]}")
    print(f"  (stored as-is from PLY; ordering must be confirmed against renderer)")

    # --- SH / color (informational) ------------------------------------
    section("8. Spherical Harmonic attributes (informational)")
    if scene.sh_dc is not None:
        check("sh_dc shape",   scene.sh_dc.shape,   (EXPECTED_N, 3))
        print(f"  sh_dc[0]   = {[round(float(v), 6) for v in scene.sh_dc[0]]}")
    else:
        print("  [INFO]  sh_dc  : not present in PLY")

    if scene.sh_rest is not None:
        print(f"  sh_rest shape: {scene.sh_rest.shape}  ({scene.sh_rest.shape[1]} higher-order coefficients)")
    else:
        print("  [INFO]  sh_rest: not present in PLY")

    # --- No object IDs / semantic IDs ----------------------------------
    section("9. Confirming object_id / semantic_class_id are absent")
    has_obj_attr = hasattr(scene, "object_id")
    has_sem_attr = hasattr(scene, "semantic_class_id")
    check("object_id attribute absent (not yet assigned)",
          has_obj_attr, False)
    check("semantic_class_id attribute absent (not yet assigned)",
          has_sem_attr, False)
    print("  (Both will be assigned by the Layer-2 association pipeline)")

    # --- Final result --------------------------------------------------
    section("RESULT")
    if failures == 0:
        print("  [PASS]  STEP 2 -- Gaussian Data Model: ALL CHECKS PASSED")
    else:
        print(f"  [FAIL]  STEP 2 -- {failures} check(s) failed")
    print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
