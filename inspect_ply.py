"""
Layer 2 â€” Drone Data Augmentation Pipeline
==========================================
STEP 1: PLY Inspection Script
File   : src/inspect_ply.py

PURPOSE
-------
Inspect the Layer-1 3D Gaussian Splatting demo PLY to understand its
structure before building any downstream modules.

This script does NOT modify the PLY.
This script does NOT assume any specific property names.
This script does NOT use the actual 4 GB IITB PLY â€” only the small demo file.

PROTOTYPE STATUS
----------------
This is the initial inspection step.  Nothing here is validated against the
real IITB dataset yet.
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check â€” plyfile must be installed
# ---------------------------------------------------------------------------
try:
    from plyfile import PlyData
except ImportError:
    print("[ERROR] 'plyfile' package is not installed.")
    print("        Run:  pip install plyfile")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve path relative to this script's location so the script can be run
# from any working directory.
SCRIPT_DIR   = Path(__file__).resolve().parent        # .../Layer2/src
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../Layer2
PLY_PATH     = PROJECT_ROOT / "data" / "ply" / "3dgs_sr_demo.ply"

# ---------------------------------------------------------------------------
# Known 3DGS property-group patterns (used for summary only, not assumed)
# ---------------------------------------------------------------------------

POSITION_PROPS     = {"x", "y", "z"}
OPACITY_PROPS      = {"opacity"}
SCALE_PROPS        = {"scale_0", "scale_1", "scale_2"}
ROTATION_PROPS     = {"rot_0", "rot_1", "rot_2", "rot_3"}
# SH / color: f_dc_*, f_rest_*, red, green, blue, nx, ny, nz â€¦
SH_PREFIXES        = ("f_dc_", "f_rest_", "features_dc", "features_rest")
COLOR_PROPS        = {"red", "green", "blue", "r", "g", "b",
                      "nx", "ny", "nz"}
OBJECT_ID_PROPS    = {"object_id", "object_idx", "objectid"}
SEMANTIC_ID_PROPS  = {"semantic_class_id", "semantic_id", "class_id",
                      "label", "category_id"}


def classify_property(name: str) -> str:
    """Return a human-readable category for a PLY property name."""
    nl = name.lower()
    if nl in POSITION_PROPS:
        return "Position (XYZ)"
    if nl in OPACITY_PROPS:
        return "Opacity"
    if nl in SCALE_PROPS:
        return "Scale"
    if nl in ROTATION_PROPS:
        return "Rotation (quaternion)"
    if any(nl.startswith(p) for p in SH_PREFIXES):
        return "Spherical Harmonic / Color"
    if nl in COLOR_PROPS:
        return "Color / Normal"
    if nl in OBJECT_ID_PROPS:
        return "Object ID  *** FOUND ***"
    if nl in SEMANTIC_ID_PROPS:
        return "Semantic Class ID  *** FOUND ***"
    return "Unknown / Other"


def check_group(all_props: set, group: set, label: str) -> bool:
    found = all_props & group
    if found:
        print(f"  [YES] {label}: {sorted(found)}")
        return True
    else:
        print(f"  [NO ] {label}")
        return False


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Layer 2 â€” PLY Inspection (STEP 1)")
    print("=" * 65)

    # 1. File existence check
    print(f"\n[1] Checking file: {PLY_PATH}")
    if not PLY_PATH.exists():
        print(f"    [ERROR] File not found: {PLY_PATH}")
        print("    Make sure 3dgs_sr_demo.ply is placed at:")
        print(f"    {PLY_PATH}")
        sys.exit(1)
    file_size_mb = PLY_PATH.stat().st_size / (1024 ** 2)
    print(f"    [OK] File found  ({file_size_mb:.2f} MB)")

    # 2. Load PLY
    print("\n[2] Loading PLY ...")
    ply_data = PlyData.read(str(PLY_PATH))
    print(f"    [OK] Loaded successfully")
    print(f"    Elements in file: {[e.name for e in ply_data.elements]}")

    # 3. Locate the vertex / Gaussian element
    element_names = [e.name for e in ply_data.elements]
    if "vertex" in element_names:
        vertex_el = ply_data["vertex"]
    else:
        # Fall back to first element if no 'vertex' key
        vertex_el = ply_data.elements[0]
        print(f"    [NOTE] No 'vertex' element found; using first element: "
              f"'{vertex_el.name}'")

    # 4. Number of Gaussians
    n_gaussians = len(vertex_el.data)
    print(f"\n[3] Number of Gaussian vertices : {n_gaussians:,}")

    # 5. All available properties
    properties = vertex_el.properties
    prop_names = [p.name for p in properties]

    print(f"\n[4] Properties ({len(prop_names)} total):")
    for i, p in enumerate(properties):
        cat = classify_property(p.name)
        print(f"    [{i:>3}]  {p.name:<35}  dtype={p.dtype}   â†’ {cat}")

    # 6. First Gaussian's values
    print(f"\n[5] First Gaussian (index 0) â€” raw values:")
    first = vertex_el.data[0]
    for name in prop_names:
        val = first[name]
        print(f"    {name:<35} = {val}")

    # 7. Summary â€” which standard groups are present
    print("\n[6] Attribute Group Summary:")
    all_props_lower = {n.lower() for n in prop_names}

    has_xyz      = check_group(all_props_lower, POSITION_PROPS,  "Position (x, y, z)")
    has_opacity  = check_group(all_props_lower, OPACITY_PROPS,   "Opacity")
    has_scale    = check_group(all_props_lower, SCALE_PROPS,     "Scale (scale_0/1/2)")
    has_rotation = check_group(all_props_lower, ROTATION_PROPS,  "Rotation (rot_0/1/2/3)")

    # SH / color: prefix-based
    sh_found = [n for n in prop_names if any(n.lower().startswith(p) for p in SH_PREFIXES)]
    if sh_found:
        print(f"  [YES] Spherical Harmonic / Color attrs: {sh_found[:6]}"
              f"{'...' if len(sh_found) > 6 else ''}")
        has_sh = True
    else:
        print(f"  [NO ] Spherical Harmonic / Color attrs (f_dc_*, f_rest_*, â€¦)")
        has_sh = False

    has_obj_id  = check_group(all_props_lower, OBJECT_ID_PROPS,   "Object ID")
    has_sem_id  = check_group(all_props_lower, SEMANTIC_ID_PROPS, "Semantic Class ID")

    # 8. Final conclusion
    print("\n[7] Conclusion:")
    print(f"    Gaussians        : {n_gaussians:,}")
    print(f"    XYZ positions    : {'YES' if has_xyz      else 'NO'}")
    print(f"    Opacity          : {'YES' if has_opacity  else 'NO'}")
    print(f"    Scale            : {'YES' if has_scale    else 'NO'}")
    print(f"    Rotation         : {'YES' if has_rotation else 'NO'}")
    print(f"    SH / Color       : {'YES' if has_sh       else 'NO'}")
    print(f"    Object ID        : {'YES â€” already in PLY' if has_obj_id else 'NO  â€” will be assigned by Layer 2'}")
    print(f"    Semantic Class ID: {'YES â€” already in PLY' if has_sem_id else 'NO  â€” will be assigned by Layer 2'}")

    if not has_obj_id and not has_sem_id:
        print("\n    NOTE: This is a standard 3DGS PLY with geometry/appearance only.")
        print("    Object IDs and semantic class IDs are NOT stored in the PLY.")
        print("    They will be derived by the Layer-2 association pipeline.")

    print("\n" + "=" * 65)
    print("  STEP 1 complete â€” PLY inspection finished.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()


