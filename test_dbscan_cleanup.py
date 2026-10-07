"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_dbscan_cleanup.py

PURPOSE
-------
Validation script for STEP 9: DBSCAN Spatial Cleanup.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from gaussian.scene import GaussianScene
from association.dbscan import (
    GaussianCandidate,
    DBSCANCleanupResult,
    cleanup_with_dbscan
)


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

def test_1_2_25_basic_and_isolated() -> None:
    section("1/2/25. Basic Cluster and Isolated Noise")

    scene = make_scene([
        [0.0, 0.0, 0.0],  # 0
        [0.01, 0.0, 0.0], # 1
        [0.0, 0.01, 0.0], # 2
        [0.01, 0.01, 0.0],# 3
        [5.0, 5.0, 5.0]   # 4 (noise)
    ])
    
    candidates = [GaussianCandidate(17, i, 0.9) for i in range(5)]
    
    res = cleanup_with_dbscan(candidates, scene, eps=0.05, min_samples=3)[0]
    
    check("Dense cluster points retained", all(i in res.cleaned_gaussian_ids for i in [0,1,2,3]))
    check("Isolated point removed as noise", 4 in res.removed_gaussian_ids and 4 not in res.cleaned_gaussian_ids)
    check("Basic one-cluster cleanup works", len(res.cluster_sizes) == 1)


def test_3_27_multiple_clusters() -> None:
    section("3/27. Multiple Clusters Retained")
    
    scene = make_scene([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], # Cluster A (IDs 0-2)
        [2.0, 2.0, 2.0], [2.01, 2.0, 2.0], [2.0, 2.01, 2.0], # Cluster B (IDs 3-5)
        [5.0, 5.0, 5.0]  # Noise (ID 6)
    ])
    
    candidates = [GaussianCandidate(1, i, 0.8) for i in range(7)]
    res = cleanup_with_dbscan(candidates, scene, eps=0.05, min_samples=3)[0]
    
    check("Multiple spatial clusters retained", len(res.cluster_sizes) == 2)
    check("All cluster points in cleaned set", len(res.cleaned_gaussian_ids) == 6)
    check("Noise point correctly isolated", 6 in res.removed_gaussian_ids)


def test_4_5_object_isolation() -> None:
    section("4/5. Object-Specific Clustering")

    scene = make_scene([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], # Near origin
        [0.02, 0.02, 0.0], [0.03, 0.02, 0.0], [0.02, 0.03, 0.0]  # Also near origin
    ])
    
    # Even though all 6 points are close, they belong to two different objects
    cands = [GaussianCandidate(1, i, 0.9) for i in range(3)] + \
            [GaussianCandidate(2, i, 0.9) for i in range(3, 6)]
            
    res = cleanup_with_dbscan(cands, scene, eps=0.05, min_samples=3)
    
    check("Returned two independent results", len(res) == 2)
    check("Nearby different objects remain separate", res[0].object_id == 1 and res[1].object_id == 2)


def test_6_to_9_candidate_counts() -> None:
    section("6/7/8/9/26. Low Counts and All-Noise")
    
    scene = make_scene([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    
    # 6. Empty candidate set
    res0 = cleanup_with_dbscan([], scene, eps=0.05, min_samples=3)
    check("Empty candidate set handled gracefully", len(res0) == 0)
    
    # 7. Single candidate
    c1 = [GaussianCandidate(1, 0, 0.9)]
    r1 = cleanup_with_dbscan(c1, scene, eps=0.05, min_samples=3)[0]
    check("Single candidate correctly identified as noise (below min_samples)", 0 in r1.removed_gaussian_ids)
    
    # 8. Two candidates (when min_samples is 3)
    c2 = [GaussianCandidate(1, 0, 0.9), GaussianCandidate(1, 1, 0.9)]
    r2 = cleanup_with_dbscan(c2, scene, eps=0.05, min_samples=3)[0]
    check("Two candidates removed when min_samples=3", len(r2.removed_gaussian_ids) == 2)
    
    # 9. All points noise
    # (points too far apart)
    scene2 = make_scene([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]])
    c3 = [GaussianCandidate(1, i, 0.9) for i in range(3)]
    r3 = cleanup_with_dbscan(c3, scene2, eps=0.05, min_samples=3)[0]
    check("All points correctly classified as noise", len(r3.cleaned_gaussian_ids) == 0 and len(r3.removed_gaussian_ids) == 3)
    check("DBSCAN noise label -1 is handled correctly", all(lbl == -1 for lbl in r3.cluster_labels.values()))


def test_10_11_parameters_affect_clustering() -> None:
    section("10/11. DBSCAN Parameters")
    
    scene = make_scene([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0] # Spaced by 0.1
    ])
    cands = [GaussianCandidate(1, i, 0.9) for i in range(3)]
    
    r_tight = cleanup_with_dbscan(cands, scene, eps=0.05, min_samples=2)[0]
    check("Tight eps separates points as noise", len(r_tight.removed_gaussian_ids) == 3)
    
    r_loose = cleanup_with_dbscan(cands, scene, eps=0.15, min_samples=2)[0]
    check("Loose eps groups them", len(r_loose.cleaned_gaussian_ids) == 3)
    
    r_high_min = cleanup_with_dbscan(cands, scene, eps=0.15, min_samples=4)[0]
    check("High min_samples rejects the group", len(r_high_min.removed_gaussian_ids) == 3)


def test_12_to_18_validation_errors() -> None:
    section("12-18. Validation and Errors")
    
    scene = make_scene([[0.0, 0.0, 0.0]])
    cands = [GaussianCandidate(1, 0, 0.9)]
    
    expect_error("Invalid eps rejected (0)", lambda: cleanup_with_dbscan(cands, scene, eps=0.0))
    expect_error("Invalid eps rejected (-1)", lambda: cleanup_with_dbscan(cands, scene, eps=-1.0))
    expect_error("Invalid eps rejected (NaN)", lambda: cleanup_with_dbscan(cands, scene, eps=float('nan')))
    expect_error("Invalid min_samples rejected", lambda: cleanup_with_dbscan(cands, scene, min_samples=0))
    
    expect_error("Duplicate Gaussian IDs rejected", lambda: cleanup_with_dbscan([GaussianCandidate(1, 0, 0.9), GaussianCandidate(1, 0, 0.9)], scene))
    expect_error("Invalid Gaussian ID rejected", lambda: cleanup_with_dbscan([GaussianCandidate(1, 99, 0.9)], scene))
    
    scene_nan = make_scene([[np.nan, 0.0, 0.0]])
    expect_error("NaN XYZ rejected", lambda: cleanup_with_dbscan(cands, scene_nan))
    
    scene_inf = make_scene([[np.inf, 0.0, 0.0]])
    expect_error("Infinite XYZ rejected", lambda: cleanup_with_dbscan(cands, scene_inf))


def test_19_to_24_28_formatting() -> None:
    section("19-24/28. Formatting, Scores, and Consistency")
    
    scene = make_scene([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [5.0, 5.0, 5.0]])
    cands = [
        GaussianCandidate(1, 0, 0.91), 
        GaussianCandidate(1, 1, 0.92), 
        GaussianCandidate(1, 2, 0.93), 
        GaussianCandidate(1, 3, 0.94)
    ]
    
    r1 = cleanup_with_dbscan(cands, scene, eps=0.05, min_samples=3)[0]
    
    # 19. Deterministic output ordering
    check("Deterministic output ordering", r1.input_gaussian_ids == sorted(r1.input_gaussian_ids))
    
    # 20. No duplicate cleaned Gaussian IDs
    check("No duplicate cleaned Gaussian IDs", len(r1.cleaned_gaussian_ids) == len(set(r1.cleaned_gaussian_ids)))
    
    # 21. Removed IDs are not in cleaned IDs
    overlap = set(r1.cleaned_gaussian_ids).intersection(set(r1.removed_gaussian_ids))
    check("Removed IDs are not in cleaned IDs", len(overlap) == 0)
    
    # 22. All input IDs are accounted for
    union = set(r1.cleaned_gaussian_ids).union(set(r1.removed_gaussian_ids))
    check("All input IDs accounted for", union == set(r1.input_gaussian_ids))
    
    # 23. Fused scores preserved
    check("Fused scores preserved for surviving IDs", r1.cleaned_scores[1] == 0.92)
    
    # 24. JSON serialization
    try:
        json.dumps(r1.to_dict())
        check("JSON serialization works", True)
    except Exception as e:
        check("JSON serialization works", False, str(e))
        
    # 28. Repeated execution identical
    r2 = cleanup_with_dbscan(cands, scene, eps=0.05, min_samples=3)[0]
    check("Repeated execution produces identical results", r1 == r2)


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 9: DBSCAN Spatial Cleanup")
    print("=" * 60)
    print()

    test_1_2_25_basic_and_isolated()
    test_3_27_multiple_clusters()
    test_4_5_object_isolation()
    test_6_to_9_candidate_counts()
    test_10_11_parameters_affect_clustering()
    test_12_to_18_validation_errors()
    test_19_to_24_28_formatting()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 9 -- DBSCAN Spatial Cleanup: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 9 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
