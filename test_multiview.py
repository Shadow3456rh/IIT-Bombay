"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : src/test_multiview.py

PURPOSE
-------
Validation script for STEP 8: Multi-View Evidence Fusion.
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from association.multiview import (
    ViewEvidence,
    FusedGaussianResult,
    fuse_multiview_evidence
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

# ---------------------------------------------------------------------------
# Test Sections
# ---------------------------------------------------------------------------

def test_1_2_3_4_observations() -> None:
    section("1/2/3/4. Observation Consistency and Fusion")

    # One observation
    ev1 = ViewEvidence("f1", 17, 42, 1.0, 0.0, True)
    r1 = fuse_multiview_evidence([ev1])[0]
    check("One perfect observation produces score 1.0", math.isclose(r1.fused_score, 1.0))

    # Two consistent views
    ev2 = ViewEvidence("f2", 17, 42, 1.0, 0.0, True)
    r2 = fuse_multiview_evidence([ev1, ev2])[0]
    check("Two consistent views support confidence", math.isclose(r2.fused_score, 1.0))

    # Multiple consistent views
    ev3 = ViewEvidence("f3", 17, 42, 1.0, 0.0, True)
    r3 = fuse_multiview_evidence([ev1, ev2, ev3])[0]
    check("Multiple consistent views fused deterministically", math.isclose(r3.fused_score, 1.0))

    # Conflicting view
    ev_conflict = ViewEvidence("f4", 17, 42, 0.0, 0.0, True)
    r4 = fuse_multiview_evidence([ev1, ev2, ev_conflict])[0]
    # (1 + 1 + 0) / 3 = 2/3 = 0.666...
    check("Conflicting views do not produce full confidence", math.isclose(r4.fused_score, 2.0 / 3.0))


def test_5_6_16_frame_weights() -> None:
    section("5/6/16. Frame Weights and Zero Weights")

    ev1 = ViewEvidence("f1", 17, 42, 1.0, 0.0, True, frame_weight=2.0)
    ev2 = ViewEvidence("f2", 17, 42, 0.0, 0.0, True, frame_weight=1.0)

    # fused_score = (1.0 * 2.0 + 0.0 * 1.0) / 3.0 = 2/3
    r1 = fuse_multiview_evidence([ev1, ev2])[0]
    check("Frame weights affect result expectedly", math.isclose(r1.fused_score, 2.0 / 3.0))

    ev_zero = ViewEvidence("f3", 17, 42, 1.0, 0.0, True, frame_weight=0.0)
    # fused_score = (1*2 + 0*1 + 1*0) / 3.0 = 2/3
    r2 = fuse_multiview_evidence([ev1, ev2, ev_zero])[0]
    check("Zero frame weight contributes no evidence", math.isclose(r2.fused_score, 2.0 / 3.0))

    ev_only_zero = ViewEvidence("f4", 17, 42, 1.0, 0.0, True, frame_weight=0.0)
    r3 = fuse_multiview_evidence([ev_only_zero])[0]
    check("Zero total frame weight handled safely (returns 0.0)", r3.fused_score == 0.0)


def test_7_8_object_separation() -> None:
    section("7/8. Multiple Objects and Gaussians Separation")

    evs = [
        ViewEvidence("f1", 1, 10, 1.0, 0.0, True),
        ViewEvidence("f1", 1, 20, 0.5, 0.0, True),
        ViewEvidence("f1", 2, 10, 0.2, 0.0, True),
    ]

    r = fuse_multiview_evidence(evs)
    check("Output length matches unique pairs", len(r) == 3)
    
    # Should be sorted by object_id, then gaussian_id
    check("Sorted output: 1st is obj 1, g 10", r[0].object_id == 1 and r[0].gaussian_id == 10)
    check("Sorted output: 2nd is obj 1, g 20", r[1].object_id == 1 and r[1].gaussian_id == 20)
    check("Sorted output: 3rd is obj 2, g 10", r[2].object_id == 2 and r[2].gaussian_id == 10)


def test_9_10_missing_depth() -> None:
    section("9/10. Missing Depth Policies")

    ev = ViewEvidence("f1", 17, 42, 1.0, None, False)
    
    r_zero = fuse_multiview_evidence([ev], missing_depth_policy="zero")[0]
    check("Default 'zero' policy sets confidence to 0", r_zero.fused_score == 0.0)

    r_one = fuse_multiview_evidence([ev], missing_depth_policy="one")[0]
    check("Alternative 'one' policy sets confidence to 1.0", r_one.fused_score == 1.0)


def test_11_17_depth_confidence() -> None:
    section("11/17. Depth Confidence Calculation")

    # relative error = 0.1, depth_scale = 0.1 -> conf = exp(-1) = 0.3678...
    ev = ViewEvidence("f1", 17, 42, 1.0, 0.1, True)
    
    r = fuse_multiview_evidence([ev], depth_scale=0.1)[0]
    check("Depth confidence bounded and calculated via exp()", math.isclose(r.fused_score, math.exp(-1.0)))
    
    r2 = fuse_multiview_evidence([ev], depth_scale=0.05)[0]
    # relative error = 0.1, depth_scale = 0.05 -> conf = exp(-2) = 0.1353...
    check("Configurable depth_scale works", math.isclose(r2.fused_score, math.exp(-2.0)))


def test_12_to_15_validation() -> None:
    section("12/13/14/15. Validation of Numeric Inputs")

    expect_error("Association ratio < 0", lambda: ViewEvidence("f1", 1, 1, -0.1, 0.0, True))
    expect_error("Association ratio > 1", lambda: ViewEvidence("f1", 1, 1, 1.1, 0.0, True))
    expect_error("Negative relative depth error", lambda: ViewEvidence("f1", 1, 1, 1.0, -0.1, True))
    expect_error("Negative frame weight", lambda: ViewEvidence("f1", 1, 1, 1.0, 0.0, True, -1.0))
    expect_error("NaN association ratio", lambda: ViewEvidence("f1", 1, 1, float('nan'), 0.0, True))
    expect_error("NaN depth error", lambda: ViewEvidence("f1", 1, 1, 1.0, float('nan'), True))
    expect_error("NaN frame weight", lambda: ViewEvidence("f1", 1, 1, 1.0, 0.0, True, float('nan')))


def test_18_19_thresholds() -> None:
    section("18/19. Score Threshold and Determinism")

    ev = ViewEvidence("f1", 17, 42, 0.5, 0.0, True) # fused score = 0.5
    
    r_acc = fuse_multiview_evidence([ev], fused_score_threshold=0.5)[0]
    check("Accepted exactly at threshold", r_acc.accepted == True)
    
    r_rej = fuse_multiview_evidence([ev], fused_score_threshold=0.51)[0]
    check("Rejected slightly below threshold", r_rej.accepted == False)
    
    # Check stable threshold behavior
    for _ in range(3):
        r = fuse_multiview_evidence([ev], fused_score_threshold=0.5)[0]
        if not r.accepted:
            check("Threshold determinism", False)
            return
    check("Exact threshold behavior is deterministic", True)


def test_20_21_22_counts() -> None:
    section("20/21/22. Observation Counts")

    ev1 = ViewEvidence("f1", 1, 1, 1.0, 0.0, True)   # positive, valid depth
    ev2 = ViewEvidence("f2", 1, 1, 0.0, 0.0, True)   # not positive, valid depth
    ev3 = ViewEvidence("f3", 1, 1, 1.0, None, False) # positive (via policy "one"), invalid depth
    
    r = fuse_multiview_evidence([ev1, ev2, ev3], missing_depth_policy="one")[0]
    
    check("Output contains correct total observation count", r.observation_count == 3)
    check("Positive observation count is correct", r.positive_observation_count == 2)
    check("Depth-valid observation count is correct", r.depth_valid_observation_count == 2)


def test_23_24_formatting() -> None:
    section("23/24. Format and JSON")

    ev = ViewEvidence("f1", 1, 1, 0.8, 0.1, True)
    r1 = fuse_multiview_evidence([ev])
    r2 = fuse_multiview_evidence([ev])
    
    check("Repeated runs produce identical output", r1 == r2)
    
    try:
        json.dumps(r1[0].to_dict())
        check("Output is JSON serialisable", True)
    except Exception as e:
        check("Output is JSON serialisable", False, str(e))


def test_25_demonstration() -> None:
    section("25. Key Multi-View Benefit Demonstration")

    print("  Scenario:")
    print("  View 1: strong evidence (assoc=1.0, err=0.0)")
    print("  View 2: strong evidence (assoc=0.9, err=0.02)")
    print("  View 3: weak conflicting evidence (assoc=0.1, err=0.5)")
    
    ev1 = ViewEvidence("f1", 17, 42, 1.0, 0.0, True)
    ev2 = ViewEvidence("f2", 17, 42, 0.9, 0.02, True)
    ev3 = ViewEvidence("f3", 17, 42, 0.1, 0.5, True)
    
    r = fuse_multiview_evidence([ev1, ev2, ev3], fused_score_threshold=0.5)[0]
    
    print(f"  Final Fused Score: {r.fused_score:.3f}")
    check("Robust against conflicting view", r.accepted == True)


def main() -> int:
    print("=" * 60)
    print("  Layer 2 -- STEP 8: Multi-View Evidence Fusion")
    print("=" * 60)
    print()

    test_1_2_3_4_observations()
    test_5_6_16_frame_weights()
    test_7_8_object_separation()
    test_9_10_missing_depth()
    test_11_17_depth_confidence()
    test_12_to_15_validation()
    test_18_19_thresholds()
    test_20_21_22_counts()
    test_23_24_formatting()
    test_25_demonstration()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  STEP 8 -- Multi-View Evidence Fusion: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  STEP 8 -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
