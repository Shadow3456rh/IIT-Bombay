"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
File : test_gnn_training.py

PURPOSE
-------
Validation test suite for Stage D: RelationshipGNN Training Infrastructure.
Tests metric calculations (accuracy, macro-F1, missing classes), training loss
reduction on a small controlled synthetic fixture, validation-based checkpoint
selection, held-out test evaluation, and input safety guards.
"""

import sys
import os
import math
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from relationships.gnn_model import (
    RelationshipGNN,
    RelationshipClass,
    NUM_RELATIONSHIP_CLASSES,
)
from relationships.gnn_dataset import (
    generate_synthetic_dataset,
    split_dataset_by_scene,
    RelationshipDataset,
)
from relationships.train_gnn import (
    TrainingConfig,
    train_gnn_model,
    compute_relationship_metrics,
)


_failures = 0


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"  [PASS]  {label}")
        if detail:
            print(f"         {detail}")
    else:
        _failures += 1
        print(f"  [FAIL]  {label}")
        if detail:
            print(f"         DETAIL: {detail}")


def expect_error(label: str, fn, expected_exception=ValueError) -> None:
    global _failures
    try:
        fn()
        _failures += 1
        print(f"  [FAIL]  {label} (did not raise {expected_exception.__name__})")
    except expected_exception as e:
        print(f"  [PASS]  {label}")
        print(f"         {type(e).__name__}: {e}")
    except Exception as e:
        _failures += 1
        print(f"  [FAIL]  {label} (raised unexpected {type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


def test_1_metrics_calculation_and_edge_cases() -> None:
    section("1. Relationship Metrics Calculation and Missing Classes")

    # Case A: Perfect predictions
    y_true = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    # One-hot logits
    logits = torch.zeros((4, NUM_RELATIONSHIP_CLASSES))
    for i, cls in enumerate([0, 1, 2, 4]):
        logits[i, cls] = 10.0

    metrics = compute_relationship_metrics(y_true, logits)
    check("Perfect accuracy is 1.0", math.isclose(metrics["accuracy"], 1.0))
    check("Perfect macro-F1 is 1.0", math.isclose(metrics["macro_f1"], 1.0))
    check("Per-class F1 for class 1 is 1.0", math.isclose(metrics["per_class_f1"][1], 1.0))
    # Unrepresented class 6 (BEHIND) is 0.0 without crash
    check("Unrepresented class 6 has F1 0.0", math.isclose(metrics["per_class_f1"][6], 0.0))

    # Case B: Empty tensors
    empty_metrics = compute_relationship_metrics(
        torch.zeros(0, dtype=torch.long),
        torch.zeros((0, NUM_RELATIONSHIP_CLASSES)),
    )
    check("Empty metrics returns accuracy 0.0", empty_metrics["accuracy"] == 0.0)
    check("Empty metrics returns macro_f1 0.0", empty_metrics["macro_f1"] == 0.0)


def test_2_training_loss_reduction() -> None:
    section("2. Training Loss Reduction on Synthetic Fixture")

    # Generate small controlled synthetic dataset
    full_dataset = generate_synthetic_dataset(num_scenes=6, samples_per_scene=2, seed=42)
    train_set, val_set, test_set = split_dataset_by_scene(
        full_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    check("Train set has samples", len(train_set) > 0)
    check("Val set has samples", len(val_set) > 0)
    check("Test set has samples", len(test_set) > 0)

    cfg = TrainingConfig(
        epochs=8,
        learning_rate=5e-3,
        batch_size=2,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_class_weights=True,
        seed=101,
    )

    result = train_gnn_model(
        train_dataset=train_set,
        val_dataset=val_set,
        config=cfg,
    )

    history = result["history"]
    check("Training executed 8 epochs", len(history) == 8)

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    check(
        f"Training loss decreased ({initial_loss:.4f} -> {final_loss:.4f})",
        final_loss < initial_loss,
    )
    check("best_epoch is within range [1, 8]", 1 <= result["best_epoch"] <= 8)
    check("best_val_score is positive", result["best_val_score"] >= 0.0)


def test_3_checkpoint_saving_and_selection() -> None:
    section("3. Best Checkpoint Selection and Persistence")

    dataset = generate_synthetic_dataset(num_scenes=4, samples_per_scene=2, seed=7)
    train_set, val_set, test_set = split_dataset_by_scene(
        dataset,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            epochs=5,
            learning_rate=2e-3,
            batch_size=2,
            hidden_dim=32,
            num_layers=2,
            checkpoint_dir=tmpdir,
            seed=42,
        )

        result = train_gnn_model(
            train_dataset=train_set,
            val_dataset=val_set,
            test_dataset=test_set,
            config=cfg,
        )

        best_ckpt_path = os.path.join(tmpdir, "best_model.pt")
        check("Best checkpoint file saved to disk", os.path.exists(best_ckpt_path))

        # Restore checkpoint and verify output
        loaded_model = RelationshipGNN.load_checkpoint(best_ckpt_path)
        check("Loaded model restored successfully", isinstance(loaded_model, RelationshipGNN))

        # Check test evaluation was performed
        test_metrics = result.get("test_metrics")
        check("Held-out test set evaluated", test_metrics is not None)
        if test_metrics:
            check("Test accuracy is valid float", 0.0 <= test_metrics["accuracy"] <= 1.0)
            check("Test macro-F1 is valid float", 0.0 <= test_metrics["macro_f1"] <= 1.0)


def test_4_input_safety_guards() -> None:
    section("4. Training Input Safety Guards")

    empty_set = RelationshipDataset([])
    expect_error("Training on empty dataset raises ValueError", lambda: train_gnn_model(empty_set))


def main() -> int:
    import math

    print("=" * 60)
    print("  Layer 2 -- Stage D: Relationship Training Unit Tests")
    print("=" * 60)

    test_1_metrics_calculation_and_edge_cases()
    test_2_training_loss_reduction()
    test_3_checkpoint_saving_and_selection()
    test_4_input_safety_guards()

    section("RESULT")
    if _failures == 0:
        print("  [PASS]  Stage D -- Relationship Training: ALL TESTS PASSED")
    else:
        print(f"  [FAIL]  Stage D -- {_failures} test(s) failed")
    print()

    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
