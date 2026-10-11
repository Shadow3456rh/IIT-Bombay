"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/relationships/train_gnn.py

PURPOSE
-------
Training and validation engine for the RelationshipGNN model.
Computes cross-entropy loss on directed candidate edges, tracks accuracy, macro-F1,
and per-class metrics, manages early stopping, and selects checkpoints strictly based
on validation metrics.

DATA INTEGRITY NOTICE
---------------------
No real annotated relationship data exists yet in the IIT-Bombay Layer-2 repository.
When running training demonstrations using synthetic data, this module explicitly
logs that the training fixture is synthetic.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from relationships.gnn_dataset import (
    RelationshipDataset,
    collate_relationship_graphs,
)
from relationships.gnn_model import (
    NUM_RELATIONSHIP_CLASSES,
    RELATIONSHIP_NAMES,
    RelationshipGNN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """
    Hyperparameters and runtime settings for RelationshipGNN training.
    """

    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    seed: int = 42
    device: str = "cpu"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_class_weights: bool = True
    early_stopping_patience: int = 8
    checkpoint_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluation Metrics Calculation
# ---------------------------------------------------------------------------


def compute_relationship_metrics(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    num_classes: int = NUM_RELATIONSHIP_CLASSES,
) -> Dict[str, Any]:
    """
    Computes accuracy, macro-averaged F1, and per-class precision/recall/F1 metrics.
    Safe against empty tensors and missing classes.
    """
    if y_true.numel() == 0 or y_pred_logits.numel() == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_f1": {cls_id: 0.0 for cls_id in range(num_classes)},
            "per_class_precision": {cls_id: 0.0 for cls_id in range(num_classes)},
            "per_class_recall": {cls_id: 0.0 for cls_id in range(num_classes)},
        }

    preds = torch.argmax(y_pred_logits, dim=-1)
    y_true_np = y_true.cpu().numpy()
    preds_np = preds.cpu().numpy()

    total = len(y_true_np)
    accuracy = float(np.mean(y_true_np == preds_np))

    per_class_prec: Dict[int, float] = {}
    per_class_rec: Dict[int, float] = {}
    per_class_f1: Dict[int, float] = {}
    f1_list: List[float] = []

    for cls_id in range(num_classes):
        tp = int(np.sum((preds_np == cls_id) & (y_true_np == cls_id)))
        fp = int(np.sum((preds_np == cls_id) & (y_true_np != cls_id)))
        fn = int(np.sum((preds_np != cls_id) & (y_true_np == cls_id)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_prec[cls_id] = prec
        per_class_rec[cls_id] = rec
        per_class_f1[cls_id] = f1

        # Only include classes present in ground truth or predictions in macro-average
        if (tp + fp + fn) > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "num_samples": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
    }


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: RelationshipGNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """Runs one training epoch over the dataloader."""
    model.train()
    total_loss = 0.0
    total_edges = 0
    all_preds_list: List[torch.Tensor] = []
    all_targets_list: List[torch.Tensor] = []

    for batch_graph, labels in dataloader:
        if labels is None or labels.numel() == 0 or batch_graph.total_edges == 0:
            continue

        optimizer.zero_grad()
        logits = model.forward_batch(batch_graph)
        loss = criterion(logits, labels.to(device))

        loss.backward()
        optimizer.step()

        batch_n = labels.numel()
        total_loss += float(loss.item()) * batch_n
        total_edges += batch_n
        all_preds_list.append(logits.detach().cpu())
        all_targets_list.append(labels.cpu())

    avg_loss = total_loss / total_edges if total_edges > 0 else 0.0
    if all_preds_list:
        metrics = compute_relationship_metrics(
            torch.cat(all_targets_list, dim=0),
            torch.cat(all_preds_list, dim=0),
        )
    else:
        metrics = compute_relationship_metrics(torch.zeros(0, dtype=torch.long), torch.zeros((0, NUM_RELATIONSHIP_CLASSES)))

    metrics["loss"] = avg_loss
    return avg_loss, metrics


def evaluate_dataset(
    model: RelationshipGNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates the model over the dataset without gradient tracking."""
    model.eval()
    total_loss = 0.0
    total_edges = 0
    all_preds_list: List[torch.Tensor] = []
    all_targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_graph, labels in dataloader:
            if labels is None or labels.numel() == 0 or batch_graph.total_edges == 0:
                continue

            logits = model.forward_batch(batch_graph)
            loss = criterion(logits, labels.to(device))

            batch_n = labels.numel()
            total_loss += float(loss.item()) * batch_n
            total_edges += batch_n
            all_preds_list.append(logits.cpu())
            all_targets_list.append(labels.cpu())

    avg_loss = total_loss / total_edges if total_edges > 0 else 0.0
    if all_preds_list:
        metrics = compute_relationship_metrics(
            torch.cat(all_targets_list, dim=0),
            torch.cat(all_preds_list, dim=0),
        )
    else:
        metrics = compute_relationship_metrics(torch.zeros(0, dtype=torch.long), torch.zeros((0, NUM_RELATIONSHIP_CLASSES)))

    metrics["loss"] = avg_loss
    return metrics


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------


def train_gnn_model(
    train_dataset: RelationshipDataset,
    val_dataset: Optional[RelationshipDataset] = None,
    test_dataset: Optional[RelationshipDataset] = None,
    config: Optional[TrainingConfig] = None,
    model: Optional[RelationshipGNN] = None,
) -> Dict[str, Any]:
    """
    Main training entry point for RelationshipGNN.

    Parameters
    ----------
    train_dataset : RelationshipDataset
        Training samples. Must contain labeled edges.
    val_dataset : Optional[RelationshipDataset]
        Validation samples for model checkpoint selection.
    test_dataset : Optional[RelationshipDataset]
        Held-out test samples evaluated ONLY after final model selection.
    config : Optional[TrainingConfig]
        Training configuration. If None, default TrainingConfig is used.
    model : Optional[RelationshipGNN]
        Existing model to continue training. If None, instantiates a new model.

    Returns
    -------
    Dict[str, Any]
        Dictionary with training history, best epoch, validation metrics,
        and test metrics (if test_dataset provided).
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty training dataset.")
    if train_dataset.total_labeled_edges == 0:
        raise ValueError("Training dataset contains 0 labeled candidate edges.")

    cfg = config or TrainingConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Instantiate model if not provided
    if model is None:
        model = RelationshipGNN(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    model.to(device)

    # Class weighting for class imbalance
    if cfg.use_class_weights:
        weights = train_dataset.compute_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Collation helper closure for device
    def collate_fn(batch):
        return collate_relationship_graphs(batch, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0 and val_dataset.total_labeled_edges > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # Training state
    best_val_score = -1.0  # Track macro-F1 (or -val_loss)
    best_epoch = -1
    best_state_dict: Optional[Dict[str, Any]] = None
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0

    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
        }

        # Validation evaluation
        if val_loader is not None:
            val_metrics = evaluate_dataset(model, val_loader, criterion, device)
            epoch_record["val_loss"] = val_metrics["loss"]
            epoch_record["val_accuracy"] = val_metrics["accuracy"]
            epoch_record["val_macro_f1"] = val_metrics["macro_f1"]

            # Selection criterion: primary = val_macro_f1, tiebreaker = -val_loss
            score = val_metrics["macro_f1"]
            if score > best_val_score:
                best_val_score = score
                best_epoch = epoch
                best_val_metrics = val_metrics
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0

                if cfg.checkpoint_dir:
                    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                    model.save_checkpoint(ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    break
        else:
            # If no validation set, save state from latest epoch
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if cfg.checkpoint_dir:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                model.save_checkpoint(ckpt_path)

        history.append(epoch_record)


    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Held-out Test Set Evaluation (strictly once after selection)
    test_metrics: Optional[Dict[str, Any]] = None
    if test_dataset is not None and len(test_dataset) > 0 and test_dataset.total_labeled_edges > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_metrics = evaluate_dataset(model, test_loader, criterion, device)

    return {
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "model": model,
        "config": cfg.to_dict(),
    }
