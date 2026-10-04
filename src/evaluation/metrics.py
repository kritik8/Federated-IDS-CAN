"""
Performance Evaluation and Profiling Module for Automotive CAN IDS
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import torch


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, list, dict]]:
    """
    Compute comprehensive scientific classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_pred: Predicted binary labels.
        y_prob: Optional predicted probabilities for positive class (Attack).
        
    Returns:
        Dictionary of metrics including Accuracy, Precision, Recall, F1, FPR, FNR, CM.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # False Negative Rate (FNR) = FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def measure_inference_latency(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    num_warmup: int = 20,
    num_iterations: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure model inference latency per batch and per individual sample.
    
    Args:
        model: PyTorch model in eval mode.
        sample_tensor: Input tensor of shape (Batch, Seq_Len, Features).
        num_warmup: Warmup iterations.
        num_iterations: Timed iterations.
        device: 'cpu' or 'cuda'.
        
    Returns:
        Dict with mean batch latency (ms) and sample latency (microseconds).
    """
    model.eval()
    model.to(device)
    x = sample_tensor.to(device)
    batch_size = x.size(0)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # Benchmarking
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_batch_ms = float(np.mean(latencies))
    std_batch_ms = float(np.std(latencies))
    mean_sample_us = float((mean_batch_ms / batch_size) * 1000.0)

    return {
        "batch_size": batch_size,
        "mean_batch_latency_ms": mean_batch_ms,
        "std_batch_latency_ms": std_batch_ms,
        "mean_per_sample_latency_us": mean_sample_us,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "1D-CNN IDS Training & Validation Dynamics",
) -> plt.Figure:
    """
    Plot Loss and F1 score evolution over training epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # F1 curve
    ax2.plot(epochs, history["train_f1"], "o-", label="Train F1", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_f1"], "s--", label="Val F1", color="#d62728", linewidth=2)
    ax2.set_title("F1-Score vs Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, list],
    class_names: List[str] = ["Normal", "Attack"],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Centralized 1D-CNN IDS Confusion Matrix",
) -> plt.Figure:
    """
    Render normalized and count confusion matrix heatmap.
    """
    cm = np.asarray(cm)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.2f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
