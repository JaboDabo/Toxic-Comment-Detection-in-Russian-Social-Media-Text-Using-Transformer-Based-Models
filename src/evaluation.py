"""
Shared evaluation utilities: metrics, plots, and save functions.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


TARGET_NAMES = ["Negative", "Positive"]


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics. Returns dict."""
    acc = accuracy_score(y_true, y_pred)

    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    metrics = {
        "accuracy": acc,
        "precision_negative": prec_per[0],
        "recall_negative": rec_per[0],
        "f1_negative": f1_per[0],
        "precision_positive": prec_per[1],
        "recall_positive": rec_per[1],
        "f1_positive": f1_per[1],
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
    }

    if y_prob is not None:
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def print_report(y_true, y_pred):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", ax=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def plot_roc_curve(y_true, y_prob, label="Model", ax=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")


def save_metrics(metrics_dict, filepath):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=convert)
    print(f"Metrics saved to {filepath}")
