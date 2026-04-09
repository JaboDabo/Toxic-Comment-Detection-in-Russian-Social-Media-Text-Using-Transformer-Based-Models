"""
Shared evaluation utilities for all models.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all evaluation metrics. Returns a dict."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_nontoxic": float(precision[0]),
        "recall_nontoxic": float(recall[0]),
        "f1_nontoxic": float(f1[0]),
        "precision_toxic": float(precision[1]),
        "recall_toxic": float(recall[1]),
        "f1_toxic": float(f1[1]),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def print_report(y_true, y_pred):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=["Non-toxic", "Toxic"]))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", ax=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-toxic", "Toxic"],
                yticklabels=["Non-toxic", "Toxic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curve(y_true, y_prob, label="Model", ax=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def save_metrics(metrics_dict, filepath):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {filepath}")
