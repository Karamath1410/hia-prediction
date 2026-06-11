"""
evaluation.py
-------------
Shared evaluation utilities: metrics, ROC curve plotting, confusion matrix.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, classification_report
)


SEED = 42


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return a dict of all key classification metrics."""
    return {
        "roc_auc":  round(roc_auc_score(y_true, y_proba), 4),
        "f1":       round(f1_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "mcc":      round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: str = "results/figures/roc_curves.png"):
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    models_dict : dict  {model_name: fitted_model}
    X_test      : array-like
    y_test      : array-like
    save_path   : str
    """
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — HIA Classification Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Plot and save a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low (0)", "High (1)"],
        yticklabels=["Low (0)", "High (1)"]
    )
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
