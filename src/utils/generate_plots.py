"""
generate_plots.py
------------------
Generates ROC curves (all classical models on one plot) and individual
confusion matrices for every trained classical ML model, using the
saved models and scaler from train_classical.py.

Run this AFTER train_classical.py has been run at least once.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

SEED = 42
FEATURE_COLS = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotBonds", "RingCount", "AromaticRings",
    "HeavyAtoms", "FractionCSP3", "MolMR", "NumHeteroatoms"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest":       "random_forest.joblib",
    "XGBoost":             "xgboost.joblib",
    "LightGBM":            "lightgbm.joblib",
}

os.makedirs("results/figures", exist_ok=True)


def load_test_data():
    test_df = pd.read_csv("data/processed/hia_test_features.csv").dropna(subset=FEATURE_COLS)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["Y"].values

    scaler = joblib.load("results/metrics/scaler.joblib")
    X_test_sc = scaler.transform(X_test)

    return X_test_sc, y_test


def plot_combined_roc_curves(X_test_sc, y_test):
    """Overlay ROC curves for all 4 classical models on a single plot."""
    plt.figure(figsize=(8, 7))
    colors = ["#3498DB", "#9B59B6", "#E67E22", "#27AE60"]

    for (name, fname), color in zip(MODEL_FILES.items(), colors):
        model_path = f"results/metrics/{fname}"
        if not os.path.exists(model_path):
            print(f"  Skipping {name} — model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2.2, color=color)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.500)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Classical ML Models (HIA Test Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = "results/figures/roc_curves_all_classical_models.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined ROC curve saved to: {save_path}")


def plot_confusion_matrices(X_test_sc, y_test):
    """Generate a confusion matrix heatmap for each classical model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for ax, (name, fname) in zip(axes, MODEL_FILES.items()):
        model_path = f"results/metrics/{fname}"
        if not os.path.exists(model_path):
            print(f"  Skipping {name} — model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_sc)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["Low (0)", "High (1)"],
            yticklabels=["Low (0)", "High (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrices — Classical ML Models (HIA Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = "results/figures/confusion_matrices_all_classical_models.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined confusion matrices saved to: {save_path}")

    # Also save individual confusion matrices
    for name, fname in MODEL_FILES.items():
        model_path = f"results/metrics/{fname}"
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_sc)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 4.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low (0)", "High (1)"],
            yticklabels=["Low (0)", "High (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        plt.title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        safe_name = name.replace(" ", "_").lower()
        individual_path = f"results/figures/confusion_matrix_{safe_name}.png"
        plt.savefig(individual_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Individual confusion matrices saved to results/figures/confusion_matrix_*.png")


def main():
    print("Loading test data...")
    X_test_sc, y_test = load_test_data()
    print(f"Test samples: {len(y_test)}\n")

    print("Generating combined ROC curve plot...")
    plot_combined_roc_curves(X_test_sc, y_test)

    print("\nGenerating confusion matrices...")
    plot_confusion_matrices(X_test_sc, y_test)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
