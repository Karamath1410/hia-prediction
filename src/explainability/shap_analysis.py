"""
shap_analysis.py
----------------
SHAP explainability analysis for the best-performing tree model (XGBoost/LightGBM).
Generates:
  - Global SHAP summary plot (feature importance across all molecules)
  - SHAP bar plot (mean absolute SHAP values)
  - Local SHAP waterfall plot (single molecule explanation)
  - SHAP values CSV for further analysis
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

SEED = 42
FEATURE_COLS = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotBonds", "RingCount", "AromaticRings",
    "HeavyAtoms", "FractionCSP3", "MolMR", "NumHeteroatoms"
]
FEATURE_LABELS = {
    "MolWt":          "Molecular Weight",
    "LogP":           "LogP (Lipophilicity)",
    "TPSA":           "TPSA (Polar Surface Area)",
    "HBD":            "H-Bond Donors",
    "HBA":            "H-Bond Acceptors",
    "RotBonds":       "Rotatable Bonds",
    "RingCount":      "Ring Count",
    "AromaticRings":  "Aromatic Rings",
    "HeavyAtoms":     "Heavy Atom Count",
    "FractionCSP3":   "Fraction CSP3",
    "MolMR":          "Molar Refractivity",
    "NumHeteroatoms": "Heteroatom Count",
}

os.makedirs("results/shap", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


def run_shap_analysis(model_name: str = "xgboost"):
    # ── Load model and data ───────────────────────────────────────────────────
    model_path = f"results/metrics/{model_name}.joblib"
    scaler_path = "results/metrics/scaler.joblib"

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    test_df = pd.read_csv("data/processed/hia_test_features.csv").dropna(subset=FEATURE_COLS)
    X_test  = test_df[FEATURE_COLS].values
    X_test_sc = scaler.transform(X_test)
    y_test  = test_df["Y"].values

    print(f"Running SHAP on: {model_name} | Test samples: {len(X_test_sc)}")

    # ── SHAP TreeExplainer ────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sc)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # SHAP values for class 1 (high absorption)
    else:
        sv = shap_values

    # ── Save SHAP values ──────────────────────────────────────────────────────
    shap_df = pd.DataFrame(sv, columns=FEATURE_COLS)
    shap_df["true_label"] = y_test
    shap_df.to_csv(f"results/shap/{model_name}_shap_values.csv", index=False)
    print(f"SHAP values saved to results/shap/{model_name}_shap_values.csv")

    # ── Plot 1: Summary dot plot ──────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test_sc,
        feature_names=[FEATURE_LABELS[f] for f in FEATURE_COLS],
        show=False
    )
    plt.title(f"SHAP Summary Plot — {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/figures/{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/figures/{model_name}_shap_summary.png")

    # ── Plot 2: Bar plot (mean |SHAP|) ────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        sv, X_test_sc,
        feature_names=[FEATURE_LABELS[f] for f in FEATURE_COLS],
        plot_type="bar", show=False
    )
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {model_name.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/figures/{model_name}_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/figures/{model_name}_shap_bar.png")

    # ── Plot 3: Waterfall plot for a single molecule ──────────────────────────
    idx = 0  # First test molecule — change to any index for a specific molecule
    explanation = shap.Explanation(
        values=sv[idx],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_test_sc[idx],
        feature_names=[FEATURE_LABELS[f] for f in FEATURE_COLS]
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP Waterfall — Molecule {idx} (True label: {y_test[idx]})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/figures/{model_name}_shap_waterfall_mol{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: results/figures/{model_name}_shap_waterfall_mol{idx}.png")

    # ── Print top features ────────────────────────────────────────────────────
    mean_shap = np.abs(sv).mean(axis=0)
    importance = sorted(zip(FEATURE_COLS, mean_shap), key=lambda x: x[1], reverse=True)
    print("\nTop Features by Mean |SHAP|:")
    for feat, val in importance:
        print(f"  {FEATURE_LABELS[feat]:<30} {val:.4f}")


if __name__ == "__main__":
    run_shap_analysis(model_name="xgboost")
