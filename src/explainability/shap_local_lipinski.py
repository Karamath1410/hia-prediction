"""
shap_local_lipinski.py
-----------------------
Extends SHAP analysis with:
  1. Local (per-molecule) explanations for representative compounds
     (a correctly predicted "High" absorption molecule, a correctly
     predicted "Low" absorption molecule, and a misclassified molecule)
  2. Validation of top SHAP features against Lipinski's Rule of Five
     and known pharmacokinetic thresholds (TPSA, LogP, HBD, HBA, MolWt)
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

# ── Established pharmacokinetic thresholds for oral absorption ────────────────
# Lipinski's Rule of Five (1997) + Veber's Rules (2002) for oral bioavailability
THRESHOLDS = {
    "MolWt":    {"rule": "Lipinski",  "limit": 500,  "direction": "<=", "desc": "Molecular weight should be <= 500 Da"},
    "LogP":     {"rule": "Lipinski",  "limit": 5,    "direction": "<=", "desc": "LogP should be <= 5 (not too lipophilic)"},
    "HBD":      {"rule": "Lipinski",  "limit": 5,    "direction": "<=", "desc": "H-bond donors should be <= 5"},
    "HBA":      {"rule": "Lipinski",  "limit": 10,   "direction": "<=", "desc": "H-bond acceptors should be <= 10"},
    "TPSA":     {"rule": "Veber",     "limit": 140,  "direction": "<=", "desc": "TPSA should be <= 140 Å² for good oral absorption"},
    "RotBonds": {"rule": "Veber",     "limit": 10,   "direction": "<=", "desc": "Rotatable bonds should be <= 10 for good oral bioavailability"},
}

os.makedirs("results/shap", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


def load_model_and_data(model_name="xgboost"):
    model  = joblib.load(f"results/metrics/{model_name}.joblib")
    scaler = joblib.load("results/metrics/scaler.joblib")

    test_df = pd.read_csv("data/processed/hia_test_features.csv").dropna(subset=FEATURE_COLS)
    X_test  = test_df[FEATURE_COLS].values
    X_test_sc = scaler.transform(X_test)
    y_test  = test_df["Y"].values
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    return model, scaler, test_df, X_test, X_test_sc, y_test, y_pred, y_proba


def pick_representative_molecules(test_df, y_test, y_pred, y_proba):
    """
    Select 3 representative molecules:
      1. High-confidence correct "High absorption" prediction
      2. High-confidence correct "Low absorption" prediction
      3. A misclassified molecule (model got it wrong)
    """
    correct = (y_test == y_pred)

    # Correct High (true=1, pred=1), highest confidence
    high_mask = correct & (y_test == 1)
    idx_high = np.where(high_mask)[0]
    idx_high_best = idx_high[np.argmax(y_proba[idx_high])] if len(idx_high) > 0 else None

    # Correct Low (true=0, pred=0), highest confidence (lowest proba)
    low_mask = correct & (y_test == 0)
    idx_low = np.where(low_mask)[0]
    idx_low_best = idx_low[np.argmin(y_proba[idx_low])] if len(idx_low) > 0 else None

    # Misclassified
    incorrect = np.where(~correct)[0]
    idx_wrong = incorrect[0] if len(incorrect) > 0 else None

    return {
        "high_absorption_correct": idx_high_best,
        "low_absorption_correct":  idx_low_best,
        "misclassified":           idx_wrong,
    }


def explain_molecule(explainer, sv, X_sc, idx, test_df, y_test, y_pred, y_proba, label, model_name):
    """Generate a SHAP waterfall plot + text explanation for one molecule."""
    if idx is None:
        print(f"  No molecule found for category: {label}")
        return None

    smiles = test_df.iloc[idx]["SMILES"] if "SMILES" in test_df.columns else "N/A"
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    proba = y_proba[idx]

    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(np.atleast_1d(explainer.expected_value)) > 1 else explainer.expected_value

    explanation = shap.Explanation(
        values=sv[idx],
        base_values=base_val,
        data=X_sc[idx],
        feature_names=[FEATURE_LABELS[f] for f in FEATURE_COLS]
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP Local Explanation — {label}\nTrue: {true_label} | Predicted: {pred_label} | P(High)={proba:.3f}", fontsize=11)
    plt.tight_layout()
    fname = f"results/figures/{model_name}_local_{label.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # Top 3 contributing features for this molecule
    feat_contributions = sorted(
        zip(FEATURE_COLS, sv[idx]),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]

    print(f"\n  --- {label} ---")
    print(f"  SMILES: {smiles}")
    print(f"  True label: {true_label} | Predicted: {pred_label} | P(High absorption): {proba:.3f}")
    print(f"  Top 3 contributing features:")
    for feat, val in feat_contributions:
        direction = "pushes toward HIGH absorption" if val > 0 else "pushes toward LOW absorption"
        print(f"    {FEATURE_LABELS[feat]:<28} SHAP={val:+.4f}  ({direction})")
    print(f"  Plot saved: {fname}")

    return {
        "category": label,
        "smiles": smiles,
        "true_label": true_label,
        "predicted_label": pred_label,
        "probability_high": round(float(proba), 4),
        "top_features": [(FEATURE_LABELS[f], round(float(v), 4)) for f, v in feat_contributions],
    }


def validate_lipinski_rules(test_df):
    """
    Check what fraction of molecules in each absorption class violate
    Lipinski/Veber thresholds, and compare against SHAP-identified importance.
    """
    print(f"\n{'='*60}")
    print("VALIDATION AGAINST LIPINSKI'S RULE OF FIVE / VEBER'S RULES")
    print(f"{'='*60}\n")

    results = []
    for feat, rule_info in THRESHOLDS.items():
        limit = rule_info["limit"]
        high_df = test_df[test_df["Y"] == 1]
        low_df  = test_df[test_df["Y"] == 0]

        high_violation_rate = (high_df[feat] > limit).mean() * 100
        low_violation_rate  = (low_df[feat] > limit).mean() * 100

        high_mean = high_df[feat].mean()
        low_mean  = low_df[feat].mean()

        results.append({
            "Feature": FEATURE_LABELS[feat],
            "Rule": rule_info["rule"],
            "Threshold": f"<= {limit}",
            "Mean (High Absorption)": round(high_mean, 2),
            "Mean (Low Absorption)": round(low_mean, 2),
            "% Violating in High-Absorption Class": round(high_violation_rate, 1),
            "% Violating in Low-Absorption Class": round(low_violation_rate, 1),
        })

        print(f"{FEATURE_LABELS[feat]} ({rule_info['rule']}'s Rule, threshold {rule_info['direction']} {limit}):")
        print(f"  Mean in HIGH absorption molecules: {high_mean:.2f}")
        print(f"  Mean in LOW absorption molecules:  {low_mean:.2f}")
        print(f"  % exceeding threshold — High absorption class: {high_violation_rate:.1f}%")
        print(f"  % exceeding threshold — Low absorption class:  {low_violation_rate:.1f}%")
        if low_violation_rate > high_violation_rate:
            print(f"  ✓ CONFIRMS pharmacokinetic theory: low-absorption molecules violate this rule more often\n")
        else:
            print(f"  ⚠ Does not clearly confirm expected pharmacokinetic trend in this dataset\n")

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/shap/lipinski_validation.csv", index=False)
    print(f"Lipinski/Veber validation table saved to results/shap/lipinski_validation.csv")
    return results_df


def run_full_analysis(model_name="xgboost"):
    print(f"Loading model and data for: {model_name}")
    model, scaler, test_df, X_test, X_test_sc, y_test, y_pred, y_proba = load_model_and_data(model_name)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sc)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # ── 1. Local explanations for representative molecules ───────────────────
    print(f"\n{'='*60}")
    print("LOCAL SHAP EXPLANATIONS — REPRESENTATIVE MOLECULES")
    print(f"{'='*60}")

    indices = pick_representative_molecules(test_df, y_test, y_pred, y_proba)
    local_results = []
    for label_key, idx in indices.items():
        label_display = label_key.replace("_", " ").title()
        res = explain_molecule(explainer, sv, X_test_sc, idx, test_df, y_test, y_pred, y_proba, label_display, model_name)
        if res:
            local_results.append(res)

    pd.DataFrame(local_results).to_csv("results/shap/local_explanations_summary.csv", index=False)
    print(f"\nLocal explanations summary saved to results/shap/local_explanations_summary.csv")

    # ── 2. Lipinski / Veber validation ────────────────────────────────────────
    lipinski_df = validate_lipinski_rules(test_df)

    return local_results, lipinski_df


if __name__ == "__main__":
    run_full_analysis(model_name="xgboost")
