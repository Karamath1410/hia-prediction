"""
split_comparison.py
-------------------
Compares model performance across different data splitting strategies:
  - Random split
  - Scaffold split (structurally different molecules in test)

This addresses the research question: "Which models genuinely generalise
to unseen chemical scaffolds?"

A large drop in performance from random to scaffold split indicates a model
is memorising similar molecules rather than learning generalisable patterns.

Author: Mohammad Karamath Fardeen (25251265)
Supervisor: Kolawole Adebayo | Maynooth University | 2025-2026
"""

import os
import numpy as np
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

SEED = 42
FEATURE_COLS = ["MolWt","LogP","TPSA","HBD","HBA","RotBonds",
                "RingCount","AromaticRings","HeavyAtoms",
                "FractionCSP3","MolMR","NumHeteroatoms"]

DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt, "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA, "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings, "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "HeavyAtoms": Descriptors.HeavyAtomCount, "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolMR": Descriptors.MolMR, "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
}


def compute_features(df):
    records = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["Drug"])
        if mol is None:
            continue
        feats = {name: fn(mol) for name, fn in DESCRIPTOR_FUNCTIONS.items()}
        feats["Y"] = int(row["Y"])
        records.append(feats)
    return pd.DataFrame(records)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1),
    }


def evaluate_split(split_method):
    print(f"\n{'='*55}")
    print(f"SPLIT METHOD: {split_method.upper()}")
    print(f"{'='*55}")

    data = ADME(name="HIA_Hou")
    split = data.get_split(method=split_method, seed=SEED)

    train_feat = compute_features(split["train"]).dropna(subset=FEATURE_COLS)
    test_feat  = compute_features(split["test"]).dropna(subset=FEATURE_COLS)

    X_train, y_train = train_feat[FEATURE_COLS].values, train_feat["Y"].values
    X_test,  y_test  = test_feat[FEATURE_COLS].values,  test_feat["Y"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

    results = []
    for name, model in get_models().items():
        model.fit(X_train_res, y_train_res)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        y_pred = model.predict(X_test_sc)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        results.append({"split": split_method, "model": name,
                        "roc_auc": round(auc, 4), "f1": round(f1, 4), "mcc": round(mcc, 4)})
        print(f"  {name:<22} AUC: {auc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    return results


def main():
    all_results = []
    for method in ["random", "scaffold"]:
        all_results.extend(evaluate_split(method))

    results_df = pd.DataFrame(all_results)
    os.makedirs("results/metrics", exist_ok=True)
    results_df.to_csv("results/metrics/split_comparison.csv", index=False)

    print(f"\n{'='*55}")
    print("SUMMARY: RANDOM vs SCAFFOLD SPLIT (ROC-AUC)")
    print(f"{'='*55}")
    pivot = results_df.pivot(index="model", columns="split", values="roc_auc")
    pivot["drop"] = (pivot["random"] - pivot["scaffold"]).round(4)
    print(pivot.to_string())
    print("\nA larger 'drop' means the model relied more on memorising similar")
    print("molecules and generalises less well to unseen chemical scaffolds.")
    print(f"\nResults saved to results/metrics/split_comparison.csv")


if __name__ == "__main__":
    main()
