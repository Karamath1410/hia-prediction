"""
train_classical.py
------------------
Train and evaluate classical ML models for HIA binary classification.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Saves results to results/metrics/classical_results.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    matthews_corrcoef, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

SEED = 42
FEATURE_COLS = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotBonds", "RingCount", "AromaticRings",
    "HeavyAtoms", "FractionCSP3", "MolMR", "NumHeteroatoms"
]


def load_data():
    train = pd.read_csv("data/processed/hia_train_features.csv").dropna(subset=FEATURE_COLS)
    valid = pd.read_csv("data/processed/hia_valid_features.csv").dropna(subset=FEATURE_COLS)
    test  = pd.read_csv("data/processed/hia_test_features.csv").dropna(subset=FEATURE_COLS)

    X_train, y_train = train[FEATURE_COLS].values, train["Y"].values
    X_valid, y_valid = valid[FEATURE_COLS].values, valid["Y"].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test["Y"].values
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=SEED, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=SEED,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=SEED, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=-1,
            num_leaves=31, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1
        ),
    }


def evaluate(model, X, y, split_name: str) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "split":    split_name,
        "roc_auc":  round(roc_auc_score(y, y_proba), 4),
        "f1":       round(f1_score(y, y_pred), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "mcc":      round(matthews_corrcoef(y, y_pred), 4),
    }


def train_and_evaluate():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, "results/metrics/scaler.joblib")

    # Apply SMOTE to training set only
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
    print(f"After SMOTE — train size: {len(y_train_res)}, class dist: {np.bincount(y_train_res)}")

    all_results = []
    models = get_models()

    for name, model in models.items():
        print(f"\n{'='*50}\nTraining: {name}")

        # Logistic Regression and scaled data; tree models use unscaled
        X_tr = X_train_res
        X_va = X_valid_sc
        X_te = X_test_sc

        model.fit(X_tr, y_train_res)
        joblib.dump(model, f"results/metrics/{name.replace(' ', '_').lower()}.joblib")

        for split_name, X, y in [("valid", X_va, y_valid), ("test", X_te, y_test)]:
            metrics = evaluate(model, X, y, split_name)
            metrics["model"] = name
            all_results.append(metrics)
            print(f"  {split_name.upper()} — AUC: {metrics['roc_auc']}  F1: {metrics['f1']}  MCC: {metrics['mcc']}")

    results_df = pd.DataFrame(all_results)[["model", "split", "roc_auc", "f1", "accuracy", "mcc"]]
    os.makedirs("results/metrics", exist_ok=True)
    results_df.to_csv("results/metrics/classical_results.csv", index=False)
    print(f"\nResults saved to results/metrics/classical_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate()
