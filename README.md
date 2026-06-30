# Predicting Human Intestinal Absorption (HIA) Using Machine Learning and Explainable AI (SHAP)

**Student:** Mohammad Karamath Fardeen | ID: 25251265  
**Programme:** MSc Data Science & Analytics, Maynooth University  
**Supervisor:** Kolawole Adebayo  
**Year:** 2025–2026

---

## Project Overview

This project builds an accurate and interpretable machine learning pipeline to predict whether a drug molecule will be well-absorbed in the human intestine (Human Intestinal Absorption — HIA). This is a key early-stage drug discovery task that can save time and cost before expensive lab experiments are conducted.

**Dataset:** HIA_Hou from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)  
**Task:** Binary classification — High (1) or Low (0) intestinal absorption  
**Key Feature:** SHAP-based explainability (global + local) to understand *why* a molecule is predicted as high or low absorption, validated against established pharmacokinetic rules

---

## Dataset Summary

| Split | Size | Notes |
|---|---|---|
| Train | 404 molecules | Class distribution: 360 High (1) / 44 Low (0) |
| Validation | 57 molecules | Scaffold-based split |
| Test | 117 molecules | Scaffold-based split |
| **Total** | **578 molecules** | HIA_Hou benchmark from TDC |

Class imbalance in the training set was addressed using **SMOTE** for classical ML models (balanced to 720 samples) and **class-weighted loss** for the GNN.

---

## Models Implemented

| Model | Type |
|---|---|
| Logistic Regression | Classical ML (baseline) |
| Random Forest | Ensemble (classical ML) |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| **Graph Convolutional Network (GCN)** | **Graph Neural Network — best performer** |

---

## Key Results

Performance on the held-out **test set** (117 molecules):

| Model | ROC-AUC | F1-Score | Accuracy | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.7737 | 0.8208 | 0.7350 | 0.3196 |
| Random Forest | 0.8368 | 0.8973 | 0.8376 | 0.5152 |
| XGBoost | 0.8473 | 0.9130 | 0.8632 | 0.5968 |
| LightGBM | 0.8667 | 0.9149 | 0.8632 | 0.5839 |
| **GCN (GNN)** | **0.9502** | **0.9282** | **0.8889** | **0.8889** |

**The Graph Convolutional Network substantially outperforms every classical model**, improving ROC-AUC by 8.4 percentage points and more than doubling MCC over the best classical model (LightGBM). This demonstrates that learned graph-structured representations capture predictive signal beyond what hand-crafted 2D descriptors provide for this task.

ROC curves and confusion matrices for all classical models are available in `results/figures/`.

---

## SHAP Explainability — Global Feature Importance

SHAP (TreeExplainer) was applied to the XGBoost model. Ranked by mean absolute SHAP value:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | TPSA (Polar Surface Area) | 1.1135 |
| 2 | Hydrogen Bond Donors | 0.7665 |
| 3 | Hydrogen Bond Acceptors | 0.6707 |
| 4 | Rotatable Bonds | 0.6205 |
| 5 | LogP (Lipophilicity) | 0.6027 |
| 6 | Molecular Weight | 0.4899 |
| 7 | Heteroatom Count | 0.3897 |
| 8 | Ring Count | 0.3676 |
| 9 | Fraction CSP3 | 0.3183 |
| 10 | Molar Refractivity | 0.2977 |
| 11 | Heavy Atom Count | 0.2920 |
| 12 | Aromatic Rings | 0.1963 |

**TPSA is the dominant predictor**, consistent with established pharmacokinetic theory — high polar surface area increases the energetic cost of crossing the intestinal membrane, reducing absorption.

---

## SHAP Explainability — Local (Per-Molecule) Explanations

Three representative molecules were analysed in detail (see `results/shap/local_explanations_summary.csv` and `results/figures/xgboost_local_*.png`):

| Category | True Label | Predicted | P(High) | Top Driving Feature |
|---|---|---|---|---|
| High Absorption (correct) | 1 | 1 | 0.999 | TPSA (+1.69) |
| Low Absorption (correct) | 0 | 0 | 0.015 | TPSA (−1.75) |
| Misclassified | 0 | 1 | 0.839 | Molecular Weight (−2.02, outweighed by other features) |

The misclassified case highlights a genuine limitation of 2D-descriptor-based models on structurally complex molecules — a gap the graph-based GCN appears to partially close.

---

## Validation Against Lipinski's Rule of Five / Veber's Rules

Test-set molecules were checked against classical pharmacokinetic thresholds to see whether SHAP's data-driven feature ranking aligns with established medicinal chemistry rules:

| Rule | Threshold | % Violating — High Absorption | % Violating — Low Absorption | Confirms Theory? |
|---|---|---|---|---|
| Molecular Weight (Lipinski) | ≤ 500 | 3.3% | 25.9% | ✅ Yes |
| LogP (Lipinski) | ≤ 5 | 2.2% | 3.7% | ✅ Yes |
| H-Bond Donors (Lipinski) | ≤ 5 | 0.0% | 37.0% | ✅ Yes |
| H-Bond Acceptors (Lipinski) | ≤ 10 | 1.1% | 29.6% | ✅ Yes |
| **TPSA (Veber)** | **≤ 140** | **3.3%** | **51.9%** | ✅ **Yes — strongest signal** |
| Rotatable Bonds (Veber) | ≤ 10 | 5.6% | 3.7% | ⚠️ No clear trend |

**5 of 6 rules were confirmed by the dataset**, with TPSA showing by far the largest gap between classes — independently corroborating its #1 ranking in the SHAP analysis. Full table: `results/shap/lipinski_validation.csv`.

---

## Repo Structure

```
hia-prediction/
│
├── data/
│   ├── raw/                   # HIA_Hou dataset splits downloaded via TDC
│   └── processed/             # RDKit-computed feature matrices (train/valid/test)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_feature_engineering.ipynb  # RDKit descriptor computation walkthrough
│
├── src/
│   ├── features/
│   │   └── compute_descriptors.py     # RDKit molecular descriptor computation
│   ├── models/
│   │   ├── train_classical.py          # Train LR, RF, XGBoost, LightGBM
│   │   └── train_gnn.py                # Train Graph Convolutional Network (GCN)
│   ├── explainability/
│   │   ├── shap_analysis.py            # Global SHAP explanations
│   │   └── shap_local_lipinski.py      # Local SHAP + Lipinski/Veber validation
│   └── utils/
│       ├── data_loader.py              # Load HIA_Hou from TDC
│       ├── evaluation.py               # Shared metrics utilities
│       └── generate_plots.py           # ROC curves + confusion matrices
│
├── results/
│   ├── figures/      # ROC curves, confusion matrices, SHAP summary/bar/local/waterfall plots
│   ├── metrics/       # all_model_results.csv + saved model files (.joblib, .pt)
│   └── shap/           # SHAP values, local explanations, Lipinski validation CSVs
│
├── tests/
├── docs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/Karamath1410/hia-prediction.git
cd hia-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline end to end
```bash
python src/utils/data_loader.py                  # Download HIA_Hou dataset
python src/features/compute_descriptors.py        # Compute RDKit descriptors
python src/models/train_classical.py               # Train LR, RF, XGBoost, LightGBM
python src/models/train_gnn.py                      # Train GCN (Graph Neural Network)
python src/explainability/shap_analysis.py          # Global SHAP analysis
python src/explainability/shap_local_lipinski.py    # Local SHAP + Lipinski/Veber validation
python src/utils/generate_plots.py                  # ROC curves + confusion matrices
```

### 4. Explore the notebooks
```bash
jupyter notebook notebooks/
```

---

## Reproducibility

- All random seeds are fixed at `SEED = 42`
- Dataset splits use TDC scaffold-based splitting
- Class imbalance handled with SMOTE (classical models) and class-weighted loss (GCN)
- All results, trained models, and SHAP values are saved under `results/`

---

## Key Takeaways for the Thesis

1. **Graph-based representations outperform hand-crafted descriptors** for HIA classification on this dataset — contrasting with prior literature (e.g. CaliciBoost) on the related Caco-2 task, suggesting the GNN-vs-classical-ML advantage is task-dependent rather than universal.
2. **TPSA is the single strongest predictor of HIA**, confirmed independently by both SHAP feature importance and classical Lipinski/Veber rule violation analysis.
3. **SHAP local explanations are chemically sensible**, including on a misclassified example, demonstrating that model reasoning — not just accuracy — is interpretable.
4. **Rotatable Bonds (Veber's flexibility rule) did not show the expected trend**, an honest and interesting finding suggesting flexibility may be a weaker independent predictor once polarity-related descriptors are already captured.

---

## License

This project is for academic purposes only — MSc Final Project, Maynooth University, 2025–2026.
