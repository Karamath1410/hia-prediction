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
**Key Feature:** SHAP-based explainability to understand *why* a molecule is predicted as high or low absorption

---

## Dataset Summary

| Split | Size | Notes |
|---|---|---|
| Train | 404 molecules | Class distribution: 360 High (1) / 44 Low (0) |
| Validation | 57 molecules | Scaffold-based split |
| Test | 117 molecules | Scaffold-based split |
| **Total** | **578 molecules** | HIA_Hou benchmark from TDC |

Class imbalance in the training set was addressed using **SMOTE**, which balanced the training data to 720 samples (360 per class).

---

## Models Implemented

| Model | Type |
|---|---|
| Logistic Regression | Classical ML (baseline) |
| Random Forest | Ensemble (classical ML) |
| XGBoost | Gradient Boosting |
| **LightGBM** | Gradient Boosting (best performer) |

---

## Key Results

Performance on the held-out **test set** (117 molecules):

| Model | ROC-AUC | F1-Score | Accuracy | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.7737 | 0.8208 | 0.7350 | 0.3196 |
| Random Forest | 0.8368 | 0.8973 | 0.8376 | 0.5152 |
| XGBoost | 0.8473 | 0.9130 | 0.8632 | 0.5968 |
| **LightGBM** | **0.8667** | **0.9149** | **0.8632** | 0.5839 |

**Best model: LightGBM**, with the highest ROC-AUC (0.8667) and F1-score (0.9149) on the test set. XGBoost is a close second and was used for the SHAP explainability analysis due to its strong, well-documented `TreeExplainer` support.

---

## SHAP Explainability Results

SHAP (TreeExplainer) was applied to the XGBoost model to identify the molecular features driving HIA predictions. Ranked by mean absolute SHAP value:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | TPSA (Topological Polar Surface Area) | 1.1135 |
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

**TPSA emerged as the single most important predictor**, consistent with established pharmacokinetic theory — molecules with high polar surface area struggle to cross the lipid-rich intestinal membrane, resulting in lower absorption.

SHAP visualisations (summary plot, bar plot, and per-molecule waterfall plot) are saved in `results/figures/`.

---

## Repo Structure

```
hia-prediction/
│
├── data/
│   ├── raw/                  # HIA_Hou dataset splits downloaded via TDC
│   └── processed/            # RDKit-computed feature matrices (train/valid/test)
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb  # LR, RF, XGBoost, LightGBM
│   └── 05_shap_analysis.ipynb # SHAP explainability
│
├── src/
│   ├── features/
│   │   └── compute_descriptors.py   # RDKit molecular descriptor computation
│   ├── models/
│   │   └── train_classical.py        # Train LR, RF, XGBoost, LightGBM
│   ├── explainability/
│   │   └── shap_analysis.py          # SHAP global + local explanations
│   └── utils/
│       ├── data_loader.py            # Load HIA_Hou from TDC
│       └── evaluation.py             # Metrics: AUC, F1, MCC, etc.
│
├── results/
│   ├── figures/               # ROC curves, SHAP summary/bar/waterfall plots
│   ├── metrics/                # classical_results.csv + saved model files
│   └── shap/                   # Raw SHAP values (CSV)
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

### 3. Run the pipeline end to end
```bash
python src/utils/data_loader.py            # Download HIA_Hou dataset
python src/features/compute_descriptors.py # Compute RDKit descriptors
python src/models/train_classical.py        # Train LR, RF, XGBoost, LightGBM
python src/explainability/shap_analysis.py  # Run SHAP analysis
```

---

## Reproducibility

- All random seeds are fixed at `SEED = 42`
- Dataset splits use TDC scaffold-based splitting
- Class imbalance handled with SMOTE on the training set only
- All results, trained models, and SHAP values are saved under `results/`

---

## Next Steps

- Implement and benchmark Graph Neural Network (GNN/MPNN) models against classical ML
- Expand SHAP analysis with local (per-molecule) explanations for representative compounds
- Validate top SHAP features against Lipinski's Rule of Five and known pharmacokinetic thresholds
- Write up full results and discussion in the project thesis

---

## License

This project is for academic purposes only — MSc Final Project, Maynooth University, 2025–2026.
