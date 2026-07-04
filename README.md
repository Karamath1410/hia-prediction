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
**Key Feature:** SHAP-based explainability (global + local) validated against Lipinski's Rule of Five and Veber's Rules

---

## Dataset Summary

| Split | Size | Notes |
|---|---|---|
| Train | 404 molecules | 360 High (1) / 44 Low (0) — class imbalance handled via SMOTE |
| Validation | 57 molecules | Scaffold-based split |
| Test | 117 molecules | Scaffold-based split |
| **Total** | **578 molecules** | HIA_Hou benchmark from TDC |

---

## Models Implemented

Four tiers of models were implemented and benchmarked — from simple baselines to state-of-the-art 2024/2025 methods:

| Tier | Model | Type |
|---|---|---|
| Baseline | Logistic Regression | Linear ML |
| Classical Ensemble | Random Forest | Ensemble ML |
| Classical Ensemble | XGBoost | Gradient Boosting |
| Classical Ensemble | LightGBM | Gradient Boosting |
| Graph Neural Network | GCN (Graph Convolutional Network) | GNN |
| **State-of-the-Art** | **Attentive FP** | **Attention-based GNN (Xiong et al., J. Med. Chem. 2020)** |
| **State-of-the-Art** | **ChemBERTa** | **Pretrained Transformer on 77M SMILES (Chithrananda et al., 2020)** |

---

## Key Results (Test Set)

| Model | ROC-AUC | F1-Score | Accuracy | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.7737 | 0.8208 | 0.7350 | 0.3196 |
| Random Forest | 0.8368 | 0.8973 | 0.8376 | 0.5152 |
| XGBoost | 0.8473 | 0.9130 | 0.8632 | 0.5968 |
| LightGBM | 0.8667 | 0.9149 | 0.8632 | 0.5839 |
| GCN (GNN) | 0.9502 | 0.9282 | 0.8889 | 0.6831 |
| ChemBERTa | 0.9128 | 0.8824 | 0.8291 | 0.5873 |
| **Attentive FP** | **0.9568** | **0.9006** | **0.8547** | **0.6458** |

**Best model: Attentive FP** (ROC-AUC: 0.9568) — an attention-based GNN specifically designed for molecular property prediction, outperforming all classical ML models and the GCN.

**Key finding:** All three graph/transformer-based models substantially outperform classical descriptor-based models, confirming that learned molecular representations capture predictive signal beyond hand-crafted 2D descriptors.

---

## SHAP Explainability — Global Feature Importance

SHAP (TreeExplainer) applied to XGBoost. Top features by mean absolute SHAP value:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | TPSA (Polar Surface Area) | 1.1135 |
| 2 | Hydrogen Bond Donors | 0.7665 |
| 3 | Hydrogen Bond Acceptors | 0.6707 |
| 4 | Rotatable Bonds | 0.6205 |
| 5 | LogP (Lipophilicity) | 0.6027 |
| 6 | Molecular Weight | 0.4899 |

**TPSA is the dominant predictor** — consistent with pharmacokinetic theory and independently confirmed by Lipinski/Veber rule violation analysis.

---

## Lipinski/Veber Validation

| Rule | Threshold | % Violating — High Absorption | % Violating — Low Absorption | Confirmed? |
|---|---|---|---|---|
| Molecular Weight (Lipinski) | ≤ 500 | 3.3% | 25.9% | ✅ |
| LogP (Lipinski) | ≤ 5 | 2.2% | 3.7% | ✅ |
| H-Bond Donors (Lipinski) | ≤ 5 | 0.0% | 37.0% | ✅ |
| H-Bond Acceptors (Lipinski) | ≤ 10 | 1.1% | 29.6% | ✅ |
| **TPSA (Veber)** | **≤ 140** | **3.3%** | **51.9%** | ✅ **Strongest** |
| Rotatable Bonds (Veber) | ≤ 10 | 5.6% | 3.7% | ⚠️ No clear trend |

5 of 6 rules confirmed — SHAP findings independently validated by classical pharmacokinetic rules.

---

## Repo Structure

```
hia-prediction/
│
├── data/
│   ├── raw/                   # HIA_Hou dataset splits (downloaded via TDC)
│   └── processed/             # RDKit-computed feature matrices
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── features/
│   │   └── compute_descriptors.py
│   ├── models/
│   │   ├── train_classical.py         # LR, RF, XGBoost, LightGBM
│   │   ├── train_gnn.py               # Graph Convolutional Network (GCN)
│   │   ├── train_attentivefp.py       # Attentive FP (state-of-the-art)
│   │   └── train_chemberta.py         # ChemBERTa (state-of-the-art)
│   ├── explainability/
│   │   ├── shap_analysis.py           # Global SHAP
│   │   └── shap_local_lipinski.py     # Local SHAP + Lipinski/Veber validation
│   └── utils/
│       ├── data_loader.py
│       ├── evaluation.py
│       └── generate_plots.py          # ROC curves + confusion matrices
│
├── results/
│   ├── figures/       # ROC curves, confusion matrices, SHAP plots
│   ├── metrics/        # all_model_results.csv + saved model files
│   └── shap/            # SHAP values, local explanations, Lipinski CSV
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

```bash
git clone https://github.com/Karamath1410/hia-prediction.git
cd hia-prediction
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python src/utils/data_loader.py                    # Download HIA_Hou dataset
python src/features/compute_descriptors.py          # Compute RDKit descriptors
python src/models/train_classical.py                # Train classical ML models
python src/models/train_gnn.py                      # Train GCN
python src/models/train_attentivefp.py              # Train Attentive FP
python src/models/train_chemberta.py                # Fine-tune ChemBERTa
python src/explainability/shap_analysis.py          # Global SHAP
python src/explainability/shap_local_lipinski.py    # Local SHAP + Lipinski validation
python src/utils/generate_plots.py                  # ROC curves + confusion matrices
```

---

## Reproducibility

- All random seeds fixed at `SEED = 42`
- TDC scaffold-based splitting for realistic out-of-distribution evaluation
- Class imbalance: SMOTE (classical models), class-weighted loss (GNN/Attentive FP/ChemBERTa)
- All results saved to `results/metrics/all_model_results.csv`

---

## Key Takeaways

1. **Attentive FP achieves best overall AUC (0.9568)** — attention mechanism on molecular graphs outperforms all other approaches
2. **All graph/transformer models substantially outperform classical ML** — confirming graph-based representations are superior for this task
3. **TPSA is the single strongest predictor** — confirmed by both SHAP and Lipinski/Veber rule analysis independently
4. **ChemBERTa (pretrained Transformer) reaches AUC 0.9128** — competitive without any molecular graph construction, using only raw SMILES
5. **5/6 pharmacokinetic rules validated empirically** — data-driven SHAP findings align with established medicinal chemistry knowledge

---

## License

Academic purposes only — MSc Final Project, Maynooth University, 2025–2026.
