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

## Models Implemented

| Model | Type |
|---|---|
| Logistic Regression | Classical ML (baseline) |
| Random Forest | Ensemble (classical ML) |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| GNN (MPNN/GCN) | Graph Neural Network |

---

## Repo Structure

```
hia-prediction/
│
├── data/
│   ├── raw/                  # Original HIA_Hou dataset (downloaded via TDC)
│   └── processed/            # Cleaned + feature-engineered data (CSV)
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb  # LR, RF, XGBoost, LightGBM
│   ├── 04_gnn.ipynb           # Graph Neural Network
│   └── 05_shap_analysis.ipynb # SHAP explainability
│
├── src/
│   ├── features/
│   │   └── compute_descriptors.py   # RDKit molecular descriptor computation
│   ├── models/
│   │   ├── train_classical.py        # Train LR, RF, XGBoost, LightGBM
│   │   └── train_gnn.py              # Train GNN model
│   ├── explainability/
│   │   └── shap_analysis.py          # SHAP global + local explanations
│   └── utils/
│       ├── data_loader.py            # Load HIA_Hou from TDC
│       └── evaluation.py             # Metrics: AUC, F1, MCC, etc.
│
├── results/
│   ├── figures/              # All plots (ROC curves, SHAP plots, etc.)
│   ├── metrics/              # Model performance CSVs
│   └── shap/                 # SHAP values and summary plots
│
├── tests/
│   └── test_descriptors.py   # Unit tests
│
├── docs/
│   └── project_documentation.docx
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hia-prediction.git
cd hia-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Mac/Linux
venv\Scripts\activate           # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run data processing
```bash
python src/features/compute_descriptors.py
```

### 5. Train models
```bash
python src/models/train_classical.py
python src/models/train_gnn.py
```

### 6. Run SHAP analysis
```bash
python src/explainability/shap_analysis.py
```

---

## Reproducibility

- All random seeds are fixed at `SEED = 42`
- Dataset splits use TDC scaffold-based splitting
- All results are saved to `results/` with versioned filenames

---

## Key Results

> *(To be updated as experiments are completed)*

| Model | ROC-AUC | F1-Score | MCC |
|---|---|---|---|
| Logistic Regression | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |
| GNN (MPNN) | - | - | - |

---

## License

This project is for academic purposes only — MSc Final Project, Maynooth University, 2025–2026.
