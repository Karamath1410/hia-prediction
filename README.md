# 🧪 Predicting Human Intestinal Absorption (HIA)
### Using Machine Learning and Explainable Artificial Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**Mohammad Karamath Fardeen | Student ID: 25251265**

MSc Data Science and Analytics — Maynooth University, Ireland

Supervisor: Dr. Kolawole Adebayo | Academic Year: 2025–2026

</div>

---

## 📋 Project Overview

Human Intestinal Absorption (HIA) is a critical pharmacokinetic property in the ADME
(Absorption, Distribution, Metabolism, Excretion) framework that determines whether an
orally administered drug can reach systemic circulation in sufficient quantities to be
therapeutically effective.

This project develops a **comprehensive, reproducible machine learning pipeline** for
binary HIA classification using the standardised **HIA_Hou dataset (578 molecules)**
from the Therapeutics Data Commons (TDC).

Seven models across four tiers of complexity were implemented, evaluated, and compared.
A novel **hybrid multimodal architecture** was proposed that fuses molecular descriptors,
graph representations and transformer embeddings for superior predictive performance.

---

## 🏆 Results Summary

### Full Model Comparison

| Model | Type | Test AUC | F1 Score | Accuracy | MCC |
|---|---|---|---|---|---|
| Logistic Regression | Linear baseline | 0.7737 | 0.8208 | 0.7350 | 0.3196 |
| Random Forest | Ensemble ML | 0.8368 | 0.8973 | 0.8376 | 0.5152 |
| XGBoost | Ensemble ML | 0.8473 | 0.9130 | 0.8632 | 0.5968 |
| LightGBM | Ensemble ML | 0.8667 | 0.9149 | 0.8632 | 0.5839 |
| ChemBERTa | Pretrained Transformer | 0.9128 | 0.8824 | 0.8291 | 0.5873 |
| GCN | Graph Neural Network | 0.9502 | 0.9282 | 0.8889 | 0.6831 |
| AttentiveFP | Attention GNN | 0.9568 | 0.9006 | 0.8547 | 0.6458 |
| **Hybrid Model** | **Descriptor + Graph + BERT** | **0.9588** | **0.9249** | **0.8889** | **0.7217** |

### Hybrid Model Performance
| Metric | Value |
|---|---|
| Validation AUC | 0.9829 |
| Test AUC | 0.9588 |
| Accuracy | 0.8889 |
| F1 Score | 0.9249 |
| MCC | 0.7217 |

---

## 🔬 Key Contributions

1. **Complete benchmarking** of 7 models across 4 tiers on the standardised TDC HIA_Hou dataset
2. **Novel hybrid multimodal architecture** fusing molecular descriptors, AttentiveFP graph embeddings and ChemBERTa transformer embeddings via late fusion
3. **First systematic SHAP global and local explainability** analysis for binary HIA classification on HIA_Hou
4. **Scaffold vs random split comparison** demonstrating realistic molecular generalisation evaluation
5. **Integrated Gradients atom-level explainability** for GNN predictions identifying positive and negative absorption contributors
6. **Empirical validation** of Lipinski Rule of Five and Veber Rules using data-driven analysis — 5 of 6 rules confirmed

---

## 📊 SHAP Feature Importance

TPSA was identified as the dominant predictor of HIA by SHAP analysis,
independently validated by Lipinski/Veber rule violation analysis.

| Rank | Feature | Mean |SHAP| | Pharmacokinetic Role |
|---|---|---|---|
| 1 | TPSA (Polar Surface Area) | 1.1135 | Membrane permeability barrier |
| 2 | H-Bond Donors | 0.7665 | Desolvation energy cost |
| 3 | H-Bond Acceptors | 0.6707 | Polarity indicator |
| 4 | Rotatable Bonds | 0.6205 | Molecular flexibility |
| 5 | LogP (Lipophilicity) | 0.6027 | Membrane partitioning |
| 6 | Molecular Weight | 0.4899 | Size constraint |
| 7 | Heteroatom Count | 0.3897 | Polarity indicator |
| 8 | Ring Count | 0.3676 | Structural rigidity |
| 9 | Fraction CSP3 | 0.3183 | 3D character |
| 10 | Molar Refractivity | 0.2977 | Polarisability |
| 11 | Heavy Atom Count | 0.2920 | Molecular size |
| 12 | Aromatic Rings | 0.1963 | Lipophilicity contribution |

---

## 🏗️ Hybrid Multimodal Architecture

```
Molecule (SMILES)
       |
  ┌────┴─────────────────────────────────────────┐
  │              │                               │
  ▼              ▼                               ▼
RDKit         Molecular Graph              SMILES Tokens
Descriptors   (atoms + bonds)             (ChemBERTa)
(12 features) (Local topology)            (Sequence context)
  │              │                               │
  ▼              ▼                               ▼
MLP Encoder  AttentiveFP Encoder     Frozen ChemBERTa
(12→64→64)   (3 attention layers)    Projection Layer
  │              │                               │
  ▼              ▼                               ▼
64-dim        64-dim                        64-dim
embedding     embedding                     embedding
  │              │                               │
  └──────────────┴───────────────────────────────┘
                         │
                         ▼
              Late Fusion — Feature Concatenation
                  (192-dimensional representation)
                         │
                         ▼
              Fully Connected Classification Head
                  (192 → 128 → 64 → 1)
                         │
                         ▼
                  HIA Prediction
             (High / Low absorption)
```

---

## 📁 Repository Structure

```
hia-prediction/
│
├── 📓 notebooks/
│   ├── 01_data_loading_EDA.ipynb       # Data loading and exploratory analysis
│   ├── 02_classical_ML.ipynb           # Logistic Regression, RF, XGBoost, LightGBM
│   ├── 03_GCN.ipynb                    # Graph Convolutional Network
│   ├── 04_AttentiveFP.ipynb            # Attentive Fingerprint Network
│   ├── 05_ChemBERTa.ipynb              # ChemBERTa Transformer model
│   ├── 06_SHAP_Analysis.ipynb          # SHAP global and local explainability
│   └── 07_Hybrid_Model.ipynb           # Proposed hybrid multimodal model
│
├── 🐍 src/
│   ├── models/
│   │   ├── train_classical.py          # Classical ML training pipeline
│   │   ├── train_gnn.py                # GCN training pipeline
│   │   ├── train_attentivefp.py        # AttentiveFP training pipeline
│   │   ├── train_chemberta.py          # ChemBERTa fine-tuning pipeline
│   │   ├── train_hybrid.py             # Hybrid model training pipeline
│   │   └── split_comparison.py         # Random vs scaffold split analysis
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py            # SHAP TreeExplainer analysis
│   │   ├── shap_local_lipinski.py      # Local SHAP + Lipinski validation
│   │   └── gnn_explainability.py       # Integrated Gradients for GNN
│   │
│   ├── features/
│   │   └── compute_descriptors.py      # RDKit descriptor computation
│   │
│   └── utils/
│       ├── data_loader.py              # TDC dataset loading utilities
│       ├── evaluation.py               # Metrics computation
│       └── generate_plots.py           # Visualisation utilities
│
├── 🖼️ figures/
│   ├── figure_4_1_methodology_flowchart.png
│   ├── figure_4_2_dataset_distribution.png
│   ├── figure_4_3_descriptor_distributions.png
│   ├── figure_4_4_molecular_graph_representation.png
│   ├── figure_4_5_smiles_representation.png
│   ├── figure_4_6_roc_classical_models.png
│   ├── figure_4_7_deep_learning_roc_curves.png
│   ├── figure_4_8_hybrid_architecture.png
│   ├── figure_4_9_scaffold_vs_random_split.png
│   ├── figure_4_10_shap_detailed_plot.png
│   ├── figure_4_11_high_absorption_integrated_gradients.png
│   ├── figure_4_12_low_absorption_integrated_gradients.png
│   ├── figure_4_13_false_positive_integrated_gradients.png
│   ├── figure_4_14_false_negative_integrated_gradients.png
│   └── figure_4_15_auc_all_models_comparison.png
│
├── 📊 data/
│   └── (HIA_Hou dataset loaded automatically via TDC API)
│
├── 📈 results/
│   ├── metrics/                        # Model performance metrics
│   └── hybrid_model_best.pt            # Best hybrid model weights
│
├── requirements.txt                    # All dependencies
└── README.md                           # This file
```

---

## 🚀 How to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/Karamath1410/hia-prediction
cd hia-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```

Open and run notebooks in order:
- `01_data_loading_EDA.ipynb` — Start here
- `02_classical_ML.ipynb`
- `03_GCN.ipynb`
- `04_AttentiveFP.ipynb`
- `05_ChemBERTa.ipynb`
- `06_SHAP_Analysis.ipynb`
- `07_Hybrid_Model.ipynb` — Proposed hybrid model

> The HIA_Hou dataset is automatically downloaded via the TDC API when notebooks are executed. No manual data download required.

---

## 🛠️ Technologies Used

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch, PyTorch Geometric |
| Classical ML | Scikit-learn, XGBoost, LightGBM |
| Cheminformatics | RDKit |
| Transformers | HuggingFace Transformers (ChemBERTa) |
| Explainability | SHAP, Integrated Gradients |
| Dataset | Therapeutics Data Commons (TDC) |
| Visualisation | Matplotlib, Seaborn |
| Data Processing | NumPy, Pandas |

---

## 📖 Dataset

**HIA_Hou Dataset** from Therapeutics Data Commons (TDC)
- **578** drug-like molecules
- **Binary labels**: High (1) or Low (0) intestinal absorption
- **Split method**: Scaffold-based split (Bemis-Murcko scaffolds)
- **Split ratio**: Train 404 / Validation 57 / Test 117
- **Class distribution**: ~89% high absorption / ~11% low absorption

---

## 📚 Key References

- Chithrananda et al. (2020). ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. arXiv:2010.09885
- Huang et al. (2021). Therapeutics Data Commons. NeurIPS Datasets and Benchmarks
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS
- Wessel et al. (1998). Prediction of HIA from molecular structure. J. Chem. Inf. Comput. Sci.
- Xiong et al. (2020). Pushing the boundaries of molecular representation with graph attention. J. Med. Chem.
- Lipinski et al. (1997). Rule of Five. Advanced Drug Delivery Reviews
- Veber et al. (2002). Molecular properties for oral bioavailability. J. Med. Chem.

---

## 📝 Citation

If you use this work please cite:

```bibtex
@mastersthesis{fardeen2026hia,
  author    = {Mohammad Karamath Fardeen},
  title     = {Predicting Human Intestinal Absorption Using Machine Learning
               and Explainable Artificial Intelligence},
  school    = {Maynooth University},
  year      = {2026},
  note      = {MSc Data Science and Analytics},
  url       = {https://github.com/Karamath1410/hia-prediction}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
<b>Department of Computer Science — Maynooth University, Co. Kildare, Ireland</b><br>
<i>EE648 Project — MSc Data Science and Analytics — 2025/2026</i>
</div>