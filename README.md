# \# Predicting Human Intestinal Absorption (HIA)

# \## Using Machine Learning and Explainable Artificial Intelligence

# 

# \*\*Mohammad Karamath Fardeen | Student ID: 25251265\*\*  

# MSc Data Science and Analytics — Maynooth University  

# Supervisor: Dr. Kolawole Adebayo | Academic Year: 2025–2026

# 

# \---

# 

# \## Project Overview

# 

# This project develops and benchmarks a comprehensive machine learning pipeline 

# for binary Human Intestinal Absorption (HIA) classification using the standardised 

# HIA\_Hou dataset (578 molecules) from the Therapeutics Data Commons (TDC).

# 

# Seven models across four tiers of complexity were implemented and evaluated:

# 

# | Model | Type | Test AUC |

# |---|---|---|

# | Logistic Regression | Linear baseline | 0.7737 |

# | Random Forest | Ensemble ML | 0.8368 |

# | XGBoost | Ensemble ML | 0.8473 |

# | LightGBM | Ensemble ML | 0.8667 |

# | ChemBERTa | Pretrained Transformer | 0.9128 |

# | GCN | Graph Neural Network | 0.9502 |

# | AttentiveFP | Attention GNN | 0.9568 |

# | \*\*Hybrid Model\*\* | \*\*Descriptor + Graph + BERT\*\* | \*\*0.9588\*\* |

# 

# \---

# 

# \## Key Contributions

# 

# \- Complete benchmarking of 7 models across 4 tiers on TDC HIA\_Hou

# \- Novel hybrid multimodal architecture fusing descriptors, graph and transformer embeddings

# \- First systematic SHAP global and local explainability for binary HIA classification

# \- Scaffold vs random split comparison demonstrating generalisation evaluation

# \- Integrated Gradients atom-level explainability for GNN predictions

# \- Empirical validation of Lipinski Rule of Five and Veber Rules using data-driven analysis

# 

# \---

# 

# \## Repository Structure

