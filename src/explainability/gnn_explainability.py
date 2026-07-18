"""
gnn_explainability.py
---------------------
Makes the graph neural network (Attentive FP) explainable by computing
atom-level importance using Integrated Gradients.

This complements the SHAP analysis on XGBoost, extending explainability to
the graph-based model. It answers the questions:
  - Which atoms/substructures increase predicted absorption?
  - Which atoms/substructures decrease predicted absorption?
  - Are the explanations chemically meaningful?

Integrated Gradients (Sundararajan et al., 2017) attributes the model's
prediction to each input atom feature by integrating gradients along a path
from a baseline (zero features) to the actual input.

Author: Mohammad Karamath Fardeen (25251265)
Supervisor: Kolawole Adebayo | Maynooth University | 2025-2026
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATOM_LIST = ['C','N','O','S','F','Cl','Br','I','P','B','Si','H','Other']

def atom_features(atom):
    symbol = atom.GetSymbol()
    enc = [1 if symbol == s else 0 for s in ATOM_LIST[:-1]]
    enc.append(1 if symbol not in ATOM_LIST[:-1] else 0)
    return enc + [atom.GetDegree(), atom.GetFormalCharge(),
                  int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                  atom.GetTotalNumHs()]

def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == Chem.rdchem.BondType.SINGLE), int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE), int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()), int(bond.IsInRing())]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_indices += [[i,j],[j,i]]
        edge_feats += [f, f]
    if not edge_indices:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.zeros((0,6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), mol


def load_model(num_node_features, num_edge_features):
    model = AttentiveFP(
        in_channels=num_node_features, hidden_channels=64, out_channels=1,
        edge_dim=num_edge_features, num_layers=3, num_timesteps=2, dropout=0.2
    ).to(DEVICE)
    model.load_state_dict(torch.load("results/metrics/attentivefp_model.pt", map_location=DEVICE))
    model.eval()
    return model


def integrated_gradients(model, data, steps=50):
    """
    Compute atom-level importance using Integrated Gradients.
    Returns one importance score per atom.
    """
    data = data.to(DEVICE)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

    baseline = torch.zeros_like(data.x)
    total_grads = torch.zeros_like(data.x)

    for alpha in np.linspace(0, 1, steps):
        interp = (baseline + alpha * (data.x - baseline)).clone().detach().requires_grad_(True)
        out = model(interp, data.edge_index, data.edge_attr, batch)
        model.zero_grad()
        out.backward()
        total_grads += interp.grad

    avg_grads = total_grads / steps
    ig = (data.x - baseline) * avg_grads
    atom_importance = ig.sum(dim=1).detach().cpu().numpy()  # sum over features per atom
    return atom_importance


def explain_molecule(model, smiles, label_name, save_path):
    data, mol = smiles_to_graph(smiles)
    if data is None:
        print(f"Could not parse {smiles}")
        return

    atom_importance = integrated_gradients(model, data)

    # Normalise for visualisation
    max_abs = np.abs(atom_importance).max() + 1e-9
    norm_importance = atom_importance / max_abs

    print(f"\n{label_name}")
    print(f"SMILES: {smiles}")
    print(f"Atom importance (positive = increases absorption prediction):")
    for i, atom in enumerate(mol.GetAtoms()):
        print(f"  Atom {i} ({atom.GetSymbol()}): {atom_importance[i]:+.4f}")

    # Draw molecule with atom importance heatmap
    try:
        fig = SimilarityMaps.GetSimilarityMapFromWeights(
            mol, list(norm_importance.astype(float)), colorMap='coolwarm', size=(400, 400))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved atom importance map: {save_path}")
    except Exception as e:
        print(f"Could not draw map: {e}")

    return atom_importance


def main():
    os.makedirs("results/figures", exist_ok=True)

    # Determine feature dimensions
    sample_data, _ = smiles_to_graph("CCO")
    num_node_features = sample_data.x.shape[1]
    num_edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr.shape[0] > 0 else 6

    model = load_model(num_node_features, num_edge_features)

    # Load some test molecules
    test_df = pd.read_csv("data/raw/hia_test.csv")

    # Pick one high absorption and one low absorption molecule
    high_mol = test_df[test_df["Y"] == 1].iloc[0]["Drug"]
    low_mol  = test_df[test_df["Y"] == 0].iloc[0]["Drug"]

    print("="*55)
    print("ATTENTIVE FP EXPLAINABILITY VIA INTEGRATED GRADIENTS")
    print("="*55)

    explain_molecule(model, high_mol, "HIGH ABSORPTION MOLECULE",
                     "results/figures/gnn_explain_high_absorption.png")
    explain_molecule(model, low_mol, "LOW ABSORPTION MOLECULE",
                     "results/figures/gnn_explain_low_absorption.png")

    print("\n" + "="*55)
    print("INTERPRETATION GUIDE")
    print("="*55)
    print("Atoms with POSITIVE importance push the prediction toward HIGH absorption.")
    print("Atoms with NEGATIVE importance push toward LOW absorption.")
    print("Polar atoms (N, O) in the low-absorption molecule are expected to")
    print("show negative importance, consistent with the SHAP finding that")
    print("polar surface area (TPSA) reduces absorption.")


if __name__ == "__main__":
    main()
