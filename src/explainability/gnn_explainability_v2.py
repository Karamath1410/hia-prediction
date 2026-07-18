"""
gnn_explainability_v2.py
------------------------
Improved atom-level explainability for the Attentive FP graph neural network
using Integrated Gradients (Sundararajan et al., 2017).

Improvements over v1:
  - Fixed RDKit drawing for newer versions
  - Reports true label, predicted probability, predicted class, correctness
  - Explains a balanced set of molecules (correct high, correct low,
    false positive, false negative)
  - Aggregates atom importance by element type to identify general patterns
  - Neutral wording (no assumed conclusions)

Author: Mohammad Karamath Fardeen (25251265)
Supervisor: Kolawole Adebayo | Maynooth University | 2025-2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


def predict(model, data):
    """Return predicted probability and class."""
    data = data.to(DEVICE)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, batch)
        prob = torch.sigmoid(out).item()
    return prob, int(prob > 0.5)


def integrated_gradients(model, data, steps=50):
    """Compute atom-level importance using Integrated Gradients."""
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
    return ig.sum(dim=1).detach().cpu().numpy()


def draw_molecule_with_importance(mol, importance, save_path, title=""):
    """Draw molecule with atoms coloured by importance (RDKit v2023+ compatible)."""
    max_abs = np.abs(importance).max() + 1e-9
    norm = importance / max_abs

    atom_colors = {}
    highlight_atoms = []
    for i, val in enumerate(norm):
        highlight_atoms.append(int(i))
        v = float(val)  # convert numpy float32 to Python float
        if v > 0:
            # Positive = blue (increases absorption)
            atom_colors[int(i)] = (float(1 - v), float(1 - v), 1.0)
        else:
            # Negative = red (decreases absorption)
            atom_colors[int(i)] = (1.0, float(1 + v), float(1 + v))

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 450)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"  Saved molecule map: {save_path}")


def analyse_molecule(model, smiles, true_label, category, save_dir):
    data, mol = smiles_to_graph(smiles)
    if data is None:
        return None

    prob, pred = predict(model, data)
    correct = (pred == true_label)
    importance = integrated_gradients(model, data)

    print(f"\n{'-'*60}")
    print(f"CATEGORY: {category}")
    print(f"SMILES: {smiles[:70]}{'...' if len(smiles) > 70 else ''}")
    print(f"True label: {true_label} | Predicted: {pred} | P(High)={prob:.4f} | {'CORRECT' if correct else 'INCORRECT'}")

    # Aggregate importance by element type
    element_importance = {}
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        element_importance.setdefault(sym, []).append(importance[i])

    print(f"\n  Mean atom importance by element:")
    summary = []
    for sym, vals in sorted(element_importance.items(), key=lambda kv: np.mean(kv[1])):
        mean_val = np.mean(vals)
        direction = "toward HIGH absorption" if mean_val > 0 else "toward LOW absorption"
        print(f"    {sym:<3} (n={len(vals):>2}): {mean_val:+.4f}  ({direction})")
        summary.append({"element": sym, "count": len(vals), "mean_importance": round(float(mean_val), 4)})

    # Top 5 most influential atoms
    top_idx = np.argsort(-np.abs(importance))[:5]
    print(f"\n  Top 5 most influential atoms:")
    for idx in top_idx:
        sym = mol.GetAtomWithIdx(int(idx)).GetSymbol()
        print(f"    Atom {idx} ({sym}): {importance[idx]:+.4f}")

    safe_name = category.replace(" ", "_").lower()
    save_path = os.path.join(save_dir, f"gnn_explain_{safe_name}.png")
    try:
        draw_molecule_with_importance(mol, importance, save_path, category)
    except Exception as e:
        print(f"  Could not draw molecule: {e}")

    return {"category": category, "smiles": smiles, "true_label": true_label,
            "predicted": pred, "probability": round(prob, 4), "correct": correct,
            "element_summary": summary}


def main():
    save_dir = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    sample_data, _ = smiles_to_graph("CCO")
    num_node_features = sample_data.x.shape[1]
    num_edge_features = 6

    model = load_model(num_node_features, num_edge_features)
    test_df = pd.read_csv("data/raw/hia_test.csv")

    print("="*60)
    print("ATTENTIVE FP EXPLAINABILITY VIA INTEGRATED GRADIENTS")
    print("="*60)

    # Classify all test molecules to find the 4 categories
    records = []
    for _, row in test_df.iterrows():
        data, mol = smiles_to_graph(row["Drug"])
        if data is None:
            continue
        prob, pred = predict(model, data)
        records.append({"smiles": row["Drug"], "true": int(row["Y"]),
                        "pred": pred, "prob": prob})
    rec_df = pd.DataFrame(records)

    # Select representative molecules
    selections = []
    tp = rec_df[(rec_df["true"] == 1) & (rec_df["pred"] == 1)].nlargest(2, "prob")
    tn = rec_df[(rec_df["true"] == 0) & (rec_df["pred"] == 0)].nsmallest(2, "prob")
    fp = rec_df[(rec_df["true"] == 0) & (rec_df["pred"] == 1)].nlargest(1, "prob")
    fn = rec_df[(rec_df["true"] == 1) & (rec_df["pred"] == 0)].nsmallest(1, "prob")

    for i, (_, r) in enumerate(tp.iterrows()):
        selections.append((r["smiles"], r["true"], f"Correct High Absorption {i+1}"))
    for i, (_, r) in enumerate(tn.iterrows()):
        selections.append((r["smiles"], r["true"], f"Correct Low Absorption {i+1}"))
    for _, r in fp.iterrows():
        selections.append((r["smiles"], r["true"], "False Positive"))
    for _, r in fn.iterrows():
        selections.append((r["smiles"], r["true"], "False Negative"))

    all_results = []
    for smiles, true_label, category in selections:
        res = analyse_molecule(model, smiles, true_label, category, save_dir)
        if res:
            all_results.append(res)

    # Save summary
    os.makedirs("results/shap", exist_ok=True)
    summary_rows = []
    for r in all_results:
        for e in r["element_summary"]:
            summary_rows.append({
                "category": r["category"], "true_label": r["true_label"],
                "predicted": r["predicted"], "probability": r["probability"],
                "correct": r["correct"], **e
            })
    pd.DataFrame(summary_rows).to_csv("results/shap/gnn_atom_importance_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Atoms with positive attribution push the prediction toward high absorption.")
    print("Atoms with negative attribution push toward low absorption.")
    print("These atom-level attributions can be compared with established")
    print("physicochemical determinants of intestinal absorption, including")
    print("polarity, hydrogen bonding, lipophilicity and molecular flexibility.")
    print("\nSummary saved to results/shap/gnn_atom_importance_summary.csv")


if __name__ == "__main__":
    main()
