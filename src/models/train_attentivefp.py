"""
train_attentivefp.py
---------------------
Implements Attentive FP (Xiong et al., Nature Chemistry 2020) for HIA binary classification.
Attentive FP uses graph attention mechanisms at both atom and molecule level,
allowing the model to focus on the most chemically relevant parts of the molecule.

Reference: Xiong et al. (2020) "Pushing the Boundaries of Molecular Representation
for Drug Discovery with the Graph Attention Mechanism", J. Med. Chem.

Uses PyTorch Geometric's built-in AttentiveFP implementation.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Atom features (same as GCN for consistency) ───────────────────────────────
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H', 'Other']

def atom_features(atom):
    symbol = atom.GetSymbol()
    symbol_enc = [1 if symbol == s else 0 for s in ATOM_LIST[:-1]]
    symbol_enc.append(1 if symbol not in ATOM_LIST[:-1] else 0)
    return symbol_enc + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_feats += [feat, feat]
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float))


class HIAGraphDataset(InMemoryDataset):
    def __init__(self, csv_path):
        super().__init__(None)
        df = pd.read_csv(csv_path)
        data_list, skipped = [], 0
        for _, row in df.iterrows():
            g = smiles_to_graph(row["Drug"], int(row["Y"]))
            if g is not None: data_list.append(g)
            else: skipped += 1
        print(f"  Loaded {len(data_list)} graphs from {csv_path} (skipped {skipped})")
        self.data, self.slices = self.collate(data_list)


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(out.squeeze(-1))
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    return {
        "roc_auc":  round(roc_auc_score(all_labels, all_probs), 4),
        "f1":       round(f1_score(all_labels, all_preds), 4),
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "mcc":      round(matthews_corrcoef(all_labels, all_preds), 4),
    }


def train_attentivefp(epochs=150, lr=0.0005, batch_size=32, patience=20):
    print(f"Device: {DEVICE}")
    print("Loading datasets...")
    train_ds = HIAGraphDataset("data/raw/hia_train.csv")
    valid_ds = HIAGraphDataset("data/raw/hia_valid.csv")
    test_ds  = HIAGraphDataset("data/raw/hia_test.csv")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    num_node_features = train_ds[0].x.shape[1]
    num_edge_features = train_ds[0].edge_attr.shape[1]
    print(f"Node features: {num_node_features} | Edge features: {num_edge_features}")

    # AttentiveFP from PyTorch Geometric
    # in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout
    model = AttentiveFP(
        in_channels=num_node_features,
        hidden_channels=64,
        out_channels=1,
        edge_dim=num_edge_features,
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Class weighting
    train_labels = [int(d.y.item()) for d in train_ds]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    print("\nTraining Attentive FP...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out.squeeze(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, valid_loader)
        scheduler.step(1 - val_metrics["roc_auc"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    os.makedirs("results/metrics", exist_ok=True)
    torch.save(model.state_dict(), "results/metrics/attentivefp_model.pt")

    valid_final = evaluate(model, valid_loader)
    test_final  = evaluate(model, test_loader)

    print(f"\n{'='*50}")
    print("FINAL ATTENTIVE FP RESULTS")
    print(f"{'='*50}")
    print(f"  VALID — AUC: {valid_final['roc_auc']}  F1: {valid_final['f1']}  MCC: {valid_final['mcc']}")
    print(f"  TEST  — AUC: {test_final['roc_auc']}  F1: {test_final['f1']}  MCC: {test_final['mcc']}")

    # Append to all_model_results
    new_rows = pd.DataFrame([
        {"model": "Attentive FP", "split": "valid", **valid_final},
        {"model": "Attentive FP", "split": "test",  **test_final},
    ])[["model", "split", "roc_auc", "f1", "accuracy", "mcc"]]

    results_path = "results/metrics/all_model_results.csv"
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        # Remove old AttentiveFP rows if re-running
        existing = existing[existing["model"] != "Attentive FP"]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(results_path, index=False)
    print(f"\nUpdated results saved to {results_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    train_attentivefp(epochs=150, lr=0.0005, batch_size=32, patience=20)
