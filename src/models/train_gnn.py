"""
train_gnn.py
------------
Train a Message Passing Neural Network (MPNN) / Graph Convolutional Network (GCN)
for HIA binary classification, using molecular graphs built directly from SMILES.

Atoms = nodes (features: atomic number, degree, formal charge, hybridisation,
                aromaticity, H count)
Bonds = edges (features: bond type, conjugation, ring membership)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Atom featurisation ────────────────────────────────────────────────────────
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H', 'Other']

def atom_features(atom):
    symbol = atom.GetSymbol()
    symbol_enc = [1 if symbol == s else 0 for s in ATOM_LIST[:-1]]
    symbol_enc.append(1 if symbol not in ATOM_LIST[:-1] else 0)  # 'Other'

    features = symbol_enc + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]
    return features


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


def smiles_to_graph(smiles: str, label: int):
    """Convert a SMILES string into a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge index + edge features (undirected: add both directions)
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_feats += [feat, feat]

    if len(edge_indices) == 0:  # single-atom molecule edge case
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class HIAGraphDataset(InMemoryDataset):
    """In-memory PyG dataset built from a TDC-style CSV (Drug_ID, Drug=SMILES, Y)."""
    def __init__(self, csv_path):
        super().__init__(None)
        df = pd.read_csv(csv_path)
        data_list = []
        skipped = 0
        for _, row in df.iterrows():
            g = smiles_to_graph(row["Drug"], int(row["Y"]))
            if g is not None:
                data_list.append(g)
            else:
                skipped += 1
        print(f"  Loaded {len(data_list)} graphs from {csv_path} (skipped {skipped} invalid SMILES)")
        self.data, self.slices = self.collate(data_list)


# ── GCN Model ──────────────────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)  # graph-level readout

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x.squeeze(-1)


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits)
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


def train_gnn(epochs=100, lr=0.001, batch_size=32, patience=15):
    print(f"Using device: {DEVICE}")

    print("Loading graph datasets...")
    train_ds = HIAGraphDataset("data/raw/hia_train.csv")
    valid_ds = HIAGraphDataset("data/raw/hia_valid.csv")
    test_ds  = HIAGraphDataset("data/raw/hia_test.csv")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    num_node_features = train_ds[0].x.shape[1]
    print(f"Node feature dimension: {num_node_features}")

    model = GCN(num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Class weighting for imbalance (positive class weight)
    train_labels = [int(d.y.item()) for d in train_ds]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    print("\nTraining GCN...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, valid_loader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model
    model.load_state_dict(best_state)
    os.makedirs("results/metrics", exist_ok=True)
    torch.save(model.state_dict(), "results/metrics/gcn_model.pt")

    valid_final = evaluate(model, valid_loader)
    test_final  = evaluate(model, test_loader)

    print(f"\n{'='*50}")
    print("FINAL GCN RESULTS")
    print(f"{'='*50}")
    print(f"  VALID — AUC: {valid_final['roc_auc']}  F1: {valid_final['f1']}  MCC: {valid_final['mcc']}")
    print(f"  TEST  — AUC: {test_final['roc_auc']}  F1: {test_final['f1']}  MCC: {test_final['mcc']}")

    # Save results
    results_df = pd.DataFrame([
        {"model": "GCN (GNN)", "split": "valid", **valid_final},
        {"model": "GCN (GNN)", "split": "test",  **test_final},
    ])[["model", "split", "roc_auc", "f1", "accuracy", "mcc"]]

    # Append to classical results if it exists
    classical_path = "results/metrics/classical_results.csv"
    if os.path.exists(classical_path):
        classical_df = pd.read_csv(classical_path)
        combined = pd.concat([classical_df, results_df], ignore_index=True)
    else:
        combined = results_df

    combined.to_csv("results/metrics/all_model_results.csv", index=False)
    print(f"\nAll results (classical + GNN) saved to results/metrics/all_model_results.csv")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    train_gnn(epochs=100, lr=0.001, batch_size=32, patience=15)
