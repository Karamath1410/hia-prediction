"""
train_hybrid.py
---------------
Hybrid Multimodal Architecture for HIA Prediction.

Combines three complementary molecular representations using LATE FUSION:
  1. RDKit physicochemical descriptors (12 features) — global properties
  2. Attentive FP graph embedding — local atomic/bond topology
  3. ChemBERTa SMILES embedding — sequence-level chemical context

Each stream is encoded separately, the representations are concatenated,
and a final classifier head produces the prediction.

This is the NOVEL contribution proposed by the supervisor — a multimodal
framework for HIA prediction.

Author: Mohammad Karamath Fardeen (25251265)
Supervisor: Kolawole Adebayo | Maynooth University | 2025-2026
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP
from torch_geometric.loader import DataLoader as GeoLoader
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHEMBERTA_NAME = "seyonec/ChemBERTa-zinc-base-v1"

FEATURE_COLS = ["MolWt","LogP","TPSA","HBD","HBA","RotBonds",
                "RingCount","AromaticRings","HeavyAtoms",
                "FractionCSP3","MolMR","NumHeteroatoms"]

DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt, "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA, "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings, "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "HeavyAtoms": Descriptors.HeavyAtomCount, "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolMR": Descriptors.MolMR, "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
}

# ── Atom / bond features for the graph stream ─────────────────────────────────
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


class MultimodalDataset(Dataset):
    """Dataset returning all three representations for each molecule."""
    def __init__(self, csv_path, tokenizer, scaler=None, fit_scaler=False):
        df = pd.read_csv(csv_path)
        self.smiles = df["Drug"].tolist()
        self.labels = df["Y"].astype(int).tolist()
        self.tokenizer = tokenizer

        # 1. Descriptor stream
        desc = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            desc.append([fn(mol) for fn in DESCRIPTOR_FUNCTIONS.values()])
        desc = np.array(desc, dtype=np.float32)
        if fit_scaler:
            self.scaler = StandardScaler().fit(desc)
        else:
            self.scaler = scaler
        self.descriptors = self.scaler.transform(desc).astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def get_graph(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
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
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([label], dtype=torch.float))

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        # Graph
        graph = self.get_graph(smiles, label)
        # SMILES tokens
        enc = self.tokenizer(smiles, max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt")
        # Descriptors
        desc = torch.tensor(self.descriptors[idx], dtype=torch.float)
        return {
            "graph": graph,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "descriptors": desc,
            "label": torch.tensor(label, dtype=torch.float),
        }


def collate_fn(batch):
    from torch_geometric.data import Batch
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    descriptors = torch.stack([b["descriptors"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return graphs, input_ids, attention_mask, descriptors, labels


# ── Hybrid Multimodal Model ───────────────────────────────────────────────────
class HybridMultimodalModel(nn.Module):
    """
    Late-fusion multimodal model combining:
      - Descriptor MLP encoder
      - Attentive FP graph encoder
      - ChemBERTa SMILES encoder
    """
    def __init__(self, num_node_features, num_edge_features, fusion_dim=64):
        super().__init__()

        # Stream 1: Descriptor encoder (MLP)
        self.desc_encoder = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, fusion_dim), nn.ReLU()
        )

        # Stream 2: Attentive FP graph encoder (outputs fusion_dim embedding)
        self.graph_encoder = AttentiveFP(
            in_channels=num_node_features, hidden_channels=64,
            out_channels=fusion_dim, edge_dim=num_edge_features,
            num_layers=3, num_timesteps=2, dropout=0.2
        )

        # Stream 3: ChemBERTa encoder (frozen backbone + projection)
        self.chemberta = AutoModel.from_pretrained(CHEMBERTA_NAME)
        # Freeze ChemBERTa to keep training efficient
        for param in self.chemberta.parameters():
            param.requires_grad = False
        self.smiles_proj = nn.Linear(self.chemberta.config.hidden_size, fusion_dim)

        # Late fusion head: concatenate 3 x fusion_dim -> prediction
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, graph, input_ids, attention_mask, descriptors):
        # Stream 1
        desc_emb = self.desc_encoder(descriptors)
        # Stream 2
        graph_emb = self.graph_encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        # Stream 3
        with torch.no_grad():
            bert_out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]  # CLS token
        smiles_emb = F.relu(self.smiles_proj(cls_emb))

        # Late fusion: concatenate all three
        fused = torch.cat([desc_emb, graph_emb, smiles_emb], dim=1)
        out = self.fusion_head(fused)
        return out.squeeze(-1)


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for graph, input_ids, attention_mask, descriptors, labels in loader:
            graph = graph.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            descriptors = descriptors.to(DEVICE)
            out = model(graph, input_ids, attention_mask, descriptors)
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.numpy())
    return {
        "roc_auc": round(roc_auc_score(all_labels, all_probs), 4),
        "f1": round(f1_score(all_labels, all_preds), 4),
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "mcc": round(matthews_corrcoef(all_labels, all_preds), 4),
    }


def train_hybrid(epochs=50, lr=0.001, batch_size=16, patience=15):
    print(f"Device: {DEVICE}")
    print("Loading ChemBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_NAME)

    print("Building multimodal datasets (this computes descriptors + graphs)...")
    train_ds = MultimodalDataset("data/raw/hia_train.csv", tokenizer, fit_scaler=True)
    valid_ds = MultimodalDataset("data/raw/hia_valid.csv", tokenizer, scaler=train_ds.scaler)
    test_ds  = MultimodalDataset("data/raw/hia_test.csv",  tokenizer, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_graph = train_ds[0]["graph"]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1]

    model = HybridMultimodalModel(num_node_features, num_edge_features).to(DEVICE)

    # Only train non-frozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-5)

    # Class weighting
    labels = train_ds.labels
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc, best_state, patience_counter = 0.0, None, 0

    print("\nTraining Hybrid Multimodal Model (late fusion)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for graph, input_ids, attention_mask, descriptors, labels_b in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            graph = graph.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            descriptors = descriptors.to(DEVICE)
            labels_b = labels_b.to(DEVICE)

            optimizer.zero_grad()
            out = model(graph, input_ids, attention_mask, descriptors)
            loss = criterion(out, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, valid_loader)
        print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    os.makedirs("results/metrics", exist_ok=True)
    torch.save(model.state_dict(), "results/metrics/hybrid_model.pt")

    test_metrics = evaluate(model, test_loader)
    print(f"\n{'='*50}")
    print("HYBRID MULTIMODAL MODEL — FINAL TEST RESULTS")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Append to results
    new_row = pd.DataFrame([
        {"model": "Hybrid (Multimodal)", "split": "test", **test_metrics}
    ])[["model", "split", "roc_auc", "f1", "accuracy", "mcc"]]

    results_path = "results/metrics/all_model_results.csv"
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "Hybrid (Multimodal)"]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    train_hybrid(epochs=50, lr=0.001, batch_size=16, patience=15)
