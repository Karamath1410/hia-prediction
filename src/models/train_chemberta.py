"""
train_chemberta.py
-------------------
Fine-tunes ChemBERTa (Chithrananda et al., 2020) for HIA binary classification.
ChemBERTa is a RoBERTa-based Transformer pretrained on 77 million SMILES strings
from PubChem, making it one of the most powerful pretrained models for
molecular property prediction from SMILES.

Model used: seyonec/ChemBERTa-zinc-base-v1 (from HuggingFace)

This represents genuine state-of-the-art transfer learning for chemistry —
analogous to using BERT for NLP tasks.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class HIASmilesDataset(Dataset):
    """Dataset that tokenizes SMILES strings for ChemBERTa."""
    def __init__(self, csv_path, tokenizer, max_length=128):
        df = pd.read_csv(csv_path)
        self.smiles = df["Drug"].tolist()
        self.labels = df["Y"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds   = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    return {
        "roc_auc":  round(roc_auc_score(all_labels, all_probs), 4),
        "f1":       round(f1_score(all_labels, all_preds), 4),
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "mcc":      round(matthews_corrcoef(all_labels, all_preds), 4),
    }


def train_chemberta(epochs=20, lr=2e-5, batch_size=16, patience=5):
    print(f"Device: {DEVICE}")
    print(f"Loading ChemBERTa tokenizer and model from: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    print("Loading datasets...")
    train_ds = HIASmilesDataset("data/raw/hia_train.csv", tokenizer)
    valid_ds = HIASmilesDataset("data/raw/hia_valid.csv", tokenizer)
    test_ds  = HIASmilesDataset("data/raw/hia_test.csv",  tokenizer)

    # Class weights for imbalance
    train_labels = [d["label"].item() for d in train_ds]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    print(f"\nFine-tuning ChemBERTa ({MODEL_NAME})...")
    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, valid_loader)

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    os.makedirs("results/metrics", exist_ok=True)
    model.save_pretrained("results/metrics/chemberta_finetuned")
    tokenizer.save_pretrained("results/metrics/chemberta_finetuned")

    valid_final = evaluate(model, valid_loader)
    test_final  = evaluate(model, test_loader)

    print(f"\n{'='*50}")
    print("FINAL CHEMBERTA RESULTS")
    print(f"{'='*50}")
    print(f"  VALID — AUC: {valid_final['roc_auc']}  F1: {valid_final['f1']}  MCC: {valid_final['mcc']}")
    print(f"  TEST  — AUC: {test_final['roc_auc']}  F1: {test_final['f1']}  MCC: {test_final['mcc']}")

    # Append to all_model_results
    new_rows = pd.DataFrame([
        {"model": "ChemBERTa", "split": "valid", **valid_final},
        {"model": "ChemBERTa", "split": "test",  **test_final},
    ])[["model", "split", "roc_auc", "f1", "accuracy", "mcc"]]

    results_path = "results/metrics/all_model_results.csv"
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "ChemBERTa"]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(results_path, index=False)
    print(f"\nUpdated results saved to {results_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    train_chemberta(epochs=20, lr=2e-5, batch_size=16, patience=5)
