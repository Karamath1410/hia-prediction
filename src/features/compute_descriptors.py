"""
compute_descriptors.py
----------------------
Compute RDKit molecular descriptors from SMILES strings.
Saves processed feature matrices to data/processed/.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

SEED = 42

# ── Descriptor definitions ────────────────────────────────────────────────────
DESCRIPTOR_FUNCTIONS = {
    "MolWt":        Descriptors.MolWt,
    "LogP":         Descriptors.MolLogP,
    "TPSA":         Descriptors.TPSA,
    "HBD":          rdMolDescriptors.CalcNumHBD,       # H-bond donors
    "HBA":          rdMolDescriptors.CalcNumHBA,       # H-bond acceptors
    "RotBonds":     rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount":    rdMolDescriptors.CalcNumRings,
    "AromaticRings":rdMolDescriptors.CalcNumAromaticRings,
    "HeavyAtoms":   Descriptors.HeavyAtomCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolMR":        Descriptors.MolMR,                 # Molar refractivity
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
}


def smiles_to_descriptors(smiles: str) -> dict | None:
    """
    Compute molecular descriptors for a single SMILES string.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}
    for name, fn in DESCRIPTOR_FUNCTIONS.items():
        try:
            features[name] = fn(mol)
        except Exception:
            features[name] = np.nan
    return features


def compute_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptors for all molecules in a DataFrame.
    Drops rows where SMILES could not be parsed.

    Parameters
    ----------
    df : DataFrame with columns ['Drug_ID', 'Drug', 'Y']
         'Drug' column contains SMILES strings

    Returns
    -------
    DataFrame with descriptor columns + 'Y' label column
    """
    records = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing descriptors"):
        feats = smiles_to_descriptors(row["Drug"])
        if feats is None:
            failed += 1
            continue
        feats["Drug_ID"] = row["Drug_ID"]
        feats["SMILES"]  = row["Drug"]
        feats["Y"]       = int(row["Y"])
        records.append(feats)

    print(f"Processed: {len(records)} | Failed (invalid SMILES): {failed}")
    return pd.DataFrame(records)


def process_and_save():
    """Load raw splits, compute descriptors, save processed CSVs."""
    for split in ["train", "valid", "test"]:
        raw_path = f"data/raw/hia_{split}.csv"
        out_path = f"data/processed/hia_{split}_features.csv"

        df_raw = pd.read_csv(raw_path)
        df_feat = compute_features_for_df(df_raw)
        df_feat.to_csv(out_path, index=False)
        print(f"Saved: {out_path}  ({len(df_feat)} rows, {len(df_feat.columns)} cols)")


if __name__ == "__main__":
    process_and_save()
