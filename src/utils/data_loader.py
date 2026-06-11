"""
data_loader.py
--------------
Load the HIA_Hou dataset from Therapeutics Data Commons (TDC)
and return train/validation/test splits.
"""

from tdc.single_pred import ADME
import pandas as pd

SEED = 42


def load_hia_dataset(save_raw: bool = True) -> dict:
    """
    Download and return HIA_Hou dataset splits from TDC.

    Returns
    -------
    dict with keys: 'train', 'valid', 'test'
    Each value is a pandas DataFrame with columns: ['Drug_ID', 'Drug', 'Y']
        Drug  : SMILES string
        Y     : Binary label (1 = high absorption, 0 = low absorption)
    """
    data = ADME(name="HIA_Hou")
    split = data.get_split(method="scaffold", seed=SEED)

    train_df = split["train"]
    valid_df = split["valid"]
    test_df  = split["test"]

    if save_raw:
        train_df.to_csv("data/raw/hia_train.csv", index=False)
        valid_df.to_csv("data/raw/hia_valid.csv", index=False)
        test_df.to_csv("data/raw/hia_test.csv",  index=False)
        print(f"Raw data saved to data/raw/")

    print(f"Train size : {len(train_df)}")
    print(f"Valid size : {len(valid_df)}")
    print(f"Test  size : {len(test_df)}")
    print(f"Label distribution (train):\n{train_df['Y'].value_counts()}")

    return {"train": train_df, "valid": valid_df, "test": test_df}


if __name__ == "__main__":
    splits = load_hia_dataset(save_raw=True)
