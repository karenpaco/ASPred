"""
data_utils.py
Utility functions for loading datasets and preparing batches for ASPred.
This file is cluster-agnostic and safe for public sharing.
"""

import pandas as pd
from pathlib import Path

def load_dataset(dataset_root: str, dataset_name: str):
    """
    Loads train.csv and test.csv from a dataset folder.

    Parameters
    ----------
    dataset_root : str
        Path to ./datasets
    dataset_name : str
        Name of the dataset folder (e.g., FLU_heavy)

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    dataset_path = Path(dataset_root) / dataset_name

    train_file = dataset_path / "train.csv"
    test_file = dataset_path / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} missing train.csv/test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    required_cols = ["sequence_id", "sequence_aa", "label"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"{col} missing from {dataset_name}")

    return train_df, test_df
