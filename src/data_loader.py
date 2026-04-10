"""
KazSAnDRA dataset loader.
Loads polarity classification CSVs from the data/ directory.
"""

import os
import pandas as pd


def load_kazsandra(data_dir=None):
    """Load KazSAnDRA polarity classification from local CSVs.

    Returns dict with 'train', 'val', 'test' DataFrames.
    Columns: custom_id, text, text_cleaned, label, domain
    Labels: 0 = negative, 1 = positive
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    splits = {}
    for name, filename in [("train", "train_pc.csv"), ("val", "valid_pc.csv"), ("test", "test_pc.csv")]:
        df = pd.read_csv(os.path.join(data_dir, filename))
        splits[name] = df
    return splits


if __name__ == "__main__":
    splits = load_kazsandra()
    for name, df in splits.items():
        pos_pct = df["label"].mean() * 100
        print(f"{name}: {len(df):,} rows | positive: {pos_pct:.1f}% | negative: {100-pos_pct:.1f}%")
