"""
Preprocessing utilities for Russian Toxic Comment Detection.
Handles text cleaning and train/val/test splitting.
"""

import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clean_text(text: str) -> str:
    """Clean a single text string: lowercase, remove URLs/mentions, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load CSV, drop empty comments, clean text, cast labels to int."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    df["toxic"] = df["toxic"].astype(int)
    df["clean_comment"] = df["comment"].apply(clean_text)
    df = df.reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame, output_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Saves train.csv, val.csv, test.csv."""
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO), stratify=df["toxic"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["toxic"], random_state=RANDOM_SEED
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        toxic_pct = split_df["toxic"].mean() * 100
        print(f"{name}: {len(split_df)} rows ({toxic_pct:.1f}% toxic) -> {path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", "labeled.csv")
    output_dir = os.path.join(base, "data")

    df = load_and_clean(data_path)
    print(f"Loaded {len(df)} rows ({df['toxic'].mean()*100:.1f}% toxic)")
    train, val, test = split_data(df, output_dir)
