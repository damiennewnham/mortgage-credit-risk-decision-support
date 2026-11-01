import pandas as pd
from .config import FEATURES, TARGET

def load_dataset(path):
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df

def split_inputs(df):
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()
