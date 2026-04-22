import pandas as pd

def preprocess_classification(df, config):
    target = config["target"]   # ✅ FIXED (not train_labels!)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {df.columns}")

    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    return X, y