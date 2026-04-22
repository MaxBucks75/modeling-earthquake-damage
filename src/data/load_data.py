import pandas as pd
from pathlib import Path

def load_training_data(config):
    X_path = Path(config["train_data"]["path"])
    y_path = Path(config["train_labels"]["path"])

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # merge ON building_id (this is critical)
    df = X.merge(y, on="building_id")

    return df