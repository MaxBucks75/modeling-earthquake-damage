from sklearn.model_selection import train_test_split

def split_data(X, y, config):
    return train_test_split(
        X,
        y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )