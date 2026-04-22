from src.data.load_data import load_training_data
from src.preprocessing.preprocessing import preprocess_classification
from src.models.logistic_regression import train_logistic_regression
from src.evaluation.metrics import evaluate_classification
from sklearn.model_selection import train_test_split


config = {
    "train_data": {
        "path": "data/raw/train_values.csv"
    },
    "train_labels": {
        "path": "data/raw/train_labels.csv"
    },
    "target": "damage_grade",
    "train": {
        "test_size": 0.2,
        "random_state": 42
    }
}


def run_experiment_log_regression():
    print("Running logistic regression experiment...")

    # 1. Load data
    df = load_training_data(config)

    # 2. Preprocess
    X, y = preprocess_classification(df, config)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )

    # 4. Train model with CV
    model, best_params = train_logistic_regression(X_train, y_train, config)

    print("Best Params:", best_params)

    # 5. Predict
    preds = model.predict(X_test)

    # 6. Evaluate
    results = evaluate_classification(y_test, preds)

    print("Results:")
    print(f"Accuracy: {results['accuracy']}")
    print("Confusion Matrix:")
    for row in results["confusion_matrix"]:
        print(row)

    return results
