from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_classification(y_true, y_pred):

    print("Evaluating classification results...")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }