from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train, config):
    """
    Train logistic regression with hyperparameter tuning using GridSearchCV
    """

    print("Training logistic regression with GridSearchCV...")

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000]
    }

    model = LogisticRegression()

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_