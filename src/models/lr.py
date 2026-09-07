from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def create_lr_grid_search(model):
    param_grid = {
        "logistic_regression__C": [0.01, 0.1, 1, 10, 100],
        "logistic_regression__class_weight": [None, "balanced"],
    }

    return GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1)