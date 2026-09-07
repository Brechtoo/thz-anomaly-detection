from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_random_forest():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])