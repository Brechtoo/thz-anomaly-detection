from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


def create_hgb():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.03,
                max_iter=800,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            ),
        ),
    ])


def train_hgb(model, X_train, y_train):

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model.fit(X_train, y_train, model__sample_weight=sample_weight)

    return model